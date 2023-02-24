import torch
import torch.nn as nn
from models.Final_model_2d_detect import H_Net
from utils import *
import random
import numpy as np
from tqdm import tqdm
import time
import cv2

from torchvision import datasets, transforms
import matplotlib.pyplot as plt

############################################################################################################################################
import torch
import torch.nn as nn
import heapq
import numpy as np
import cv2
import time

def calculate_edge_mask(mask):
    boundry_mask = np.zeros_like(mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)#[1][0]
    cv2.drawContours(boundry_mask, contours, -1, (255,255,255), 30)

    final_boundry = mask*boundry_mask
    return final_boundry

def Calculate_Affinity_GT(mask):
    height, width = mask.shape
    affinity_mask = torch.zeros(4, height, width)
    
    for i in range(height):
        for j in range(width):
            up = mask[i - 1, j]
            right = mask[i, j + 1]
            down = mask[i + 1, j]
            left = mask[i, j - 1]
            
            current = mask[i, j]
            
            if current == up:
                affinity_mask[0, i, j] = 1
                
            if current == right:
                affinity_mask[1, i, j] = 1
                
            if current == down:
                affinity_mask[2, i, j] = 1
                
            if current == left:
                affinity_mask[3, i, j] = 1
    
    return affinity_mask


class Partition:

    def __init__(self, nodes):
        
        self.nodes = nodes
        self.links = {}
        self.merged = False

    def __repr__(self):
        return 'Partition(nodes={})'.format(self.nodes)

    def connect(self, edge):
        
        pair = edge.get_pair(self)
        self.links[id(pair)] = (edge, pair)

    def disconnect(self, edge):
        
        pair = edge.get_pair(self)
        self.links.pop(id(pair))

    def merge_nodes(self, partition):
        
        self.nodes += partition.nodes
        partition.nodes = []
        partition.merged = True

    def get_edge(self, partition):
        
        if id(partition) not in self.links:
            return None
        return self.links[id(partition)][0]


class Edge:

    def __init__(self, partition0, partition1, weight):
        
        self.pair = (partition0, partition1)
        self.weight = weight
        self.removed = False
        partition0.connect(self)
        partition1.connect(self)

    def __lt__(self, edge):
        # priority que のため逆にしている
        return self.weight > edge.weight

    def __repr__(self):
        return 'Edge({}<=>{})'.format(*self.pair)

    def get_pair(self, partition):
        
        if partition is self.pair[0]:
            return self.pair[1]
        else:
            return self.pair[0]

    def remove(self):
        
        partition0, partition1 = self.pair
        partition0.disconnect(self)
        partition1.disconnect(self)
        self.removed = True

    def contract(self):
        partition0, partition1 = self.pair
        new_edge = []

        # edgeが多くつながっているNodeを元Nodeにする.
        if len(partition1.links) > len(partition0.links):
            partition0, partition1 = partition1, partition0

        for edge12, partition2 in list(partition1.links.values()):
            # すでにつながっている場合は何もしない.
            if partition0 is partition2:
                continue
            edge02 = partition0.get_edge(partition2)
            # つながっていない場合は新しくEdgeを作成.
            if edge02 is None:
                edge02 = Edge(partition0, partition2, 0)
                new_edge.append(edge02)

            # それぞれのEdgeの重みを更新.
            edge02.weight += edge12.weight
            edge12.remove()

        partition0.merge_nodes(partition1)
        self.remove()

        return new_edge


def greedy_additive(edges, partitions):

    heapq.heapify(edges)

    while edges:
        edge = heapq.heappop(edges)

        if edge.removed:
            continue

        # 全てのedgeの重みが0以下になったら終了.
        if edge.weight < 0:
            heapq.heappush(edges, edge)
            break

        new_edges = edge.contract()

        for new_edge in new_edges:
            heapq.heappush(edges, new_edge)

    # 結合して不要になったedgeとpartitionを取り除く.
    edges = list(filter(lambda e: not e.removed, edges))
    partitions = list(filter(lambda p: not p.merged, partitions))

    return edges, partitions


def calc_js_div(p_, q_):
    
    p_q_ = (p_+q_)/2+1e-5
    kl_1 = np.sum(p_ * np.log(p_/p_q_ + 1e-5))
    kl_2 = np.sum(q_ * np.log(q_/p_q_ + 1e-5))
    js_d = 0.5 * (kl_1 + kl_2)
    refine = np.exp(-js_d)
    refine = np.clip(refine, 0, 1)

    return refine

def make_ins(p, det_segment, aff, cls_segment):
    ins = np.zeros((det_segment.shape[0],
                    aff.shape[1], aff.shape[2], 3), dtype=int)
    ins_list = [[] for i in range(det_segment.shape[0])]
    pre_color = []
    print('len p: ', len(p))
    for area in p:
        pos = sorted(list(area.nodes))
        if len(pos) < 5:
            continue
        cls_value = np.array([0 for i in range(det_segment.shape[0])])
        # avoid generating the same color
        while(True):
            color = np.random.randint(1, 255, 3)
            if not [i for i in range(len(pre_color))
                    if np.sum(pre_color[i] == color) == 3]:
                pre_color.append(color)
                break
            
        for i in pos:
            cls_value[cls_segment[i[0], i[1]]] += 1
        cls_num = np.argmax(cls_value)
        
        ins_list[cls_num].append(pos)
        for i in pos:
            ins[cls_num, i[0], i[1]] = color
        
        return ins

def make_ins_seg(outputs, b=0, st_for=0, en_for=5, min_size=5):
    
    p = []
    p_list = []
    e = []
    mids = []
    
    # Indicates the position where the data of the previous layer is present (0 is the data of the previous layer).
    pre_detect = np.ones((1, 1))
    
    start_time = time.time()
    for mag in range(st_for, en_for):
        det_segment = outputs[mag].cpu().detach().numpy()[b]
        #print('det shape: ', det_segment.shape)
        back = 0#det_segment.shape[0] - 1
        cls_segment = np.argmax(det_segment, axis=0)
        foreground = np.where(cls_segment != back, 1, 0)
        aff = outputs[mag+5].cpu().detach().numpy()[b] # ====================== change 4 to 5
        
        # The number of Instances inherited from the previous layer.
        pre_node = len(p_list)
        # Find new Nodes only from positions where there is no data in the previous layer.
        print('foreground: ', foreground.shape)
        print('pre_detect: ', pre_detect.shape)
        foreground = foreground * pre_detect
        
        # Coordinates above the segmentation threshold are treated as nodes.
        #print(aff.shape)
        for i in range(aff.shape[1]):
            for j in range(aff.shape[2]):
                if foreground[i, j] == 1:
                    p.append(Partition([(i, j)]))
                    p_list.append([(i, j)])
                    
        # Create an Edge between new Nodes.
        for i in range(pre_node, len(p)):
            for j in range(i+1, len(p)):
                i_y, i_x = p_list[i][0]
                j_y, j_x = p_list[j][0]
                sub_y = j_y - i_y
                sub_x = j_x - i_x
                if (sub_y <= 2 and sub_x >= -2 and sub_x <= 2):
                    # Segmentation Refinement
                    refine = calc_js_div(det_segment[:, i_y, i_x],
                                         det_segment[:, j_y, j_x])
                    ind = 12+sub_x+sub_y*5
                    aff_a = aff[ind, i_y, i_x]
                    aff_b = aff[24-ind, j_y, j_x]
                    aff_ = (aff_a+aff_b)/2
                    aff_ = aff_ * refine
                    aff_ = np.log((aff_+1e-5)/(1-aff_+1e-5))
                    
                    e.append(Edge(p[i], p[j], aff_))
        end_time = time.time()
        #print(mag, ' time: ', end_time-start_time)
                    
        # Create a new Node and an Edge of the previous Node.
        for j in range(pre_node, len(p)):
            for i in range(0, pre_node):
                # An indicator if there is already an Edge between two Nodes.
                flag = False
                for pre in p_list[i]:
                    i_y, i_x = pre
                    j_y, j_x = p_list[j][0]
                    sub_y = j_y - i_y
                    sub_x = j_x - i_x
                    if not (sub_y <= 2 and sub_y >= -2
                            and sub_x >= -2 and sub_x <= 2):
                        continue
                    # Segmentation Refinement
                    refine = calc_js_div(det_segment[:, i_y, i_x],
                                         det_segment[:, j_y, j_x])

                    ind = 12+sub_x+sub_y*5
                    aff_a = aff[ind, i_y, i_x]
                    aff_b = aff[24-ind, j_y, j_x]
                    aff_ = (aff_a+aff_b)/2
                    aff_ = aff_ * refine
                    aff_ = np.log((aff_+1e-5)/(1-aff_+1e-5))
                    if flag is False:
                        e.append(Edge(p[i], p[j], aff_))
                        flag = True
                    else:
                        e[-1].weight += aff_
                        
        e, p = greedy_additive(e, p)
        #print('len p: ', len(p))
        mid_ins = make_ins(p, det_segment, aff, cls_segment)
        #('mid_ins shape: ', mid_ins.shape)
        mids.append(mid_ins)
        # Decide which Node to pass to the next layer except for the final layer.
        if mag != en_for-1:
            pre_detect = np.ones((aff.shape[1], aff.shape[2]))
            p_list = []
            for i in range(len(p)):
                area_l = p[i].nodes
                
                # Extract only nodes whose top, bottom, left, and right graphs are the same.
                area_l = [area_l[i] for i in range(len(area_l))
                          if (((area_l[i][0], area_l[i][1]+1) in area_l)
                              and ((area_l[i][0]+1, area_l[i][1]) in area_l)
                              and ((area_l[i][0]-1, area_l[i][1]) in area_l)
                              and ((area_l[i][0], area_l[i][1]-1) in area_l))]
                p_ = []
                # The size of the graph is doubled vertically and horizontally for the next layer..
                for area_ in area_l:
                    pre_detect[area_[0], area_[1]] = 0
                    p_.append((area_[0]*2, area_[1]*2))
                    p_.append((area_[0]*2+1, area_[1]*2))
                    p_.append((area_[0]*2, area_[1]*2+1))
                    p_.append((area_[0]*2+1, area_[1]*2+1))
                    
                p[i].nodes = p_
                if p_:
                    p_list.append(p_)
                    
            # Delete Nodes and Edges that are not inherited to the next layer.
            p = [p[i] for i in range(len(p)) if p[i].nodes]
            e = [e[i] for i in range(len(e))
                 if ((e[i].pair[0].nodes) and (e[i].pair[1].nodes))]
            
            pre_detect = pre_detect.repeat(2, axis=0).repeat(2, axis=1)
            
            
            
    # Create instance segmentation image and node list.
    ins = np.zeros((det_segment.shape[0],
                    aff.shape[1], aff.shape[2], 3), dtype=int)
    ins_list = [[] for i in range(det_segment.shape[0])]
    pre_color = []
    #print('len p: ', len(p))
    for area in p:
        pos = sorted(list(area.nodes))
        if len(pos) < min_size:
            continue
        cls_value = np.array([0 for i in range(det_segment.shape[0])])
        # avoid generating the same color
        while(True):
            color = np.random.randint(1, 255, 3)
            if not [i for i in range(len(pre_color))
                    if np.sum(pre_color[i] == color) == 3]:
                pre_color.append(color)
                break
            
        for i in pos:
            cls_value[cls_segment[i[0], i[1]]] += 1
        cls_num = np.argmax(cls_value)
        
        ins_list[cls_num].append(pos)
        for i in pos:
            ins[cls_num, i[0], i[1]] = color
            
    ins = ins.repeat(2**(5-en_for), axis=1).repeat(2**(5-en_for), axis=2)
    
    return ins, ins_list, mids
  

def make_ins_seg_new(outputs, b=0, st_for=0, en_for=5, min_size=5):
    
    p = []
    p_list = []
    e = []
    mids = []
    
    # Indicates the position where the data of the previous layer is present (0 is the data of the previous layer).
    pre_detect = np.ones((1, 1))
    
    start_time = time.time()
    for mag in range(st_for, en_for):
        det_segment = outputs[mag].cpu().detach().numpy()[b]
        #print('det shape: ', det_segment.shape)
        back = 0#det_segment.shape[0] - 1
        cls_segment = np.argmax(det_segment, axis=0)
        foreground = np.where(cls_segment != back, 1, 0)
        aff = outputs[mag+5].cpu().detach().numpy()[b] # ====================== change 4 to 5
        
        # The number of Instances inherited from the previous layer.
        pre_node = len(p_list)
        # Find new Nodes only from positions where there is no data in the previous layer.
        print('foreground: ', foreground.shape)
        print('pre_detect: ', pre_detect.shape)
        foreground = foreground * pre_detect
        
        # Coordinates above the segmentation threshold are treated as nodes.
        #print(aff.shape)
        for i in range(aff.shape[1]):
            for j in range(aff.shape[2]):
                if foreground[i, j] == 1:
                    p.append(Partition([(i, j)]))
                    p_list.append([(i, j)])
                    
        # Create an Edge between new Nodes.
        for i in range(pre_node, len(p)):
            for j in range(i+1, len(p)):
                i_y, i_x = p_list[i][0]
                j_y, j_x = p_list[j][0]
                sub_y = j_y - i_y
                sub_x = j_x - i_x
                if (sub_y <= 2 and sub_x >= -2 and sub_x <= 2):
                    # Segmentation Refinement
                    refine = calc_js_div(det_segment[:, i_y, i_x],
                                         det_segment[:, j_y, j_x])
                    ind = 12+sub_x+sub_y*5
                    aff_a = aff[ind, i_y, i_x]
                    aff_b = aff[24-ind, j_y, j_x]
                    aff_ = (aff_a+aff_b)/2
                    aff_ = aff_ * refine
                    aff_ = np.log((aff_+1e-5)/(1-aff_+1e-5))
                    
                    e.append(Edge(p[i], p[j], aff_))
        end_time = time.time()
        #print(mag, ' time: ', end_time-start_time)
        
                        
        e, p = greedy_additive(e, p)
        #print('len p: ', len(p))
        mid_ins = make_ins(p, det_segment, aff, cls_segment)
        #('mid_ins shape: ', mid_ins.shape)
        mids.append(mid_ins)
        # Decide which Node to pass to the next layer except for the final layer.
        if mag != en_for-1:
            pre_detect = np.ones((aff.shape[1], aff.shape[2]))
            p_list = []
            for i in range(len(p)):
                area_l = p[i].nodes
                
                # Extract only nodes whose top, bottom, left, and right graphs are the same.
                area_l = [area_l[i] for i in range(len(area_l))
                          if (((area_l[i][0], area_l[i][1]+1) in area_l)
                              and ((area_l[i][0]+1, area_l[i][1]) in area_l)
                              and ((area_l[i][0]-1, area_l[i][1]) in area_l)
                              and ((area_l[i][0], area_l[i][1]-1) in area_l))]
                p_ = []
                # The size of the graph is doubled vertically and horizontally for the next layer..
                for area_ in area_l:
                    pre_detect[area_[0], area_[1]] = 0
                    p_.append((area_[0]*2, area_[1]*2))
                    p_.append((area_[0]*2+1, area_[1]*2))
                    p_.append((area_[0]*2, area_[1]*2+1))
                    p_.append((area_[0]*2+1, area_[1]*2+1))
                    
                p[i].nodes = p_
                if p_:
                    p_list.append(p_)
                    
            # Delete Nodes and Edges that are not inherited to the next layer.
            p = [p[i] for i in range(len(p)) if p[i].nodes]
            e = [e[i] for i in range(len(e))
                 if ((e[i].pair[0].nodes) and (e[i].pair[1].nodes))]
            
            pre_detect = pre_detect.repeat(2, axis=0).repeat(2, axis=1)
            
            
            
    # Create instance segmentation image and node list.
    ins = np.zeros((det_segment.shape[0],
                    aff.shape[1], aff.shape[2], 3), dtype=int)
    ins_list = [[] for i in range(det_segment.shape[0])]
    pre_color = []
    #print('len p: ', len(p))
    for area in p:
        pos = sorted(list(area.nodes))
        if len(pos) < min_size:
            continue
        cls_value = np.array([0 for i in range(det_segment.shape[0])])
        # avoid generating the same color
        while(True):
            color = np.random.randint(1, 255, 3)
            if not [i for i in range(len(pre_color))
                    if np.sum(pre_color[i] == color) == 3]:
                pre_color.append(color)
                break
            
        for i in pos:
            cls_value[cls_segment[i[0], i[1]]] += 1
        cls_num = np.argmax(cls_value)
        
        ins_list[cls_num].append(pos)
        for i in pos:
            ins[cls_num, i[0], i[1]] = color
            
    ins = ins.repeat(2**(5-en_for), axis=1).repeat(2**(5-en_for), axis=2)
    
    return ins, ins_list, mids


def teeth_class(ind):
    if ind == 0:
        tclass = 18
    if ind == 1:
        tclass = 17
    if ind == 2:
        tclass = 16
    if ind == 3:
        tclass = 15
    if ind == 4:
        tclass = 14
    if ind == 5:
        tclass = 13
    if ind == 6:
        tclass = 12
    if ind == 7:
        tclass = 11
    if ind == 8:
        tclass = 21
    if ind == 9:
        tclass = 22
    if ind == 10:
        tclass = 23
    if ind == 11:
        tclass = 24
    if ind == 12:
        tclass = 25
    if ind == 13:
        tclass = 26
    if ind == 14:
        tclass = 27
    if ind == 15:
        tclass = 28
    if ind == 16:
        tclass = 38
    if ind == 17:
        tclass = 37
    if ind == 18:
        tclass = 36
    if ind == 19:
        tclass = 35
    if ind == 20:
        tclass = 34
    if ind == 21:
        tclass = 33
    if ind == 22:
        tclass = 32
    if ind == 23:
        tclass = 31
    if ind == 24:
        tclass = 41
    if ind == 25:
        tclass = 42
    if ind == 26:
        tclass = 43
    if ind == 27:
        tclass = 44
    if ind == 28:
        tclass = 45
    if ind == 29:
        tclass = 46
    if ind == 30:
        tclass = 47
    if ind == 31:
        tclass = 48
    
    return tclass


def post_process_multi_channel_numbering(channeled_input):
    
    print('full channeled unique: ', np.unique(channeled_input))
    unique_vals = np.unique(channeled_input)
    
    for i in unique_vals:
        mask = np.copy(channeled_input)
        print('mask unique: ', np.unique(mask))
        mask[mask != i] = 0
        mask[mask == i] = 255
        
        plt.imshow(mask, cmap = 'gray')
        plt.show()

############################################################################################################################################

device = torch.device('cuda:0')

model = H_Net(in_channels=1, num_classes=2, image_size=256).to(device)
model_path = "C:\\Users\\Sharjeel\\Desktop\\codes\\teeth_segmentation_saves\\weight_saves\\experiment_6_numbering_with_onehot_labels_second\\Model0.9554452896118164.pth"
model.load_state_dict(torch.load(model_path))

model.eval()

#data_path = 'C:\\Users\\Sharjeel\\Desktop\\datasets\\Raw_data_array.npy'
#data_path = 'C:\\Users\\Sharjeel\\Desktop\\codes\\teeth_numbered_data\\Total_numbered_data.npy'
data_path = 'C:\\Users\\Sharjeel\\Desktop\\codes\\teeth_numbered_data\\Data_with_bbox.npy'
data = np.load(data_path, allow_pickle = True)
print('Data shape: ', data.shape)

dlen = len(data) - 20
data = data[dlen:len(data)]

#image, mask, _ = data[2]
#image, img_t, _, img_t_aff, weights, _ = data[0]
image, img_t, _, img_t_aff, weights, all_teeths, _ = data[8] # problematic 18


orig_image_copy = np.copy(image)

plt.imshow(image)
plt.show()

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = (image/np.max(image))



transform = transforms.ToTensor()

image = transform(image)
image = torch.unsqueeze(image, dim = 0).to(device)
print('image tensor shape: ', image.shape)

output, out_instance, out_detection = model(image.float())



dot1 = torch.zeros([1, 2, 16, 16]).to(device)
dins1 = torch.zeros([1, 25, 16, 16]).to(device)

dot2 = torch.zeros([1, 2, 32, 32]).to(device)
dins2 = torch.zeros([1, 25, 32, 32]).to(device)

dot3 = torch.zeros([1, 2, 64, 64]).to(device)
dins3 = torch.zeros([1, 25, 64, 64]).to(device)

dot4 = torch.zeros([1, 2, 128, 128]).to(device)
dins4 = torch.zeros([1, 25, 128, 128]).to(device)

tot_outs = [dot1, dot2, dot3, dot4, output, dins1, dins2, dins3, dins4, out_instance]

st_for = 4
en_for = 5
min_size = 4

ins, ins_list, mids = make_ins_seg(tot_outs, st_for=st_for,
                                   en_for=en_for, min_size=min_size)


instance_pred = ins[1]
plt.imshow(ins[1])
plt.show()

flat_instance = ins[1].reshape(256*256, 3)
un = np.unique(flat_instance, axis = 0)
print('instance unique len: ', len(un))

output = torch.argmax(output, dim = 1)
output = torch.squeeze(output, dim = 0)
print('output shape: ', output.shape)

output = output.detach().cpu().numpy()
plt.imshow(output)
plt.show()


out_detection = torch.argmax(out_detection, dim = 1)

out_detection = out_detection.detach().cpu().numpy()
print('Out detection shape: ', out_detection.shape)
out_detection = np.squeeze(out_detection)
print('Out detection shape: ', out_detection.shape)

orig_image_copy = np.array(orig_image_copy*255,  dtype=np.uint8)

alpha = 0.5
cv2.addWeighted(ins[1].astype(np.uint8), alpha, orig_image_copy, 1 - alpha, 0, orig_image_copy)

post_process_multi_channel_numbering(out_detection)

# for i in range(0, len(out_detection), 5):
    
#     vals = out_detection[i:i+5]
#     print('In vals shape: ', vals.shape)
    
#     conf, x, y, w, h = vals
#     x = int(x*256)
#     y = int(y*256)
#     w = int(w*256)
#     h = int(h*256)
    
#     print('all vals: ', conf, x, y, w, h)
    
#     cv2.rectangle(orig_image_copy, (x, y), (x+w, y+h), (0, 0, 0), 1)


# from scipy import stats as st

# #orig_image_copy = cv2.resize(orig_image_copy, (320, 320))
# i = 0
# for k, d in enumerate(out_detection):
#     conf, x, y, w, h = d
    
#     if conf > 0.5:
    
#         x = int(x*256)
#         y = int(y*256)
#         w = int(w*256)
#         h = int(h*256)
        
#         cx = x + (w/2)
#         cy = y + (h/2)
        
#         tc = teeth_class(k)
        
#         font = cv2.FONT_HERSHEY_PLAIN 
#         if i == 0:
#             org = (int(cx), int(cy))
#             i = 1
#         else:
#             org = (int(cx), int(cy+10))
#             i = 0
    
#         fontScale = 0.5
    
#         color = (0, 0, 0)
    
#         thickness = 1
           
#         # Using cv2.putText() method
#         #image = cv2.putText(orig_image_copy, str(tc), org, font, fontScale, color, thickness, cv2.LINE_AA)
#         img_teeth_crop = orig_image_copy[y:y+h, x:x+w, :]
#         plt.imshow(img_teeth_crop)
#         plt.show()
#         sx, sy, sc = img_teeth_crop.shape
        
        
#         instance_copy = np.copy(instance_pred)
#         instance_crop = instance_copy[y:y+h, x:x+w, :]
        
#         instance_crop = instance_crop.reshape(int(sx*sy), 3)
#         mode_colour = st.mode(instance_crop, axis = 0)
#         un = np.unique(instance_crop, axis = 0)
        
#         mode_val = np.squeeze(mode_colour[0])
        
#         mr, mg, mb = mode_val
        
#         if mr != 0 and mg != 0 and mb != 0:
#             lower = np.array([mr - 1, mg - 1, mb - 1])
#             upper = np.array([mr + 1, mg + 1, mb + 1])
#             masked = cv2.inRange(instance_copy, lower, upper)
            
#             contours, _ = cv2.findContours(masked, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#             for c in contours:
#                 rect = cv2.boundingRect(c)
#                 x, y, w, h = rect
#                 cv2.rectangle(orig_image_copy, (x, y), (x+w, y+h), (int(0), int(0), int(0)), 1)
                
#                 M = cv2.moments(c)
#                 cX = int(M["m10"] / M["m00"])
#                 cY = int(M["m01"] / M["m00"])
                
#                 org = (int(cX), int(cY))
#                 #cv2.putText(orig_image_copy, str(tc), org, font, fontScale, color, thickness, cv2.LINE_AA)
        
#         #print('mode colour: ', mode_val)
#         #print('Crop unique: ', un)
#         #print('\n')
    
#         #cv2.rectangle(orig_image_copy, (x, y), (x+w, y+h), (0, 0, 0), 1)
    

plt.imshow(orig_image_copy)
plt.show()
