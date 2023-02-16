import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed

def re_instance(image, mask):
    all_colours = [
    [51, 0, 0],
    [51, 25, 0],
    [51, 51, 0],
    [0, 51, 0],
    [0, 51, 51],
    [0, 0, 51],
    [51, 0, 51],
    [102, 0, 0],
    [102, 51, 0],
    [102, 102, 0],
    [0, 102, 0],
    [0, 102, 102],
    [0, 51, 102],
    [102, 0, 102],
    [102, 0, 51],
    [102, 102, 255],
    [204, 0, 0],
    [204, 102, 0],
    [204, 204, 0],
    [102, 204, 0],
    [0, 204, 0],
    [0, 204, 102],
    [0, 204, 204],
    [255, 51, 51],
    [255, 153, 51],
    [255, 255, 51],
    [153, 255, 51],
    [102, 255, 102],
    [102, 255, 255],
    [0, 0, 102],
    [102, 178, 255],
    [178, 102, 255],
    [255, 102, 178],
    [178, 255, 102]]
    
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)#[1][0]
    instance_mask = np.zeros_like(image)
    
    k = 0
    for i, contour in enumerate(contours):
        if len(contour) > 200:
            cv2.drawContours(instance_mask, [contour], -1, color=[255, 0, 0], thickness=-1)
            k = k + 1
    
    
    rch = instance_mask[:, :, 0]
    fig, ax = plt.subplots(1)
    ax.imshow(rch)
    ax.text(5, 5, 'r channel', bbox={'facecolor': 'white', 'pad': 10})
    plt.show()

    kernel = np.ones((2, 2), np.uint8)
    img_erosion = cv2.erode(rch, kernel, iterations=1)
    
    modified_semantic = np.where(rch == 255, img_erosion, mask)#mask# - erroded
    
    fig, ax = plt.subplots(1)
    ax.imshow(img_erosion)
    ax.text(5, 5, 'erroded_semantic', bbox={'facecolor': 'white', 'pad': 10})
    plt.show()
    
     ########### re instance
    _, contours, hierarchy = cv2.findContours(modified_semantic, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)#[1][0]
    instance_mask = np.zeros_like(image)
    
    k = 0
    for i, contour in enumerate(contours):
        #print(len(contour))
        if len(contour) > 80:
            #print(i)
            cv2.drawContours(instance_mask, [contour], -1, color=all_colours[k], thickness=-1)
            #cv2.drawContours(instance_mask, [contour], -1, color=[255, 0, 0], thickness=0)
            k = k + 1
    
    fig, ax = plt.subplots(1)
    ax.imshow(instance_mask)
    ax.text(5, 5, 'ReInstance mask', bbox={'facecolor': 'white', 'pad': 10})
    plt.show()
    
    return instance_mask
    
def partitioned_marking(convc_part):
    
    for j in range(0, 256):
        for i in range(50, 95):
            combined_copy = np.copy(conc_part)
            pix = combined_copy[i, j, :]
            
            rr, gg, bb = pix
            
            if rr != 0 or gg != 0 or bb != 0:
                teeths_in_order.append([rr, gg, bb])


number_colour_code = [
        [0, 204, 204], # 18
        [255, 153, 51], # 17
        [0, 204, 0], # 16
        [102, 255, 255], # 15
        [102, 255, 102], # 14
        [178, 102, 255], # 13
        [255, 255, 51], # 12
        [0, 0, 102], # 11
        [255, 51, 51], # 21
        [153, 255, 51], # 22
        [102, 178, 255], # 23
        [204, 0, 0], # 24
        [0, 204, 102], # 25
        [102, 204, 0], # 26
        [204, 102, 0], # 27
        [204, 204, 0], # 28
        [102, 102, 255], # 38
        [102, 0, 102], # 37
        [0, 102, 102], # 36
        [0, 51, 102], # 35
        [0, 102, 0], # 34
        [102, 102, 0], # 33
        [102, 51, 0], # 32
        [0, 0, 51], # 31
        [51, 51, 0], # 41
        [51, 0, 0], # 42
        [0, 51, 51], # 43
        [51, 25, 0], # 44
        [0, 51, 0], # 45
        [51, 0, 51], # 46
        [102, 0, 0], # 47
        [102, 0, 51]  # 48
    ]

number_colour_code = np.array(number_colour_code)

data_path = 'C:\\Users\\Sharjeel\\Desktop\\codes\\teeth_numbered_data\\corrected_connected_images.npy'
data = np.load(data_path, allow_pickle=True)

print('data shape: ', data.shape)

numbered_data = []

#for sample in data:
image, mask, instance, numbered, weights, _ = data[1] # 0, 2

plt.imshow(image)
plt.show()
plt.imshow(mask)
plt.show()
plt.imshow(instance)
plt.show()

instance = re_instance(image, mask)

flat_instance = instance.reshape(65536, 3)
un = np.unique(flat_instance, axis = 0)

print('length unique: ', len(un))
#print('all unique vals: ', un)

if len(un) == 33:
    
    semantic_up_part = mask[0:128, :]
    semantic_down_part = mask[128:256, :]
    semantic_down_part = np.flip(semantic_down_part, 0)
    semantic_conc_part = np.concatenate((semantic_up_part, semantic_down_part), axis = 1)
    
    
    up_part = instance[0:128, :, :]
    down_part = instance[128:256, :, :]
    down_part = np.flip(down_part, 0)
    conc_part = np.concatenate((up_part, down_part), axis = 1)
    
    plt.imshow(conc_part)
    plt.show()
    
    teeths_in_order = []
    
    x_window = 100
    
    for j in range(512):
        for i in range(50, 110):
            combined_copy = np.copy(conc_part)
            pix = combined_copy[i, j, :]
            
            rr, gg, bb = pix
            
            if rr != 0 or gg != 0 or bb != 0:
                teeths_in_order.append([rr, gg, bb])
                
    teeths_in_order = np.array(teeths_in_order)
    unique_teeths_sorted = np.unique(teeths_in_order, axis = 0, return_index = True)[1]
    unique_teeths_order = [teeths_in_order[ind, :] for ind in sorted(unique_teeths_sorted)]
    
    numbered_mask = np.zeros((256, 256, 3))
    print('len teeths: ', len(unique_teeths_order))
    
    red_buffer = []
    blue_buffer = []
    green_buffer = []
    
    for i, teeth in enumerate(unique_teeths_order):
        r, g, b = teeth
        #print('r, g, b: ', r, g, b)
        
        #print('instance shape: ', instance.shape)
        #plt.imshow(instance)
        #plt.show()
        instance_copy = np.copy(instance)
        #numbered_mask_copy = np.copy(numbered_mask)
        
        lower = np.array([r-1, g-1, b-1])
        upper = np.array([r+1, g+1, b+1])
        masked = cv2.inRange(instance_copy, lower, upper)
        #print('uniques in mask: ', np.unique(masked))
        #plt.imshow(masked)
        #plt.show()

        numbered_mask_copy = np.copy(numbered_mask)
        
        #numbered_mask_r = np.zeros((256, 256)) #numbered_mask_copy[:, :, 0]
        #numbered_mask_g = np.zeros((256, 256)) #numbered_mask_copy[:, :, 1]
        #numbered_mask_b = np.zeros((256, 256)) #numbered_mask_copy[:, :, 2]
        
        
        #unn_mask = np.unique(numbered_mask_g)
        
        numbered_mask_r = np.where(masked == 255, number_colour_code[i, 0], 0)
        numbered_mask_g = np.where(masked == 255, number_colour_code[i, 1], 0)
        numbered_mask_b = np.where(masked == 255, number_colour_code[i, 2], 0)
        
        #numbered_mask_r[masked] = number_colour_code[i, 0]
        #numbered_mask_g[masked] = number_colour_code[i, 1]
        #numbered_mask_b[masked] = number_colour_code[i, 2]
        
        #plt.imshow(numbered_mask_g)
        #plt.show()
        
        numbered_mask_r = np.expand_dims(numbered_mask_r, axis = 2)
        numbered_mask_g = np.expand_dims(numbered_mask_g, axis = 2)
        numbered_mask_b = np.expand_dims(numbered_mask_b, axis = 2)
        
        if i == 0:
            numbered_mask = np.concatenate((numbered_mask_r, numbered_mask_g, numbered_mask_b), axis = 2)
        else:
            numbered_mask_c = np.concatenate((numbered_mask_r, numbered_mask_g, numbered_mask_b), axis = 2)
            numbered_mask = numbered_mask + numbered_mask_c
            
        #print('Numbered mask shape: ', numbered_mask.shape)
        #plt.imshow(numbered_mask)
        #plt.show()
        
        flat_numbered = numbered_mask.reshape(65536, 3)
        unnn = np.unique(flat_numbered, axis = 0)
        #print('intermediate vals: ', unnn)
        
        # red_buffer.append(numbered_mask_r)
        # blue_buffer.append(numbered_mask_g)
        # green_buffer.append(numbered_mask_b)
        
        # numbered_mask[:, :, 0] = numbered_mask_r
        # numbered_mask[:, :, 1] = numbered_mask_g
        # numbered_mask[:, :, 2] = numbered_mask_b
        
    # red_buffer = np.array(red_buffer)
    # print('red buffer shape: ', red_buffer.shape)
    # red_added = np.sum(red_buffer, axis = 0)
    # blue_added = np.sum(blue_buffer, axis = 0)
    # green_added = np.sum(green_buffer, axis = 0)
    
    # red_added = np.expand_dims(red_added, axis = 2)
    # blue_added = np.expand_dims(blue_added, axis = 2)
    # green_added = np.expand_dims(green_added, axis = 2)
    
    # all_combined = np.concatenate((red_added, blue_added, green_added), axis = 2)
    
    flat_numbered = numbered_mask.reshape(65536, 3)
    unnn = np.unique(flat_numbered, axis = 0)
    print('vals: ', unnn)
    print('unique numbered: ', len(unnn))
    numbered_data.append([image, mask, instance, numbered_mask, weights, 1])
    
    plt.imshow(numbered_mask)
    plt.show()

numbered_data = np.array(numbered_data)

# flat_instance = instance.reshape(65536, 3)
# un = np.unique(flat_instance, axis = 0)
# print('unique vals: ', len(un))

# plt.imshow(instance)
# plt.show()

# plt.imshow(up_part)
# plt.show()

# plt.imshow(down_part)
# plt.show()

# plt.imshow(conc_part)
# plt.show()
