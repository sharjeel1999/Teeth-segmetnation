import torch
import torch.nn as nn
#from models.swin import SwinTransformer
from models.Final_model import H_Net
#from models.Full_model_pooling_assist import H_Net
from utils import *
import random
import numpy as np
from tqdm import tqdm
import time

random.seed(0)  
np.random.seed(0)  
torch.manual_seed(0)

device = torch.device('cuda:0')

#model = SwinTransformer(in_chans=1).to(device)
model = H_Net(in_channels=1, num_classes=2, image_size=128).to(device)


data_path = 'C:\\Users\\Sharjeel\\Desktop\\datasets\\teeth_data\\proper_final_data.npy'

dataset = DataPrep_affinity(data_path)
Train_loader = torch.utils.data.DataLoader(dataset, batch_size = 3, shuffle = True)

test_dataset = Test_DataPrep_affinity(data_path)
Test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 3, shuffle = True)
base_lr = 0.001#0.001 # 3e-4
optimizer = torch.optim.AdamW(model.parameters(), lr = base_lr, weight_decay = 1e-5)
#optimizer = torch.optim.SGD(model.parameters(), lr = base_lr, momentum = 0.9, nesterov = True)

#model_path = "/content/drive/MyDrive/Colab Notebooks/teeth_segmentation_research/ours_instance0.9554141759872437.pth"
#model.load_state_dict(torch.load(model_path))

print('params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

EPOCHS = 150
dice_latch = 0#0.95145

for epoch in range(0, EPOCHS):
    training_loss = []
    training_dice = []
    training_iou = []
    train_precision = []
    train_recall = []
    train_specificity = []
    train_accuracy = []
    
    model.train()
    for data in Train_loader:
        inputs, labels, aff_labels, weights = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        aff_labels = aff_labels.cuda()
        weights = weights.cuda()
        
        optimizer.zero_grad()
        #print('inputs shape: ', inputs.shape)
        outputs4, out_aff = model(inputs.float())
        
        labels = torch.squeeze(labels, dim = 1)
        #print('outputs: ', outputs.shape)
        #print('labels: ', labels.shape)
        loss4, dsc4, iouscore4 = comined_loss(outputs4, labels.long(), weights, epoch, out_aff, aff_labels)
        loss = loss4

        loss.backward()
        optimizer.step()

        outputs4 = torch.argmax(outputs4, dim = 1)
        outputs4 = torch.squeeze(outputs4, dim = 0)
        #print('Predicted shape: ', outputs4.shape)
        #print('Lables shape: ', labels.shape)
        prec, rec, spec, accuracy, iou = prec_rec(outputs4.detach().cpu().numpy(), labels.detach().cpu().numpy())

        #instance_dice, instance_iou = score_after_ins()
        
        train_precision.append(prec)
        train_recall.append(rec)
        train_specificity.append(spec)
        train_accuracy.append(accuracy)

        training_loss.append(loss.item())
        training_dice.append(dsc4)
        training_iou.append(iouscore4)
    
    print('\n')
    print('Epoch: ', epoch+1)
    print('Training loss: ', np.mean(training_loss))
    print('Training Dice: ', np.mean(training_dice))
    print('Training IOU: ', np.mean(training_iou))

    print('Training Precision: ', np.mean(train_precision))
    print('Training Recall: ', np.mean(train_recall))
    print('Training Specificity: ', np.mean(train_specificity))
    print('Training Accuracy: ', np.mean(train_accuracy))
    
    validation_loss = []
    validation_dice = []
    validation_iou = []
    validation_precision = []
    validation_recall = []
    validation_specificity = []
    validation_accuracy = []

    score_ap_50 = []
    score_ap_60 = []
    score_ap_75 = []
    score_ap_80 = []
    score_ap_95 = []
    score_ap = []

    model.eval()
    for data in Test_loader:
        inputs, labels, aff_labels, weights = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        aff_labels = aff_labels.cuda()
        weights = weights.cuda()
        
        optimizer.zero_grad()

        outputs4, out_aff = model(inputs.float())
        
        out_copy, aff_copy = outputs4.clone(), out_aff.clone()

        labels = torch.squeeze(labels, dim = 1)
        loss4, dsc4, iouscore4 = comined_loss(outputs4, labels.long(), weights, epoch, out_aff, aff_labels)
        loss = loss4
        
        loss.backward()
        optimizer.step()
        
        outputs4 = torch.argmax(outputs4, dim = 1)
        outputs4 = torch.squeeze(outputs4, dim = 0)
        prec, rec, spec, accuracy, iou = prec_rec(outputs4.detach().cpu().numpy(), labels.detach().cpu().numpy())
        
        validation_precision.append(prec)
        validation_recall.append(rec)
        validation_specificity.append(spec)
        validation_accuracy.append(accuracy)

        validation_loss.append(loss.item())
        validation_dice.append(dsc4)
        validation_iou.append(iouscore4)

        # =========================================================================================
        b, c, h, w = out_copy.shape

        outputs_1 = out_copy
        outputs_1 = outputs_1[:, 1, :, :]
        
        b_out_95 = torch.zeros(b, h, w)
        b_out_95[outputs_1 > 0.95] = 1 # ----
        
        b_out_80 = torch.zeros(b, h, w)
        b_out_80[outputs_1 > 0.80] = 1 # ----
        
        b_out_75 = torch.zeros(b, h, w)
        b_out_75[outputs_1 > 0.75] = 1 # ----
        
        b_out_60 = torch.zeros(b, h, w)
        b_out_60[outputs_1 > 0.60] = 1 # ----
        
        out = torch.argmax(out_copy, dim = 1)#.detach().cpu().numpy()
        lbl = torch.squeeze(labels, dim = 1)#.detach().cpu().numpy()
        #print('output: ', out.shape)
        #print('GT: ', lbl.shape)

        ap_50 = average_precision_score(out.detach().cpu().numpy().ravel(), lbl.detach().cpu().numpy().ravel())
        ap_60 = average_precision_score(b_out_60.detach().cpu().numpy().ravel(), lbl.detach().cpu().numpy().ravel())
        ap_75 = average_precision_score(b_out_75.detach().cpu().numpy().ravel(), lbl.detach().cpu().numpy().ravel())
        ap_80 = average_precision_score(b_out_80.detach().cpu().numpy().ravel(), lbl.detach().cpu().numpy().ravel())
        ap_95 = average_precision_score(b_out_95.detach().cpu().numpy().ravel(), lbl.detach().cpu().numpy().ravel())
        
        score_ap_50.append(ap_50)
        score_ap_60.append(ap_60)
        score_ap_75.append(ap_75)
        score_ap_80.append(ap_80)
        score_ap_95.append(ap_95)
        score_ap.append(np.mean([ap_50, ap_60, ap_75, ap_80, ap_95]))
        # ==========================================================================================
        
    print('Validation loss: ', np.mean(validation_loss))
    print('Validation Dice: ', np.mean(validation_dice))
    print('Validation IOU: ', np.mean(validation_iou))
    print('Validation Precision: ', np.mean(validation_precision))
    print('Validation Recall: ', np.mean(validation_recall))
    print('Validation Specificity: ', np.mean(validation_specificity))
    print('Validation Accuracy: ', np.mean(validation_accuracy))

    print('Average Precision: ', np.mean(score_ap))
    print('Average Precision 50: ', np.mean(score_ap_50))
    print('Average Precision 75: ', np.mean(score_ap_75))

    if np.mean(training_dice) > 0.85:
      if dice_latch < np.mean(training_dice):
          torch.save(model.state_dict(), "C:\\Users\\Sharjeel\\Desktop\\codes\\teeth_segmentation_saves\\weight_saves\\experiment_1_semantic_and_instance_only\\Model{}.pth".format(np.mean(training_dice)))
          dice_latch = np.mean(training_dice)
    
    with open('C:\\Users\\Sharjeel\\Desktop\\codes\\teeth_segmentation_saves\\record_saves\\experiment_1_semantic_and_instance_only.txt', 'a') as f:
        f.write(f'Epoch: {epoch+1}')
        f.write('\n')
        f.write(f'Train Loss: {np.mean(training_loss)} Train IOU: {np.mean(training_iou)} Train Dice: {np.mean(training_dice)}')
        f.write('\n')
        f.write(f'Train precision: {np.mean(train_precision)} Train recall: {np.mean(train_recall)} Train specificity: {np.mean(train_specificity)} Train accuracy: {np.mean(train_accuracy)}')
        f.write('\n')
        f.write(f'Val Loss: {np.mean(validation_loss)} Val IOU: {np.mean(validation_iou)} Val Dice: {np.mean(validation_dice)}')
        f.write('\n')
        f.write(f'Val precision: {np.mean(validation_precision)} Val recall: {np.mean(validation_recall)} Val specificity: {np.mean(validation_specificity)} Val accuracy: {np.mean(validation_accuracy)}')
        f.write('\n')
        f.write('\n')
    

    '''lr_ = base_lr * (1.0 - epoch / EPOCHS) ** 0.9
    #lr_ = base_lr * min(epoch**(-0.5), epoch*8**(-1.5))
    if lr_ < 0.0001:
        lr_ = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_'''