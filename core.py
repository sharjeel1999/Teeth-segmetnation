import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
import time
import os
from sklearn.metrics import confusion_matrix, average_precision_score

from utils import prec_rec

class Run_model():
    def __init__(self):
        print('...')
        
    def validation(self, model, validation_loader, loss_function, record_save_path, weights_save_path, dice_latch):
        validation_loss = []
        validation_dice = []
        validation_iou = []
        validation_precision = []
        validation_recall = []
        validation_specificity = []
        validation_accuracy = []
        validation_detection_loss = []
        
        score_ap_50 = []
        score_ap_60 = []
        score_ap_75 = []
        score_ap_80 = []
        score_ap_95 = []
        score_ap = []
        
        model.eval()
        
        for data in validation_loader:
            inputs, labels, aff_labels, weights, numb_label = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            aff_labels = aff_labels.cuda()
            weights = weights.cuda()
            numb_label = numb_label.cuda()
            
            bb = inputs.shape[0]
            
            if bb > 1:
                
                with torch.no_grad():
                    outputs4, numb_pred = model(inputs.float())
                
                out_copy = outputs4.clone()
                
                labels = torch.squeeze(labels, dim = 1)
                loss4, dsc4, iouscore4, det_loss = loss_function(outputs4, labels.long(), numb_pred, weights, numb_label.long())
                loss = loss4
                
                
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
                validation_detection_loss.append(det_loss.detach().cpu().numpy())
                
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
                
                out = torch.argmax(out_copy, dim = 1)
                lbl = torch.squeeze(labels, dim = 1)
                
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
                
        print('Validation loss: ', np.mean(validation_loss))
        print('Validation detection loss: ', np.mean(validation_detection_loss))
        print('Validation Dice: ', np.mean(validation_dice))
        print('Validation IOU: ', np.mean(validation_iou))
        print('Validation Precision: ', np.mean(validation_precision))
        print('Validation Recall: ', np.mean(validation_recall))
        print('Validation Specificity: ', np.mean(validation_specificity))
        print('Validation Accuracy: ', np.mean(validation_accuracy))
        
        print('Average Precision: ', np.mean(score_ap))
        print('Average Precision 50: ', np.mean(score_ap_50))
        print('Average Precision 75: ', np.mean(score_ap_75))
        
        if np.mean(validation_dice) > 0.85:
          if dice_latch < np.mean(validation_dice):
              save_name = 'Model' + str(np.mean(validation_dice)) + '.pth'
              save_file = os.path.join(weights_save_path, save_name)
              torch.save(model.state_dict(), save_file)
              dice_latch = np.mean(validation_dice)
      
        with open(record_save_path, 'a') as f:
            f.write(f'Val Loss: {np.mean(validation_loss)} Val IOU: {np.mean(validation_iou)} Val Dice: {np.mean(validation_dice)}')
            f.write('\n')
            f.write(f'Val precision: {np.mean(validation_precision)} Val recall: {np.mean(validation_recall)} Val specificity: {np.mean(validation_specificity)} Val accuracy: {np.mean(validation_accuracy)}')
            f.write('\n')
            f.write('\n')
      
        return dice_latch
                
        
    def train(self, model, train_loader, validation_loader, optimizer, loss_function, num_epochs, record_save_path, weights_save_path, base_lr):
        for epoch in range(num_epochs):
            training_loss = []
            training_dice = []
            training_iou = []
            training_precision = []
            training_recall = []
            training_specificity = []
            training_accuracy = []
            
            model.train()
            dice_latch = 0
            
            for data in train_loader:
                inputs, labels, aff_labels, weights, numb_label = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                aff_labels = aff_labels.cuda()
                weights = weights.cuda()
                numb_label = numb_label.cuda()
                
                bb = inputs.shape[0]
                if bb > 1:
                    
                    optimizer.zero_grad()
                    
                    outputs4, numb_pred = model(inputs.float())
                    labels = torch.squeeze(labels, dim = 1)

                    loss4, dsc4, iouscore4, det_loss = loss_function(outputs4, labels.long(), numb_pred, weights, numb_label.long())
                    loss = loss4
                    
                    loss.backward()
                    optimizer.step()
                    
                    outputs4 = torch.argmax(outputs4, dim = 1)
                    outputs4 = torch.squeeze(outputs4, dim = 0)
                    
                    prec, rec, spec, accuracy, iou = prec_rec(outputs4.detach().cpu().numpy(), labels.detach().cpu().numpy())

                    training_precision.append(prec)
                    training_recall.append(rec)
                    training_specificity.append(spec)
                    training_accuracy.append(accuracy)
                    
                    training_loss.append(loss.item())
                    training_dice.append(dsc4)
                    training_iou.append(iouscore4)
                    
            print('\n')
            print('Epoch: ', epoch+1)
            print('Training loss: ', np.mean(training_loss))
            print('Training Dice: ', np.mean(training_dice))
            print('Training IOU: ', np.mean(training_iou))
            
            print('Training Precision: ', np.mean(training_precision))
            print('Training Recall: ', np.mean(training_recall))
            print('Training Specificity: ', np.mean(training_specificity))
            print('Training Accuracy: ', np.mean(training_accuracy))
            
            with open(record_save_path, 'a') as f:
                f.write(f'Epoch: {epoch+1}')
                f.write('\n')
                f.write(f'Train Loss: {np.mean(training_loss)} Train IOU: {np.mean(training_iou)} Train Dice: {np.mean(training_dice)}')
                f.write('\n')
                f.write(f'Train precision: {np.mean(training_precision)} Train recall: {np.mean(training_recall)} Train specificity: {np.mean(training_specificity)} Train accuracy: {np.mean(training_accuracy)}')
                f.write('\n')
            
            dice_latch = self.validation(model, validation_loader, loss_function, record_save_path, weights_save_path, dice_latch)
            
            lr_ = base_lr * (1.0 - epoch / num_epochs) ** 0.9
            if lr_ < 0.0001:
                lr_ = 0.0001
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
    
        return model
        