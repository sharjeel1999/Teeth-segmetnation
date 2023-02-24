import torch
import torch.nn as nn
from models.Final_model_2d_detect import H_Net

from utils import *
import random
import numpy as np
from tqdm import tqdm
import time

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

device = torch.device('cuda:0')

data_path = 'C:\\Users\\Sharjeel\\Desktop\\codes\\teeth_numbered_data\\Data_with_center.npy'

dataset = DataPrep_affinity(data_path)
Train_loader = torch.utils.data.DataLoader(dataset, batch_size = 3, shuffle = True)

test_dataset = Test_DataPrep_affinity(data_path)
Test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 3, shuffle = True)

#print('params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))


from core import Run_model

trainer = Run_model()

model = H_Net(in_channels=1, num_classes=2, image_size=256).to(device)

num_epochs = 100
record_save_path = 'C:\\Users\\Sharjeel\\Desktop\\codes\\fold_training_code\\record_saves\\normal_training.txt'
weights_save_folder = 'C:\\Users\\Sharjeel\\Desktop\\codes\\fold_training_code\\weight_saves\\normal_training'

base_lr = 0.001#0.001 # 3e-4
optimizer = torch.optim.AdamW(model.parameters(), lr = base_lr, weight_decay = 1e-5)

model = trainer.train(model, Train_loader, Test_loader, optimizer, comined_loss_normal, num_epochs, record_save_path, weights_save_folder, base_lr)

num_epochs = 50

base_lr = 0.0001#0.001 # 3e-4
optimizer = torch.optim.AdamW(model.parameters(), lr = base_lr, weight_decay = 1e-5)
model = trainer.train(model, Train_loader, Test_loader, optimizer, comined_loss_refinement, num_epochs, record_save_path, weights_save_folder, base_lr)

