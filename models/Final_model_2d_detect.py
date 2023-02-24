import torch
import torch.nn as nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from attentions import Attention, CSA, SwinTransformerBlock
from convolutional_attention import ConvEmbed, Block

class Conv_2d(nn.Module):
    def __init__(self, in_channels, out_channels, p):
        super().__init__()
        
        self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 3, dilation = p, padding = p),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
    def forward(self, x):
        return self.block(x)

class Feature_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.block = nn.Sequential(
                Conv_2d(in_channels, out_channels, 1),
                Conv_2d(out_channels, out_channels, 2),
                Conv_2d(out_channels, out_channels, 4)
            )
    def forward(self, x):
        return self.block(x)

class Fixed_Partitions(nn.Module):
    def __init__(self, window_size, channels, patch, image_size):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.patch = patch
        self.image_size = image_size
        
        self.divide_regions = nn.Sequential(
            Rearrange('b c (h1 h) (w1 w) -> (b h1 w1) c h w', h = self.window_size, w = self.window_size),
        )
        
        #self.att = CSA(self.channels, self.channels)
        self.att = Attention(self.channels, self.patch, img_size = self.window_size)
        
        self.combine_regions = nn.Sequential(
            Rearrange('(b h1 w1) c h w -> b c (h1 h) (w1 w)', h = self.window_size, w = self.window_size,
                      h1 = int(self.image_size/self.window_size), w1 = int(self.image_size/self.window_size)),
        )
        
    def forward(self, x):
        
        #print('input shape: ', x.shape)
        x = self.divide_regions(x)
        #print('divided shape: ', x.shape)
        x = self.att(x)
        #print('after attention: ', x.shape)
        x = self.combine_regions(x)
        #print('output shape: ', x.shape)
        return x

class Affinity_Block(nn.Module):
    def __init__(self, in_channels):
        super(Affinity_Block, self).__init__()
        
        self.conv_1 = Conv_2d(in_channels, in_channels, 3) # change activation to leaky ReLU
        self.conv_2 = Conv_2d(in_channels, in_channels, 3)
        #self.conv_3 = Conv_2d(in_channels, 4, 1)
        self.conv_layer = nn.Conv2d(in_channels, 25, kernel_size = 3, padding = 1)
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_layer(x)
        x = self.act(x)
        
        return x
    
class Numbering_Block(nn.Module):
    def __init__(self, in_channels):
        super(Numbering_Block, self).__init__()
        
        self.conv_1 = Conv_2d(in_channels, in_channels, 3) # change activation to leaky ReLU
        self.conv_2 = Conv_2d(in_channels, in_channels, 3)
        #self.conv_3 = Conv_2d(in_channels, 4, 1)
        self.conv_layer = nn.Conv2d(in_channels, 33, kernel_size = 3, padding = 1)
        #self.act = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_layer(x)
        #x = self.act(x)
        
        return x

class H_Net(nn.Module):
    def __init__(self, in_channels, num_classes, image_size):
        super().__init__()
        self.device = torch.device('cuda:0')
        self.image_size = image_size
        self.network_channs = 64
        
        self.pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.dout = nn.Dropout2d(0.2)
        
        self.block_1 = nn.Sequential(
                Feature_Block(in_channels, self.network_channs),
                Feature_Block(self.network_channs, self.network_channs)
            )
        
        #self.red_chan_1 = nn.Conv2d(self.network_channs, int(self.network_channs / 2), kernel_size = 1)
        '''self.red_chan_1 = self.block = nn.Sequential(
                nn.Conv2d(self.network_channs, int(self.network_channs / 2), kernel_size = 1),
                nn.BatchNorm2d(int(self.network_channs / 2)),
                nn.ReLU()
            )'''
        self.attention_mechanism_1 = nn.Sequential(
                SwinTransformerBlock(dim = self.network_channs, input_resolution = [self.image_size, self.image_size], num_heads = 8,
                                     window_size = 8, shift_size = 0),
                
                SwinTransformerBlock(dim = self.network_channs, input_resolution = [self.image_size, self.image_size], num_heads = 8,
                                     window_size = 8, shift_size = 8//2)
            )
        self.attention_mechanism_1 = self.attention_mechanism_1.cuda()
        
        self.aft_chan_1 = nn.Sequential(
                nn.Conv2d(int(self.network_channs*2), self.network_channs, kernel_size = 1),
                nn.BatchNorm2d(self.network_channs),
                nn.ReLU()
            )
        
        self.block_2 = nn.Sequential(
                Feature_Block(self.network_channs, self.network_channs),
                Feature_Block(self.network_channs, self.network_channs)
            )
        
        #self.red_chan_2 = nn.Conv2d(self.network_channs, int(self.network_channs / 2), kernel_size = 1)
        '''self.red_chan_2 = self.block = nn.Sequential(
                nn.Conv2d(self.network_channs, int(self.network_channs / 2), kernel_size = 1),
                nn.BatchNorm2d(int(self.network_channs / 2)),
                nn.ReLU()
            )'''
        self.attention_mechanism_2 = nn.Sequential(
                SwinTransformerBlock(dim = self.network_channs, input_resolution = [self.image_size, self.image_size], num_heads = 8,
                                     window_size = 8, shift_size = 0),
                
                SwinTransformerBlock(dim = self.network_channs, input_resolution = [self.image_size, self.image_size], num_heads = 8,
                                     window_size = 8, shift_size = 8//2)
            )
        
        self.aft_chan_2 = nn.Sequential(
                nn.Conv2d(int(self.network_channs*2), self.network_channs, kernel_size = 1),
                nn.BatchNorm2d(self.network_channs),
                nn.ReLU()
            )
        
        self.block_3 = nn.Sequential(
                Feature_Block(self.network_channs, self.network_channs),
                Feature_Block(self.network_channs, self.network_channs)
            )
        
        #self.red_chan_3 = nn.Conv2d(self.network_channs, int(self.network_channs / 2), kernel_size = 1)
        '''self.red_chan_3 = self.block = nn.Sequential(
                nn.Conv2d(self.network_channs, int(self.network_channs / 2), kernel_size = 1),
                nn.BatchNorm2d(int(self.network_channs / 2)),
                nn.ReLU()
            )'''
        self.attention_mechanism_3 = nn.Sequential(
                SwinTransformerBlock(dim = self.network_channs, input_resolution = [self.image_size, self.image_size], num_heads = 8,
                                     window_size = 8, shift_size = 0),
                
                SwinTransformerBlock(dim = self.network_channs, input_resolution = [self.image_size, self.image_size], num_heads = 8,
                                     window_size = 8, shift_size = 8//2)
            )
        
        self.aft_chan_3 = nn.Sequential(
                nn.Conv2d(int(self.network_channs*2), self.network_channs, kernel_size = 1),
                nn.BatchNorm2d(self.network_channs),
                nn.ReLU()
            )
        
        self.block_4 = nn.Sequential(
                Feature_Block(self.network_channs, self.network_channs),
                Feature_Block(self.network_channs, self.network_channs)
            )
        
        #self.red_chan_4 = nn.Conv2d(self.network_channs, int(self.network_channs / 2), kernel_size = 1)
        '''self.red_chan_4 = self.block = nn.Sequential(
                nn.Conv2d(self.network_channs, int(self.network_channs / 2), kernel_size = 1),
                nn.BatchNorm2d(int(self.network_channs / 2)),
                nn.ReLU()
            )'''
        self.attention_mechanism_4 = nn.Sequential(
                SwinTransformerBlock(dim = self.network_channs, input_resolution = [self.image_size, self.image_size], num_heads = 8,
                                     window_size = 8, shift_size = 0),
                
                SwinTransformerBlock(dim = self.network_channs, input_resolution = [self.image_size, self.image_size], num_heads = 8,
                                     window_size = 8, shift_size = 8//2)
            )
        
        self.aft_chan_4 = nn.Sequential(
                nn.Conv2d(int(self.network_channs*2), self.network_channs, kernel_size = 1),
                nn.BatchNorm2d(self.network_channs),
                nn.ReLU()
            )
        
        #self.affinity_prediction = Affinity_Block(int(self.network_channs*2))
        
        self.prediction_block = nn.Sequential(
                nn.Conv2d(int(self.network_channs*2), num_classes, kernel_size = 1)
            )
        
        '''self.spatial_reduction = nn.Sequential(
                
                nn.Conv2d(int(self.network_channs*2), self.network_channs, kernel_size = 1),
                nn.BatchNorm2d(self.network_channs),
                nn.ReLU(),
            
                nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
                
                nn.Conv2d(self.network_channs, int(self.network_channs*2), kernel_size = 3, dilation = 1, padding = 1),
                nn.BatchNorm2d(int(self.network_channs*2)),
                nn.ReLU(),
                
                nn.Conv2d(int(self.network_channs*2), int(self.network_channs*2), kernel_size = 3, dilation = 1, padding = 1),
                nn.BatchNorm2d(int(self.network_channs*2)),
                nn.ReLU(),
                
                nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
                
                nn.Conv2d(int(self.network_channs*2), int(self.network_channs*4), kernel_size = 3, dilation = 1, padding = 1),
                nn.BatchNorm2d(int(self.network_channs*4)),
                nn.ReLU(),
                
                nn.Conv2d(int(self.network_channs*4), int(self.network_channs*4), kernel_size = 3, dilation = 1, padding = 1),
                nn.BatchNorm2d(int(self.network_channs*4)),
                nn.ReLU(),
                
                # nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
                
                # nn.Conv2d(int(self.network_channs*4), int(self.network_channs*8), kernel_size = 3, dilation = 1, padding = 1),
                # nn.BatchNorm2d(int(self.network_channs*8)),
                # nn.ReLU(),
                
                # nn.Conv2d(int(self.network_channs*8), int(self.network_channs*8), kernel_size = 3, dilation = 1, padding = 1),
                # nn.BatchNorm2d(int(self.network_channs*8)),
                # nn.ReLU(),
                
                nn.Conv2d(int(self.network_channs*4), 32, kernel_size = 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )
        
        self.detection_block = nn.Sequential(
                nn.Linear(4096, 2046),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                
                nn.Linear(2046, 1024),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                
                nn.Linear(1024, 512),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                
                nn.Linear(512, 256),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                
                nn.Linear(256, 128),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                
                nn.Linear(128, 64),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                
                nn.Linear(64, 3),
                nn.BatchNorm1d(32),
                nn.Sigmoid(),
                
            )'''
        
        #self.numbering_prediction = Numbering_Block(int(self.network_channs*2 + 25))
        self.numbering_prediction = Numbering_Block(int(self.network_channs*2))
        
    def forward(self, x):
        x1 = self.block_1(x)
        ####x1_div = self.red_chan_1(x1)
        x1_self = self.attention_mechanism_1(x1.cuda()).cuda()
        x1 = torch.cat([x1, x1_self], dim = 1)
        #x1 = x1 + x1_self
        x1 = self.aft_chan_1(x1)
        
        x2 = self.dout(x1)
        x2 = self.block_2(x2)
        ####x2_div = self.red_chan_2(x2)
        x2_self = self.attention_mechanism_2(x2)
        x2 = torch.cat([x2, x2_self], dim = 1)
        #x2 = x2 + x2_self
        x2 = self.aft_chan_2(x2)
        
        x3 = self.dout(x2)
        x3 = self.block_3(x3)
        ####x3_div = self.red_chan_3(x3)
        x3_self = self.attention_mechanism_3(x3)
        x3 = torch.cat([x3, x3_self], dim = 1)
        #x3 = x3 + x3_self
        x3 = self.aft_chan_3(x3)
        
        x4 = self.dout(x3)
        x4 = self.block_4(x4)
        ####x4_div = self.red_chan_4(x4)
        x4_self = self.attention_mechanism_4(x4)
        x4 = torch.cat([x4, x4_self], dim = 1)
        #x4 = x4 + x4_self
        x4 = self.aft_chan_4(x4)
        
        xcomb = torch.cat([x4, x2], dim = 1)
        #print('comb out shape: ', xcomb.shape)
        
        #aff_out = self.affinity_prediction(xcomb)
        #print('affinity shape: ', aff_out.shape)
        semantic = self.prediction_block(xcomb)
        sem_mxed = torch.argmax(semantic, dim = 1)
        
        #for_numb = torch.cat([xcomb, aff_out], dim = 1)
        onehot_numbering = self.numbering_prediction(xcomb)
        
        #xreduced = self.spatial_reduction(xcomb)
        #print('reduced shape: ', xreduced.shape)
        #xcomb_flat = torch.flatten(xreduced, start_dim=2)
        #print('flattented shape: ', xcomb_flat.shape)
        #det_out = self.detection_block(xcomb_flat)
        
        return semantic, onehot_numbering