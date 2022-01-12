from torch.nn.modules.utils import _pair
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

import torch
import torch.nn as nn
import math
import copy
import torch.nn.functional as F
import timm

import torchvision

def _round_to_multiple_of(val, divisor, round_up_bias = 0.9):
    
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. 
    """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val+divisor / 2) // divisor * divisor)
    
    return new_val if new_val >= round_up_bias *val else new_val + divisor

def _get_depths(alpha):
    """
    Scales tensor depths as in reference MobileNet code,
    prefers rounding up rather than down
    """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth*alpha, 8) for depth in depths]

class Backbone_Embedding(nn.Module):
    
    def __init__(self, alpha = 1.0, config=None, img_size = (480,640)):
        super(Backbone_Embedding, self).__init__()
        depths = _get_depths(alpha)
       
        if alpha == 1.0:
            MNASNet = torchvision.models.mnasnet1_0(pretrained= True, progress =True)
        else:
            MNASNet = torchvision.models.MNASNet(alpha=alpha)
        
        c = [80]
        self.cnn = nn.Sequential(MNASNet.layers._modules['0'],
                                 MNASNet.layers._modules['1'],
                                 MNASNet.layers._modules['2'],
                                 MNASNet.layers._modules['3'],
                                 MNASNet.layers._modules['4'],
                                 MNASNet.layers._modules['5'],
                                 MNASNet.layers._modules['6'],
                                 MNASNet.layers._modules['7'],
                                 MNASNet.layers._modules['8'],
                                 MNASNet.layers._modules['9'],
                                 MNASNet.layers._modules['10'])
        # ---- Embedding ----
        patch_size = _pair(config.patches["size"])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        
        self.patch_embeddings = nn.Conv2d(in_channels=c[0], out_channels=config.hidden_size,kernel_size= 1)
    
        self.position_embeddings = nn.Parameter(torch.zeros(1, 1201, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])
        
        
    def forward(self, x):
        
        B = x.shape[0]
        Backbone = self.cnn(x)
        cls_tokens = self.cls_token.expand(B,-1,-1)
        x = self.patch_embeddings(Backbone)
        
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim = 1)
        
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
    
        return embeddings 
    
    
class Transformer_3D(nn.Module):
    
    def __init__(self, config, model_name):
        super(Transformer_3D, self).__init__()
        
        
        model_name = model_name
        pre_model = timm.create_model(model_name, pretrained=True)
        
        out_f = [21843, 1200]
        out_c = [1201, 80, 40, 24]
        
        self.img_embedding  = Backbone_Embedding(alpha= 1.0 , config= config, img_size=(480,640))
    
        self.blocks0 = nn.Sequential(pre_model.blocks[0],
                                     pre_model.blocks[1],
                                     pre_model.blocks[2],
                                     pre_model.blocks[3],
                                     pre_model.blocks[4],
                                     pre_model.blocks[5],
                                     pre_model.blocks[6],
                                     pre_model.blocks[7],
                                     pre_model.blocks[8],
                                     pre_model.blocks[9],
                                     pre_model.blocks[10],
                                     pre_model.blocks[11])
        self.head = pre_model.head
        self.head1 = nn.Linear(out_f[0], out_f[1]) 
            
        self.conv0 = nn.Sequential(nn.Conv2d(out_c[0], out_c[1], 1, bias = False),
                                   nn.BatchNorm2d(out_c[1]),
                                   nn.LeakyReLU())
        
        self.conv1 = nn.Sequential(nn.Conv2d(out_c[1], out_c[2], 1, bias = False),
                                   nn.BatchNorm2d(out_c[2]),
                                   nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_c[2], out_c[3], 1, bias = False),
                                   nn.BatchNorm2d(out_c[3]),
                                   nn.LeakyReLU())
        
        self.inner0 = nn.Conv2d(out_c[0], out_c[1], 1, bias = False)
        self.inner1 = nn.Conv2d(out_c[1], out_c[2], 1, bias = False)
        
        self.upsampling = nn.Upsample(scale_factor=2 , mode = 'nearest')
 
        
    def forward(self, img):
        
        B = img.shape[0]
    
        outputs = []
        embedding_output  =  self.img_embedding(img)
        trans_out = self.blocks0(embedding_output)
        trans_out = self.head(trans_out)
        intra_feat0 = self.head1(trans_out).view(B, -1, 30, 40)
        out0 = self.conv0(intra_feat0)
        outputs.append(out0)
        
        intra_feat1 = out0 + self.inner0(intra_feat0)
        out1 = self.upsampling(self.conv1(intra_feat1))
        outputs.append(out1)
        
        intra_feat2 = out1 + self.upsampling(self.inner1(intra_feat1))
        out2 = self.upsampling(self.conv2(intra_feat2))
        outputs.append(out2)
        
        return outputs[::-1] 
