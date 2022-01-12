from torch.nn.modules.utils import _pair
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

import torch
import torch.nn as nn
import math
import copy
import torch.nn.functional as F
import timm

class Embeddings(nn.Module):
    
    def __init__(self, config, img_size, in_channels):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)
        
        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False
            
        ##if self.hybrid:
        ##    self.hybrid_model = ResNetV2(block_units = config.resnet.num_layers, width_fatcor = config.resnet.width_factor)
        ##    in_channels = self.hybrid_modle.width * 16
        
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])
        

    def forward(self, x):
        
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B,-1,-1)
   
        if self.hybrid:
            x = self.hybrid_model(x)
        
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim = 1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
class Transformer_3D(nn.Module):
    
    def __init__(self, config, model_name):
        super(Transformer_3D, self).__init__()
        
        
        self.model_name = model_name
        pre_model = timm.create_model(self.model_name, pretrained=True)
        
        out_f = [21843, 1200]
        out_c = [301, 80, 40, 24]
        
        self.img_embedding  = Embeddings(config, (480,640), 3)
        #self.edge_embedding = Embeddings(config, (480,640), 1)    
        
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
 
        
    def forward(self, img):#, edge):
        
        B = img.shape[0]
    
        outputs = []
        #img_output  =  self.img_embedding(img)
        #edge_output =  self.edge_embedding(edge)
        #embedding_output = torch.cat((img_output, edge_output), dim =1)
        embedding_output = self.img_embedding(img)
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
