from torch.nn.modules.utils import _pair
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

import torch
import torch.nn as nn
import math
import copy
import timm
import torchvision.transforms.functional as TF
import torchvision
from pytorch_pretrained_vit import ViT
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

class FeatureMap(nn.Module):
    
    def __init__(self, alpha = 1.0, config=None, img_size = (480,640)):
        super(FeatureMap, self).__init__()
        depths = _get_depths(alpha)
       
        if alpha == 1.0:
            MNASNet = torchvision.models.mnasnet1_0(pretrained= True, progress =True)
        else:
            MNASNet = torchvision.models.MNASNet(alpha=alpha)
        
        c = [1200]      
        self.Encoder = nn.Sequential(MNASNet.layers._modules['0'],
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
        
        self.Deconv1 = nn.Sequential(nn.ConvTranspose2d(in_channels= 80, out_channels= 80, kernel_size = 5, 
                                                        stride=2, padding=2, output_padding=1, bias = False),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(in_channels= 80, out_channels= 40, kernel_size = 1,bias = False),
                                     nn.ReLU())
        
        self.Deconv2 = nn.Sequential(nn.ConvTranspose2d(in_channels= 40, out_channels= 40, kernel_size = 5, 
                                                        stride=2, padding=2, output_padding=1, bias = False),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(in_channels= 40, out_channels= 20, kernel_size = 1,bias = False),
                                     nn.ReLU())
        self.Deconv3 = nn.Sequential(nn.ConvTranspose2d(in_channels= 20, out_channels= 20, kernel_size = 5, 
                                                        stride=2, padding=2, output_padding=1, bias = False),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(in_channels= 20, out_channels= 10, kernel_size = 1,bias = False),
                                     nn.ReLU())
        self.Deconv4 = nn.Sequential(nn.ConvTranspose2d(in_channels= 10, out_channels= 10, kernel_size = 5, 
                                                        stride=2, padding=2, output_padding=1, bias = False),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(in_channels= 10, out_channels= 3, kernel_size = 1,bias = False))        
        
    def forward(self, x):
        
        B = x.shape[0]

        x = self.Encoder(x)
        feature_map = self.Deconv4(self.Deconv3(self.Deconv2(self.Deconv1(x))))

        return feature_map 
    
class Transformer_3D(nn.Module):
    
    def __init__(self, model_name):
        super(Transformer_3D, self).__init__()
        
        
        model_name = model_name
        pre_model = ViT(model_name, pretrained=True)
        
        out_f = [21843, 6000]
        out_c = [20, 80, 40, 24]
        
        self.Map  = FeatureMap(alpha= 1.0)
    
        self.Transformer = pre_model
               
        self.linear = nn.Linear(out_f[0], out_f[1]) 
            
        self.deconv0 = nn.Sequential(nn.ConvTranspose2d(out_c[0], out_c[0], 5 , 2, 2, 1, bias = False),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(out_c[0], out_c[1], 1 ,  bias = False))

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(out_c[1], out_c[1], 5 , 2, 2, 1, bias = False),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(out_c[1], out_c[2], 1 ,  bias = False))
        
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(out_c[2], out_c[2], 5 , 2, 2, 1,  bias = False),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(out_c[2], out_c[3], 1 ,  bias = False))
 
    def forward(self, img):
        
        B = img.shape[0]
    
        outputs = []
        feature_map = self.Map(img)
        feature_map = TF.resize(feature_map,(224,224))

        T_output = self.Transformer(feature_map)
 
        out = self.linear(T_output).view(B,20,15,20)
    
        output0 = self.deconv0(out)
        outputs.append(output0)
        
        output1 = self.deconv1(output0)
        outputs.append(output1)
        
        output2 = self.deconv2(output1)
        outputs.append(output2)        

        return outputs[::-1] 


