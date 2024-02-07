import torch
import torchvision
import torch.nn as nn
from guided_diffusion import dist_util, logger

def featurepreprocess(frame):

    G=torchvision.models.resnet50(pretrained=False).to(dist_util.dev())
    checkpoint = torch.load('./PATH TO CONTRASTIVE PRETRAINING WEIGHT', map_location=dist_util.dev())
    G.load_state_dict(checkpoint)
    Gbackbone = torch.nn.Sequential(*list(G.children())[:-2]).to(dist_util.dev())
    
    frame=frame.to(dist_util.dev())

    #改尺寸为 224,224 的
    frame = torch.nn.functional.interpolate(frame, size=(224,224), mode='bilinear', align_corners=False)
    frame = Gbackbone(frame)
    

    # 使用 interpolate 函数进行上采样
    frame = torch.nn.functional.interpolate(frame, size=(112, 112), mode='bilinear', align_corners=False)
    conv = torch.nn.Conv2d(2048, 3, kernel_size=1).to(dist_util.dev())
    frame = conv(frame)
    return frame