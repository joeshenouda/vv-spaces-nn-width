'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def uniform(w):
    if isinstance(w, torch.nn.BatchNorm2d):
        w.weight.data = torch.rand(w.weight.data.shape)
        w.bias.data = torch.zeros_like(w.bias.data)

def kaiming_normal(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(w.weight)


def param_counter(model):
    param_counter = 0
    for p_index, (p_name, p) in enumerate(model.named_parameters()):
        param_counter += p.numel()
    return param_counter

class FullRankVGG19(nn.Module):
    '''
    FullRankVGG19 Model 
    '''
    def __init__(self, num_classes=10):
        super(FullRankVGG19, self).__init__()
        # based on the literature, we don't touch the first conv layer
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        #self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(64)
        #self.conv2 = nn.Conv2d(64, 64, 3, 1, padding=1, bias=True)
        
        #self.max_pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(128)
        #self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1, bias=False)
        self.batch_norm4 = nn.BatchNorm2d(128)
        #self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1, bias=True)
        
        #self.max_pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(128, 256, 3, 1, padding=1, bias=False)
        self.batch_norm5 = nn.BatchNorm2d(256)
        #self.conv5 = nn.Conv2d(128, 256, 3, 1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, padding=1, bias=False)
        #self.conv6 = nn.Conv2d(256, 256, 3, 1, padding=1, bias=True)
        self.batch_norm6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, padding=1, bias=False)
        #self.conv7 = nn.Conv2d(256, 256, 3, 1, padding=1, bias=True)
        self.batch_norm7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, 3, 1, padding=1, bias=False)
        #self.conv8 = nn.Conv2d(256, 256, 3, 1, padding=1, bias=True)
        self.batch_norm8 = nn.BatchNorm2d(256)
        
        #self.max_pooling3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv9 = nn.Conv2d(256, 512, 3, 1, padding=1, bias=False)
        #self.conv9 = nn.Conv2d(256, 512, 3, 1, padding=1, bias=True)
        self.batch_norm9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=False)
        #self.conv10 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=True)
        self.batch_norm10 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=False)
        #self.conv11 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=True)
        self.batch_norm11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=False)
        #self.conv12 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=True)
        self.batch_norm12 = nn.BatchNorm2d(512)

        #self.max_pooling4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv13 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=False)
        #self.conv13 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=True)
        self.batch_norm13 = nn.BatchNorm2d(512)
        self.conv14 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=False)
        #self.conv14 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=True)
        self.batch_norm14 = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=False)
        #self.conv15 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=True)
        self.batch_norm15 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=False)
        #self.conv16 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=True)
        self.batch_norm16 = nn.BatchNorm2d(512)
        self.max_pooling5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv5(x)
        x = self.batch_norm5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.batch_norm6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = self.batch_norm7(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = self.batch_norm8(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv9(x)
        x = self.batch_norm9(x)
        x = F.relu(x)
        x = self.conv10(x)
        x = self.batch_norm10(x)
        x = F.relu(x)
        x = self.conv11(x)
        x = self.batch_norm11(x)
        x = F.relu(x)
        x = self.conv12(x)
        x = self.batch_norm12(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv13(x)
        x = self.batch_norm13(x)
        x = F.relu(x)
        x = self.conv14(x)
        x = self.batch_norm14(x)
        x = F.relu(x)
        x = self.conv15(x)
        x = self.batch_norm15(x)
        x = F.relu(x)
        x = self.conv16(x)
        x = self.batch_norm16(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


LR_FACOR = 4

if __name__ == "__main__":
    net = LowrankVGG19LTH()
    print("#### Model arch: {}, num params: {}".format(net, param_counter(net)))