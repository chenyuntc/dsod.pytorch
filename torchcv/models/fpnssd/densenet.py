import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from torch.autograd import Variable


def stem():
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, 2, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 3,1,1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, 3,1,1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2,ceil_mode=True)
                         )


class DenseLayer(nn.Module):

    def __init__(self, inC, midC=192, growth_rate=48):
        super(DenseLayer, self).__init__()
        self.model_name = 'DenseLayer'
        self.conv = nn.Sequential(
            nn.BatchNorm2d(inC),
            nn.ReLU(inplace=True),
            nn.Conv2d(inC, midC, 1),
            nn.BatchNorm2d(midC),
            nn.ReLU(inplace=True),
            nn.Conv2d(midC, growth_rate, 3, padding=1),
        )

    def forward(self, x):
        y = self.conv(x)
        y = t.cat([x, y], 1)
        return y


class DenseBlock(nn.Module):
    def __init__(self, layer_num, inC, midC=192,growth_rate=48):
        super(DenseBlock, self).__init__()
        self.model_name = 'DenseBlock'
        layers = []
        layers.append(DenseLayer(inC,midC,growth_rate))
        for layer_idx in range(1, layer_num):
            layers.append(DenseLayer(inC+growth_rate*layer_idx,midC,growth_rate))
        self.dense = nn.Sequential(*layers)

    def forward(self, x):
        return self.dense(x)


class TransitionLayer(nn.Module):

    def __init__(self, inC, outC,pool=False):
        super(TransitionLayer, self).__init__()
        self.model_name = 'TransitionLayer'
        self.conv = nn.Sequential(
            nn.BatchNorm2d(inC),
            nn.ReLU(inplace=True),
            nn.Conv2d(inC,outC,1)
        )
        self.pool = nn.MaxPool2d(2,2,ceil_mode=True) if pool else lambda x: x


    def forward(self, x):
        x = self.conv(x)
        return (x,self.pool(x))
    
class DenseSupervision1(nn.Module):

    def __init__(self,inC,outC=256):
        super(DenseSupervision1, self).__init__()
        self.model_name='DenseSupervision'

        self.right = nn.Sequential(
            nn.BatchNorm2d(inC),
            nn.ReLU(inplace=True),
            nn.Conv2d(inC,outC,1),
            nn.BatchNorm2d(outC),
            nn.ReLU(inplace=True),
            nn.Conv2d(outC,outC,3,2,1)
        )

    def forward(self,x1,x2):
        # x1 should be f1
        right = self.right(x1)
        return t.cat([x2,right],1)


class DenseSupervision(nn.Module):

    def __init__(self,inC,outC=128):
        super(DenseSupervision, self).__init__()
        self.model_name='DenseSupervision'
        self.left = nn.Sequential(
            nn.MaxPool2d(2,2,ceil_mode=True),
            nn.BatchNorm2d(inC),
            nn.ReLU(inplace=True),
            nn.Conv2d(inC,outC,1)
        )
        self.right = nn.Sequential(
            nn.BatchNorm2d(inC),
            nn.ReLU(inplace=True),
            nn.Conv2d(inC,outC,1),
            nn.BatchNorm2d(outC),
            nn.ReLU(inplace=True),
            nn.Conv2d(outC,outC,3,2,1)
        )

    def forward(self,x):
        left = self.left(x)
        right = self.right(x)
        return t.cat([left,right],1)

class DenseNet(nn.Module):

    def __init__(self):
        super(DenseNet, self).__init__()
        self.model_name='DenseNet'

        self.stem = stem()
        self.dense1 = DenseBlock(6,128)
        self.trans1 = TransitionLayer(416,416,pool=True)
        self.dense2 = DenseBlock(8,416)
        self.trans2 = TransitionLayer(800,800,pool=True)
        self.dense3 = DenseBlock(8,800)
        self.trans3 = TransitionLayer(1184,1184) #TODO：论文中是1120但是代码中是 1184， 而且这样才符合下一个densenetblock的growthrate
        self.dense4 = DenseBlock(8,1184)
        # TODO: 论文中的输出 形状是1568
        self.trans4 = TransitionLayer(1568,256)

        self.dense_sup1 = DenseSupervision1(800,256)
        self.dense_sup2 = DenseSupervision(512,256)
        self.dense_sup3 = DenseSupervision(512,128)
        self.dense_sup4 = DenseSupervision(256,128)
        self.dense_sup5 = DenseSupervision(256,128)


    def forward(self,x):
        x = self.stem(x)

        x = self.dense1(x)
        x,x = self.trans1(x)

        x = self.dense2(x)
        f1,x = self.trans2(x)
        
        x = self.dense3(x)
        x,x = self.trans3(x)

        x = self.dense4(x)
        x,x = self.trans4(x)
        
        f2 = self.dense_sup1(f1,x)
        f3 = self.dense_sup2(f2)
        f4 = self.dense_sup3(f3)
        f5 = self.dense_sup4(f4)
        f6 = self.dense_sup5(f5)
        return f1,f2,f3,f4,f5,f6

if __name__ == '__main__':
    m = DenseNet()
    input = t.autograd.Variable(t.randn(1,3,300,300))
    o = m(input)
    for ii in o:
        print(ii.shape)
