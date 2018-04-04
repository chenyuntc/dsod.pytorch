import torch
t = torch
import torch.nn as nn
import torch.nn.functional as F

from torchcv.models.dsod.densenet import DenseNet
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.init as init

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

class DSOD(nn.Module):
    steps = (8, 16, 32, 64, 100, 300)
    # default bounding box sizes for each feature map.
    box_sizes = (30.0, 60.0, 111., 162., 213., 264., 315.)
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
    # fm_sizes = (64, 32, 16, 8, 4, 2, 1)
    fm_sizes = (38, 19, 10, 5, 3, 2)


    def __init__(self, num_classes):
        super(DSOD, self).__init__()
        self.num_classes = num_classes
        self.extractor = DenseNet()
        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()

        in_channels = [800,512,512,256,256,256]
        num_anchors = (4, 6, 6, 6, 4, 4)
        for inC,num_anchor in zip(in_channels,num_anchors):
            # self.loc_layers += [nn.Conv2d(inC, num_anchor*4, kernel_size=3, padding=1)]
            # self.cls_layers += [nn.Conv2d(inC, num_anchor* num_classes, kernel_size=3, padding=1)
            #                                   ]
            self.loc_layers += [nn.Sequential(nn.Conv2d(inC,
                                          num_anchor*4, kernel_size=3, padding=1,bias=False),
                                              nn.BatchNorm2d(num_anchor*4)
                                              )]
            self.cls_layers += [nn.Sequential(nn.Conv2d(inC,
                                        num_anchor* num_classes, kernel_size=3, padding=1,bias=False),
                                              nn.BatchNorm2d(num_anchor* num_classes)
                                              )]
        self.normalize = nn.ModuleList([L2Norm(chan,20) for chan in in_channels])

        self.reset_parameters()
    
    def forward(self, x):
        loc_preds = []
        cls_preds = []
        xs = self.extractor(x)
        for i, x in enumerate(xs):
            x = self.normalize[i](x)
            loc_pred = self.loc_layers[i](x)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_preds.append(loc_pred.view(loc_pred.size(0), -1, 4))

            cls_pred = self.cls_layers[i](x)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_preds.append(cls_pred.view(
                cls_pred.size(0), -1, self.num_classes))

        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)
        return loc_preds, cls_preds
    
    def reset_parameters(self):
        for name,param in self.extractor.named_parameters():
            if hasattr(param,'weight'):
                nn.init.xavier_uniform(param.weight.data,gain=nn.init.calculate_gain('relu'))

        for name,param in self.loc_layers.named_parameters():
            if hasattr(param,'weight'):
                nn.init.normal(param.weight.data,std=0.01)

        for name,param in self.cls_layers.named_parameters():
            if hasattr(param,'weight'):
                nn.init.normal(param.weight.data,std=0.01)

if __name__ == '__main__':
    m = DSOD(21)
    input = t.autograd.Variable(t.randn(1,3,300,300))
    o = m(input)
    for ii in o:
        print(ii.shape)
