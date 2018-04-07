from __future__ import print_function

import os
import random

import matplotlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFile
from torch.autograd import Variable
from torchnet.meter import AverageValueMeter
from tqdm import tqdm

from torchcv.datasets import ListDataset
from torchcv.evaluations.voc_eval import voc_eval
from torchcv.loss import SSDLoss
from torchcv.models import DSOD
from torchcv.models.ssd import SSDBoxCoder
from torchcv.transforms import (random_crop, random_distort, random_flip,
                                random_paste, resize)
from torchcv.utils.config import opt
from torchcv.visualizations import Visualizer

ImageFile.LOAD_TRUNCATED_IMAGES = True
matplotlib.use('agg')






def caffe_normalize(x):
         return transforms.Compose([
            transforms.Lambda(lambda x:255*x[[2,1,0]]) ,
            transforms.Normalize([104,117,123], (1,1,1)), # make it the same as caffe
                  # bgr and 0-255
        ])(x)
def Transform(box_coder, train=True):
    def train_(img, boxes, labels):
        img = random_distort(img)
        if random.random() < 0.5:
            img, boxes = random_paste(img, boxes, max_ratio=4, fill=(123, 116, 103))
        img, boxes, labels = random_crop(img, boxes, labels)
        img, boxes = resize(img, boxes, size=(opt.img_size, opt.img_size), random_interpolation=True)
        img, boxes = random_flip(img, boxes)
        img = transforms.Compose([
            transforms.ToTensor(),
            caffe_normalize
        ])(img)
        boxes, labels = box_coder.encode(boxes, labels)
        return img, boxes, labels

    def test_(img, boxes, labels):
        img, boxes = resize(img, boxes, size=(opt.img_size, opt.img_size))
        img = transforms.Compose([
            transforms.ToTensor(),
            caffe_normalize
        ])(img)
        boxes, labels = box_coder.encode(boxes, labels)
        return img, boxes, labels

    return train_ if train else test_


def eval(net,test_num=10000):
    net.eval()

    def transform(img, boxes, labels):
        img, boxes = resize(img, boxes, size=(opt.img_size, opt.img_size))
        img = transforms.Compose([
            transforms.ToTensor(),
            caffe_normalize
        ])(img)
        return img, boxes, labels

    dataset = ListDataset(root=opt.data_root, \
                          list_file=opt.voc07_test,
                          transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    box_coder = SSDBoxCoder(net)

    pred_boxes = []
    pred_labels = []
    pred_scores = []
    gt_boxes = []
    gt_labels = []

    with open('torchcv/datasets/voc/voc07_test_difficult.txt') as f:
        gt_difficults = []
        for line in f.readlines():
            line = line.strip().split()
            d = np.array([int(x) for x in line[1:]])
            gt_difficults.append(d)

    for i, (inputs, box_targets, label_targets) in tqdm(enumerate(dataloader)):
        gt_boxes.append(box_targets.squeeze(0))
        gt_labels.append(label_targets.squeeze(0))

        loc_preds, cls_preds = net(Variable(inputs.cuda(), volatile=True))
        box_preds, label_preds, score_preds = box_coder.decode(
            loc_preds.cpu().data.squeeze(),
            F.softmax(cls_preds.squeeze(), dim=1).cpu().data,
            score_thresh=0.01)

        pred_boxes.append(box_preds)
        pred_labels.append(label_preds)
        pred_scores.append(score_preds)
        if i==test_num:break

    aps = (voc_eval(
        pred_boxes, pred_labels, pred_scores,
        gt_boxes, gt_labels, gt_difficults,
        iou_thresh=0.5, use_07_metric=True))
    net.train()
    return aps


def predict(net, box_coder, img):
    net.eval()
    if isinstance(img, str):
        img = Image.open(img)
        ow = oh = 300
        img = img.resize((ow, oh))
    transform = transforms.Compose([
        transforms.ToTensor(),
        caffe_normalize
    ])
    x = transform(img).cuda()
    x = Variable(x, volatile=True)
    loc_preds, cls_preds = net(x.unsqueeze(0))
    try:
        boxes, labels, scores = box_coder.decode(
            loc_preds.data.cpu().squeeze(), F.softmax(cls_preds.squeeze().cpu(), dim=1).data)
    except:print('except in predict')
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle(list(box), outline='red')
    net.train()
    return img


def main(**kwargs):
    opt._parse(kwargs)

    vis = Visualizer(env=opt.env)

    # Model
    print('==> Building model..')
    net = DSOD(num_classes=21)
    start_epoch = 0  # start from epoch 0 or last epoch



    # Dataset
    print('==> Preparing dataset..')
    box_coder = SSDBoxCoder(net)

    trainset = ListDataset(root=opt.data_root,
                           list_file=[opt.voc07_trainval, opt.voc12_trainval],
                           transform=Transform(box_coder, True))

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=opt.batch_size,
                                              shuffle=True,
                                              num_workers=8,
                                              pin_memory=True)

    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    if opt.load_path is not None:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(opt.load_path)
        net.module.load_state_dict(checkpoint['net'])

    criterion = SSDLoss(num_classes=21)
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

    best_map_ = 0
    best_loss = 1e100
    for epoch in range(start_epoch, start_epoch + 200):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        optimizer.zero_grad()
        ix = 0
        for batch_idx, (inputs, loc_targets, cls_targets) in tqdm(enumerate(trainloader)):
            inputs = Variable(inputs.cuda())
            loc_targets = Variable(loc_targets.cuda())
            cls_targets = Variable(cls_targets.cuda())

            loc_preds, cls_preds = net(inputs)
            ix+=1
            loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss.backward()
            train_loss += loss.data[0]
            if (batch_idx+1) % (opt.iter_size) == 0:
            # if True:
                for name,p in net.named_parameters():p.grad.data.div_(ix)
                ix = 0
                optimizer.step()
                optimizer.zero_grad()


                if (batch_idx + 1) % opt.plot_every == 0:
                    vis.plot('loss', train_loss / (batch_idx + 1))

                    img = predict(net, box_coder, os.path.join(opt.data_root, trainset.fnames[batch_idx]))
                    vis.img('predict', np.array(img).transpose(2, 0, 1))

                    if os.path.exists(opt.debug_file):
                        import ipdb
                        ipdb.set_trace()

        # if (epoch+1)%10 == 0 :
        #     state = {
        #             'net': net.module.state_dict(),
        #             # 'map': best_map_,
        #             'epoch': epoch,
        #     }
        #     torch.save(state, opt.checkpoint + '/%s.pth' % epoch)
        # if (epoch+1) % 30 == 0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.1
        current_loss = train_loss/(1+batch_idx)
        if current_loss< best_loss:
            best_loss = current_loss
            torch.save(net.module.state_dict(), '/tmp/dsod.pth')
        

        if (epoch+1)%opt.eval_every ==0:
            net.module.load_state_dict(torch.load('/tmp/dsod.pth'))

            aps = eval(net.module)
            map_ = aps['map']
            if map_ > best_map_:
                print('Saving..')
                state = {
                    'net': net.module.state_dict(),
                    'map': best_map_,
                    'epoch': epoch,
                }
                best_map_ = map_
                if not os.path.isdir(os.path.dirname(opt.checkpoint)):
                    os.mkdir(os.path.dirname(opt.checkpoint))
                best_path = opt.checkpoint + '/%s.pth' % best_map_
                torch.save(state, best_path)
            else:
                net.module.load_state_dict(torch.load(best_path)['net'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
            vis.log(dict(epoch=(epoch+1),map=map_,loss=train_loss / (batch_idx + 1)))
            
            


if __name__ == '__main__':
    import fire

    fire.Fire()
