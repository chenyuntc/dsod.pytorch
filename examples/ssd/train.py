from __future__ import print_function
import matplotlib
matplotlib.use('agg')
import os
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
b
import torchvision
import torchvision.transforms as transforms

from PIL import Image
from torch.autograd import Variable

# from torchcv.models.fpnssd import FPNSSD512
from torchcv.models.ssd import  SSDBoxCoder
import numpy as np
from torchcv.loss import SSDLoss
from torchcv.datasets import ListDataset
from torchcv.transforms import resize, random_flip, random_paste, random_crop, random_distort
from torchcv.models import DSOD
from torchcv.evaluations.voc_eval import voc_eval

class Config:




parser = argparse.ArgumentParser(description='PyTorch SSD Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# parser.add_argument('--model', default='/home/claude.cy/code/tmp/torchcv/fpnssd512_21_fpn50.pth', type=str, help='initialized model path')
parser.add_argument('--checkpoint', default='/home/claude.cy/file/ssd/checkpoint/ckpt.pth', type=str, help='checkpoint path')
args = parser.parse_args()
import ipdb
ipdb.set_trace()
# Model
print('==> Building model..')
net = DSOD(num_classes=21)
start_epoch = 0  # start from epoch 0 or last epoch
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

# Dataset
print('==> Preparing dataset..')
box_coder = SSDBoxCoder(net)
img_size = 300
def transform_train(img, boxes, labels):
    img = random_distort(img)
    if random.random() < 0.5:
        img, boxes = random_paste(img, boxes, max_ratio=4, fill=(123,116,103))
    img, boxes, labels = random_crop(img, boxes, labels)
    img, boxes = resize(img, boxes, size=(img_size,img_size), random_interpolation=True)
    img, boxes = random_flip(img, boxes)
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels

trainset = ListDataset(root='/home/claude.cy/.data/all_images',
                       list_file=['torchcv/datasets/voc/voc07_trainval.txt',
                                  'torchcv/datasets/voc/voc12_trainval.txt'],
                       transform=transform_train)

def transform_test(img, boxes, labels):
    img, boxes = resize(img, boxes, size=(img_size,img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels

testset = ListDataset(root='/home/claude.cy/.data/all_images',
                      list_file='torchcv/datasets/voc/voc07_test.txt',
                      transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=8)

net.cuda()
#net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True

criterion = SSDLoss(num_classes=21)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        print('train_loss: %.3f | avg_loss: %.3f [%d/%d]'
              % (loss.data[0], train_loss/(batch_idx+1), batch_idx+1, len(trainloader)))

# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
        inputs = Variable(inputs.cuda(), volatile=True)
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        test_loss += loss.data[0]
        print('test_loss: %.3f | avg_loss: %.3f [%d/%d]'
              % (loss.data[0], test_loss/(batch_idx+1), batch_idx+1, len(testloader)))

    # Save checkpoint
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir(os.path.dirname(args.checkpoint)):
            os.mkdir(os.path.dirname(args.checkpoint))
        torch.save(state, args.checkpoint)
        best_loss = test_loss


def eval(net):

    net.eval()
    def transform(img, boxes, labels):
        img, boxes = resize(img, boxes, size=(img_size,img_size))
        img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        ])(img)
        return img, boxes, labels

    dataset = ListDataset(root='/home/claude.cy/.data/all_images', \
                        list_file='torchcv/datasets/voc/voc07_test.txt',
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

   
    for i, (inputs, box_targets, label_targets) in enumerate(dataloader):
        print('%d/%d' % (i, len(dataloader)))
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

    aps = (voc_eval(
        pred_boxes, pred_labels, pred_scores,
        gt_boxes, gt_labels, gt_difficults,
        iou_thresh=0.5, use_07_metric=True))
    net.train()
    return aps


for epoch in range(start_epoch, start_epoch+200):
    best_map_ = 0
    train(epoch)
    # test(epoch)
    aps = eval(net)
    map_ = aps['map']

    # Save checkpoint

    if map_ < best_map_:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'map': best_map_,
            'epoch': epoch,
        }
        best_map_ = map_
        if not os.path.isdir(os.path.dirname(args.checkpoint)):
            os.mkdir(os.path.dirname(args.checkpoint))
        torch.save(state, args.checkpoint)

    

