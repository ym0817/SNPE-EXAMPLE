# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 9:23  2022-02-26
from darknet19 import DarkNet
from darknet53 import darknet53
from dataloder import ImageLoader
from torch.utils.data import DataLoader
import dataloder
import torch
import numpy as np


def main():
    class_num = len(dataloder.TYPE)+len(dataloder.COLOR)
    net = DarkNet(class_num)
    # net = darknet53(class_num=class_num)
    stat_dict = torch.load("darkent19_best.pt")
    del stat_dict["layer7.weight"]
    net.load_state_dict(stat_dict, strict=False)
    print("load pretrained weight")
    batch_size = 10
    EPOCH = 200
    device = "cuda"

    traindir = r"./train_dataset"
    valdir = r"./test_data"
    trainset = ImageLoader(traindir)
    trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    valset = ImageLoader(valdir)
    valLoader = DataLoader(valset, batch_size=batch_size, shuffle=True)

    net.cuda()
    momentum = 0.9
    weight_decay = 5.0e-4
    base_lr = 0.001
    optimizer = torch.optim.SGD(net.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)

    net.to(device)
    loss_fn = torch.nn.BCELoss(size_average=False)
    best_acc = np.inf
    for epoch in range(EPOCH):
        net.train()
        if epoch in [100, 150, 180]:
            base_lr = base_lr*0.1
            for param in optimizer.param_groups:
                param["lr"] = base_lr

        for iter_i, batch in enumerate(trainLoader):
            img, label = batch[0].float().to(device), batch[1].float().to(device)
            label = label.squeeze(1)
            pred = net(img)
            loss = loss_fn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter_i % 5 == 0:
                print("Epoch: %d , loss : %.3f , lr : %.8f" %
                                        (epoch, loss.item(),
                                         [param_group['lr'] for param_group in optimizer.param_groups][0]))

        net.eval()
        with torch.no_grad():
            tempLoss = 0
            for batch in valLoader:
                img, label = batch[0].float().to(device), batch[1].float().to(device)
                label = label.squeeze(1)
                pred = net(img)
                tempLoss += loss_fn(pred, label)

            tempLoss = tempLoss / len(valLoader)
            print("val Epoch: %d , val loss : %.3f" % (epoch, tempLoss))
            if best_acc > tempLoss:
                best_acc = tempLoss
                print('get best test loss %.5f' % best_acc)
                torch.save(net.state_dict(), 'best.pt')


if __name__ == '__main__':
    main()
