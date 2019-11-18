# -*- coding: utf-8 -*-
import os
import sys
import argparse
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
sys.path.append('../')
from ialgebra.benchmark.data_utils import load_data
from ialgebra.benchmark.model_utils import load_model, train, test


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(args):
    # load dataset     parallel_model not done
    trainloader = load_data(args.data_dir, args.dataset, 'train', batch_size = args.batch_size, shuffle=True)
    testloader = load_data(args.data_dir, args.dataset, 'test', batch_size = args.batch_size, shuffle=False)
    model, parallel_model = load_model(args)

    if args.dataset in ['imagenet']:       # imagenet
        model.load_official_weights()
        print("###########################this is a test###########################")
        for epoch in range(args.epoch):
            criterion = nn.CrossEntropyLoss()

            test(args, epoch, model, testloader, criterion)

    else:    # cifar10
        
        # load model checkpoint
        if args.resume:
            assert os.path.isdir(os.path.join(args.model_dir)), 'Error: no checkpoint directory found!'
            model_save_dir = os.path.join(args.model_dir, 'ckpt_{}_{}_{}.t7'.format(args.dataset, args.model_name, args.layer))
            model.load_state_dict(torch.load(model_save_dir))
            model = model.to(device)
            
        # start train model
        else:
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
            if args.lr_scheduler:
                lr_milestones = [int(i) for i in args.lr_milestones]
                scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=args.gamma)
            start = time()
            for epoch in range(args.epoch):
                train(epoch, model, trainloader, criterion, optimizer)
                test(args, epoch, model, testloader, criterion)
                if args.lr_scheduler:
                    scheduler.step()
            stop = time()
            print("Model: {} - Total Time: {} sec".format(args.model_dir, str(stop-start)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir', default='../data/')
    parser.add_argument('--model_dir', dest='model_dir', default='../checkpoints/')
    parser.add_argument('-d', '--dataset', dest='dataset', default='cifar10', help='{mnist, cifar10, imagenet}')
    parser.add_argument('-m', '--model_name', dest='model_name', default=None)
    parser.add_argument('--layer', dest='layer', default=None, type=int)
    parser.add_argument('-e', '--epoch', dest='epoch', default=100, type=int)
    parser.add_argument('-b', '--batch_size', dest='batch_size', default=128, type=int)
    parser.add_argument('-lr', '--learning_rate', dest='learning_rate', default=None, type=float)
    parser.add_argument('--lr_scheduler', action='store_true', dest='lr_scheduler', default=False)
    parser.add_argument('--gamma', dest='gamma', default=0.1, type=float)
    parser.add_argument('--lr_milestones', dest='lr_milestones', default=[], nargs='+')
    parser.add_argument('--resume', action='store_true', dest='resume', default=False)
    args = parser.parse_args()
    print(args.__dict__)
    main(args)