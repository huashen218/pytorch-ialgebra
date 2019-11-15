import torch.nn.functional as F
from torch import optim
from models.resnet import resnet50
import torch.nn as nn
import torchvision
import torch
import navi.utils
import os
from navi.utils import Model


def train(model, train_data, val_data, layer):
    optimizer = optim.SGD(model.fc.parameters(), lr=1e-2)
    model.to('cuda')
    model.fc.to('cuda')

    for e in range(10):
        model.train(True)
        model.fc.train(True)
        train_loss = 0
        correct = 0
        total = 0
        test_loss = 0
        correct_test = 0
        total_test = 0
        best_acc = 0
        for batch_idx, (bx, by) in enumerate(train_data):

            bx, by = torch.tensor(bx).to('cuda'), torch.tensor(by).to('cuda')
            pre_bx = navi.utils.imagenet_normalize(bx)
            # print(pre_bx.shape)
            layer_ = model.get_layer(pre_bx)
            logits = model.linear_layer(layer_)
            loss = F.nll_loss(F.log_softmax(logits), by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += by.size(0)
            correct += predicted.eq(by).sum().item()
            if batch_idx % 50 == 0:
                print('training epoch: %d, batch_id: %d, loss: %.2f, accuracy: %.5f' % (
                e, batch_idx, (train_loss / (batch_idx + 1)), 100.0 * float(correct) / float(total)))
        model.train(False)
        model.fc.train(False)
        if e % 1 == 0:
            # if batch_idx % 10 == 0:
                # model.train(False)
            for batch_idx_val, (bx_val, by_val) in enumerate(val_data):
                bx_val, by_val = torch.tensor(bx_val).to('cuda'), torch.tensor(by_val).to('cuda')
                pre_bx_val = navi.utils.imagenet_normalize(bx_val)
                layer_val = model.get_layer(pre_bx_val)
                logits_val = model.linear_layer(layer_val)
                loss_val = F.nll_loss(F.log_softmax(logits_val), by_val)

                test_loss += loss_val.item()
                _, predicted_val = logits_val.max(1)
                total_test += by_val.size(0)
                correct_test += predicted_val.eq(by_val).sum().item()
                # if batch_idx_val == 6:
                #     break
                if batch_idx_val % 5 == 0:
                    print('testing epoch: %d, batch_id: %d, loss: %.2f, accuracy: %.5f' % (
                    e, batch_idx_val, (test_loss / (batch_idx_val + 1)), 100.0 * float(correct_test) / float(total_test)))
            acc = 100. * correct_test / total_test
            if acc > best_acc:
                print('Saving..')
                state = {
                    'net': model.state_dict(),
                    'net_fc': model.fc.state_dict(),
                    'acc': acc,
                    'epoch': e,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, '/home/ningfei/interpretalg/data/navi_layer2.t7')
                best_acc = acc
    #     if batch_idx == 1000:
    #         break
    # break



if __name__ == '__main__':
    # imagenet_data = torchvision.datasets.ImageNet('/home/xinyang/Datasets/imagenet_1000/', split='train')
    imagenet_data = torchvision.datasets.ImageFolder('/home/xinyang/Datasets/imagenet_1000/', transform=navi.utils.TEST_TRANSFORM)
    data_loader_train = torch.utils.data.DataLoader(imagenet_data,
                                                batch_size=64,
                                                shuffle=True)
    # imagenet_data = torchvision.datasets.ImageNet('/home/xinyang/Datasets/imagenet_val/', split='val')
    imagenet_data = torchvision.datasets.ImageFolder('/home/xinyang/Datasets/imagenet_val/',
                                            transform=navi.utils.TEST_TRANSFORM)
    data_loader_val = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=32,
                                              shuffle=True)
    print('training')
    layer = 'layer2'
    model = Model(layer, 512)
    # layer = 1
    train(model, data_loader_train,data_loader_val, layer)

