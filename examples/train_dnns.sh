# CUDA_VISIBLE_DEVICES=2,3 python train_dnns.py -d 'mnist' -m 'lenet' -e 200 -b 64 -lr 0.01 
# CUDA_VISIBLE_DEVICES=2,3 python train_dnns.py -d 'cifar10' -m 'resnet18' -e 350 -b 64 -lr 0.1 --lr_scheduler --lr_milestones 150 250
# CUDA_VISIBLE_DEVICES=2,3 python train_dnns.py -d 'cifar10' -m 'resnet50' -e 350 -b 64 -lr 0.1 --lr_scheduler --lr_milestones 150 250
# CUDA_VISIBLE_DEVICES=2,3 python train_dnns.py -d 'cifar10' -m 'vgg19' -e 350 -b 128 -lr 0.1 --lr_scheduler --lr_milestones 150 250
# CUDA_VISIBLE_DEVICES=2,3 python train_dnns.py -d 'cifar10' -m 'densenet121' -e 350 -b 128 -lr 0.1 --lr_scheduler --lr_milestones 150 250
# CUDA_VISIBLE_DEVICES=2,3 python train_dnns.py -d 'imagenet' -m 'resnet18' -e 350 -b 128 -lr 0.1 --lr_scheduler --lr_milestones 150 250

# CUDA_VISIBLE_DEVICES=2,3 python train_dnns.py -d 'cifar10' -m 'resnet50' -e 350 -b 128 -lr 0.1 --lr_scheduler --lr_milestones 150 250
# CUDA_VISIBLE_DEVICES=2,3 python train_dnns.py -d 'cifar10' -m 'resnet18' -e 350 -b 128 -lr 0.1 --lr_scheduler --lr_milestones 150 250
# CUDA_VISIBLE_DEVICES=2,3 python train_dnns.py -d 'imagenet' -m 'densenet121' -e 350 -b 128 -lr 0.1 --lr_scheduler --lr_milestones 150 250 --layer 4


######################### CIFAR10 #########################
# CIFAR10-vgg19  wrong XXXX
CUDA_VISIBLE_DEVICES=2,3 python train_dnns.py -d 'cifar10' -m 'vgg19' -e 350 -b 128 -lr 0.1 --lr_scheduler --gamma 0.1 --lr_milestones 150 250

# CIFAR10-resnet18
CUDA_VISIBLE_DEVICES=2,3 python train_dnns.py -d 'cifar10' -m 'resnet18' -e 350 -b 128 -lr 0.1 --lr_scheduler --gamma 0.1 --lr_milestones 150 250 --layer 3

# CIFAR10-resnet50 ing
CUDA_VISIBLE_DEVICES=2,3 python train_dnns.py -d 'cifar10' -m 'resnet50' -e 350 -b 128 -lr 0.1 --lr_scheduler --gamma 0.1 --lr_milestones 150 250

# CIFAR10-densenet121 ing
CUDA_VISIBLE_DEVICES=2,3 python train_dnns.py -d 'cifar10' -m 'densenet121' -e 350 -b 128 -lr 0.1 --lr_scheduler --gamma 0.1 --lr_milestones 150 250 --layer 2

######################### ImageNet #########################
# ImageNet-vgg19

# ImageNet-resnet18

# ImageNet-resnet50

# ImageNet-densenet121



######################### Mnist #########################
# MNist-LeNet
CUDA_VISIBLE_DEVICES=2,3 python train_dnns.py -d 'mnist' -m 'lenet' -e 350 -b 128 -lr 0.1 --lr_scheduler --gamma 0.1 --lr_milestones 150 250 --layer 1