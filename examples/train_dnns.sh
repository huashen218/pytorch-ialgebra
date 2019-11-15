# CUDA_VISIBLE_DEVICES=2,3 python train_dnns.py -d 'mnist' -m 'lenet' -e 200 -b 64 -lr 0.01 
# CUDA_VISIBLE_DEVICES=2,3 python train_dnns.py -d 'cifar10' -m 'resnet18' -e 350 -b 64 -lr 0.1 --lr_scheduler --lr_milestones 150 250
# CUDA_VISIBLE_DEVICES=2,3 python train_dnns.py -d 'cifar10' -m 'resnet50' -e 350 -b 64 -lr 0.1 --lr_scheduler --lr_milestones 150 250
# CUDA_VISIBLE_DEVICES=2,3 python train_dnns.py -d 'cifar10' -m 'vgg19' -e 350 -b 128 -lr 0.1 --lr_scheduler --lr_milestones 150 250
# CUDA_VISIBLE_DEVICES=2,3 python train_dnns.py -d 'cifar10' -m 'densenet121' -e 350 -b 128 -lr 0.1 --lr_scheduler --lr_milestones 150 250

# CUDA_VISIBLE_DEVICES=2,3 python train_dnns.py -d 'imagenet' -m 'resnet18' -e 350 -b 128 -lr 0.1 --lr_scheduler --lr_milestones 150 250

CUDA_VISIBLE_DEVICES=2,3 python train_dnns.py -d 'cifar10' -m 'resnet50' -e 350 -b 128 -lr 0.1 --lr_scheduler --lr_milestones 150 250 --layer 'layer2'

CUDA_VISIBLE_DEVICES=2,3 python train_dnns.py -d 'cifar10' -m 'd_resnet50' -e 350 -b 128 -lr 0.1 --lr_scheduler --lr_milestones 150 250 --layer 'layer2'