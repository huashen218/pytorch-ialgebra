
CUDA_VISIBLE_DEVICES=2,3 python run_ialgebra.py --dataset 'cifar10' --model_name 'vgg19' --layer '4' --identity_name 'smoothgrad' --operator_name 'join'


