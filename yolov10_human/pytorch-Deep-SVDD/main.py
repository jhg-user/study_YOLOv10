# train.py에 정의된 TrainerDeepSVDD 불러서 사용
import numpy as np
import argparse 
import torch

from train import TrainerDeepSVDD
# from preprocess import get_mnist
from preprocess import custom_yolo_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=150,
                        help="number of epochs")
    parser.add_argument("--num_epochs_ae", type=int, default=150,
                        help="number of epochs for the pretraining")
    parser.add_argument("--patience", type=int, default=50, 
                        help="Patience for Early Stopping")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.5e-6,
                        help='Weight decay hyperparameter for the L2 regularization')
    parser.add_argument('--weight_decay_ae', type=float, default=0.5e-3,
                        help='Weight decay hyperparameter for the L2 regularization')
    parser.add_argument('--lr_ae', type=float, default=1e-4,
                        help='learning rate for autoencoder')
    parser.add_argument('--lr_milestones', type=list, default=[50],
                        help='Milestones at which the scheduler multiply the lr by 0.1')
    parser.add_argument("--batch_size", type=int, default=200, 
                        help="Batch size")
    parser.add_argument('--pretrain', type=bool, default=True,
                        help='Pretrain the network using an autoencoder')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Dimension of the latent variable z')
    parser.add_argument('--normal_class', type=int, default=0,
                        help='Class to be treated as normal. The rest will be considered as anomalous.')
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                        help='Interval for saving checkpoints')
    #parsing arguments.
    args = parser.parse_args() 

    #check if cuda is available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # data = get_mnist(args)
    data = custom_yolo_loader(args)

    deep_SVDD = TrainerDeepSVDD(args, data, device)

    if args.pretrain:
        deep_SVDD.pretrain()
    deep_SVDD.train()


# --num_epochs: 전체 학습 에포크 수
# --num_epochs_ae: 오토인코더 사전 학습 에포크 수
# --patience: 조기 종료를 위한 patience
# --lr: 학습률
# --weight_decay: L2 정규화를 위한 weight decay
# --weight_decay_ae: 오토인코더의 L2 정규화를 위한 weight decay
# --lr_ae: 오토인코더의 학습률
# --lr_milestones: 학습률 조정 단계
# --batch_size: 배치 크기
# --pretrain: 사전 학습 여부 (True 또는 False)
# --latent_dim: 잠재 변수의 차원
# --normal_class: 정상 클래스로 간주할 클래스 이름 (여기서는 'human-face')
