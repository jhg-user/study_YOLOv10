import torch
from torch import optim
import torch.nn.functional as F

import numpy as np
# from barbar import Bar

from model import autoencoder, network
from utils.utils import weights_init_normal
import os

class TrainerDeepSVDD:
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device
        self.net = None  # 추가: 초기화
        self.optimizer = None  # 추가: 초기화
        self.scheduler = None  # 추가: 초기화
        self.c = None
    

    # autoencoder를 사용해 사전 학습 수행
    def pretrain(self):
        """ Pretraining the weights for the deep SVDD network using autoencoder"""
        ae = autoencoder(self.args.latent_dim).to(self.device)
        # Deep SVDD에 적용할 가중치 W를 학습하기 위해 autoencoder를 학습함
        # ae = C_AutoEncoder(self.args.latent_dim).to(self.device)

        ae.apply(weights_init_normal)
        optimizer = optim.Adam(ae.parameters(), lr=self.args.lr_ae,
                               weight_decay=self.args.weight_decay_ae)
        # 지정한 step마다 learning rate를 줄여감
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=self.args.lr_milestones, gamma=0.1)

        # AE 학습
        ae.train()
        for epoch in range(self.args.num_epochs_ae):
            total_loss = 0
            for x, _ in (self.train_loader):
            # for x, _ in Bar(self.train_loader):
                x = x.float().to(self.device)
                
                optimizer.zero_grad()
                x_hat = ae(x)

                # 재구축 오차를 최소화하는 방향으로 학습함
                # AE 모델을 통해 그 데이터를 잘 표현할 수 있는 common features를 찾는 것이 목표임
                reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                optimizer.step()
                
                total_loss += reconst_loss.item()
            scheduler.step()
            print('Pretraining Autoencoder... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.train_loader)))
        self.save_weights_for_DeepSVDD(ae, self.train_loader) 
    

    # 가중치 저장
    # autoencoder의 가중치를 Deep SVDD 모델로 옮기고
    # 하이퍼스피어 중심을 설정하여 가중치 저장
    def save_weights_for_DeepSVDD(self, model, dataloader):
        """Initialize Deep SVDD weights using the encoder weights of the pretrained autoencoder."""
        # AE의 encoder 구조의 가중치를 Deep SVDD에 초기화하기 위함임
        c = self.set_c(model, dataloader)
        net = network(self.args.latent_dim).to(self.device)
        state_dict = model.state_dict()
        # 구조가 맞는 부분만 가중치를 load함
        net.load_state_dict(state_dict, strict=False)
        torch.save({'center': c.cpu().data.numpy().tolist(),
                    'net_dict': net.state_dict()}, 'weights/pretrained_parameters.pth')
    

    # 하이퍼스피어의 중심 초기화
    # autoencoder의 인코더 부분을 사용하여 데이터의 평균계산, 작은 값(eps) 조정
    def set_c(self, model, dataloader, eps=0.1):
        """Initializing the center for the hypersphere"""
        # 구의 중심점을 초기화함
        model.eval()
        z_ = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.float().to(self.device)
                z = model.encode(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c


    # Deep SVDD 모델 학습
    def train(self):
        """Training the Deep SVDD model"""
        # AE의 학습을 마치고 그 가중치를 적용한 Deep SVDD를 학습함
        # net = network().to(self.device)
        #net = network(self.args.latent_dim).to(self.device)
        self.net = network(self.args.latent_dim).to(self.device)  # 수정: self.net 초기화
        
        if self.args.pretrain==True: # 사전 학습된 가중치 있으면 로드
            state_dict = torch.load('weights/pretrained_parameters.pth')
            #net.load_state_dict(state_dict['net_dict'])
            self.net.load_state_dict(state_dict['net_dict'])
            #c = torch.Tensor(state_dict['center']).to(self.device)
            self.c = torch.Tensor(state_dict['center']).to(self.device)
        else: # 사전 학습 가중치 없으면 초기화
            # pretrain을 하지 않았을 경우 가중치를 초기화함
            #net.apply(weights_init_normal)
            self.net.apply(weights_init_normal)
            #c = torch.randn(self.args.latent_dim).to(self.device)
            self.c = torch.randn(self.args.latent_dim).to(self.device)  # 수정: self.c 초기화

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=self.args.lr_milestones, gamma=0.1)

        # optimizer = optim.Adam(net.parameters(), lr=self.args.lr,
        #                        weight_decay=self.args.weight_decay)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
        #             milestones=self.args.lr_milestones, gamma=0.1)

        # 평가 메트릭 ->
        self.net.train()
        #net.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _ in (self.train_loader):
            # for x, _ in Bar(self.train_loader):
                x = x.float().to(self.device)

                self.optimizer.zero_grad()
                # optimizer.zero_grad()
                #z = net(x)
                z = self.net(x)
                loss = torch.mean(torch.sum((z - self.c) ** 2, dim=1))
                loss.backward()
                ######self.optimizer.zero_grad()
                #optimizer.step()
                self.optimizer.step()

                total_loss += loss.item()
            self.scheduler.step()
            # scheduler.step()
            print('Training Deep SVDD... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.train_loader)))

            # 주기적으로 체크포인트 저장
            if (epoch + 1) % self.args.checkpoint_interval == 0:
                self.save_checkpoint(epoch)

        # 모델과 하이퍼스피어 중심 저장
        self.save_model_and_center('weights/final_deep_svdd.pth')

        #self.net = net
        #self.c = c

    # 결과 모델 저장
    def save_model_and_center(self, path):
        """Save the trained model and center to a file."""
        torch.save({'net_dict': self.net.state_dict(),
                    'center': self.c.cpu().data.numpy().tolist()},
                   path)
        print(f'Model and center saved to {path}')


    def save_checkpoint(self, epoch, path='weights/checkpoint.pth'):
        """Save a checkpoint of the current training state."""
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'epoch': epoch,
            'net_state_dict': self.net.state_dict(),
            'center': self.c.cpu().data.numpy().tolist(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, path)
        print(f'Checkpoint saved to {path}')

