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
        self.net = None
        self.optimizer = None
        self.scheduler = None
        self.c = None  # 추가: 초기화


    def pretrain(self):
        """ Pretraining the weights for the deep SVDD network using autoencoder"""
        ae = autoencoder(self.args.latent_dim).to(self.device)

        ae.apply(weights_init_normal)
        optimizer = optim.Adam(ae.parameters(), lr=self.args.lr_ae,
                               weight_decay=self.args.weight_decay_ae)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                    milestones=self.args.lr_milestones, gamma=0.1)

        ae.train()
        for epoch in range(self.args.num_epochs_ae):
            total_loss = 0
            for x, _ in self.train_loader:
                x = x.float().to(self.device)

                optimizer.zero_grad()
                x_hat = ae(x)

                reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                optimizer.step()

                total_loss += reconst_loss.item()
            scheduler.step()
            print('Pretraining Autoencoder... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.train_loader)))
        # Autoencoder 모델 저장
        torch.save(ae.state_dict(), 'autoencoder_human_face2.pth')
        print('Autoencoder model saved to autoencoder_human_face.pth')

        self.save_weights_for_DeepSVDD(ae, self.train_loader)


    def save_weights_for_DeepSVDD(self, model, dataloader):
        """Initialize Deep SVDD weights using the encoder weights of the pretrained autoencoder."""
        c = self.set_c(model, dataloader)
        net = network(self.args.latent_dim).to(self.device)
        state_dict = model.state_dict()
        net.load_state_dict(state_dict, strict=False)
        torch.save({'center': c.cpu().data.numpy().tolist(),
                    'net_dict': net.state_dict()}, 'weights/pretrained_parameters2.pth')


    def set_c(self, model, dataloader, eps=0.1):
        """Initializing the center for the hypersphere"""
        model.eval()
        z_ = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.float().to(self.device)
                z = model.encode(x)
                z_.append(z.detach())
        z_ = torch.cat(z_).to(self.device)  # 추가: GPU로 이동
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c


    def train(self):
        """Training the Deep SVDD model"""
        self.net = network(self.args.latent_dim).to(self.device)

        if self.args.pretrain:
            state_dict = torch.load('weights/pretrained_parameters2.pth')
            self.net.load_state_dict(state_dict['net_dict'])
            self.c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            self.net.apply(weights_init_normal)
            self.c = torch.randn(self.args.latent_dim).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=self.args.lr_milestones, gamma=0.1)

        self.net.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _ in self.train_loader:
                x = x.float().to(self.device)

                self.optimizer.zero_grad()
                z = self.net(x)
                loss = torch.mean(torch.sum((z - self.c) ** 2, dim=1))
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            self.scheduler.step()
            print('Training Deep SVDD... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.train_loader)))

            if (epoch + 1) % self.args.checkpoint_interval == 0:
                self.save_checkpoint(epoch)

        self.save_model_and_center('weights/final_deep_svdd2.pth')


    def save_model_and_center(self, path):
        """Save the trained model and center to a file."""
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({'net_dict': self.net.state_dict(),
                    'center': self.c.cpu().data.numpy().tolist()},
                   path)
        print(f'Model and center saved to {path}')


    def save_checkpoint(self, epoch, path='weights/checkpoint2.pth'):
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

