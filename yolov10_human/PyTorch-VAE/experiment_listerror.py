import os
import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        #self.hold_graph = False
        self.hold_graph = self.params.get('retain_first_backpass', False)
        self.automatic_optimization = False  # Set to False for manual optimization
        #try:
        #    self.hold_graph = self.params['retain_first_backpass']
        #except:
        #    pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    #def training_step(self, batch, batch_idx, optimizer_idx = 0):
    def training_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              #M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              M_N=self.params.get('kld_weight', 1.0),
                                              #optimizer_idx=optimizer_idx,
                                              optimizer_idx=0,
                                              batch_idx = batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        ##
        # Manual optimization step
        opt1, *rest = self.optimizers()
        opt1.zero_grad()
        self.manual_backward(train_loss['loss'])
        opt1.step()

        if len(rest) > 0:
            opt2 = rest[0]
            opt2.zero_grad()
            # Example: You might want to call a backward pass for the second optimizer here
            self.manual_backward(train_loss['loss'])
            opt2.step()
        ##
        return train_loss['loss']

    #def validation_step(self, batch, batch_idx, optimizer_idx = 0):
    def validation_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            #optimizer_idx = optimizer_idx,
                                            optimizer_idx=0,
                                            batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

        
    def on_validation_end(self) -> None:
        self.sample_images()
        
    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

#         test_input, test_label = batch
        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir , 
                                           "Samples",      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):

        #optims = []
        #scheds = []
        optimizers = []
        schedulers = []

        optimizer = optim.Adam(self.model.parameters(),
                               #lr=self.params['LR'],
                               lr=self.params.get('LR', 0.001),
                               #weight_decay=self.params['weight_decay'])
                               #weight_decay=self.params.get('weight_decay', 0))
                               weight_decay=self.params.get('weight_decay', 0.0))
        #optims.append(optimizer)
        optimizers.append(optimizer)
        ##
        if self.params.get('LR_2') is not None:
            optimizer2 = optim.Adam(getattr(self.model, self.params.get('submodel')).parameters(),
                                    lr=self.params.get('LR_2'))
            optimizers.append(optimizer2)

        #scheduler_configs = []

        if self.params.get('scheduler_gamma') is not None:
            #scheduler_configs.append({
            schedulers.append({
                'scheduler': optim.lr_scheduler.ExponentialLR(optimizers[0],
                                                             gamma=self.params.get('scheduler_gamma')),
                'interval': 'epoch'
            })

        if self.params.get('scheduler_gamma_2') is not None and len(optimizers) > 1:
             #scheduler_configs.append({
             schedulers.append({
                'scheduler': optim.lr_scheduler.ExponentialLR(optimizers[1],
                                                                 gamma=self.params.get('scheduler_gamma_2')),
                'interval': 'epoch'
            })

        return {
            'optimizer': optimizers,
            'lr_scheduler': schedulers
            #'lr_scheduler': scheduler_configs
        }

'''
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
'''
