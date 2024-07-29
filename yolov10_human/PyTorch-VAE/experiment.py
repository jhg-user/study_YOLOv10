import os
import torch
from torch import optim
import pytorch_lightning as pl
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA
from models import BaseVAE
from models.types_ import Tensor
from utils import data_loader

class VAEXperiment(pl.LightningModule):
    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = self.params.get('retain_first_backpass', False)
        self.automatic_optimization = False  # Set to False for manual optimization

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        real_img = batch
        self.curr_device = real_img.device

        results = self.forward(real_img)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params.get('kld_weight', 1.0),
                                              optimizer_idx=0,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        # Manual optimization step
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(train_loss['loss'])
        optimizer.step()

        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        real_img = batch
        self.curr_device = real_img.device

        results = self.forward(real_img)
        val_loss = self.model.loss_function(*results,
                                            M_N=1.0,
                                            optimizer_idx=0,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_images()

    def sample_images(self):
        # Get sample reconstruction image
        test_input = next(iter(self.trainer.datamodule.val_dataloader()))
        test_input = test_input.to(self.curr_device)

        recons = self.model.generate(test_input)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir,
                                       "Reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir,
                                           "Samples",
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params.get('LR', 0.001),
                               weight_decay=self.params.get('weight_decay', 0.0))

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.params.get('scheduler_gamma', 0.95))

        return [optimizer], [scheduler]

