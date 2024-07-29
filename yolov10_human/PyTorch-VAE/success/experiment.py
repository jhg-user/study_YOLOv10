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
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params.get('kld_weight', 1.0),
                                              optimizer_idx=0,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        # Manual optimization step
        opt1, *rest = self.optimizers()
        opt1.zero_grad()
        self.manual_backward(train_loss['loss'])
        opt1.step()

        if len(rest) > 0:
            opt2 = rest[0]
            opt2.zero_grad()
            self.manual_backward(train_loss['loss'])
            opt2.step()

        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        # 배치에서 이미지 데이터만 추출
        if isinstance(batch, tuple) and len(batch) == 2:
            real_img, labels = batch
        else:
            real_img = batch
            labels = None

        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(*results,
                                            M_N=1.0,
                                            optimizer_idx=0,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_images()

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        if test_label is not None:
            test_label = test_label.to(self.curr_device)

        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir,
                                       "Reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels=test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir,
                                           "Samples",
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):
        optimizers = []
        schedulers = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params.get('LR', 0.001),
                               weight_decay=self.params.get('weight_decay', 0.0))
        optimizers.append(optimizer)

        if self.params.get('LR_2') is not None:
            optimizer2 = optim.Adam(getattr(self.model, self.params.get('submodel')).parameters(),
                                    lr=self.params.get('LR_2'))
            optimizers.append(optimizer2)

        if self.params.get('scheduler_gamma') is not None:
            schedulers.append({
                'scheduler': optim.lr_scheduler.ExponentialLR(optimizers[0],
                                                              gamma=self.params.get('scheduler_gamma')),
                'interval': 'epoch'
            })

        if self.params.get('scheduler_gamma_2') is not None and len(optimizers) > 1:
            schedulers.append({
                'scheduler': optim.lr_scheduler.ExponentialLR(optimizers[1],
                                                              gamma=self.params.get('scheduler_gamma_2')),
                'interval': 'epoch'
            })

        # 반드시 모든 schedulers가 optimizer를 참조할 수 있도록 딕셔너리로 반환해야 합니다.
        return optimizers, schedulers

# 데이터셋과 데이터 로더 설정
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# CelebA 데이터셋 사용
dataset = CelebA(root='data/', split='train', transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

# 모델 설정
model = BetaVAE()

# 실험 파라미터 설정
params = {
    'LR': 0.001,
    'kld_weight': 0.00025,
    'retain_first_backpass': False,
}

# PyTorch Lightning 트레이너 설정
trainer = pl.Trainer(gpus=1, max_epochs=50, progress_bar_refresh_rate=20)

# 실험 실행
experiment = VAEXperiment(model, params)
trainer.fit(experiment, dataloader)

