import yaml
from pytorch_lightning import Trainer
from models import BetaVAE  # PyTorch-VAE에서 제공하는 모델 사용
from dataset import VAEDataset
import torch

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Set random seed
torch.manual_seed(config['exp_params']['manual_seed'])

# Get data loaders
data_module = VAEDataset(
    data_path=config['data_params']['root_dir'],
    train_batch_size=config['data_params']['batch_size'],
    val_batch_size=config['data_params']['batch_size'],
    image_size=config['data_params']['image_size']
)

# Initialize model
model = BetaVAE(
    image_size=config['data_params']['image_size'],
    in_channels=3,  # Assuming RGB images
    latent_dim=128,
    beta=4  # Hyperparameter for BetaVAE
)

# Initialize trainer
trainer = Trainer(
    max_epochs=config['trainer_params']['max_epochs'],
    gpus=config['trainer_params']['gpus'],
    accelerator=config['trainer_params']['accelerator']
)

# Train the model
trainer.fit(model, datamodule=data_module)

