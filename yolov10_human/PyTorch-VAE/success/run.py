import os
import yaml
import argparse
from pathlib import Path
from models import vae_models
from experiment import VAEXperiment
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset

# Argument parsing
parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='config.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# Logger setup
tb_logger = TensorBoardLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['model_params']['name']
)

# Ensure logger directory exists
log_dir = Path(tb_logger.log_dir)
log_dir.mkdir(parents=True, exist_ok=True)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

# Check if the model name exists in vae_models
model_name = config['model_params']['name']
if model_name not in vae_models:
    raise ValueError(f"Model name '{model_name}' not found in vae_models")

# Model and experiment setup
model = vae_models[model_name](**config['model_params'])
#experiment = VAEXperiment(model, config['exp_params'])
experiment = VAEXperiment(model, config['exp_params']['optimizer_params'])
print(experiment)

# Data setup
data_params = config["data_params"]
data_params['pin_memory'] = len(config['trainer_params']['gpus']) != 0
data = VAEDataset(**data_params)
data.setup()
print(data)

# Trainer setup
runner = Trainer(
    logger=tb_logger,
    callbacks=[
        LearningRateMonitor(),
        ModelCheckpoint(
            save_top_k=2,
            dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
            monitor="val_loss",
            save_last=True
        ),
    ],
    strategy='ddp',
    devices=config['trainer_params']['gpus'],
    max_epochs=config['trainer_params']['max_epochs'],
    accelerator='gpu'
)

# Create directories for samples and reconstructions
Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

# Start training
print(f"======= Training {model_name} =======")
runner.fit(experiment, datamodule=data)

