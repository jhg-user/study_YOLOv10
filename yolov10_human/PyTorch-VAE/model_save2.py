import os
import yaml
import torch
from models.beta_vae import BetaVAE
from experiment import VAEXperiment
from pytorch_lightning.loggers import TensorBoardLogger

# Argument parsing (필요에 따라 argparse를 사용해도 됩니다)
config_path = "config.yaml"  # 설정 파일 경로를 지정하세요

# Load the config file
with open(config_path, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        exit(1)

# Load the best checkpoint
#checkpoint_path = "/home/hkj/yolov10pj/yolov10_human/PyTorch-VAE/logs/BetaVAE/version_60/checkpoints/last.ckpt"  # 가장 좋은 체크포인트 경로를 지정하세요
checkpoint_path = "/home/hkj/yolov10pj/yolov10_human/PyTorch-VAE/logs/BetaVAE/version_57/checkpoints/last.ckpt"  # 가장 좋은 체크포인트 경로를 지정하세요

# Logger setup
tb_logger = TensorBoardLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['model_params']['name']
)

# 모델 및 실험 객체를 다시 초기화합니다.
model = BetaVAE(**config['model_params'])
experiment = VAEXperiment.load_from_checkpoint(
    checkpoint_path, 
    vae_model=model, 
    params=config['exp_params']
)

# Ensure the model directory exists
model_dir = os.path.join(os.path.dirname(checkpoint_path), "model")
os.makedirs(model_dir, exist_ok=True)

# 최종 모델 저장
final_model_path = os.path.join(model_dir, "final_model.pth")
#torch.save(model.state_dict(), final_model_path)
torch.save(model, final_model_path)
print(f"Final model saved to {final_model_path}")

