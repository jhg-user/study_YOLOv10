import torch
from models.beta_vae import BetaVAE
from experiment import VAEXperiment

# Load the best checkpoint
checkpoint_path = "/home/hkj/yolov10pj/yolov10_human/PyTorch-VAE/logs/BetaVAE/version_60/checkpoints/epoch=0-step=18444.ckpt"  # 가장 좋은 체크포인트 경로를 지정하세요

# 모델 및 실험 객체를 다시 초기화합니다.
model = BetaVAE(**config['model_params'])
experiment = VAEXperiment.load_from_checkpoint(checkpoint_path, model=model, exp_params=config['exp_params'])

# 최종 모델 저장
final_model_path = os.path.join(tb_logger.log_dir, "final_model.pth")
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")

