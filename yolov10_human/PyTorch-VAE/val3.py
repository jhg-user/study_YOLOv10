import os
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from models.beta_vae import BetaVAE
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from PIL import Image

# 데이터셋 클래스 정의
class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = sorted(self.data_dir.glob('*.jpg'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = default_loader(img_path)
        if self.transform:
            img = self.transform(img)
        return img

# 모델 로드 함수
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

# 이미지 전처리 함수
def preprocess_image(image, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image)
    return image.unsqueeze(0)

# 재구성 오류 계산 함수
def calculate_reconstruction_error(model, image_tensor):
    with torch.no_grad():
        recons, _, _, _ = model(image_tensor)
        recons_error = torch.mean((recons - image_tensor) ** 2, dim=[1, 2, 3])
    return recons_error.item()

# 모델 평가 함수
def evaluate_model(model, normal_loader, anomal_loader):
    normal_errors = []
    anomal_errors = []

    for batch in normal_loader:
        batch = batch.to(torch.device('cpu'))
        error = calculate_reconstruction_error(model, batch)
        normal_errors.append(error)

    for batch in anomal_loader:
        batch = batch.to(torch.device('cpu'))
        error = calculate_reconstruction_error(model, batch)
        anomal_errors.append(error)

    avg_normal_error = np.mean(normal_errors)
    avg_anomal_error = np.mean(anomal_errors)

    return avg_normal_error, avg_anomal_error, normal_errors, anomal_errors

# 메인 함수
def main():
    config_path = "config.yaml"
    model_path = "/home/hkj/yolov10pj/yolov10_human/PyTorch-VAE/logs/BetaVAE/version_57/checkpoints/model/final_model.pth"
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    model = load_model(model_path)
    
    val_transforms = transforms.Compose([
        transforms.Resize((config['data_params']['image_size'], config['data_params']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    normal_dataset = MyDataset(
        data_dir=os.path.join(config['data_params']['data_path'], 'val/human-face'),
        transform=val_transforms
    )

    anomal_dataset = MyDataset(
        data_dir=os.path.join(config['data_params']['data_path'], 'val/non-human-face'),
        transform=val_transforms
    )

    normal_loader = DataLoader(normal_dataset, batch_size=1, shuffle=False)
    anomal_loader = DataLoader(anomal_dataset, batch_size=1, shuffle=False)

    avg_normal_error, avg_anomal_error, normal_errors, anomal_errors = evaluate_model(model, normal_loader, anomal_loader)

    print(f"Validation Results - Normal: {avg_normal_error:.4f}, Anomal: {avg_anomal_error:.4f}")

    # ROC AUC 계산
    y_true = [0] * len(normal_errors) + [1] * len(anomal_errors)
    y_scores = normal_errors + anomal_errors
    roc_auc = roc_auc_score(y_true, y_scores)
    print(f"ROC AUC: {roc_auc:.4f}")

    # 히스토그램 시각화 및 저장
    plt.hist(normal_errors, bins=50, alpha=0.5, label='Normal')
    plt.hist(anomal_errors, bins=50, alpha=0.5, label='Anomal')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.title('Histogram of Reconstruction Errors')
    plt.savefig('reconstruction_errors_histogram.png')
    plt.close()

    # ROC Curve 시각화 및 저장
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png')
    plt.close()

if __name__ == "__main__":
    main()

