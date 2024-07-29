import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

# Define the Autoencoder class (same as provided)
class autoencoder(nn.Module):
    def __init__(self, z_dim=32):
        super(autoencoder, self).__init__()
        self.z_dim = z_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, z_dim, bias=False)

        #self.deconv1 = nn.ConvTranspose2d(z_dim // 16, 4, 5, bias=False, padding=2)
        self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)

    def encode(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

    def decode(self, x):
        x = x.view(x.size(0), int(self.z_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        return torch.sigmoid(x)


    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


def load_model(model_path, device):
    model = Autoencoder(z_dim=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def calculate_reconstruction_error(original, reconstructed):
    return F.mse_loss(reconstructed, original, reduction='none').mean(dim=[1,2,3])

def anomaly_detection(model, dataloader, device):
    reconstruction_errors = []
    labels = []

    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            
            reconstructed = model(data)
            error = calculate_reconstruction_error(data, reconstructed)
            
            reconstruction_errors.extend(error.cpu().numpy())
            labels.extend(target.cpu().numpy())

    return np.array(reconstruction_errors), np.array(labels)

def evaluate_anomaly_detection(errors, labels):
    auc_score = roc_auc_score(labels, errors)
    return auc_score

# Usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'autoencoder.pth'
model = load_model(model_path, device)

# Assuming test_loader is defined and provides (data, labels) tuples
errors, labels = anomaly_detection(model, test_loader, device)
auc_score = evaluate_anomaly_detection(errors, labels)

print(f'ROC AUC Score: {auc_score}')

