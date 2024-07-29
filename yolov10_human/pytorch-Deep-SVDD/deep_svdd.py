import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, z_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), # Conv1
            nn.BatchNorm2d(16), # BN1
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # Conv2
            nn.BatchNorm2d(32), # BN2
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, z_dim)  # FC1 (adjust size accordingly)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 32 * 8 * 8), # FC2
            nn.ReLU(),
            nn.Unflatten(1, (32, 8, 8)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # Deconv1
            nn.BatchNorm2d(16), # BN3
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1) # Deconv2
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class DeepSVDD(nn.Module):
    def __init__(self, center, radius, autoencoder_model):
        super(DeepSVDD, self).__init__()
        self.center = center
        self.radius = radius
        self.autoencoder_model = autoencoder_model

    def forward(self, x):
        # Example forward pass to compute anomaly score
        encoded = self.autoencoder_model.encode(x)
        distance = torch.norm(encoded - self.center, dim=1)
        return distance
