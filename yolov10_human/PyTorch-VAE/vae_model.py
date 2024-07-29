import torch
from models.beta_vae import BetaVAE  # VAE 모델을 import

'''
def load_vae_model(vae_checkpoint_path):
    model = BetaVAE(in_channels=3, latent_dim=64)
    checkpoint = torch.load(vae_checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model
'''

import torch
from models.beta_vae import BetaVAE

def load_vae_model(checkpoint_path):
    model = torch.load(checkpoint_path)
    model.eval()
    return model

