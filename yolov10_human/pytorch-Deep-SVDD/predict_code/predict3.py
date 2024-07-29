import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from ultralytics import YOLO  # Adjust according to YOLO version

class Autoencoder(nn.Module):
    def __init__(self, z_dim=32):
        super(Autoencoder, self).__init__()
        self.z_dim = z_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, z_dim, bias=False)

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
        x = x.view(x.size(0), 2, 4, 4)  # Adjusted to match output channels of deconv1
        x = F.leaky_relu(self.deconv1(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.leaky_relu(self.bn3(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.leaky_relu(self.bn4(self.deconv2(x)))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.deconv3(x)
        return torch.sigmoid(x)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

class DeepSVDD(nn.Module):
    def __init__(self, autoencoder, center, radius):
        super(DeepSVDD, self).__init__()
        self.autoencoder = autoencoder
        self.center = center
        self.radius = radius

    def forward(self, x):
        encoded = self.autoencoder.encode(x)
        distance = torch.norm(encoded - self.center, p=2, dim=1)
        return distance

def load_autoencoder_and_svdd(autoencoder_path, svdd_path, device):
    autoencoder = Autoencoder(z_dim=32).to(device)
    autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))
    autoencoder.eval()

    # Load svdd model and its parameters
    svdd_center_radius = torch.load(svdd_path, map_location=device)
    
    # Print the type and content of svdd_center_radius to understand its format
    print(f"Type of svdd_center_radius: {type(svdd_center_radius)}")
    if isinstance(svdd_center_radius, dict):
        for key, value in svdd_center_radius.items():
            print(f"Key: {key}, Type: {type(value)}")
    else:
        print("svdd_center_radius is not a dictionary")

    # Check if svdd_center_radius contains 'center' and 'radius'
    if isinstance(svdd_center_radius, dict):
        center = svdd_center_radius.get('center')
        radius = svdd_center_radius.get('radius')

        if center is None or radius is None:
            raise ValueError("Missing 'center' or 'radius' in svdd_center_radius.")

        # Convert lists to tensors if necessary
        if isinstance(center, list):
            center = torch.tensor(center).to(device)
        else:
            center = center.to(device)
        
        if isinstance(radius, list):
            radius = torch.tensor(radius).to(device)
        else:
            radius = radius.to(device)

        svdd = DeepSVDD(autoencoder, center, radius).to(device)
        svdd.eval()
        return svdd

    raise ValueError("svdd_center_radius is not in the expected format")

def detect_objects(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo_model = YOLO('path_to_yolo_model.pt').to(device)  # Adjust as needed
    yolo_model.load_state_dict(torch.load('/home/hkj/yolov10pj/yolov10_human/runs/detect/train4/best.pth', map_location=device))
    yolo_model.eval()

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        predictions = yolo_model(image_tensor)

    boxes = predictions[0]['boxes'].cpu().numpy()
    confidences = predictions[0]['scores'].cpu().numpy()

    return image, boxes, confidences

def extract_patches(image, boxes, size=(64, 64)):
    patches = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        patch = image[y1:y2, x1:x2]
        if patch.shape[0] > 0 and patch.shape[1] > 0:
            patch_resized = cv2.resize(patch, size)
            patches.append(patch_resized)
    return patches

def calculate_reconstruction_error(original, reconstructed):
    return F.mse_loss(reconstructed, original, reduction='none').mean(dim=[1,2,3])

def anomaly_detection_deep_svdd(model, patches, device):
    reconstruction_errors = []
    patches_tensor = torch.stack([torch.tensor(patch).float().unsqueeze(0).to(device) for patch in patches])

    with torch.no_grad():
        encoded_patches = model.autoencoder.encode(patches_tensor)
        distances = model(encoded_patches)
        reconstruction_errors = distances.cpu().numpy()

    return reconstruction_errors

def process_results(reconstruction_errors, threshold=0.5):
    anomalies = reconstruction_errors > threshold
    return anomalies

# Main process
image_path = '/home/hkj/yolov10pj/yolov10_human/dataset/test2/*.jpg'  # Update to a specific image file or handle multiple files
autoencoder_model_path = 'autoencoder_human_face.pth'
svdd_model_path = '/home/hkj/yolov10pj/yolov10_human/pytorch-Deep-SVDD/weights/final_deep_svdd.pth'

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
svdd_model = load_autoencoder_and_svdd(autoencoder_model_path, svdd_model_path, device)

# Object detection
image, boxes, confidences = detect_objects(image_path)
patches = extract_patches(image, boxes)

# Anomaly detection
errors = anomaly_detection_deep_svdd(svdd_model, patches, device)
anomalies = process_results(errors)

# Print results
for idx, (error, is_anomaly) in enumerate(zip(errors, anomalies)):
    print(f'Patch {idx}: Reconstruction Error = {error}, Anomaly Detected = {is_anomaly}')

