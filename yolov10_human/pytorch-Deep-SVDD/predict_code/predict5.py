import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from ultralytics import YOLOv10

# Autoencoder 정의
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

# Deep SVDD 정의
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

# 모델 불러오기
def load_autoencoder(model_path, device):
    model = Autoencoder(z_dim=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def load_deep_svdd(model_path, autoencoder, device):
    # Load the dictionary containing 'center' and possibly 'radius'
    svdd_center_radius = torch.load(model_path, map_location=device)

    # Print the dictionary to understand its structure
    print("Loaded dictionary keys and types:")
    for key, value in svdd_center_radius.items():
        print(f"Key: {key}, Type: {type(value)}")

    # Convert 'center' to tensor and move to the specified device
    if 'center' not in svdd_center_radius:
        raise ValueError("Missing 'center' in svdd_center_radius.")
    
    center_list = svdd_center_radius['center']
    center_tensor = torch.tensor(center_list, dtype=torch.float32).to(device)
    
    # Check for 'radius' key; if not present, use a default value or handle accordingly
    if 'radius' in svdd_center_radius:
        radius = torch.tensor(svdd_center_radius['radius'], dtype=torch.float32).to(device)
    else:
        # Default value for radius, or handle this case as needed
        radius = torch.tensor([1.0], dtype=torch.float32).to(device)  # Example default value

    return DeepSVDD(autoencoder, center_tensor, radius).to(device)

# YOLOv10 모델로 객체 감지
def detect_objects(image_path, device):
    yolo_model = YOLOv10('/home/hkj/yolov10pj/yolov10_human/runs/detect/train4/weights/best.pt').to(device)
    #yolo_model.load_state_dict(torch.load('/home/hkj/yolov10pj/yolov10_human/runs/detect/train4/best.pt', map_location=device))
    yolo_model.eval()

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        predictions = yolo_model(image_tensor)

    boxes = predictions[0]['boxes'].cpu().numpy()
    confidences = predictions[0]['scores'].cpu().numpy()

    return image, boxes, confidences

# 이미지 패치 추출
def extract_patches(image, boxes, size=(64, 64)):
    patches = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        patch = image[y1:y2, x1:x2]
        if patch.shape[0] > 0 and patch.shape[1] > 0:
            patch_resized = cv2.resize(patch, size)
            patches.append(patch_resized)
    return patches

# 패치 전처리
def preprocess_patch(patch, device):
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert to PIL Image
        transforms.Grayscale(),  # Ensure the image is in grayscale
        transforms.Resize((64, 64)),  # Resize to match the model's input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize
    ])
    return transform(patch).unsqueeze(0).to(device)

# 재구성 오류 계산
def calculate_reconstruction_error(original, reconstructed):
    return F.mse_loss(reconstructed, original, reduction='none').mean(dim=[1, 2, 3])

# 이상 탐지
def anomaly_detection_deep_svdd(model, patches, device):
    reconstruction_errors = []
    patches_tensor = torch.stack([preprocess_patch(patch, device) for patch in patches])

    with torch.no_grad():
        encoded_patches = model.autoencoder.encode(patches_tensor)
        distances = model(encoded_patches)
        reconstruction_errors = distances.cpu().numpy()

    return reconstruction_errors

def process_results(reconstruction_errors, threshold=0.5):
    anomalies = reconstruction_errors > threshold
    return anomalies

# Main process
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autoencoder_model_path = 'autoencoder_human_face.pth'  # Update this path
svdd_model_path = '/home/hkj/yolov10pj/yolov10_human/pytorch-Deep-SVDD/weights/final_deep_svdd.pth'  # Update this path
image_path = '/home/hkj/yolov10pj/yolov10_human/dataset/test2/*.jpg'  # Update this path

# Load models
autoencoder_model = load_autoencoder(autoencoder_model_path, device)
svdd_model = load_deep_svdd(svdd_model_path, autoencoder_model, device)

# Object detection
image, boxes, confidences = detect_objects(image_path, device)
patches = extract_patches(image, boxes)

# Anomaly detection
errors = anomaly_detection_deep_svdd(svdd_model, patches, device)
anomalies = process_results(errors)

# Print results
for idx, (error, is_anomaly) in enumerate(zip(errors, anomalies)):
    print(f'Patch {idx}: Reconstruction Error = {error}, Anomaly Detected = {is_anomaly}')

