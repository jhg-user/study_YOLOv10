import torch
import numpy as np
import cv2
from ultralytics import YOLOv10
from torchvision import transforms
from deep_svdd import Autoencoder, DeepSVDD
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def load_autoencoder(model_path, device):
    """
    Load the autoencoder model and center from the specified path.

    Parameters:
    - model_path: Path to the autoencoder model file.
    - device: The device to load the model onto.

    Returns:
    - model: The loaded autoencoder model.
    - center: The center tensor from the model state.
    """
    # Initialize the autoencoder model
    model = Autoencoder(z_dim=32).to(device)  # Adjust z_dim as needed
    
    # Load the model's state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Load the state dict into the model
    model.load_state_dict(state_dict['net_dict'])
    
    # Get the center tensor
    center = torch.Tensor(state_dict['center']).to(device)
    
    model.eval()
    return model, center

def load_deep_svdd(model_path, autoencoder_model, device):
    """
    Load the Deep SVDD model from the specified path.

    Parameters:
    - model_path: Path to the Deep SVDD model file.
    - autoencoder_model: The autoencoder model used for feature extraction.
    - device: The device to load the model onto.

    Returns:
    - deep_svdd: The loaded Deep SVDD model.
    """
    # Load the Deep SVDD model parameters
    svdd_state_dict = torch.load(model_path, map_location=device)
    
    # Extract the parameters
    center = torch.tensor(svdd_state_dict['center'], dtype=torch.float32).to(device)
    radius = torch.tensor(svdd_state_dict['radius'], dtype=torch.float32).to(device)
    
    # Initialize the Deep SVDD model
    deep_svdd = DeepSVDD(center=center, radius=radius, autoencoder_model=autoencoder_model).to(device)
    deep_svdd.eval()
    
    return deep_svdd

def detect_objects(model, image_path, device):
    """
    Perform object detection on the given image using YOLOv10.

    Parameters:
    - model: The YOLOv10 model.
    - image_path: Path to the image for detection.
    - device: The device to perform the detection on.

    Returns:
    - image: The input image.
    - boxes: Detected bounding boxes.
    - confidences: Detection confidences.
    """
    # Perform detection
    results = model.predict(image_path, imgsz=640, conf=0.5, device=device)
    
    # Process results
    image = cv2.imread(image_path)
    boxes = []
    confidences = []
    
    for result in results:
        boxes.extend(result.boxes.xyxy.cpu().numpy())  # Bounding boxes
        confidences.extend(result.boxes.conf.cpu().numpy())  # Confidences
    
    return image, boxes, confidences

def preprocess_image(image, box, transform):
    """
    Preprocess an image based on the bounding box and transform.

    Parameters:
    - image: The input image.
    - box: The bounding box for cropping.
    - transform: The transformation to apply.

    Returns:
    - tensor: The preprocessed image tensor.
    """
    x1, y1, x2, y2 = map(int, box)
    cropped_image = image[y1:y2, x1:x2]
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(cropped_image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    return image_tensor

def detect_anomalies(deep_svdd, autoencoder_model, image_tensor, device):
    """
    Detect anomalies in the image tensor using Deep SVDD.

    Parameters:
    - deep_svdd: The Deep SVDD model.
    - autoencoder_model: The autoencoder model.
    - image_tensor: The input image tensor.
    - device: The device to perform the detection on.

    Returns:
    - anomaly_score: The anomaly score.
    """
    with torch.no_grad():
        # Encode the image using the autoencoder
        encoded = autoencoder_model.encode(image_tensor.to(device))
        
        # Use Deep SVDD to get the anomaly score
        anomaly_score = deep_svdd(encoded)
        
    return anomaly_score

def main():
    # Paths
    autoencoder_model_path = 'weights/pretrained_parameters.pth'
    deep_svdd_model_path = 'weights/final_deep_svdd.pth'
    yolo_model_path = '/home/hkj/yolov10pj/yolov10_human/runs/detect/train4/weights/best.pt'
    image_path = '/home/hkj/yolov10pj/yolov10_human/dataset/test2/*.jpg'
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load models
    autoencoder_model, center = load_autoencoder(autoencoder_model_path, device)
    deep_svdd = load_deep_svdd(deep_svdd_model_path, autoencoder_model, device)
    yolo_model = YOLOv10(yolo_model_path).to(device)
    
    # Detect objects
    image, boxes, confidences = detect_objects(yolo_model, image_path, device)
    
    # Transformation for autoencoder
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    # Process each detected object
    for box, conf in zip(boxes, confidences):
        # Preprocess the image
        image_tensor = preprocess_image(image, box, transform)
        
        # Detect anomalies
        anomaly_score = detect_anomalies(deep_svdd, autoencoder_model, image_tensor, device)
        
        print(f"Bounding box: {box}, Confidence: {conf}, Anomaly score: {anomaly_score.item()}")

if __name__ == "__main__":
    main()

