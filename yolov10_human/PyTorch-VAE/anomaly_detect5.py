import cv2
import numpy as np
import torch
import os
import glob
from vae_model import load_vae_model
from yolov10_model import load_yolov10_model, detect_faces_yolo

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def preprocess_face(face_img, size=64):
    face_img = cv2.resize(face_img, (size, size))
    face_img = face_img / 255.0
    face_img = (face_img - 0.5) / 0.5  # Normalize to [-1, 1]
    face_img = torch.tensor(face_img).float().permute(2, 0, 1).unsqueeze(0)  # Convert to Tensor
    return face_img.to(device)

def detect_faces_batch(yolov10_model, batch_images):
    batch_results = []
    for img in batch_images:
        results = detect_faces_yolo(yolov10_model, img)
        batch_results.append(results)
    return batch_results
def detect_faces_yolo(model, img, device='cpu'):
    img = torch.from_numpy(img).to(device)
    results = model.predict(img, imgsz=640, conf=0.5, device=device)
    return results

def main(vae_checkpoint_path, yolov10_checkpoint_path, input_image_path_pattern, output_dir, threshold=0.1):
    # Load models
    vae_model = load_vae_model(vae_checkpoint_path).to(device)
    yolov10_model = load_yolov10_model(yolov10_checkpoint_path).to(device)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get list of input images
    image_paths = glob.glob(input_image_path_pattern)
    print(f'Found {len(image_paths)} images')

    batch_size = 2  # Set a batch size for YOLO predictions

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in batch_paths if cv2.imread(p) is not None]

        if len(batch_images) == 0:
            continue

        # Detect faces in batch
        batch_results = detect_faces_batch(yolov10_model, batch_images)

        for image_path, img_rgb, results in zip(batch_paths, batch_images, batch_results):
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face_img = img_rgb[y1:y2, x1:x2]
                    if face_img.size == 0:
                        print(f'Failed to extract face region from image {image_path}')
                        continue
                    face_tensor = preprocess_face(face_img)

                    with torch.no_grad():
                        recon_img, _, mu, log_var = vae_model(face_tensor)
                        recon_img = recon_img.squeeze().permute(1, 2, 0).cpu().numpy()
                        recon_img = (recon_img * 0.5 + 0.5) * 255.0  # Denormalize to [0, 255]
                        recon_img = recon_img.astype(np.uint8)

                    # Compute reconstruction error
                    error = np.mean((face_img - recon_img) ** 2)

                    if error > threshold:
                        # Anomaly detected
                        color = (0, 0, 255)  # Red for anomaly
                    else:
                        # Normal
                        color = (255, 0, 0)  # Blue for normal

                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Save output image
            output_image_path = os.path.join(output_dir, os.path.basename(image_path))
            cv2.imwrite(output_image_path, img)
            print(f'Successfully saved {output_image_path}')

if __name__ == "__main__":
    vae_checkpoint_path = "/home/hkj/yolov10pj/yolov10_human/PyTorch-VAE/logs/BetaVAE/version_60/checkpoints/model/final_model.pth"
    yolov10_checkpoint_path = "/home/hkj/yolov10pj/yolov10_human/runs/detect/train4/weights/best.pt"
    input_image_path_pattern = "/home/hkj/yolov10pj/yolov10_human/dataset/test/*.jpg"
    output_dir = "/home/hkj/yolov10pj/yolov10_human/PyTorch-VAE/runs/"
    main(vae_checkpoint_path, yolov10_checkpoint_path, input_image_path_pattern, output_dir)
