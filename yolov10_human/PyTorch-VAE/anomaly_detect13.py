import cv2
import numpy as np
import torch
import os
import glob
from vae_model import load_vae_model
from yolov10_model import load_yolov10_model, detect_faces_yolo
from PIL import Image, ImageDraw, ImageFont

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def preprocess_face(face_img, size=64):
    face_img = cv2.resize(face_img, (size, size))
    face_img = face_img / 255.0
    face_img = (face_img - 0.5) / 0.5  # Normalize to [-1, 1]
    face_img = torch.tensor(face_img).float().permute(2, 0, 1).unsqueeze(0)  # Convert to Tensor
    return face_img.to('cpu')  # VAE 모델이 CPU에서 실행되므로 CPU로 전송

def preprocess_image(img, img_size=640):
    original_shape = img.shape[:2]  # H, W
    img = cv2.resize(img, (img_size, img_size))
    img = img.transpose(2, 0, 1)  # Convert to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize to [0, 1]
    img = torch.tensor(img).float().to(device)
    return img, original_shape

def detect_faces_yolo(model, img, device='cpu'):
    img, original_shape = preprocess_image(img)
    results = model.predict(img, imgsz=640, conf=0.5, device=device)
    return results, original_shape

def detect_faces_batch(yolov10_model, batch_images):
    batch_results = []
    original_shapes = []
    for img in batch_images:
        results, original_shape = detect_faces_yolo(yolov10_model, img, device)
        batch_results.append(results)
        original_shapes.append(original_shape)
    return batch_results, original_shapes

def draw_label(image, text, position, color, font_size=20):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # 기본 폰트 사용
    draw.text(position, text, fill=color, font=font)

def main(vae_checkpoint_path, yolov10_checkpoint_path, input_image_path_pattern, output_dir, threshold=105.17):
    # Load models
    vae_model = load_vae_model(vae_checkpoint_path).to('cpu')  # VAE 모델을 CPU로 이동
    yolov10_model = load_yolov10_model(yolov10_checkpoint_path).to(device)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get list of input images
    image_paths = glob.glob(input_image_path_pattern)
    print(f'Found {len(image_paths)} images')

    batch_size = 2  # Set a batch size for YOLO predictions

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = [cv2.imread(p) for p in batch_paths if cv2.imread(p) is not None]

        if len(batch_images) == 0:
            continue

        # Detect faces in batch
        batch_results, original_shapes = detect_faces_batch(yolov10_model, batch_images)

        for image_path, img, results, original_shape in zip(batch_paths, batch_images, batch_results, original_shapes):
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # PIL 이미지를 사용하여 텍스트를 그리기 위함
            draw = ImageDraw.Draw(img_pil)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Rescale coordinates to original image
                    x1 = int(x1 * original_shape[1] / 640)
                    y1 = int(y1 * original_shape[0] / 640)
                    x2 = int(x2 * original_shape[1] / 640)
                    y2 = int(y2 * original_shape[0] / 640)

                    face_img = img[y1:y2, x1:x2]
                    if face_img.size == 0:
                        print(f'Failed to extract face region from image {image_path}')
                        continue
                    face_tensor = preprocess_face(face_img)

                    with torch.no_grad():
                        recon_img, _, mu, log_var = vae_model(face_tensor)  # VAE 모델을 CPU에서 실행
                        recon_img = recon_img.squeeze().permute(1, 2, 0).numpy()
                        recon_img = (recon_img * 0.5 + 0.5) * 255.0  # Denormalize to [0, 255]
                        recon_img = recon_img.astype(np.uint8)

                    # Resize face_img to 64x64 for comparison
                    face_img_resized = cv2.resize(face_img, (64, 64))

                    # Compute reconstruction error
                    error = np.mean((face_img_resized - recon_img) ** 2)

                    # Print reconstruction error
                    print(f'Reconstruction error for image {image_path} and box {box.xyxy[0]}: {error}')

                    if error > threshold:
                        # Anomaly detected
                        color = (0, 0, 255)  # Red for anomaly
                        label = "Anomalous"
                    else:
                        # Normal
                        color = (255, 0, 0)  # Blue for normal
                        label = "Normal"

                    # Draw bounding box and label
                    draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
                    draw_label(img_pil, label, (x1, y1 - 10), color)

            # Save output image
            output_image_path = os.path.join(output_dir, os.path.basename(image_path))
            img_pil.save(output_image_path)
            print(f'Successfully saved {output_image_path}')

if __name__ == "__main__":
    vae_checkpoint_path = "/home/hkj/yolov10pj/yolov10_human/PyTorch-VAE/logs/BetaVAE/version_57/checkpoints/model/final_model.pth"
    #yolov10_checkpoint_path = "/home/hkj/yolov10pj/yolov10_human/trainresult/result_facedetect3/weights/best.pt"
    yolov10_checkpoint_path = "/home/hkj/yolov10pj/yolov10_human/runs/detect/train4/weights/best.pt"
    input_image_path_pattern = "/home/hkj/yolov10pj/yolov10_human/dataset/test/*.jpg"
    output_dir = "/home/hkj/yolov10pj/yolov10_human/PyTorch-VAE/runs/result3_version57_thr105.17"
    main(vae_checkpoint_path, yolov10_checkpoint_path, input_image_path_pattern, output_dir)

