# celeba 데이터셋 load 부터 autoencoder 학습

import tensorflow as tf
import os
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

# TensorFlow GPU 사용 설정 예시
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # GPU 메모리 증가 설정 (선택 사항)
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

def preprocess_image(image_path, target_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = image / 255.0
    return image

# 이미지 크기 설정 (예: 64x64)
target_size = (64, 64)

# 이미지 경로 설정
image_dir = '/home/hkj/yolov10pj/autoencoder/dataset/img_align_celeba'

# 파일 목록 생성
image_files = tf.data.Dataset.list_files(os.path.join(image_dir, '*.jpg'))

# 데이터셋 생성
dataset = image_files.map(lambda x: preprocess_image(x, target_size))

# 입력과 타겟을 동일하게 설정
dataset = dataset.map(lambda x: (x, x))

# 학습/검증 데이터셋 분할
dataset_size = len(image_files)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

train_dataset = dataset.take(train_size).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = dataset.skip(train_size).take(val_size).batch(32).prefetch(tf.data.AUTOTUNE)

# 데이터셋 형태 확인
print(f"Train dataset: {tf.data.experimental.cardinality(train_dataset).numpy()} batches of size 32")
print(f"Validation dataset: {tf.data.experimental.cardinality(val_dataset).numpy()} batches of size 32")

# Autoencoder 모델 정의
input_img = Input(shape=(64, 64, 3))

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# Autoencoder 모델 구성
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# autoencoder.compile(optimizer=optimizers.Adam(), loss=losses.MSE, metrics=[metrics.mean_squared_error, 'accuracy'])

# 모델 요약 출력
autoencoder.summary()

# 모델 학습
history = autoencoder.fit(train_dataset, epochs=10, validation_data=val_dataset)
# history = autoencoder.fit(train_dataset, train_dataset, epochs=10, validation_data=[val_dataset, val_dataset])

# 학습 곡선 시각화
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()

# 재구성 오차 계산
decoded_imgs = autoencoder.predict(val_dataset)
reconstruction_error = np.mean(np.abs(np.concatenate([x for x in val_dataset], axis=0) - decoded_imgs), axis=(1, 2, 3))

# 임계값 설정 (예: 평균 재구성 오차 + 2*표준편차)
threshold = np.mean(reconstruction_error) + 2 * np.std(reconstruction_error)

# 이상 데이터 탐지
anomalies = reconstruction_error > threshold

# 결과 시각화
n = 2  # 시각화할 이미지 수
plt.figure(figsize=(20, 4))
val_images = np.concatenate([x for x in val_dataset], axis=0)
for i in range(n):
    # 원본 이미지
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(val_images[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 재구성된 이미지
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

# 학습된 모델 저장
autoencoder.save('autoencoder_face_detection.h5')

