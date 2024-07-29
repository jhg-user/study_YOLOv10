# step 1
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import glob

from PIL import Image
import numpy as np

def load_and_preprocess_image(image_path):
    # 이미지 로드
    img = Image.open(image_path)
    
    # 이미지 전처리 (크기 조정, 채널 조정, 정규화 등)
    img = img.resize((64, 64))  # 예시로 64x64로 크기 조정
    img = np.array(img)  # PIL Image를 NumPy 배열로 변환
    
    # 필요한 전처리 추가 가능
    
    # 예를 들어, 이미지를 0과 1 사이의 값으로 정규화할 수 있습니다.
    img = img / 255.0  # 0과 1 사이의 값으로 정규화
    
    return img


plot_samples = 10  # 그래프에 표시할 샘플 수
n_features = 64 * 64 * 3  # 이미지당 특성 수 (64x64 RGB)

# 검증 데이터셋에서 정상 샘플 추출
normal_samples = val_dataset.take(plot_samples)
normal_images = np.concatenate([x for x, _ in normal_samples], axis=0)

# 정상 샘플의 재구성 예측
predicted_normal = autoencoder.predict(normal_images)
mse_normal = np.mean(np.square(normal_images - predicted_normal), axis=(1, 2, 3))

# 비정상 샘플 load
abnormal_image_paths = glob.glob('/content/dataset/*.jpg')  # 예시: jpg 확장자를 가진 모든 이미지 파일 경로 가져오기
abnormal_samples = []
for path in abnormal_image_paths[:plot_samples]:  # plot_samples 개수만큼 가져오기
    image = load_and_preprocess_image(path)  # 이미지를 로드하고 필요한 전처리를 수행하는 함수 호출
    abnormal_samples.append((image, label))  # label은 필요에 따라 설정


# 비정상 이벤트 예제 (실제 비정상 데이터가 있는 경우 해당 데이터로 대체)
# abnormal_samples = val_dataset.skip(1000).take(plot_samples)  # 일반 데이터 건너뛰기 예제
abnormal_images = np.concatenate([x for x, _ in abnormal_samples], axis=0)

# 비정상 샘플의 재구성 예측
predicted_abnormal = autoencoder.predict(abnormal_images)
mse_abnormal = np.mean(np.square(abnormal_images - predicted_abnormal), axis=(1, 2, 3))

# 정상 및 비정상 MSE 값을 담은 DataFrame 생성
normal_df = pd.DataFrame({'mse': mse_normal, 'anomaly': np.zeros(plot_samples)})
abnormal_df = pd.DataFrame({'mse': mse_abnormal, 'anomaly': np.ones(plot_samples)})

mse_df = pd.concat([normal_df, abnormal_df], axis=0)


# step 2
# MSE 그래프 그리기
plt.figure(figsize=(10, 6))
sns.lineplot(x=np.arange(len(mse_df)), y='mse', hue='anomaly', data=mse_df, palette=['blue', 'red'])
plt.title('정상 및 비정상 이벤트에 대한 MSE')
plt.xlabel('샘플 인덱스')
plt.ylabel('MSE')
plt.axvline(x=plot_samples, color='gray', linestyle='--')  # 정상과 비정상 이벤트를 나누는 세로 점선
plt.legend(['정상', '비정상'])
plt.show()

# 리눅스에선 파일로 저장
plt.savefig('boston.png')
