import requests
import json
import os

# Unsplash API 키 설정
access_key = '개인 Unsplash API 키'

# 검색 쿼리 설정
query = 'close up face'
per_page = 30
num_images = 50
#pre_download_path = 'dataset/unsplash_faces'
download_path = 'dataset/unsplash_faces_plus'
downloaded_ids_path = os.path.join(download_path, 'downloaded_ids.json')

# 저장할 디렉터리 생성
os.makedirs(download_path, exist_ok=True)

# 이미지 다운로드 함수
def download_image(url, path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

# 이미 다운로드된 이미지 ID 불러오기 또는 생성하기
if os.path.exists(downloaded_ids_path):
    with open(downloaded_ids_path, 'r') as file:
        downloaded_ids = json.load(file)
else:
    downloaded_ids = []
    # 다운로드 폴더에 있는 파일들로부터 ID 추출
    for filename in os.listdir(download_path):
        if filename.endswith('.jpg'):
            image_id = os.path.splitext(filename)[0]
            downloaded_ids.append(image_id)

# API 요청 및 이미지 다운로드
downloaded = 0
page = 1

while downloaded < num_images:
    response = requests.get(f'https://api.unsplash.com/search/photos?query={query}&page={page}&per_page={per_page}&client_id={access_key}')
    data = response.json()

    for result in data['results']:
        if downloaded >= num_images:
            break
        image_id = result['id']
        if image_id not in downloaded_ids:
            image_url = result['urls']['full']
            download_image(image_url, os.path.join(download_path, f'{image_id}.jpg'))
            downloaded_ids.append(image_id)
            downloaded += 1

    page += 1

# 다운로드된 이미지 ID 저장하기
with open(downloaded_ids_path, 'w') as file:
    json.dump(downloaded_ids, file)

print(f'Downloaded {downloaded} additional images.')

