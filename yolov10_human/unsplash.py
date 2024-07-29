import requests
import json
import os

# Unsplash API 키 설정
access_key = 'hSBLfADnE9IEstS28P3jnh7fYOWjO1NKMISq2uuaKn0'

# 검색 쿼리 설정
query = 'close up face'
per_page = 30
page = 1
num_images = 50
download_path = 'dataset/unsplash_faces'

# 저장할 디렉터리 생성
os.makedirs(download_path, exist_ok=True)

# 이미지 다운로드 함수
def download_image(url, path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

# API 요청 및 이미지 다운로드
downloaded = 0
while downloaded < num_images:
    response = requests.get(f'https://api.unsplash.com/search/photos?query={query}&page={page}&per_page={per_page}&client_id={access_key}')
    data = response.json()

    for result in data['results']:
        if downloaded >= num_images:
            break
        image_url = result['urls']['full']
        image_id = result['id']
        download_image(image_url, os.path.join(download_path, f'{image_id}.jpg'))
        downloaded += 1

    page += 1

print(f'Downloaded {downloaded} images.')

