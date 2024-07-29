import requests
import os

# Pexels API 키 설정
api_key = '개인 Pexels API 키'

# 검색 쿼리 설정
query = 'close up face'
per_page = 30
page = 1
num_images = 1000
download_path = 'dataset/pexels_faces'

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
    headers = {
        'Authorization': api_key,
    }
    response = requests.get(f'https://api.pexels.com/v1/search?query={query}&per_page={per_page}&page={page}', headers=headers)
    data = response.json()

    for result in data['photos']:
        if downloaded >= num_images:
            break
        image_url = result['src']['original']
        image_id = result['id']
        download_image(image_url, os.path.join(download_path, f'{image_id}.jpg'))
        downloaded += 1

    page += 1

print(f'Downloaded {downloaded} images.')

