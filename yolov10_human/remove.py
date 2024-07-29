import os

# nonhuman_list.txt 파일 경로
nonhuman_list_file = 'nonhuman_list/nonhuman_list.txt'

# 파일을 열어 경로를 읽어옵니다.
with open(nonhuman_list_file, 'r') as f:
    file_paths = f.readlines()

# 각 경로에 해당하는 파일을 삭제합니다.
for file_path in file_paths:
    file_path = file_path.strip()  # 줄 끝의 공백 제거
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    else:
        print(f"File not found: {file_path}")

print("All files deleted successfully.")

