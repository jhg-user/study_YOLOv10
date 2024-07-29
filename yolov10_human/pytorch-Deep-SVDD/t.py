import torch

# 파일에서 state_dict와 center를 로드하여 확인합니다.
state_dict = torch.load('weights/pretrained_parameters.pth')
print(state_dict.keys())  # 'net_dict'와 'center' 확인
print(state_dict['net_dict'].keys())  # 모델 파라미터 키들 확인

