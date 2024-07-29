import torch
import torch.nn as nn

# 모델 정의 (예시)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

# 모델 초기화
model = SimpleModel()

# 가정: 병렬 학습을 위해 모델을 DataParallel 또는 DistributedDataParallel로 감쌌던 경우
# 병렬 학습 설정 (예제, 실제로는 해제할 부분)
model = torch.nn.DataParallel(model)

# 단일 GPU 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 병렬 학습 해제
if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
    model = model.module  # DataParallel 또는 DistributedDataParallel의 module 속성으로 원래 모델에 접근
model.to(device)  # 단일 GPU로 모델 이동

# 모델 정보 출력 (확인용)
print(f"Model is now on device: {next(model.parameters()).device}")

