from ultralytics import YOLOv10
import os
import matplotlib.pyplot as plt

import yaml

def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)

    return config

cfg = load_config("dataset/dataset.yaml")

print(cfg)

# 모델 불러오기
model = YOLOv10.from_pretrained('jameslahm/yolov10s')
model = YOLOv10('yolov10s.pt')

model.summary()

train_path = os.path.join(cfg["path"], cfg["train"])
train_data = model.Dataset(train_path)

# 학습 및 모델 저장 경로 설정
model.compile(optimizer='adamw', loss='', metrics=['mae'])
results = model.fit(
    x=train_data, y=train_targets,
    epochs=30,
    batch_size=8,
    validation_split=0.2)
