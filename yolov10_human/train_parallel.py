import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from ultralytics import YOLOv10

# 사용할 CPU 코어 수 설정
torch.set_num_threads(2)  # 예시로 2개의 코어 사용

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '192.168.0.123'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    # 모델 불러오기 및 DDP 래핑
    model = YOLOv10.from_pretrained('jameslahm/yolov10s')
    model = YOLOv10('yolov10s.pt').to(rank)
    model = DDP(model, device_ids=[rank])

    # 학습 및 모델 저장 경로 설정
    results = model.train(
        data="/home/hkj/yolov10pj/yolov10_human/dataset/dataset_human/dataset.yaml",
        model="yolov10s.pt",
        epochs=30,
        batch=8,
        imgsz=640,
        workers=4,
        device=rank,
        project='/home/hkj/yolov10pj/yolov10_human/trainresult',
        name='result_facedetect'
    )

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

