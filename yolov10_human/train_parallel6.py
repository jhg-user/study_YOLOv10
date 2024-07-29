import argparse
import os
from ultralytics import YOLOv10
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
import torchvision.datasets as datasets
from tqdm import tqdm


def main(rank, opts):
    init_distributed_training(rank, opts)
    local_gpu_id = opts.gpu

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    train_set = datasets.ImageFolder(root=os.path.join(opts.root, 'train'), transform=transform)
    train_sampler = DistributedSampler(dataset=train_set, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=opts.batch_size, sampler=train_sampler, num_workers=opts.num_workers)

    model_path = '/home/hkj/yolov10pj/yolov10_human/yolov10s.pt'
    model = YOLO(model_path).model
    model = model.cuda(local_gpu_id)
    model = DistributedDataParallel(module=model, device_ids=[local_gpu_id])

    criterion = torch.nn.CrossEntropyLoss().to(local_gpu_id)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9)

    writer = SummaryWriter(log_dir=f'runs/result_train')

    print(f'[INFO] : 학습 시작')
    for epoch in range(opts.epoch):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        all_labels = []
        all_preds = []
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(local_gpu_id)
            labels = labels.to(local_gpu_id)
            outputs = model(images)

            # YOLO 모델의 출력 딕셔너리에서 필요한 텐서 추출
            predictions = outputs['some_key_for_class_predictions']  # 수정 필요: 실제 모델 출력 구조에 맞게 변경
            
            optimizer.zero_grad()
            loss = criterion(predictions, labels)  # outputs 대신 predictions 사용
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(predictions, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            if i % 10 == 0:
                print(f'Epoch [{epoch}/{opts.epoch}], Step [{i}/{len(train_loader)}], Loss: {loss.item()}')

        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        # mAP 계산 (placeholder)
        mAP = 0.0  # 실제 계산 방법에 따라 변경
        
        print(f'[INFO] : {epoch} 번째 epoch 완료, 평균 손실 값: {avg_loss:.4f}, 정확도: {accuracy:.4f}, mAP: {mAP:.4f}')

        # TensorBoard에 기록
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)
        writer.add_scalar('mAP/train', mAP, epoch)

    print(f'[INFO] : Distributed 학습 테스트완료')
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Distributed training test', parents=[get_args_parser()])
    opts = parser.parse_args()

    opts.ngpus_per_node = 4
    opts.gpu_ids = list(range(opts.ngpus_per_node))
    opts.num_workers = opts.ngpus_per_node * 4

    mp.spawn(main, args=(opts,), nprocs=opts.ngpus_per_node, join=True)

