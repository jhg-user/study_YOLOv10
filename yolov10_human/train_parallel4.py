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

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--port', type=int, default=2033)
    parser.add_argument('--root', type=str, default='/home/hkj/yolov10pj/yolov10_human/dataset/dataset_human/images')
    parser.add_argument('--local_rank', type=int)
    return parser

def init_distributed_training(rank, opts):
    opts.rank = rank
    opts.gpu = opts.rank % torch.cuda.device_count()
    local_gpu_id = int(opts.gpu)
    torch.cuda.set_device(local_gpu_id)

    if opts.rank is not None:
        print("Use GPU: {} for training".format(local_gpu_id))

    torch.distributed.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:' + str(opts.port),
                            world_size=opts.ngpus_per_node,
                            rank=opts.rank)

    torch.distributed.barrier()

    setup_for_distributed(opts.rank == 0)
    print('opts :', opts)


def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def main(rank, opts):
    init_distributed_training(rank, opts)
    local_gpu_id = opts.gpu

    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # YOLO 모델의 입력 크기에 맞게 조정
        transforms.ToTensor(),
    ])

    train_set = datasets.ImageFolder(root=os.path.join(opts.root, 'train'), transform=transform)
    train_sampler = DistributedSampler(dataset=train_set, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=opts.batch_size, sampler=train_sampler, num_workers=opts.num_workers)

    # YOLO 모델 로드
    # 모델 로드
    model_path = '/home/hkj/yolov10pj/yolov10_human/yolov10s.pt'
    model = YOLO(model_path).model  # YOLO 객체에서 모델만 추출
    model = model.cuda(local_gpu_id)
    model = DistributedDataParallel(module=model, device_ids=[local_gpu_id])

    criterion = torch.nn.CrossEntropyLoss().to(local_gpu_id)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9)

    print('[INFO] : 학습 시작')
    for epoch in range(opts.epoch):
        model.train()
        train_sampler.set_epoch(epoch)
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (images, labels) in progress_bar:
            images = images.to(local_gpu_id)
            labels = labels.to(local_gpu_id)
            outputs = model(images)
            
            # 예상 출력 형식: outputs = {'output1': tensor, 'output2': tensor, ...}
            # outputs를 올바른 형태로 가공하여 사용할 필요가 있음
            
            # 예시: outputs['logits'] 대신 적절한 출력 선택 필요
            #logits = outputs['logits']
            #print(outputs)
            #print(outputs.keys())
            #print(type(outputs))
            # 클래스 확률 예측 텐서를 추출
            if 'one2one' in outputs:
                predictions = outputs['one2one']
            else:
                raise KeyError("Key 'one2one' not found in outputs dictionary.")
            logits = outputs.logits
             
            optimizer.zero_grad()
            loss = criterion(predictions, labels)  # outputs 대신 predictions 사용
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(predictions, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            #optimizer.zero_grad()
            #loss = criterion(logits, labels)
            #loss.backward()
            #optimizer.step()
            
            # 진행 바 업데이트
            progress_bar.set_description(f'Epoch {epoch}, Iteration {i}, Loss: {loss.item():.4f}')
        
        print(f'[INFO] : {epoch} 번째 epoch 완료')

    print(f'[INFO] : Distributed 학습 테스트 완료')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Distributed training test', parents=[get_args_parser()])
    opts = parser.parse_args()
    opts.ngpus_per_node = 4
    opts.gpu_ids = list(range(opts.ngpus_per_node))
    opts.num_workers = opts.ngpus_per_node * 4

    mp.spawn(main, args=(opts,), nprocs=opts.ngpus_per_node, join=True)

