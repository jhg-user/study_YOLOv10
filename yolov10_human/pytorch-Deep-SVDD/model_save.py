import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from model import autoencoder, network
from utils.utils import weights_init_normal
from preprocess import custom_yolo_loader
import numpy as np



class TrainerDeepSVDD:
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device
        self.net = None
        self.optimizer = None
        self.scheduler = None
        self.c = None  # Hyper-sphere center
        self.writer = SummaryWriter(log_dir=args.tensorboard_log_dir)  # TensorBoard SummaryWriter

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Checkpoint file {path} not found.')
        
        checkpoint = torch.load(path)
        
        # 모델과 옵티마이저 복원
        self.net = network(self.args.latent_dim).to(self.device)
        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.c = torch.Tensor(checkpoint['center']).to(self.device)
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.lr_milestones, gamma=0.1)
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # 체크포인트로부터 마지막 에포크 번호를 복원합니다 (선택적)
        self.start_epoch = checkpoint['epoch'] + 1 if 'epoch' in checkpoint else 0

        print(f'Checkpoint loaded from {path}')

    def save_model_and_center(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save({'net_dict': self.net.state_dict(), 'center': self.c.cpu().data.numpy().tolist()}, path)
        print(f'Model and center saved to {path}')

    # 추가적인 메서드들...

    def evaluate(self, dataloader):
        self.net.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for x, labels in dataloader:
                x = x.float().to(self.device)
                labels = labels.to(self.device)
                z = self.net(x)
                distances = torch.sum((z - self.c) ** 2, dim=1).cpu().numpy()

                # 이진 분류를 위해 임계값을 설정
                threshold = np.percentile(distances, 95)  # 95th percentile
                y_pred.extend(distances > threshold)
                y_true.extend(labels.cpu().numpy())
        print(threshold)

        # 평가 지표 계산
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)

        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'ROC AUC: {roc_auc:.4f}')

        # TensorBoard 기록
        self.writer.add_scalar('Evaluation/Precision', precision)
        self.writer.add_scalar('Evaluation/Recall', recall)
        self.writer.add_scalar('Evaluation/F1 Score', f1)
        self.writer.add_scalar('Evaluation/ROC AUC', roc_auc)

    def close_writer(self):
        self.writer.close()

if __name__ == "__main__":
    # 인자 파싱
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=150, help="number of epochs")
    parser.add_argument("--num_epochs_ae", type=int, default=150, help="number of epochs for the pretraining")
    parser.add_argument("--patience", type=int, default=50, help="Patience for Early Stopping")
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.5e-6, help='Weight decay hyperparameter for the L2 regularization')
    parser.add_argument('--weight_decay_ae', type=float, default=0.5e-3, help='Weight decay hyperparameter for the L2 regularization')
    parser.add_argument('--lr_ae', type=float, default=1e-4, help='learning rate for autoencoder')
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[10], help='Milestones at which the scheduler multiply the lr by 0.1')
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument('--pretrain', type=bool, default=True, help='Pretrain the network using an autoencoder')
    parser.add_argument('--latent_dim', type=int, default=32, help='Dimension of the latent variable z')
    parser.add_argument('--normal_class', type=int, default=0, help='Class to be treated as normal. The rest will be considered as anomalous.')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Interval for saving checkpoints')
    parser.add_argument('--tensorboard_log_dir', type=str, default='runs/DeepSVDD_experiment', help='Directory for TensorBoard logs')
    parser.add_argument('--checkpoint_path', type=str, default='./weights/checkpoint.pth', help='Path to checkpoint file')

    args = parser.parse_args()

    # 데이터 로딩
    data = custom_yolo_loader(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = TrainerDeepSVDD(args, data, device)

    # 체크포인트 로드
    trainer.load_checkpoint(args.checkpoint_path)

    # 모델 저장
    trainer.save_model_and_center('./weights/final_deep_svdd9.pth')


    trainer.evaluate(data[1])
    trainer.close_writer()

