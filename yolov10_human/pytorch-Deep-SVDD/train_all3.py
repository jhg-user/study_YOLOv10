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

    def pretrain(self):
        ae = autoencoder(self.args.latent_dim).to(self.device)
        ae.apply(weights_init_normal)
        optimizer = optim.Adam(ae.parameters(), lr=self.args.lr_ae, weight_decay=self.args.weight_decay_ae)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.lr_milestones, gamma=0.1)

        ae.train()
        for epoch in range(self.args.num_epochs_ae):
            total_loss = 0
            for x, _ in self.train_loader:
                x = x.float().to(self.device)
                optimizer.zero_grad()
                x_hat = ae(x)
                reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                optimizer.step()
                total_loss += reconst_loss.item()

            scheduler.step()
            avg_loss = total_loss / len(self.train_loader)
            print(f'Autoencoder train Epoch: {epoch}, Loss: {avg_loss}')

            # TensorBoard 기록
            self.writer.add_scalar('Autoencoder/Train/Loss', avg_loss, epoch)

        torch.save(ae.state_dict(), 'autoencoder7.pth')
        self.save_weights_for_DeepSVDD(ae, self.train_loader)
        print(f'Autoencoder saved')

    def save_weights_for_DeepSVDD(self, model, dataloader):
        c = self.set_c(model, dataloader)
        net = network(self.args.latent_dim).to(self.device)
        state_dict = model.state_dict()
        net.load_state_dict(state_dict, strict=False)
        torch.save({'center': c.cpu().data.numpy().tolist(), 'net_dict': net.state_dict()}, 'pretrained_parameters7.pth')

    def set_c(self, model, dataloader, eps=0.1):
        model.eval()
        z_ = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.float().to(self.device)
                z = model.encode(x)
                z_.append(z.detach())
        z_ = torch.cat(z_).to(self.device)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    def train(self):
        self.net = network(self.args.latent_dim).to(self.device)
        if self.args.pretrain:
            state_dict = torch.load('pretrained_parameters7.pth')
            self.net.load_state_dict(state_dict['net_dict'])
            self.c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            self.net.apply(weights_init_normal)
            self.c = torch.randn(self.args.latent_dim).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.lr_milestones, gamma=0.1)

        self.net.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _ in self.train_loader:
                x = x.float().to(self.device)
                self.optimizer.zero_grad()
                z = self.net(x)
                loss = torch.mean(torch.sum((z - self.c) ** 2, dim=1))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            self.scheduler.step()
            avg_loss = total_loss / len(self.train_loader)
            print(f'Deep SVDD train Epoch: {epoch}, Loss: {avg_loss}')

            # TensorBoard 기록
            self.writer.add_scalar('DeepSVDD/Train/Loss', avg_loss, epoch)

            if (epoch + 1) % self.args.checkpoint_interval == 0:
                self.save_checkpoint(epoch)

        self.save_model_and_center('./weights/final_deep_svdd7.pth')

    def save_model_and_center(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save({'net_dict': self.net.state_dict(), 'center': self.c.cpu().data.numpy().tolist()}, path)
        print(f'Model and center saved to {path}')

    def save_checkpoint(self, epoch, path='./weights/checkpoint7.pth'):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save({
            'epoch': epoch,
            'net_state_dict': self.net.state_dict(),
            'center': self.c.cpu().data.numpy().tolist(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, path)
        print(f'Checkpoint saved to {path}')

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
                threshold = np.percentile(distances, 95)  # 95th 퍼센타일
                y_pred.extend(distances > threshold)
                y_true.extend(labels.cpu().numpy())

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
    parser.add_argument("--num_epochs", type=int, default=150, help="에포크 수")
    parser.add_argument("--num_epochs_ae", type=int, default=150, help="사전 학습 에포크 수")
    parser.add_argument("--patience", type=int, default=50, help="Early Stopping을 위한 Patience")
    parser.add_argument('--lr', type=float, default=1e-4, help='학습률')
    parser.add_argument('--weight_decay', type=float, default=0.5e-6, help='L2 정규화를 위한 Weight decay 하이퍼파라미터')
    parser.add_argument('--weight_decay_ae', type=float, default=0.5e-3, help='L2 정규화를 위한 Weight decay 하이퍼파라미터')
    parser.add_argument('--lr_ae', type=float, default=1e-4, help='오토인코더 학습률')
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[10], help='스케줄러가 학습률을 0.1로 줄이는 마일스톤')
    parser.add_argument("--batch_size", type=int, default=16, help="배치 크기")
    parser.add_argument('--pretrain', type=bool, default=True, help='오토인코더를 사용하여 네트워크 사전 학습')
    parser.add_argument('--latent_dim', type=int, default=32, help='잠재 변수 z의 차원')
    parser.add_argument('--normal_class', type=int, default=0, help='정상으로 취급할 클래스. 나머지는 비정상으로 간주.')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='체크포인트 저장 간격')
    parser.add_argument('--tensorboard_log_dir', type=str, default='runs/DeepSVDD_experiment7', help='TensorBoard 로그 디렉토리')

    args = parser.parse_args()

    # 데이터 로딩
    data = custom_yolo_loader(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = TrainerDeepSVDD(args, data, device)

    trainer.pretrain()
    trainer.train()
    trainer.evaluate(data[1])
    trainer.close_writer()

