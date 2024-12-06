import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import time
from datetime import datetime, timedelta

class FaceBeautyNet(nn.Module):
    def __init__(self):
        super(FaceBeautyNet, self).__init__()
        # 使用预训练的EfficientNet，确保加载预训练权重
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # 移除原始分类器
        self.backbone.classifier = nn.Identity()
        
        # 特征维度 (EfficientNet-B0的特征维度是1280)
        num_features = 1280
        
        # 改进头部网络
        self.head = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 1),
            nn.Sigmoid()  # 将输出限制在0-1之间，然后在训练时缩放到实际分数范围
        )
        
        # 减少冻结层数，允许更多层参与训练
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
    
    def forward(self, x):
        features = self.backbone(x)
        # 输出会在0-1之间，在损失函数中进行缩放
        return self.head(features)

class FaceDataset(Dataset):
    def __init__(self, image_dir, score_file, transform=None, split='train'):
        self.image_dir = image_dir
        self.transform = transform
        self.split = split
        self.image_scores = {}
        
        # 读取分数文件
        with open(score_file, 'r') as f:
            for line in f:
                # 正确分割文件名和分数
                parts = line.strip().split()
                if len(parts) == 2:  # 确保行格式正确
                    img_name, score = parts[0], float(parts[1])
                    self.image_scores[img_name] = score
        
        # 读取训练/测试分割文件
        split_file = os.path.join(os.path.dirname(score_file), 
                                'split/train.txt' if split == 'train' else 'split/test.txt')
        
        # 读取分割文件中的图片名称，只取文件名部分
        with open(split_file, 'r') as f:
            self.images = []
            for line in f:
                # 只取第一部分作为文件名
                img_name = line.strip().split()[0]
                if img_name in self.image_scores:
                    self.images.append(img_name)
        
        print(f"Loaded {len(self.images)} images for {split} set")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        try:
            # 加载图片
            image_path = os.path.join(self.image_dir, img_name)
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            # 获取分数并转换为tensor
            score = torch.tensor([self.image_scores[img_name]], dtype=torch.float32)
            
            return image, score
        except Exception as e:
            print(f"Error loading image {img_name}: {str(e)}")
            print(f"Image path: {image_path}")
            raise

def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

def calculate_metrics(outputs, targets):
    # 将输出缩放到目标分数范围
    outputs = outputs * 4 + 1  # 缩放到1-5分范围
    
    # 确保维度匹配
    outputs = outputs.view(-1)
    targets = targets.view(-1)
    
    mse = nn.MSELoss()(outputs, targets)
    mae = nn.L1Loss()(outputs, targets)
    
    # 计算相关系数
    outputs_np = outputs.detach().cpu()
    targets_np = targets.detach().cpu()
    stacked = torch.stack([outputs_np, targets_np])
    pearson = torch.corrcoef(stacked)[0,1]
    
    return {
        'mse': mse.item(),
        'mae': mae.item(),
        'pearson': pearson.item()
    }

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, outputs, targets):
        # 将输出从0-1缩放到实际分数范围（假设是1-5分）
        outputs = outputs * 4 + 1
        
        # 组合MSE和MAE损失
        mse_loss = self.mse(outputs, targets)
        l1_loss = self.l1(outputs, targets)
        
        # 添加相关性损失
        outputs_norm = (outputs - outputs.mean()) / outputs.std()
        targets_norm = (targets - targets.mean()) / targets.std()
        correlation_loss = -torch.mean(outputs_norm * targets_norm)
        
        return mse_loss + 0.5 * l1_loss + 0.1 * correlation_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, device='cuda'):
    model = model.to(device)
    best_val_loss = float('inf')
    patience = 5  # 提前停止的耐心值
    counter = 0  # 计数器
    
    # 记录开始时间
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_metrics = {'mse': 0.0, 'mae': 0.0, 'pearson': 0.0}
        batch_count = 0
        
        for batch_idx, (images, scores) in enumerate(train_loader):
            images, scores = images.to(device), scores.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, scores)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            batch_metrics = calculate_metrics(outputs, scores)
            train_loss += loss.item()
            for k in batch_metrics:
                train_metrics[k] += batch_metrics[k]
            batch_count += 1
            
            # 打印进度
            if batch_idx % 10 == 0:
                progress = (batch_idx + 1) / len(train_loader) * 100
                print(f'\rEpoch {epoch+1}/{num_epochs} - Training: {progress:.1f}%', end='')
        
        # 计算每个epoch的用时
        epoch_time = time.time() - epoch_start_time
        
        # 估算剩余时间
        epochs_remaining = num_epochs - (epoch + 1)
        estimated_remaining_time = epoch_time * epochs_remaining
        estimated_completion_time = datetime.now() + timedelta(seconds=estimated_remaining_time)
        
        avg_train_loss = train_loss / batch_count
        for k in train_metrics:
            train_metrics[k] /= batch_count
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_metrics = {'mse': 0.0, 'mae': 0.0, 'pearson': 0.0}
        batch_count = 0
        
        with torch.no_grad():
            for images, scores in val_loader:
                images, scores = images.to(device), scores.to(device)
                outputs = model(images)
                loss = criterion(outputs, scores)
                
                batch_metrics = calculate_metrics(outputs, scores)
                val_loss += loss.item()
                for k in batch_metrics:
                    val_metrics[k] += batch_metrics[k]
                batch_count += 1
        
        avg_val_loss = val_loss / batch_count
        for k in val_metrics:
            val_metrics[k] /= batch_count
        
        # 打印训练信息
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Time per epoch: {epoch_time:.1f}s')
        print(f'Estimated completion time: {estimated_completion_time.strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'Train: {{"loss": {avg_train_loss:.4f}, "mse": {train_metrics["mse"]:.4f}, '
              f'"mae": {train_metrics["mae"]:.4f}, "pearson": {train_metrics["pearson"]:.4f}}}')
        print(f'Val: {{"loss": {avg_val_loss:.4f}, "mse": {val_metrics["mse"]:.4f}, '
              f'"mae": {val_metrics["mae"]:.4f}, "pearson": {val_metrics["pearson"]:.4f}}}')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_face_beauty_model.pth')
            counter = 0
        else:
            counter += 1
        
        # 提前停止检查
        if counter >= patience:
            print(f'\nEarly stopping at epoch {epoch+1}')
            total_time = time.time() - start_time
            print(f'Total training time: {total_time:.1f}s')
            break
    
    # 打印总训练时间
    total_time = time.time() - start_time
    print(f'\nTotal training time: {total_time:.1f}s')

def main():
    # 设置随机种子确保可重复性
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # SCUT-FBP5500数据集路径
    base_dir = 'E:/project/web_y/dev/face-score/SCUT-FBP5500_v2'
    
    # 数据加载器
    train_dataset = FaceDataset(
        image_dir=os.path.join(base_dir, 'Images'),
        score_file=os.path.join(base_dir, 'train_test_files/All_labels.txt'),
        transform=get_transforms(is_train=True),
        split='train'
    )
    
    val_dataset = FaceDataset(
        image_dir=os.path.join(base_dir, 'Images'),
        score_file=os.path.join(base_dir, 'train_test_files/All_labels.txt'),
        transform=get_transforms(is_train=False),
        split='test'
    )
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化模型
    model = FaceBeautyNet()
    model = model.to(device)
    
    # 使用自定义损失函数
    criterion = CustomLoss()
    
    # 分层学习率
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 5e-5},
        {'params': model.head.parameters(), 'lr': 5e-4}
    ], weight_decay=0.01)
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[5e-5, 5e-4],
        epochs=50,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=100.0
    )
    
    # 训练模型
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=50,
        device=device
    )

if __name__ == '__main__':
    main()