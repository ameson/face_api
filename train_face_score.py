import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

class FaceBeautyNet(nn.Module):
    def __init__(self):
        super(FaceBeautyNet, self).__init__()
        # 使用预训练的EfficientNet
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # 移除原始分类器
        self.backbone.classifier = nn.Identity()
        
        # 特征维度 (EfficientNet-B0的特征维度是1280)
        num_features = 1280
        
        # 头部网络
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
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

class FaceScorePredictor:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FaceBeautyNet().to(self.device)
        self.model.eval()
        
        # 加载模型权重
        weights_path = 'best_face_beauty_model.pth'
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        
        self.transform = get_transforms()
    
    @torch.no_grad()
    def predict(self, image):
        """
        对单张图片进行预测
        Args:
            image: PIL Image对象
        Returns:
            float: 预测的颜值分数 (0-100)
        """
        # 预处理图片
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 预测
        output = self.model(image_tensor)
        
        # 将0-1的输出转换为0-100的分数
        score = float(output.item() * 100)
        
        return max(0, min(100, score))  # 确保分数在0-100范围内
