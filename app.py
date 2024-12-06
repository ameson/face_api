from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import base64
from io import BytesIO
import jwt
import datetime
from functools import wraps
from train_face_score import FaceBeautyNet
import time
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import hashlib
import hmac
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "X-API-Key", "X-Timestamp", "X-Signature"],
        "expose_headers": ["Content-Type", "X-API-Key", "X-Timestamp", "X-Signature"]
    }
})

# API配置
class APIConfig:
    # API密钥 (32位随机字符串)
    API_KEY = "sk_test_f7HKLGNfA9XmB4Vc5DwS2Pq8RjY3TuE6"
    
    # 用于签名的密钥 (64位随机字符串)
    SECRET_KEY = "sk_secret_L9nM7vK4hJ2fX8pR5tQ6wB3cY0gN1aD9EmZbW4xU3sPyA6kC8jH7"
    
    # 请求过期时间（秒）
    REQUEST_EXPIRE_TIME = 300  # 5分钟

app.config.update(
    SECRET_KEY=APIConfig.SECRET_KEY,
    API_KEY=APIConfig.API_KEY
)

# 设置限流器
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per day", "10 per minute"]
)

class FaceScorePredictor:
    def __init__(self, model_path='best_face_beauty_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FaceBeautyNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 修改为10分制
        self.min_score = 1.0
        self.max_score = 10.0
    
    def preprocess_image(self, image):
        if isinstance(image, str):
            image = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')
        elif isinstance(image, BytesIO):
            image = Image.open(image).convert('RGB')
        
        if image.size[0] < 224 or image.size[1] < 224:
            image = image.resize((256, 256), Image.Resampling.LANCZOS)
        
        return self.transform(image)
    
    def postprocess_score(self, raw_score):
        # 将0-1的分数转换为1-10分
        scaled_score = raw_score * (self.max_score - self.min_score) + self.min_score
        scaled_score = max(self.min_score, min(self.max_score, scaled_score))
        return scaled_score
    
    def predict_image(self, image):
        try:
            image_tensor = self.preprocess_image(image)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor)
                raw_score = output.item()
                
                flipped = torch.flip(image_tensor, [3])
                output_flip = self.model(flipped)
                raw_score_flip = output_flip.item()
                
                final_raw_score = (raw_score + raw_score_flip) / 2
                scaled_score = self.postprocess_score(final_raw_score)
                
                return {
                    "code": 0,
                    "message": "success",
                    "data": {
                        "raw_score": round(final_raw_score, 4),
                        "score": round(scaled_score, 1),  # 保留一位小数
                        "percentage": round(final_raw_score * 100, 2),
                        "details": {
                            "original_score": round(raw_score, 4),
                            "flipped_score": round(raw_score_flip, 4),
                            "scoring_system": "1-10分制"
                        }
                    }
                }
        except Exception as e:
            return {
                "code": -1,
                "message": str(e),
                "data": None
            }

# 初始化预测器
predictor = FaceScorePredictor()

# 添加API安全相关的函数
def generate_signature(api_key, timestamp, params=None):
    """生成请求签名"""
    message = f"{api_key}{timestamp}"
    if params:
        # 对参数按键排序，确保签名一致性
        sorted_params = sorted(params.items())
        message += ''.join(f"{k}{v}" for k, v in sorted_params)
    
    signature = hmac.new(
        app.config['SECRET_KEY'].encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature

def verify_request_timestamp(timestamp, max_diff_seconds=300):
    """验证请求时间戳，防止重放攻击"""
    try:
        request_time = datetime.datetime.fromtimestamp(int(timestamp))
        current_time = datetime.datetime.now()
        time_diff = abs((current_time - request_time).total_seconds())
        return time_diff <= max_diff_seconds
    except (ValueError, TypeError):
        return False

def verify_api_key(api_key):
    """验证API密钥"""
    return api_key == app.config['API_KEY']

def verify_signature(api_key, timestamp, signature, params=None):
    """验证请求签名"""
    expected_signature = generate_signature(api_key, timestamp, params)
    return hmac.compare_digest(signature, expected_signature)

# 修改API密钥验证装饰器
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        timestamp = request.headers.get('X-Timestamp')
        signature = request.headers.get('X-Signature')

        # 添加调试日志
        print(f"Received headers: {dict(request.headers)}")
        print(f"API Key: {api_key}")
        print(f"Timestamp: {timestamp}")
        print(f"Signature: {signature}")

        if not all([api_key, timestamp, signature]):
            missing_headers = []
            if not api_key: missing_headers.append('X-API-Key')
            if not timestamp: missing_headers.append('X-Timestamp')
            if not signature: missing_headers.append('X-Signature')
            return jsonify({
                "code": -1,
                "message": f"Missing required headers: {', '.join(missing_headers)}",
                "data": None
            }), 401

        # 验证时间戳
        if not verify_request_timestamp(timestamp):
            return jsonify({
                "code": -1,
                "message": "Request expired or invalid timestamp",
                "data": None
            }), 401

        # 验证API密钥
        if not verify_api_key(api_key):
            return jsonify({
                "code": -1,
                "message": "Invalid API key",
                "data": None
            }), 401

        # 获取请求参数
        params = {}
        if request.is_json:
            params = request.get_json()
        elif request.form:
            params = request.form.to_dict()
        
        # 添加调试日志
        print(f"Request params: {params}")

        # 验证签名
        expected_signature = generate_signature(api_key, timestamp, params)
        print(f"Expected signature: {expected_signature}")
        print(f"Received signature: {signature}")
        
        if not verify_signature(api_key, timestamp, signature, params):
            return jsonify({
                "code": -1,
                "message": "Invalid signature",
                "data": None
            }), 401

        return f(*args, **kwargs)
    return decorated

@app.route('/api/predict', methods=['POST'])
@require_api_key
@limiter.limit("10 per minute")  # 特定接口的限流规则
def predict():
    try:
        if 'image' not in request.files and 'image_base64' not in request.json:
            return jsonify({
                "code": -1,
                "message": "No image provided",
                "data": None
            }), 400
        
        if 'image' in request.files:
            image = request.files['image']
            if image.filename == '':
                return jsonify({
                    "code": -1,
                    "message": "No selected image",
                    "data": None
                }), 400
            img_bytes = BytesIO(image.read())
        else:
            img_bytes = request.json['image_base64']
        
        result = predictor.predict_image(img_bytes)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "code": -1,
            "message": str(e),
            "data": None
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
