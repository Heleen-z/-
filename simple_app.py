"""
金属表面缺陷检测系统 - Keras模型版
"""

from flask import Flask, request, render_template_string, jsonify, send_from_directory
import os
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime
import webbrowser
from threading import Timer
import logging

# ==================== Keras模型集成 ====================
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# ✅ 模型路径 - 修改为你的实际路径
MODEL_PATH = r"C:\Users\30833\Desktop\wrokmean\实训\mo\OKComputer_金属缺陷检测软件\models\model.h5"

# ✅ 修复：添加模型输入尺寸定义
MODEL_INPUT_SIZE = (224, 224)  # 必须与训练时一致

# 全局模型变量
model = None
MODEL_LOADED = False

# 缺陷类别
DEFECT_CLASSES = [
    "pitted_surface",
    "scratches", 
    "inclusion",
    "patches",
    "rolled_scale"
]

DEFECT_INFO = {
    "pitted_surface": {"name": "麻点缺陷", "description": "表面出现小凹坑，影响外观和耐腐蚀性"},
    "scratches": {"name": "划痕缺陷", "description": "表面线性损伤，可能影响产品美观和防护性能"},
    "inclusion": {"name": "夹杂物缺陷", "description": "表面存在异物夹杂，影响材料均匀性"},
    "patches": {"name": "补丁缺陷", "description": "表面出现不规则斑块，影响外观质量"},
    "rolled_scale": {"name": "氧化皮缺陷", "description": "表面氧化层不均匀，影响后续加工"}
}
# =====================================================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_keras_model():
    """加载Keras模型"""
    global model, MODEL_LOADED
    
    try:
        print(f"正在加载Keras模型: {MODEL_PATH}")
        model = keras.models.load_model(MODEL_PATH)
        MODEL_LOADED = True
        
        # 检查模型输入尺寸
        input_shape = model.input_shape
        print(f"✅ 模型加载成功！")
        print(f"模型输入尺寸: {input_shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        MODEL_LOADED = False
        return False

def preprocess_image(image_path, target_size=None):
    """
    预处理图像
    """
    if target_size is None:
        target_size = MODEL_INPUT_SIZE
    
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(img).astype(np.float32) / 255.0
        
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
        
    except Exception as e:
        print(f"图像预处理失败: {e}")
        raise

def real_prediction(image_path):
    """使用Keras模型预测"""
    global model
    
    if not MODEL_LOADED:
        print("⚠️  模型未加载，无法预测")
        return None
    
    try:
        processed_image = preprocess_image(image_path)
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        predicted_class = DEFECT_CLASSES[predicted_class_idx]
        
        probabilities = {}
        for i, defect_class in enumerate(DEFECT_CLASSES):
            probabilities[defect_class] = float(predictions[0][i])
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'description': DEFECT_INFO[predicted_class]['description'],
            'probabilities': probabilities,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_used': 'keras',
            'filename': os.path.basename(image_path)
        }
        
    except Exception as e:
        print(f"预测失败: {e}")
        return None

# ==================== HTML模板 ====================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>金属表面缺陷检测系统</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
<style>




:root {
--primary-color: #2c3e50;
--accent-color: #3498db;
--success-color: #27ae60;
--light-bg: #ecf0f1;
}
body {
font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
min-height: 100vh;
color: var(--primary-color);
margin: 0;
padding: 100px 20px 20px;
}
.main-container {
background: rgba(255, 255, 255, 0.95);
backdrop-filter: blur(10px);
border-radius: 20px;
box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
margin: 0 auto;
padding: 40px;
max-width: 900px;
}
.header {
text-align: center;
margin-bottom: 40px;
padding-bottom: 30px;
border-bottom: 3px solid var(--accent-color);
}
.header h1 {
font-size: 2.5rem;
font-weight: 700;
color: var(--primary-color);
margin-bottom: 10px;
}
.model-status {
padding: 10px;
margin-bottom: 20px;
border-radius: 10px;
text-align: center;
font-weight: 600;
}
.model-status.loaded {
background: #d4edda;
color: #155724;
}
.model-status.failed {
background: #f8d7da;
color: #721c24;
}
.upload-section {
background: var(--light-bg);
border-radius: 15px;
padding: 40px;
margin-bottom: 30px;
text-align: center;
border: 2px dashed var(--accent-color);
cursor: pointer;
transition: all 0.3s ease;
}
.upload-section:hover {
border-color: var(--success-color);
background: #f8f9fa;
}
.upload-icon {
font-size: 4rem;
color: var(--accent-color);
margin-bottom: 20px;
}
.btn-upload {
background: var(--accent-color);
border: none;
color: white;
padding: 12px 30px;
border-radius: 25px;
font-size: 1rem;
font-weight: 600;
cursor: pointer;
transition: all 0.3s ease;
}
.btn-upload:hover {
background: #2980b9;
transform: translateY(-2px);
}
.result-section {
display: none;
margin-top: 30px;
animation: fadeIn 0.5s ease-in;
}
@keyframes fadeIn {
from { opacity: 0; transform: translateY(20px); }
to { opacity: 1; transform: translateY(0); }
}
.result-card {
background: white;
border-radius: 15px;
padding: 25px;
box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
}
.result-header {
display: flex;
align-items: center;
margin-bottom: 20px;
padding-bottom: 15px;
border-bottom: 2px solid var(--light-bg);
}
.result-icon {
font-size: 2rem;
margin-right: 15px;
}
.result-title {
font-size: 1.5rem;
font-weight: 700;
flex: 1;
}
.confidence-badge {
background: var(--success-color);
color: white;
padding: 5px 15px;
border-radius: 20px;
font-weight: 600;
margin-left: auto;
}
.image-container {
text-align: center;
margin: 20px 0;
}
.result-image {
max-width: 100%;
height: auto;
border-radius: 10px;
box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
}
.defect-description {
background: #f8f9fa;
border-left: 4px solid var(--warning-color);
padding: 15px 20px;
margin: 20px 0;
border-radius: 0 10px 10px 0;
}
.model-info {
font-size: 0.9rem;
color: #666;
margin-top: 10px;
font-style: italic;
}
.probability-chart h5 {
margin-bottom: 15px;
color: var(--primary-color);
}
.progress {
height: 25px;
margin-bottom: 10px;
border-radius: 15px;
overflow: hidden;
}
.progress-bar {
display: flex;
align-items: center;
justify-content: space-between;
padding: 0 15px;
font-weight: 600;
color: white;
transition: width 0.6s ease;
}
.loading {
display: none;
text-align: center;
margin: 30px 0;
}
.spinner {
width: 50px;
height: 50px;
border: 5px solid var(--light-bg);
border-top: 5px solid var(--accent-color);
border-radius: 50%;
animation: spin 1s linear infinite;
margin: 0 auto 20px;
}
@keyframes spin {
0% { transform: rotate(0deg); }
100% { transform: rotate(360deg); }
}
.error-message {
background: var(--danger-color);
color: white;
padding: 15px;
border-radius: 10px;
margin: 20px 0;
display: none;
}
.success-message {
background: var(--success-color);
color: white;
padding: 15px;
border-radius: 10px;
margin: 20px 0;
display: none;
}




</style>
</head>
<body>
<div class="container">
<div class="main-container">
<div class="header">
<h1><i class="fas fa-industry"></i> 金属表面缺陷检测系统</h1>
<p>基于Keras深度学习模型</p>
</div>

<!-- 模型状态显示 -->
<div id="modelStatus" class="model-status">
<i class="fas fa-spinner fa-spin"></i> 正在加载模型...
</div>

<!-- 上传区域 -->
<div class="upload-section" onclick="document.getElementById('fileInput').click()">
<div class="upload-area">
<div class="upload-icon"><i class="fas fa-cloud-upload-alt"></i></div>
<div class="upload-text">
<h4>点击上传金属表面图像</h4>
<p>支持 PNG, JPG, JPEG, BMP, TIFF 格式，最大 16MB</p>
</div>
<button type="button" class="btn-upload">
<i class="fas fa-upload"></i> 选择文件
</button>
</div>
<input type="file" id="fileInput" accept="image/*" style="display: none;">
</div>

<!-- 消息区域 -->
<div id="errorMessage" class="error-message"></div>
<div id="successMessage" class="success-message"></div>

<!-- 加载动画 -->
<div id="loading" class="loading">
<div class="spinner"></div>
<h4>正在分析图像...</h4>
<p>AI正在检测金属表面缺陷，请稍候...</p>
</div>

<!-- 结果区域 -->
<div id="resultSection" class="result-section">
<div class="result-card">
<div class="result-header">
<div class="result-icon"><i class="fas fa-search" id="resultIcon"></i></div>
<div class="result-title" id="resultTitle">检测结果</div>
<div class="confidence-badge" id="confidenceBadge">置信度</div>
</div>
<div class="image-container">
<img id="resultImage" class="result-image" src="" alt="检测结果图像">
</div>
<div class="defect-description" id="defectDescription">缺陷描述信息将在这里显示...</div>
<div class="model-info" id="modelInfo"></div>
<div class="probability-chart">
<h5><i class="fas fa-chart-bar"></i> 各类缺陷概率分布</h5>
<div id="probabilityBars"></div>
</div>
</div>
</div>

</div>
</div>

<script>
// 检查模型状态
fetch('/model_status')
.then(response => response.json())
.then(data => {
const statusDiv = document.getElementById('modelStatus');
if (data.model_loaded) {
statusDiv.className = 'model-status loaded';
statusDiv.innerHTML = '<i class="fas fa-check-circle"></i> 模型已加载: ' + data.model_path;
} else {
statusDiv.className = 'model-status failed';
statusDiv.innerHTML = '<i class="fas fa-exclamation-triangle"></i> 模型加载失败，将使用模拟检测';
}
})
.catch(error => {
document.getElementById('modelStatus').style.display = 'none';
});

// 文件上传
document.getElementById('fileInput').addEventListener('change', function(e) {
const file = e.target.files[0];
if (file) {
console.log('选中文件:', file.name);
uploadFile(file);
}
});

function uploadFile(file) {
console.log('开始上传:', file.name);
document.getElementById('loading').style.display = 'block';
document.getElementById('resultSection').style.display = 'none';
document.getElementById('errorMessage').style.display = 'none';
document.getElementById('successMessage').style.display = 'none';

const formData = new FormData();
formData.append('file', file);

fetch('/upload', {
method: 'POST',
body: formData
})
.then(response => {
console.log('收到响应,状态码:', response.status);
if (!response.ok) {
throw new Error('HTTP错误 ' + response.status);
}
return response.json();
})
.then(data => {
console.log('收到数据:', data);
document.getElementById('loading').style.display = 'none';

if (data.error) {
document.getElementById('errorMessage').textContent = data.error;
document.getElementById('errorMessage').style.display = 'block';
} else {
displayResult(data);
document.getElementById('successMessage').textContent = '检测完成！';
document.getElementById('successMessage').style.display = 'block';
}
})
.catch(error => {
console.error('请求失败:', error);
document.getElementById('loading').style.display = 'none';
document.getElementById('errorMessage').textContent = '上传失败: ' + error.message;
document.getElementById('errorMessage').style.display = 'block';
});
}

function displayResult(data) {
console.log('显示结果:', data);

document.getElementById('resultTitle').textContent = '检测结果 - ' + getChineseClassName(data.class);
    document.getElementById('confidenceBadge').textContent = (data.confidence * 100).toFixed(1) + '%';
    document.getElementById('defectDescription').textContent = data.description;
    document.getElementById('resultImage').src = '/uploads/' + data.filename;

    const modelInfo = document.getElementById('modelInfo');
    // ✅ 修复：使用 data.input_size 而不是 MODEL_INPUT_SIZE
    modelInfo.textContent = data.model_used === 'keras' && data.input_size ? 
        '检测方式: Keras模型 (' + data.input_size.join('x') + ')' : 
        '检测方式: 模拟检测（模型未加载）';

const probabilityBars = document.getElementById('probabilityBars');
probabilityBars.innerHTML = '';

const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'];
let index = 0;

for (const [className, probability] of Object.entries(data.probabilities)) {
const bar = document.createElement('div');
bar.className = 'progress';
bar.innerHTML = '<div class="progress-bar" style="width: ' + (probability * 100) + '%; background-color: ' + colors[index % colors.length] + ';">' +
'<span>' + getChineseClassName(className) + '</span>' +
'<span>' + (probability * 100).toFixed(1) + '%</span>' +
'</div>';
probabilityBars.appendChild(bar);
index++;
}

document.getElementById('resultSection').style.display = 'block';

// 3秒后隐藏成功消息
setTimeout(() => {
document.getElementById('successMessage').style.display = 'none';
}, 3000);
}

function getChineseClassName(className) {
const names = {
'pitted_surface': '麻点缺陷',
'scratches': '划痕缺陷',
'inclusion': '夹杂物缺陷',
'patches': '补丁缺陷',
'rolled_scale': '氧化皮缺陷'
};
return names[className] || className;
}
</script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传和检测"""
    if 'file' not in request.files:
        return jsonify({'error': '没有选择文件'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            # 保存文件
            file.save(filepath)
            print(f"文件已保存: {filepath}")
            
            # 预测
            result = real_prediction(filepath)
            
            if result is None:
                return jsonify({'error': '模型预测失败'})
            
            result['filename'] = filename
            result['input_size'] = MODEL_INPUT_SIZE
            print(f"检测完成: {result['class']} ({result['confidence']:.2%})")
            return jsonify(result)
            
        except Exception as e:
            print(f"处理失败: {e}")
            return jsonify({'error': f'处理失败: {str(e)}'})
    
    return jsonify({'error': '文件类型不支持'})

@app.route('/uploads/<filename>')
def get_uploaded_image(filename):
    """提供上传的图像"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/model_status')
def model_status():
    """检查模型状态"""
    return jsonify({
        'model_loaded': MODEL_LOADED,
        'model_path': MODEL_PATH,
        'input_size': MODEL_INPUT_SIZE,
        'defect_classes': DEFECT_CLASSES
    })

@app.route('/health')
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER'])
    })

def open_browser():
    """自动打开浏览器"""
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    # 启动时加载模型
    print("\n" + "="*60)
    print("金属表面缺陷检测系统启动")
    print("="*60)
    print(f"模型路径: {MODEL_PATH}")
    print(f"模型存在: {os.path.exists(MODEL_PATH)}")
    
    load_keras_model()
    
    # 延迟后自动打开浏览器
    Timer(2, open_browser).start()
    
    # 启动Flask
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)