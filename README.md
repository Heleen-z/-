# 金属表面缺陷检测系统

## 项目简介

本项目是一个基于深度学习的金属表面缺陷检测系统，能够自动识别和分类金属表面的5种主要缺陷类型。系统采用卷积神经网络（CNN）技术，通过Web界面提供图像上传、实时检测和结果可视化功能。

## 功能特性

- 🔍 **智能缺陷识别**：支持5种缺陷类型自动识别
- ⚡ **实时检测**：单次检测时间小于1秒
- 📊 **可视化结果**：直观的缺陷类型和置信度展示
- 📱 **响应式设计**：支持各种设备访问
- 📁 **批量处理**：支持多张图像同时检测

## 支持的缺陷类型

1. **麻点缺陷 (Pitted Surface)**：表面出现小凹坑
2. **划痕缺陷 (Scratches)**：表面线性损伤
3. **夹杂物缺陷 (Inclusion)**：表面存在异物夹杂
4. **补丁缺陷 (Patches)**：表面出现不规则斑块
5. **氧化皮缺陷 (Rolled Scale)**：表面氧化层不均匀

## 技术架构

### 后端技术
- **Python 3.12**：主要编程语言
- **TensorFlow/Keras**：深度学习框架
- **Flask**：Web应用框架
- **OpenCV**：图像处理库

### 前端技术
- **HTML5/CSS3/JavaScript**：基础技术栈
- **Bootstrap**：响应式UI框架
- **Font Awesome**：图标库

## 项目结构

```
metal_defect_detection/
├── data/                          # 数据集目录
│   ├── train/                     # 训练数据
│   │   ├── pitted_surface/       # 麻点缺陷
│   │   ├── scratches/            # 划痕缺陷
│   │   ├── inclusion/            # 夹杂物缺陷
│   │   ├── patches/              # 补丁缺陷
│   │   └── rolled_scale/         # 氧化皮缺陷
│   └── test/                      # 测试数据
├── models/                        # 模型文件
│   ├── metal_defect_model.h5     # 训练好的模型
│   └── class_names.json          # 类别名称
├── templates/                     # Web模板
│   ├── index.html               # 主页面
│   ├── about.html               # 关于页面
│   └── help.html                # 帮助页面
├── reports/                       # 报告文档
│   ├── group_report.md          # 小组报告
│   └── individual_report.md     # 个人报告
├── metal_defect_detection.py    # 核心检测代码
├── app.py                       # Web应用主程序
└── simple_app.py               # 简化版应用
```

## 快速开始

### 环境要求

- Python 3.8+
- pip包管理器

### 安装依赖

```bash
pip install tensorflow flask opencv-python scikit-learn seaborn werkzeug
```

### 运行系统

#### 方式1：使用简化版应用

```bash
python simple_app.py
```

然后在浏览器中访问：http://localhost:5000

#### 方式2：使用完整版应用

```bash
python app.py
```

### 使用步骤

1. **访问网站**：打开浏览器，访问 http://localhost:5000
2. **上传图像**：点击"选择文件"按钮或拖拽图像到上传区域
3. **等待检测**：系统自动分析图像（通常小于1秒）
4. **查看结果**：查看检测到的缺陷类型、置信度和详细描述
5. **批量检测**：可选择多张图像进行批量检测

## API接口

### 单张图像检测

```http
POST /upload
Content-Type: multipart/form-data

文件字段：file
```

**响应示例：**
```json
{
  "class": "scratches",
  "confidence": 0.89,
  "description": "表面线性损伤，可能影响产品美观和防护性能",
  "probabilities": {
    "pitted_surface": 0.02,
    "scratches": 0.89,
    "inclusion": 0.03,
    "patches": 0.04,
    "rolled_scale": 0.02
  },
  "timestamp": "2025-12-16 14:30:25",
  "filename": "20251216_143025_test.jpg"
}
```

### 批量图像检测

```http
POST /batch_predict
Content-Type: multipart/form-data

文件字段：files（多张图像）
```

## 模型性能

- **模型大小**：约20MB
- **参数数量**：5,241,477
- **检测时间**：< 1秒
- **支持格式**：PNG, JPG, JPEG, BMP, TIFF
- **最大文件大小**：16MB

## 图像要求

- **建议尺寸**：200×200像素以上
- **格式要求**：清晰的金属表面图像
- **光照条件**：均匀照明，避免过曝或欠曝
- **拍摄角度**：尽量垂直于表面

## 系统截图

[此处应包含系统界面截图]

## 开发团队

- **项目负责人**：[姓名]
- **团队成员**：[团队成员列表]
- **指导教师**：[指导教师]

## 项目报告

详细的项目报告请查看 `reports/` 目录：

- `group_report.md`：小组项目报告
- `individual_report.md`：个人项目报告

## 许可证

本项目仅供学习和研究使用。

## 致谢

感谢指导教师和团队成员在项目开发过程中的支持和帮助。

---

**项目完成时间：** 2025年12月  
**最后更新时间：** 2025-12-16
