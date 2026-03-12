"""
金属表面缺除检测系统
基于卷积神经网络的工业缺除检测算法
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


class MetalDefectDetector:
    """金属表面缺除检测器"""
    
    def __init__(self, data_dir="/mnt/okcomputer/data", model_dir="/mnt/okcomputer/models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model = None
        self.history = None
        self.class_names = ["pitted_surface", "scratches", "inclusion", "patches", "rolled_scale"]
        self.img_size = (128, 128)
        
        # 创建目录
        os.makedirs(model_dir, exist_ok=True)
    
    def load_data(self):
        """加载数据集"""
        print("正在加载数据集...")
        
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        
        # 加载训练数据
        for i, defect_type in enumerate(self.class_names):
            train_path = os.path.join(self.data_dir, "train", defect_type)
            if os.path.exists(train_path):
                for img_file in os.listdir(train_path):
                    if img_file.endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(train_path, img_file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, self.img_size)
                        img = img.astype(np.float32) / 255.0
                        train_data.append(img)
                        train_labels.append(i)
        
        # 加载测试数据
        for i, defect_type in enumerate(self.class_names):
            test_path = os.path.join(self.data_dir, "test", defect_type)
            if os.path.exists(test_path):
                for img_file in os.listdir(test_path):
                    if img_file.endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(test_path, img_file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, self.img_size)
                        img = img.astype(np.float32) / 255.0
                        test_data.append(img)
                        test_labels.append(i)
        
        # 转换为numpy数组
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)
        
        # 添加通道维度
        train_data = np.expand_dims(train_data, -1)
        test_data = np.expand_dims(test_data, -1)
        
        print(f"训练数据形状: {train_data.shape}")
        print(f"测试数据形状: {test_data.shape}")
        print(f"训练标签分布: {np.bincount(train_labels)}")
        print(f"测试标签分布: {np.bincount(test_labels)}")
        
        return (train_data, train_labels), (test_data, test_labels)
    
    def build_model(self):
        """构建CNN模型"""
        print("正在构建CNN模型...")
        
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("模型构建完成")
        model.summary()
        return model
    
    def train_model(self, train_data, train_labels, test_data, test_labels, epochs=30):
        """训练模型"""
        print("正在训练模型...")
        
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            keras.callbacks.ModelCheckpoint(
                os.path.join(self.model_dir, 'best_model.h5'),
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        self.history = self.model.fit(
            train_data, train_labels,
            batch_size=16,
            epochs=epochs,
            validation_data=(test_data, test_labels),
            callbacks=callbacks,
            verbose=1
        )
        
        print("模型训练完成")
        return self.history
    
    def evaluate_model(self, test_data, test_labels):
        """评估模型性能"""
        print("正在评估模型性能...")
        
        # 预测
        predictions = self.model.predict(test_data)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # 计算准确率
        accuracy = np.mean(predicted_classes == test_labels)
        print(f"测试集准确率: {accuracy:.4f}")
        
        # 生成分类报告
        report = classification_report(test_labels, predicted_classes, target_names=self.class_names)
        print("分类报告:")
        print(report)
        
        # 生成混淆矩阵
        cm = confusion_matrix(test_labels, predicted_classes)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'predicted_classes': predicted_classes
        }
    
    def plot_training_history(self):
        """绘制训练历史"""
        if self.history is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 准确率曲线
        ax1.plot(self.history.history['accuracy'], label='训练准确率')
        ax1.plot(self.history.history['val_accuracy'], label='验证准确率')
        ax1.set_title('模型准确率')
        ax1.set_xlabel('训练轮次')
        ax1.set_ylabel('准确率')
        ax1.legend()
        ax1.grid(True)
        
        # 损失曲线
        ax2.plot(self.history.history['loss'], label='训练损失')
        ax2.plot(self.history.history['val_loss'], label='验证损失')
        ax2.set_title('模型损失')
        ax2.set_xlabel('训练轮次')
        ax2.set_ylabel('损失')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('/mnt/okcomputer/output/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("训练历史图已保存")
    
    def plot_confusion_matrix(self, cm):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        plt.savefig('/mnt/okcomputer/output/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("混淆矩阵图已保存")
    
    def predict_image(self, image_path):
        """预测单张图像"""
        if self.model is None:
            print("模型未加载，请先训练或加载模型")
            return None
        
        # 读取和预处理图像
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return None
        
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
        
        # 预测
        prediction = self.model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)
        
        return {
            'class': self.class_names[predicted_class],
            'confidence': float(confidence),
            'probabilities': {self.class_names[i]: float(prob) for i, prob in enumerate(prediction[0])}
        }
    
    def save_model(self):
        """保存模型"""
        if self.model is None:
            print("模型不存在")
            return
        
        model_path = os.path.join(self.model_dir, 'metal_defect_model.h5')
        self.model.save(model_path)
        print(f"模型已保存到: {model_path}")
        
        # 保存类名
        class_names_path = os.path.join(self.model_dir, 'class_names.json')
        with open(class_names_path, 'w') as f:
            json.dump(self.class_names, f)
        print(f"类名已保存到: {class_names_path}")
    
    def load_model(self):
        """加载已保存的模型"""
        model_path = os.path.join(self.model_dir, 'metal_defect_model.h5')
        if os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
            print(f"模型已从 {model_path} 加载")
            
            # 加载类名
            class_names_path = os.path.join(self.model_dir, 'class_names.json')
            if os.path.exists(class_names_path):
                with open(class_names_path, 'r') as f:
                    self.class_names = json.load(f)
                print(f"类名已加载: {self.class_names}")
            return True
        else:
            print(f"模型文件不存在: {model_path}")
            return False


def main():
    """主函数"""
    print("开始金属表面缺除检测系统...")
    
    # 创建检测器
    detector = MetalDefectDetector()
    
    # 加载数据
    (train_data, train_labels), (test_data, test_labels) = detector.load_data()
    
    # 构建模型
    detector.build_model()
    
    # 训练模型
    detector.train_model(train_data, train_labels, test_data, test_labels)
    
    # 评估模型
    results = detector.evaluate_model(test_data, test_labels)
    
    # 绘制结果
    detector.plot_training_history()
    detector.plot_confusion_matrix(results['confusion_matrix'])
    
    # 保存模型
    detector.save_model()
    
    print("系统训练完成！")
    return detector


if __name__ == "__main__":
    detector = main()