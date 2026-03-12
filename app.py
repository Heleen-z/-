"""
金属缺陷检测模型训练脚本
使用EfficientNetB3架构 + 高级优化策略
预期准确率: >90%
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (ReduceLROnPlateau, EarlyStopping, 
                                        ModelCheckpoint, CSVLogger)
from tensorflow.keras.optimizers import AdamW
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== 配置区域（请按需修改） ====================
# 数据路径
DATA_DIR = "data"  # 确保结构为: data/train/类别名/图片.jpg
# 或：data/train/包含5个子文件夹（pitted_surface, scratches, inclusion, patches, rolled_scale）

# 模型参数
IMG_SIZE = (224, 224)  # 图像尺寸
BATCH_SIZE = 32        # 批次大小
EPOCHS = 100           # 训练轮数（会自动早停）
LEARNING_RATE = 0.001  # 初始学习率
NUM_CLASSES = 5        # 缺陷类别数

# 模型保存路径
MODEL_SAVE_PATH = "models/metal_defect_model_optimized.h5"
BACKUP_PATH = "models/backup/metal_defect_model_optimized.h5"

# 确保目录存在
os.makedirs("models", exist_ok=True)
os.makedirs("models/backup", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# 类别名称（必须与DEFECT_CLASSES顺序一致）
CLASS_NAMES = [
    "pitted_surface",
    "scratches", 
    "inclusion",
    "patches",
    "rolled_scale"
]

# 中文名称映射（用于报告）
CLASS_NAMES_CN = {
    "pitted_surface": "麻点缺陷",
    "scratches": "划痕缺陷",
    "inclusion": "夹杂物缺陷",
    "patches": "补丁缺陷",
    "rolled_scale": "氧化皮缺陷"
}
# ===============================================================

def prepare_data():
    """
    准备数据生成器
    支持从文件夹自动加载
    """
    print("="*60)
    print("准备训练数据...")
    print("="*60)
    
    # 检查数据目录结构
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"数据目录不存在: {DATA_DIR}")
    
    # 统计各类别样本数
    class_counts = {}
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(DATA_DIR, 'train', class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
            class_counts[class_name] = count
            print(f"  {CLASS_NAMES_CN[class_name]}: {count} 张")
        else:
            print(f"  ⚠️  警告: 未找到类别 {class_name} 的文件夹")
            class_counts[class_name] = 0
    
    total_samples = sum(class_counts.values())
    print(f"\n总样本数: {total_samples}")
    
    if total_samples < 500:
        print("⚠️  警告: 样本数过少，建议每类至少100张")
    
    return class_counts

def create_data_generators():
    """
    创建数据增强生成器
    """
    print("\n创建数据增强生成器...")
    
    # 训练集增强（更激进）
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        brightness_range=[0.7, 1.3],
        channel_shift_range=30.0
    )
    
    # 验证集只做归一化
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # 创建生成器
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=True,
        seed=42
    )
    
    val_generator = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'val'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False
    )
    
    print(f"训练批次: {len(train_generator)}")
    print(f"验证批次: {len(val_generator)}")
    
    return train_generator, val_generator

def compute_class_weights(train_generator):
    """
    计算类别权重（处理不均衡）
    """
    print("\n计算类别权重...")
    
    # 从generator获取所有标签
    train_labels = []
    for i in range(len(train_generator)):
        _, labels_batch = train_generator[i]
        train_labels.extend(np.argmax(labels_batch, axis=1))
        if i == len(train_generator) - 1:
            break
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(NUM_CLASSES),
        y=train_labels
    )
    
    weight_dict = dict(enumerate(class_weights))
    print("类别权重:", weight_dict)
    
    return weight_dict

def build_model():
    """
    构建优化的模型架构（EfficientNetB3）
    """
    print("\n构建模型架构...")
    
    # 加载预训练模型
    base_model = keras.applications.EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    
    # 冻结基础层（先训练头部）
    base_model.trainable = False
    
    # 构建完整模型
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    # 打印模型结构
    model.summary()
    
    return model, base_model

def train_model():
    """
    完整训练流程
    """
    # 准备数据
    class_counts = prepare_data()
    train_generator, val_generator = create_data_generators()
    class_weights = compute_class_weights(train_generator)
    
    # 构建模型
    model, base_model = build_model()
    
    # 编译模型
    model.compile(
        optimizer=AdamW(learning_rate=LEARNING_RATE, weight_decay=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    
    # 创建回调函数
    callbacks = [
        # 学习率调度
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        # 早停
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # 保存最佳模型
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        # 保存训练日志
        CSVLogger('logs/training_log.csv', append=True)
    ]
    
    print("\n" + "="*60)
    print("开始训练...")
    print("="*60)
    
    # 第一阶段：训练头部（冻结主干）
    print("\n第一阶段：训练头部（10轮）")
    history1 = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        epochs=10,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # 解冻部分层进行微调
    print("\n解冻基础模型最后30层...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # 重新编译（使用更低学习率）
    model.compile(
        optimizer=AdamW(learning_rate=LEARNING_RATE/10, weight_decay=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall', 'auc']
    )
    
    # 第二阶段：微调整个网络
    print("\n第二阶段：微调整个网络")
    history2 = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        epochs=EPOCHS,
        initial_epoch=10,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*60)
    print("训练完成！")
    print(f"最佳模型已保存至: {MODEL_SAVE_PATH}")
    print("="*60)
    
    return model, history2

def evaluate_model(model, val_generator):
    """
    详细评估模型性能
    """
    print("\n" + "="*60)
    print("评估模型性能...")
    print("="*60)
    
    # 预测
    y_pred = []
    y_true = []
    
    for i in range(len(val_generator)):
        x_batch, y_batch = val_generator[i]
        predictions = model.predict(x_batch, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(np.argmax(y_batch, axis=1))
        if i == len(val_generator) - 1:
            break
    
    # 计算准确率
    accuracy = np.mean(np.array(y_pred) == np.array(y_true))
    print(f"\n总体准确率: {accuracy:.2%}")
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(
        y_true, y_pred,
        target_names=[CLASS_NAMES_CN[c] for c in CLASS_NAMES]
    ))
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 可视化
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        xticklabels=[CLASS_NAMES_CN[c] for c in CLASS_NAMES],
        yticklabels=[CLASS_NAMES_CN[c] for c in CLASS_NAMES],
        cmap='Blues'
    )
    plt.title('混淆矩阵', fontsize=16)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('logs/confusion_matrix.png', dpi=300)
    print("\n混淆矩阵已保存至: logs/confusion_matrix.png")
    
    return accuracy, cm

def save_training_config():
    """
    保存训练配置
    """
    config = {
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "model_architecture": "EfficientNetB3",
        "class_names": CLASS_NAMES,
        "model_save_path": MODEL_SAVE_PATH,
        "data_augmentation": True,
        "class_weighted": True,
        "fine_tuning": True
    }
    
    with open('logs/training_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    print("\n训练配置已保存至: logs/training_config.json")

def plot_training_history(history):
    """
    绘制训练曲线
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 准确率曲线
    axes[0].plot(history.history['accuracy'], label='训练准确率')
    axes[0].plot(history.history['val_accuracy'], label='验证准确率')
    axes[0].set_title('训练与验证准确率', fontsize=14)
    axes[0].set_xlabel('轮次', fontsize=12)
    axes[0].set_ylabel('准确率', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 损失曲线
    axes[1].plot(history.history['loss'], label='训练损失')
    axes[1].plot(history.history['val_loss'], label='验证损失')
    axes[1].set_title('训练与验证损失', fontsize=14)
    axes[1].set_xlabel('轮次', fontsize=12)
    axes[1].set_ylabel('损失', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('logs/training_curves.png', dpi=300)
    print("\n训练曲线已保存至: logs/training_curves.png")

def main():
    """
    主函数
    """
    print("="*60)
    print("金属缺陷检测模型训练程序")
    print("="*60)
    
    # 检查GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"检测到 {len(physical_devices)} 个GPU:")
        for gpu in physical_devices:
            print(f"  - {gpu}")
        # 设置GPU内存增长
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass
    else:
        print("⚠️  未检测到GPU，将使用CPU训练（速度较慢）")
    
    # 训练
    model, history = train_model()
    
    # 评估
    train_generator, val_generator = create_data_generators()
    evaluate_model(model, val_generator)
    
    # 保存配置和绘图
    save_training_config()
    plot_training_history(history)
    
    # 备份模型
    model.save(BACKUP_PATH)
    print(f"\n备份模型已保存至: {BACKUP_PATH}")
    
    print("\n" + "="*60)
    print("✅ 所有任务完成！")
    print("下一步：将生成的模型复制到 models/ 目录")
    print("       然后重新运行 simple_app.py")
    print("="*60)

if __name__ == '__main__':
    main()