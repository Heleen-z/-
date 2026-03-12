"""
金属缺陷检测模型训练脚本 - 优化版
- 解冻顶层网络微调
- 优化类别权重
- 恢复验证集监控
- 增强评估可视化
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
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ==================== 配置区域 ====================
DATA_DIR = r"C:\Users\30833\Desktop\wrokmean\实训\mo\OKComputer_金属缺陷检测软件\data"

# 模型参数
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 80  # 增加到80轮
LEARNING_RATE = 0.00001  # 降低10倍
NUM_CLASSES = 5

# 路径
MODEL_SAVE_PATH = r"C:\Users\30833\Desktop\wrokmean\实训\mo\OKComputer_金属缺陷检测软件\models\model.keras"
BACKUP_PATH = r"C:\Users\30833\Desktop\wrokmean\实训\mo\OKComputer_金属缺陷检测软件\models"

# 确保目录存在
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(BACKUP_PATH, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# 类别名称
CLASS_NAMES = [
    "pitted_surface",
    "scratches", 
    "inclusion",
    "patches",
    "rolled_scale"
]

# 中文映射
CLASS_NAMES_CN = {
    "pitted_surface": "麻点缺陷",
    "scratches": "划痕缺陷",
    "inclusion": "夹杂物缺陷",
    "patches": "补丁缺陷",
    "rolled_scale": "氧化皮缺陷"
}
# =================================================

def prepare_data():
    """准备训练数据并统计样本分布"""
    print("="*60)
    print("准备训练数据...")
    print("="*60)
    
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"数据目录不存在: {DATA_DIR}")
    
    class_counts = {}
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(DATA_DIR, 'train', class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) 
                        if f.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff'))])
            class_counts[class_name] = count
            print(f"  {CLASS_NAMES_CN[class_name]}: {count} 张")
        else:
            print(f"  ⚠️  警告: 未找到类别 {class_name} 的文件夹")
            class_counts[class_name] = 0
    
    total_samples = sum(class_counts.values())
    print(f"\n总样本数: {total_samples}")
    
    # 识别小样本类别
    min_samples = min(class_counts.values())
    max_samples = max(class_counts.values())
    if max_samples / min_samples > 3:
        print("⚠️  警告: 类别不平衡严重，建议增加小类样本")
    
    return class_counts

def create_data_generators():
    """创建训练集和验证集生成器"""
    print("\n创建数据生成器...")
    
    # 检查是否有独立验证集
    val_dir = os.path.join(DATA_DIR, 'val')
    has_val_dir = os.path.exists(val_dir)
    
    if has_val_dir:
        print(f"✅ 找到独立验证集: {val_dir}")
        
        # 训练集增强（中等强度）
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2]
        )
        
        # 验证集不增强
        val_datagen = ImageDataGenerator(rescale=1./255)
        
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
            val_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            classes=CLASS_NAMES,
            shuffle=False
        )
        
    else:
        print("⚠️  未找到val文件夹，将从训练集划分20%验证集")
        
        # 使用validation_split
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2],
            validation_split=0.2
        )
        
        train_generator = train_datagen.flow_from_directory(
            os.path.join(DATA_DIR, 'train'),
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            classes=CLASS_NAMES,
            subset='training',
            shuffle=True,
            seed=42
        )
        
        val_generator = train_datagen.flow_from_directory(
            os.path.join(DATA_DIR, 'train'),
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            classes=CLASS_NAMES,
            subset='validation',
            shuffle=False
        )
    
    print(f"训练批次: {len(train_generator)}")
    print(f"验证批次: {len(val_generator)}")
    
    return train_generator, val_generator

def compute_class_weights(train_generator):
    """计算类别权重（根据混淆矩阵结果优化）"""
    print("\n计算类别权重...")
    
    train_labels = []
    for i in range(len(train_generator)):
        _, labels_batch = train_generator[i]
        train_labels.extend(np.argmax(labels_batch, axis=1))
    
    # 自动计算权重
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(NUM_CLASSES),
        y=train_labels
    )
    
    # 手动增强弱势类别（根据您的混淆矩阵结果）
    manual_adjustment = {
        0: 1.0,  # 麻点缺陷（表现尚可）
        1: 2.0,  # 划痕缺陷（大量误判，加倍权重）
        2: 3.0,  # 夹杂物缺陷（完全失效，三倍权重）
        3: 2.5,  # 补丁缺陷（接近失效，2.5倍权重）
        4: 0.8   # 氧化皮缺陷（已完美，轻微降权）
    }
    
    final_weights = {i: class_weights[i] * manual_adjustment[i] for i in range(NUM_CLASSES)}
    
    print("自动计算权重:", dict(enumerate(class_weights)))
    print("手动优化权重:", final_weights)
    
    return final_weights

def build_model():
    """构建并微调模型架构"""
    print("\n构建模型架构...")
    
    base_model = keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    
    # ===== 关键改进：解冻顶层用于微调 =====
    base_model.trainable = True
    # 冻结底层，微调顶层30层
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    # 强制BN在推理模式
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    # ========================================
    
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
    model.summary()
    
    return model, base_model

def evaluate_model(model, generator, set_name="测试集"):
    """
    评估模型性能（增强版）
    """
    if generator is None:
        print(f"⚠️  无{set_name}，跳过评估")
        return None, None
    
    print(f"\n" + "="*60)
    print(f"评估{set_name}性能...")
    print("="*60)
    
    # 预测
    y_pred = []
    y_true = []
    
    for i in range(len(generator)):
        x_batch, y_batch = generator[i]
        predictions = model.predict(x_batch, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(np.argmax(y_batch, axis=1))
    
    # 计算准确率
    accuracy = np.mean(np.array(y_pred) == np.array(y_true))
    
    # 评估
    loss, acc, auc = model.evaluate(generator, verbose=1)
    print(f"\n📊 {set_name}总体性能:")
    print(f"   - 损失值: {loss:.4f}")
    print(f"   - 准确率: {acc:.2%}")
    print(f"   - AUC值: {auc:.4f}")
    
    # 分类报告
    print(f"\n{set_name}分类报告:")
    report = classification_report(
        y_true, y_pred,
        target_names=[CLASS_NAMES_CN[c] for c in CLASS_NAMES],
        digits=4,
        output_dict=True  # 返回字典便于分析
    )
    print(classification_report(
        y_true, y_pred,
        target_names=[CLASS_NAMES_CN[c] for c in CLASS_NAMES],
        digits=4
    ))
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 增强可视化
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        xticklabels=[CLASS_NAMES_CN[c] for c in CLASS_NAMES],
        yticklabels=[CLASS_NAMES_CN[c] for c in CLASS_NAMES],
        cmap='Blues',
        cbar_kws={'label': '样本数量'},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title(f'{set_name}混淆矩阵', fontsize=16, fontweight='bold')
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存
    filename = f'logs/{set_name.lower()}_confusion_matrix.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ 混淆矩阵已保存至: {filename}")
    plt.show()
    
    # 各类别准确率
    print(f"\n{set_name}各类别准确率:")
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"   {CLASS_NAMES_CN[class_name]}: {class_accuracies[i]:.2%}")
    
    return (loss, acc, auc), cm

def train_model():
    """完整训练流程（带验证集监控）"""
    print("\n" + "="*60)
    print("开始训练（监控验证集，早停保护）...")
    print("="*60)
    
    # 准备数据
    prepare_data()
    train_generator, val_generator = create_data_generators()
    class_weights = compute_class_weights(train_generator)
    model, base_model = build_model()
    
    # 编译模型
    model.compile(
        optimizer=AdamW(learning_rate=LEARNING_RATE, weight_decay=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'auc']
    )
    
    # 回调函数（关键：监控验证集）
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss',  # 监控验证损失
            factor=0.5,
            patience=8,
            min_lr=1e-8,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',  # 监控验证准确率
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        CSVLogger(r'logs\training_log.csv', append=True)
    ]
    
    # 训练
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✅ 训练完成！最佳模型已保存")
    return model, history, val_generator

def plot_training_history(history=None, csv_path=r'logs\training_log.csv', 
                         save_path=r'logs\training_curves.png'):
    """可视化训练过程（含验证集）"""
    plt.figure(figsize=(18, 5))
    
    # 加载数据
    if history is not None:
        history_dict = history.history
        epochs = range(1, len(history_dict['loss']) + 1)
    else:
        try:
            df = pd.read_csv(csv_path)
            epochs = df['epoch'] + 1
            history_dict = df
        except Exception as e:
            print(f"❌ 无法加载数据: {e}")
            return
    
    # 子图1: Loss
    plt.subplot(1, 4, 1)
    plt.plot(epochs, history_dict['loss'], 'b-', label='训练损失', linewidth=2, marker='o')
    plt.plot(epochs, history_dict['val_loss'], 'r-', label='验证损失', linewidth=2, marker='s')
    plt.title('损失对比', fontsize=14, fontweight='bold')
    plt.xlabel('轮次')
    plt.ylabel('损失值')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # 子图2: Accuracy
    plt.subplot(1, 4, 2)
    plt.plot(epochs, history_dict['accuracy'], 'b-', label='训练准确率', linewidth=2, marker='o')
    plt.plot(epochs, history_dict['val_accuracy'], 'r-', label='验证准确率', linewidth=2, marker='s')
    plt.title('准确率对比', fontsize=14, fontweight='bold')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # 子图3: AUC
    plt.subplot(1, 4, 3)
    plt.plot(epochs, history_dict['auc'], 'b-', label='训练AUC', linewidth=2, marker='o')
    plt.plot(epochs, history_dict['val_auc'], 'r-', label='验证AUC', linewidth=2, marker='s')
    plt.title('AUC对比', fontsize=14, fontweight='bold')
    plt.xlabel('轮次')
    plt.ylabel('AUC值')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # 子图4: 学习率
    plt.subplot(1, 4, 4)
    if 'lr' in history_dict:
        plt.plot(epochs, history_dict['lr'], 'g-', label='学习率', linewidth=2, marker='o')
        plt.title('学习率衰减', fontsize=14, fontweight='bold')
        plt.xlabel('轮次')
        plt.ylabel('学习率')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"📊 训练曲线已保存至: {save_path}")
    plt.show()

def save_training_config():
    """保存训练配置"""
    config = {
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "model_architecture": "MobileNetV2",
        "class_names": CLASS_NAMES,
        "model_save_path": MODEL_SAVE_PATH,
        "use_validation": True,
        "fine_tune": True,
    }
    
    with open(r'logs\training_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    print("\n训练配置已保存至: logs/training_config.json")

def main():
    """主函数（完整版，含测试集强制验证）"""
    print("="*60)
    print("金属缺陷检测模型训练程序（优化版）")
    print("="*60)
    
    # 检查GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"✅ 检测到 {len(physical_devices)} 个GPU")
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("⚠️  未检测到GPU，将使用CPU训练（速度较慢）")
    
    # ========== 核心修复：确保所有步骤顺序执行 ==========
    model = None
    val_generator = None
    test_generator = None
    
    try:
        # 步骤1：训练模型
        print("\n🚀 步骤1/5: 开始训练...")
        model, history, val_generator = train_model()
        print("✅ 训练完成")
        
        # 步骤2：生成训练曲线（防阻塞）
        print("\n📊 步骤2/5: 生成训练曲线...")
        try:
            plot_training_history(history)
            # 强制关闭图表防止阻塞后续代码
            plt.close('all')
        except Exception as e:
            print(f"⚠️  训练曲线生成警告: {e}")
        
        # 步骤3：评估验证集
        print("\n🔍 步骤3/5: 评估验证集...")
        if val_generator is not None and len(val_generator) > 0:
            try:
                evaluate_model(model, val_generator, set_name="验证集")
            except Exception as e:
                print(f"❌ 验证集评估失败: {e}")
        else:
            print("⚠️  验证集生成器为空，跳过")
        
        # 步骤4：评估测试集（强制执行）
        print("\n🎯 步骤4/5: 评估测试集...")
        try:
            # 强制创建测试集生成器
            test_generator = create_test_generator()
            if test_generator is None:
                print("❌ 致命错误：无法创建测试集生成器")
            elif test_generator.samples == 0:
                print("❌ 致命错误：测试集为空，无任何图片")
            else:
                print(f"✅ 测试集加载成功，共 {test_generator.samples} 张图片")
                evaluate_model(model, test_generator, set_name="测试集")
        except Exception as e:
            print(f"❌ 测试集评估异常: {e}")
            print("💡 请检查以下路径是否存在且包含类别子文件夹：")
            print(f"   {os.path.join(DATA_DIR, 'test')}")
        
        # 步骤5：保存配置和备份
        print("\n💾 步骤5/5: 保存配置和备份...")
        try:
            save_training_config()
            backup_file = os.path.join(BACKUP_PATH, 'backup_model.keras')
            model.save(backup_file)
            print(f"✅ 备份模型已保存至: {backup_file}")
        except Exception as e:
            print(f"⚠️  保存配置/备份失败: {e}")
        
        print("\n" + "="*60)
        print("✅ 所有任务成功完成！")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⛔ 用户手动中断训练")
        print("⚠️  模型可能处于不完整状态")
        
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 排查建议：")
        print("   1. 检查GPU内存是否足够")
        print("   2. 检查数据目录结构是否正确")
        print("   3. 检查类别名称配置是否匹配文件夹名称")

# 添加测试集生成器函数（如果还没有）
def create_test_generator():
    """创建独立测试集生成器"""
    test_dir = os.path.join(DATA_DIR, 'test')
    
    if not os.path.exists(test_dir):
        print(f"❌ 致命错误：测试集目录不存在: {test_dir}")
        print("   期望的目录结构：")
        print(f"   {DATA_DIR}")
        print("   ├── train/")
        print("   └── test/          ← 此文件夹缺失")
        return None
    
    # 检查测试集类别文件夹
    missing_classes = []
    for class_name in CLASS_NAMES:
        class_path = os.path.join(test_dir, class_name)
        if not os.path.exists(class_path):
            missing_classes.append(class_name)
        else:
            # 统计图片数量
            img_count = len([f for f in os.listdir(class_path) 
                           if f.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff'))])
            if img_count == 0:
                print(f"⚠️  警告: 测试集类别 {CLASS_NAMES_CN[class_name]} 文件夹为空")
    
    if missing_classes:
        print(f"❌ 错误：测试集缺少以下类别文件夹: {missing_classes}")
        return None
    
    print(f"✅ 找到测试集: {test_dir}")
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False  # 测试集不打乱，便于分析
    )
    
    print(f"测试集批次: {len(test_generator)}")
    print(f"测试集总样本数: {test_generator.samples}")
    return test_generator

def main():
    """主函数（完整版，含测试集强制验证）"""
    print("="*60)
    print("金属缺陷检测模型训练程序（优化版）")
    print("="*60)
    
    # 检查GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"✅ 检测到 {len(physical_devices)} 个GPU")
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("⚠️  未检测到GPU，将使用CPU训练（速度较慢）")
    
    # ========== 核心修复：确保所有步骤顺序执行 ==========
    model = None
    val_generator = None
    test_generator = None
    
    try:
        # 步骤1：训练模型
        print("\n🚀 步骤1/5: 开始训练...")
        model, history, val_generator = train_model()
        print("✅ 训练完成")
        
        # 步骤2：生成训练曲线（防阻塞）
        print("\n📊 步骤2/5: 生成训练曲线...")
        try:
            plot_training_history(history)
            # 强制关闭图表防止阻塞后续代码
            plt.close('all')
        except Exception as e:
            print(f"⚠️  训练曲线生成警告: {e}")
        
        # 步骤3：评估验证集
        print("\n🔍 步骤3/5: 评估验证集...")
        if val_generator is not None and len(val_generator) > 0:
            try:
                evaluate_model(model, val_generator, set_name="验证集")
            except Exception as e:
                print(f"❌ 验证集评估失败: {e}")
        else:
            print("⚠️  验证集生成器为空，跳过")
        
        # 步骤4：评估测试集（强制执行）
        print("\n🎯 步骤4/5: 评估测试集...")
        try:
            # 强制创建测试集生成器
            test_generator = create_test_generator()
            if test_generator is None:
                print("❌ 致命错误：无法创建测试集生成器")
            elif test_generator.samples == 0:
                print("❌ 致命错误：测试集为空，无任何图片")
            else:
                print(f"✅ 测试集加载成功，共 {test_generator.samples} 张图片")
                evaluate_model(model, test_generator, set_name="测试集")
        except Exception as e:
            print(f"❌ 测试集评估异常: {e}")
            print("💡 请检查以下路径是否存在且包含类别子文件夹：")
            print(f"   {os.path.join(DATA_DIR, 'test')}")
        
        # 步骤5：保存配置和备份
        print("\n💾 步骤5/5: 保存配置和备份...")
        try:
            save_training_config()
            backup_file = os.path.join(BACKUP_PATH, 'backup_model.keras')
            model.save(backup_file)
            print(f"✅ 备份模型已保存至: {backup_file}")
        except Exception as e:
            print(f"⚠️  保存配置/备份失败: {e}")
        
        print("\n" + "="*60)
        print("✅ 所有任务成功完成！")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⛔ 用户手动中断训练")
        print("⚠️  模型可能处于不完整状态")
        
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 排查建议：")
        print("   1. 检查GPU内存是否足够")
        print("   2. 检查数据目录结构是否正确")
        print("   3. 检查类别名称配置是否匹配文件夹名称")

# 添加测试集生成器函数（如果还没有）
def create_test_generator():
    """创建独立测试集生成器"""
    test_dir = os.path.join(DATA_DIR, 'test')
    
    if not os.path.exists(test_dir):
        print(f"❌ 致命错误：测试集目录不存在: {test_dir}")
        print("   期望的目录结构：")
        print(f"   {DATA_DIR}")
        print("   ├── train/")
        print("   └── test/          ← 此文件夹缺失")
        return None
    
    # 检查测试集类别文件夹
    missing_classes = []
    for class_name in CLASS_NAMES:
        class_path = os.path.join(test_dir, class_name)
        if not os.path.exists(class_path):
            missing_classes.append(class_name)
        else:
            # 统计图片数量
            img_count = len([f for f in os.listdir(class_path) 
                           if f.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff'))])
            if img_count == 0:
                print(f"⚠️  警告: 测试集类别 {CLASS_NAMES_CN[class_name]} 文件夹为空")
    
    if missing_classes:
        print(f"❌ 错误：测试集缺少以下类别文件夹: {missing_classes}")
        return None
    
    print(f"✅ 找到测试集: {test_dir}")
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False  # 测试集不打乱，便于分析
    )
    
    print(f"测试集批次: {len(test_generator)}")
    print(f"测试集总样本数: {test_generator.samples}")
    return test_generator
if __name__ == '__main__':
    main()