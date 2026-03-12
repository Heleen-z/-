import tensorflow as tf
import numpy as np

# 测试预训练模型
print("加载EfficientNetB3...")
base_model = tf.keras.applications.EfficientNetB3(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# 测试前向传播
dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
output = base_model.predict(dummy_input, verbose=0)

print(f"输出形状: {output.shape}")  # 应为 (1, 7, 7, 1536)
print(f"输出均值: {output.mean():.4f}")  # 正常应在 0.5-2.0 之间
print(f"输出方差: {output.var():.4f}")  # 应有明显方差，不是接近0

if output.mean() < 0.1 or output.mean() > 10:
    print("❌ 预训练模型权重异常！")
else:
    print("✅ 预训练模型加载正常")