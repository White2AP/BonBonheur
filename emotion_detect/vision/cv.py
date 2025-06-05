import pandas as pd
import numpy as np
import cv2
import logging
from keras.utils import to_categorical

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def safe_preprocess(pixel_str, target_size=(128, 128)):
    """安全可靠的图像预处理函数"""
    try:
        # 1. 从字符串解析像素值
        img_array = np.fromstring(pixel_str, sep=' ', dtype=np.uint8)

        # 2. 验证数组长度
        if len(img_array) != 48 * 48:
            logger.warning(f"无效的图像长度: {len(img_array)}，应为2304")
            return None

        # 3. 重塑为48x48图像
        img = img_array.reshape(48, 48).astype(np.uint8)

        # 4. 调整大小
        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        # 5. 标准化和归一化
        normalized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.float32) / 255.0

        # 6. 添加通道维度
        return np.expand_dims(normalized, -1)
    except Exception as e:
        logger.error(f"预处理失败: {str(e)}")
        return None


# 加载数据集
df = pd.read_csv('fer2013.csv')
logger.info(f"数据集加载完成，共 {len(df)} 条记录")

# 处理所有图像
valid_images = []
valid_indices = []
invalid_count = 0

# 确保情感标签是整数类型
if not pd.api.types.is_integer_dtype(df['emotion']):
    logger.info("情感标签不是整数类型，尝试转换...")
    try:
        # 尝试转换为整数
        df['emotion'] = df['emotion'].astype(int)
        logger.info("情感标签成功转换为整数")
    except Exception as e:
        logger.error(f"情感标签转换失败: {str(e)}")
        # 如果转换失败，尝试提取数字
        df['emotion'] = pd.to_numeric(df['emotion'], errors='coerce')
        # 删除无效行
        df = df.dropna(subset=['emotion'])
        df['emotion'] = df['emotion'].astype(int)
        logger.warning(f"强制转换情感标签，删除无效行")

# 检查情感标签范围
emotion_min = df['emotion'].min()
emotion_max = df['emotion'].max()
if emotion_min < 0 or emotion_max > 6:
    logger.warning(f"情感标签超出范围 [0,6]: 最小值={emotion_min}, 最大值={emotion_max}")
    # 修正超出范围的值
    df['emotion'] = df['emotion'].clip(0, 6)

# 处理图像
for i, row in df.iterrows():
    if i % 1000 == 0:
        logger.info(f"处理中: {i}/{len(df)}")

    # 跳过情感标签无效的行
    if pd.isna(row['emotion']) or not isinstance(row['emotion'], (int, np.integer)):
        invalid_count += 1
        continue

    processed = safe_preprocess(row['pixels'])
    if processed is not None:
        valid_images.append(processed)
        valid_indices.append(i)
    else:
        invalid_count += 1

# 检查是否有有效图像
if not valid_images:
    logger.error("没有有效的图像数据，请检查数据集和预处理函数！")
    exit(1)

# 转换为NumPy数组
X = np.array(valid_images)

# 获取有效标签
valid_labels = df.iloc[valid_indices]['emotion'].values

# 确保标签是整数类型
if not np.issubdtype(valid_labels.dtype, np.integer):
    logger.warning("标签不是整数类型，尝试转换...")
    valid_labels = valid_labels.astype(int)

# 应用to_categorical
try:
    y = to_categorical(valid_labels, num_classes=7)
except Exception as e:
    logger.error(f"to_categorical失败: {str(e)}")
    # 检查标签值范围
    unique_labels = np.unique(valid_labels)
    logger.error(f"标签值: {unique_labels}")
    # 如果标签值不在0-6范围内，进行修正
    valid_labels = np.clip(valid_labels, 0, 6)
    y = to_categorical(valid_labels, num_classes=7)

logger.info(f"成功处理 {len(valid_images)} 张图像，跳过 {invalid_count} 张无效数据")
logger.info(f"X形状: {X.shape}, y形状: {y.shape}")
logger.info(f"标签分布: {np.unique(valid_labels, return_counts=True)}")