# 统一情绪识别系统

这是一个集成了语音、语言和表情三种模态的统一情绪识别系统。

## 系统架构

```
emotion_detect/
├── nlp/                    # 语言情绪识别模块
├── audio/                  # 语音情绪识别模块  
├── vision/                 # 表情情绪识别模块
├── unified_emotion_detector.py  # 统一检测器
└── README.md              # 使用说明
```

## 技术路线

**请求 → 开始录制音视频 → 解析 → 反馈情绪**

1. **音视频录制**: 同时录制音频和视频
2. **多模态分析**:
   - 语音情绪识别 (Audio)
   - 语言情绪识别 (NLP) 
   - 表情情绪识别 (Vision)
3. **融合分析**: 加权投票融合多模态结果
4. **返回结果**: 综合情绪评估

## 快速开始

### 1. 基本使用

```python
from emotion_detect.unified_emotion_detector import UnifiedEmotionDetector

# 初始化检测器
detector = UnifiedEmotionDetector()

# 检查状态
status = detector.get_status()
print("模块状态:", status['modules_ready'])

# 设置结果回调
def show_result(result):
    print(f"检测到情绪: {result['final_emotion']}")
    print(f"置信度: {result['confidence']:.2f}")

detector.set_result_callback(show_result)

# 开始5秒录制
detector.start_recording(duration=5)

# 等待完成
import time
time.sleep(6)

# 获取所有结果
results = detector.get_all_results()
print(f"共检测到 {len(results)} 个情绪结果")
```

### 2. 文本情绪分析

```python
# 分析文本情绪
result = detector.analyze_text("今天天气真好，我很开心！")

if 'error' not in result:
    print(f"文本情绪: {result['emotion']}")
    print(f"置信度: {result['confidence']:.2f}")
```

### 3. 文件分析

```python
# 分析音频文件
audio_result = detector.analyze_audio_file("audio.wav")

# 分析图像文件  
image_result = detector.analyze_image_file("photo.jpg")
```

## 支持的情绪类别

- **积极** - 正面情绪
- **消极** - 负面情绪
- **愤怒** - 生气、愤慨
- **悲伤** - 难过、沮丧
- **快乐** - 开心、高兴
- **恐惧** - 害怕、担心
- **惊讶** - 意外、震惊
- **中性** - 平静、无明显情绪

## 模块说明

### NLP模块 (nlp/)
- 基于NLTK的文本情绪分析
- 支持中英文混合处理
- 包含情绪关键词匹配和VADER情感分析

### Audio模块 (audio/)
- 基于机器学习的语音情绪识别
- 支持实时音频流处理
- 提取MFCC、频谱等音频特征

### Vision模块 (vision/)
- 基于CNN的人脸表情识别
- 使用OpenCV进行人脸检测
- 支持实时视频流处理

## 配置要求

### 依赖包
```bash
pip install nltk scikit-learn opencv-python tensorflow
pip install pyaudio pydub SpeechRecognition
```

### 模型文件
- 音频模型: `audio/models/audio_emotion_model.pkl`
- 视觉模型: `vision/emotion_detection_cnn.h5`

## 使用示例

### 完整演示
```bash
cd emotion_detect
python unified_emotion_detector.py
```

### 简单文本分析
```python
from emotion_detect.nlp.emotion_classifier_simple import SimpleEmotionClassifier

classifier = SimpleEmotionClassifier()
emotion, confidence, scores = classifier.predict_emotion("我很开心！")
print(f"情绪: {emotion}, 置信度: {confidence:.3f}")
```

## API接口

### UnifiedEmotionDetector类

#### 主要方法

- `start_recording(duration)` - 开始录制
- `stop_recording()` - 停止录制
- `analyze_text(text)` - 分析文本
- `analyze_audio_file(file)` - 分析音频文件
- `analyze_image_file(file)` - 分析图像文件
- `get_status()` - 获取状态
- `set_result_callback(callback)` - 设置回调

#### 结果格式

```python
{
    'timestamp': 1234567890.0,
    'final_emotion': '快乐',
    'confidence': 0.85,
    'modality_results': {
        'audio': {'emotion': '快乐', 'confidence': 0.8},
        'vision': {'emotion': '快乐', 'confidence': 0.9},
        'nlp': {'emotion': '积极', 'confidence': 0.7}
    },
    'fusion_method': 'weighted_voting'
}
```

## 故障排除

### 常见问题

1. **摄像头无法打开**
   - 检查摄像头是否被其他程序占用
   - 确认摄像头权限设置

2. **麦克风无法录音**
   - 检查麦克风权限
   - 安装pyaudio: `pip install pyaudio`

3. **模型文件缺失**
   - 下载预训练模型文件
   - 或运行训练脚本生成模型

4. **依赖包问题**
   - 更新pip: `pip install --upgrade pip`
   - 安装所有依赖: `pip install -r requirements.txt`

### 性能优化

- 调整录制参数减少延迟
- 使用GPU加速深度学习模型
- 优化音频采样率和帧率

## 扩展开发

### 添加新的情绪类别
1. 更新情绪映射字典
2. 重新训练相关模型
3. 调整融合算法权重

### 集成新的模态
1. 实现新的检测器类
2. 添加到统一检测器中
3. 更新融合逻辑

## 许可证

本项目采用MIT许可证。 