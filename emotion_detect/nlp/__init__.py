#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于NLTK的语音情绪分析模块

技术路线: STT -> Normalization -> NLTK -> Classification

主要组件:
- EmotionClassifier: 情绪分类器
- STTProcessor: 语音转文本处理器  
- EmotionAnalysisPipeline: 完整分析管道
- TrainingDataGenerator: 训练数据生成器
"""

from .emotion_classifier import EmotionClassifier
from .stt_processor import STTProcessor
from .emotion_pipeline import EmotionAnalysisPipeline
from .training_data_generator import TrainingDataGenerator

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    'EmotionClassifier',
    'STTProcessor', 
    'EmotionAnalysisPipeline',
    'TrainingDataGenerator'
] 