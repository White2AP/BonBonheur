#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio emotion recognition module
"""

try:
    from .audio_emotion_recognition import AudioEmotionRecognizer
    from .real_time_audio_emotion import RealTimeAudioEmotionRecognizer
    from .audio_feature_analysis import AudioFeatureAnalyzer
    
    __all__ = [
        'AudioEmotionRecognizer',
        'RealTimeAudioEmotionRecognizer', 
        'AudioFeatureAnalyzer'
    ]
    
except ImportError as e:
    print(f"Warning: Audio module import failed: {e}")
    __all__ = [] 