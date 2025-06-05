#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision emotion recognition module
"""

try:
    from .emotion_detector import VisionEmotionDetector
    from .cv import CVUtils
    
    __all__ = [
        'VisionEmotionDetector',
        'CVUtils'
    ]
    
except ImportError as e:
    print(f"Warning: Vision module import failed: {e}")
    __all__ = [] 