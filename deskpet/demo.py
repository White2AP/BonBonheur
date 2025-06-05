#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一情绪识别系统演示
"""

def main():
    print("=== 统一情绪识别系统演示 ===")
    print()
    print("系统包含三个模块:")
    print("1. NLP模块 - 语言情绪识别")
    print("2. Audio模块 - 语音情绪识别") 
    print("3. Vision模块 - 表情识别")
    print()
    
    # 演示文本情绪分析
    print("快速文本分析演示:")
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'nlp'))
        
        from emotion_classifier_simple import SimpleEmotionClassifier
        classifier = SimpleEmotionClassifier()
        
        test_texts = [
            "今天天气真好，我很开心！",
            "这个产品质量太差了，我很生气。",
            "我对考试结果感到失望。"
        ]
        
        for text in test_texts:
            emotion, confidence, _ = classifier.predict_emotion(text)
            print(f"文本: {text}")
            print(f"情绪: {emotion} (置信度: {confidence:.3f})")
            print()
            
    except Exception as e:
        print(f"文本分析演示失败: {e}")
    
    print("=" * 50)
    print("使用方法:")
    print("from unified_emotion_detector import UnifiedEmotionDetector")
    print("detector = UnifiedEmotionDetector()")
    print("detector.start_recording(duration=5)")
    print("=" * 50)

if __name__ == "__main__":
    main() 