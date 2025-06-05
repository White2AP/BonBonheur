#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单使用示例
快速上手情绪分析功能
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from emotion_detect.nlp.emotion_pipeline import EmotionAnalysisPipeline


def quick_text_analysis():
    """快速文本情绪分析"""
    print("=== 快速文本情绪分析 ===")
    
    # 初始化管道
    pipeline = EmotionAnalysisPipeline()
    
    # 测试文本
    texts = [
        "今天天气真好，我很开心！",
        "这个产品质量太差了，我很生气。",
        "我对考试结果感到失望。",
        "哇，这个消息太令人震惊了！",
        "我对明天的面试很紧张。"
    ]
    
    for text in texts:
        # 使用基于规则的分析（无需训练）
        normalized_text = pipeline.emotion_classifier.normalize_text(text)
        result = pipeline._rule_based_emotion_analysis(normalized_text)
        
        print(f"\n文本: {text}")
        print(f"情绪: {result['predicted_emotion']}")
        print(f"置信度: {result['confidence']:.3f}")
        print(f"情感分数: {result['vader_analysis'].get('compound', 0):.3f}")


def quick_audio_analysis():
    """快速音频情绪分析（需要音频文件）"""
    print("\n=== 快速音频情绪分析 ===")
    
    # 初始化管道
    pipeline = EmotionAnalysisPipeline(
        stt_engine='google',  # 可以改为 'sphinx' 进行离线识别
        stt_language='zh-CN'
    )
    
    # 示例音频文件路径（请替换为实际路径）
    audio_file = "sample_audio.wav"
    
    print(f"注意：请确保音频文件存在: {audio_file}")
    print("如果没有音频文件，可以尝试实时语音识别...")
    
    try:
        # 分析音频文件
        result = pipeline.analyze_audio_file(audio_file)
        
        if result['success']:
            summary = result['summary']
            print(f"识别文本: {summary['transcribed_text']}")
            print(f"预测情绪: {summary['predicted_emotion']}")
            print(f"置信度: {summary['confidence']:.3f}")
        else:
            print(f"分析失败: {result['error']}")
            
    except Exception as e:
        print(f"音频分析异常: {str(e)}")
        print("尝试实时语音识别（需要麦克风）...")
        
        try:
            # 实时语音识别
            print("请说话（5秒）...")
            realtime_result = pipeline.analyze_realtime(duration=5)
            
            if realtime_result['success']:
                print(f"识别文本: {realtime_result['stt_result']['text']}")
                print(f"预测情绪: {realtime_result['emotion_analysis']['predicted_emotion']}")
                print(f"置信度: {realtime_result['emotion_analysis']['confidence']:.3f}")
            else:
                print(f"实时识别失败: {realtime_result['error']}")
                
        except Exception as e:
            print(f"实时识别异常: {str(e)}")


def train_simple_model():
    """训练简单模型"""
    print("\n=== 训练简单模型 ===")
    
    from emotion_detect.nlp.training_data_generator import TrainingDataGenerator
    
    # 生成训练数据
    generator = TrainingDataGenerator()
    training_data = generator.generate_sample_data(samples_per_emotion=30)
    
    print(f"生成了 {len(training_data)} 条训练样本")
    
    # 初始化管道并训练
    pipeline = EmotionAnalysisPipeline(classifier_model='naive_bayes')
    
    print("开始训练模型...")
    results = pipeline.train_classifier(training_data)
    
    print(f"训练完成！测试集准确率: {results['test_score']:.4f}")
    
    # 测试训练好的模型
    test_text = "我今天非常开心，因为收到了好消息！"
    emotion, confidence, prob_dist = pipeline.emotion_classifier.predict(test_text)
    
    print(f"\n测试文本: {test_text}")
    print(f"预测情绪: {emotion}")
    print(f"置信度: {confidence:.3f}")
    
    return pipeline


def main():
    """主函数"""
    print("情绪分析简单示例")
    print("技术路线: STT -> Normalization -> NLTK -> Classification")
    print()
    
    try:
        # 1. 文本情绪分析
        quick_text_analysis()
        
        # 2. 音频情绪分析
        quick_audio_analysis()
        
        # 3. 训练简单模型
        pipeline = train_simple_model()
        
        print("\n=== 使用训练好的模型分析文本 ===")
        test_texts = [
            "这部电影太棒了，我很喜欢！",
            "今天的会议让我很生气。",
            "我对这个结果感到很失望。"
        ]
        
        for text in test_texts:
            emotion, confidence, _ = pipeline.emotion_classifier.predict(text)
            print(f"文本: {text}")
            print(f"情绪: {emotion}, 置信度: {confidence:.3f}")
            print()
        
        print("示例完成！")
        
    except KeyboardInterrupt:
        print("\n示例被用户中断")
    except Exception as e:
        print(f"\n示例运行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 