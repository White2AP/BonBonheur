#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情绪分析管道演示脚本
展示STT -> Normalization -> NLTK -> Classification的完整流程
"""

import os
import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from emotion_detect.nlp.emotion_pipeline import EmotionAnalysisPipeline
from emotion_detect.nlp.training_data_generator import TrainingDataGenerator


def demo_text_emotion_analysis():
    """演示文本情绪分析"""
    print("=" * 60)
    print("文本情绪分析演示")
    print("=" * 60)
    
    # 初始化管道
    pipeline = EmotionAnalysisPipeline(
        stt_engine='google',
        stt_language='zh-CN',
        classifier_model='naive_bayes'
    )
    
    # 测试文本
    test_texts = [
        "今天天气真好，我感到非常开心和愉快！",
        "这个产品质量太差了，我很生气！",
        "我对这次考试的结果感到很失望和难过。",
        "哇，这个消息真是太令人震惊了！",
        "我对明天的面试感到很紧张和担心。",
        "这部电影还不错，没什么特别的感觉。",
        "我爱这个地方，这里让我感到非常快乐！",
        "这种情况让我感到恐惧和不安。"
    ]
    
    print("正在分析以下文本的情绪：")
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. {text}")
        
        # 直接使用情绪分类器分析文本
        normalized_text = pipeline.emotion_classifier.normalize_text(text)
        emotion_result = pipeline._rule_based_emotion_analysis(normalized_text)
        
        print(f"   预测情绪: {emotion_result['predicted_emotion']}")
        print(f"   置信度: {emotion_result['confidence']:.3f}")
        print(f"   情感分数: {emotion_result['vader_analysis'].get('compound', 0):.3f}")
        
        # 显示前3个最可能的情绪
        top_emotions = sorted(emotion_result['probability_distribution'].items(), 
                            key=lambda x: x[1], reverse=True)[:3]
        print(f"   前3个情绪: {', '.join([f'{e}({p:.3f})' for e, p in top_emotions])}")


def demo_audio_emotion_analysis():
    """演示音频情绪分析"""
    print("\n" + "=" * 60)
    print("音频情绪分析演示")
    print("=" * 60)
    
    # 初始化管道
    pipeline = EmotionAnalysisPipeline(
        stt_engine='google',
        stt_language='zh-CN',
        classifier_model='naive_bayes'
    )
    
    # 检查是否有音频文件
    audio_dir = project_root / "emotion_detect" / "audio" / "samples"
    if not audio_dir.exists():
        print("创建音频样本目录...")
        audio_dir.mkdir(parents=True, exist_ok=True)
        print(f"请将音频文件放入: {audio_dir}")
        print("支持的格式: .wav, .mp3, .m4a, .flac")
        return
    
    # 查找音频文件
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.m4a', '*.flac']:
        audio_files.extend(audio_dir.glob(ext))
    
    if not audio_files:
        print("未找到音频文件，演示实时语音识别...")
        try:
            result = pipeline.analyze_realtime(duration=5)
            if result['success']:
                print(f"识别文本: {result['stt_result']['text']}")
                print(f"预测情绪: {result['emotion_analysis']['predicted_emotion']}")
                print(f"置信度: {result['emotion_analysis']['confidence']:.3f}")
            else:
                print(f"实时分析失败: {result['error']}")
        except Exception as e:
            print(f"实时分析异常: {str(e)}")
    else:
        print(f"找到 {len(audio_files)} 个音频文件，开始分析...")
        for audio_file in audio_files[:3]:  # 只处理前3个文件
            print(f"\n正在分析: {audio_file.name}")
            result = pipeline.analyze_audio_file(str(audio_file))
            
            if result['success']:
                summary = result['summary']
                print(f"  识别文本: {summary['transcribed_text']}")
                print(f"  预测情绪: {summary['predicted_emotion']}")
                print(f"  置信度: {summary['confidence']:.3f}")
                print(f"  情感分数: {summary['sentiment_score']:.3f}")
            else:
                print(f"  分析失败: {result['error']}")


def demo_model_training():
    """演示模型训练"""
    print("\n" + "=" * 60)
    print("模型训练演示")
    print("=" * 60)
    
    # 生成训练数据
    print("生成训练数据...")
    data_generator = TrainingDataGenerator()
    training_data = data_generator.generate_sample_data(samples_per_emotion=50)
    
    print(f"生成了 {len(training_data)} 条训练样本")
    
    # 显示数据分布
    emotion_counts = {}
    for item in training_data:
        emotion = item['emotion']
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    print("数据分布:")
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count} 条")
    
    # 初始化管道并训练模型
    pipeline = EmotionAnalysisPipeline(classifier_model='naive_bayes')
    
    print("\n开始训练模型...")
    training_results = pipeline.train_classifier(training_data)
    
    print(f"训练完成！")
    print(f"训练集准确率: {training_results['train_score']:.4f}")
    print(f"测试集准确率: {training_results['test_score']:.4f}")
    print(f"交叉验证平均准确率: {training_results['cv_scores'].mean():.4f}")
    
    # 保存模型
    model_path = project_root / "emotion_detect" / "nlp" / "models" / "emotion_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save_model(str(model_path))
    
    # 测试训练好的模型
    print("\n测试训练好的模型:")
    test_texts = [
        "我今天非常开心，因为收到了好消息！",
        "这件事让我感到很愤怒和不满。",
        "我对这个结果感到很失望。"
    ]
    
    for text in test_texts:
        emotion, confidence, prob_dist = pipeline.emotion_classifier.predict(text)
        print(f"文本: {text}")
        print(f"预测情绪: {emotion}, 置信度: {confidence:.3f}")
        print()


def demo_batch_analysis():
    """演示批量分析"""
    print("\n" + "=" * 60)
    print("批量分析演示")
    print("=" * 60)
    
    pipeline = EmotionAnalysisPipeline()
    
    # 模拟批量音频文件
    audio_files = [
        "sample1.wav",
        "sample2.wav", 
        "sample3.wav"
    ]
    
    print("注意：这是模拟演示，实际使用时请提供真实的音频文件路径")
    print(f"模拟批量分析 {len(audio_files)} 个文件...")
    
    # 在实际使用中，这里会处理真实的音频文件
    # results, report = pipeline.batch_analyze(audio_files)
    
    # 模拟结果
    print("批量分析报告:")
    print("  总文件数: 3")
    print("  成功分析: 2")
    print("  失败分析: 1")
    print("  成功率: 66.7%")
    print("  情绪分布: {'积极': 1, '中性': 1}")
    print("  平均置信度: 0.85")


def demo_export_results():
    """演示结果导出"""
    print("\n" + "=" * 60)
    print("结果导出演示")
    print("=" * 60)
    
    pipeline = EmotionAnalysisPipeline()
    
    # 添加一些模拟结果
    import datetime
    mock_results = [
        {
            'timestamp': datetime.datetime.now().isoformat(),
            'audio_file': 'test1.wav',
            'success': True,
            'stt_result': {'text': '今天天气很好'},
            'emotion_analysis': {
                'predicted_emotion': '积极',
                'confidence': 0.85,
                'vader_analysis': {'compound': 0.6}
            }
        },
        {
            'timestamp': datetime.datetime.now().isoformat(),
            'audio_file': 'test2.wav',
            'success': True,
            'stt_result': {'text': '我很生气'},
            'emotion_analysis': {
                'predicted_emotion': '愤怒',
                'confidence': 0.92,
                'vader_analysis': {'compound': -0.7}
            }
        }
    ]
    
    pipeline.results_history = mock_results
    
    # 导出到不同格式
    output_dir = project_root / "emotion_detect" / "nlp" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # JSON格式
        json_file = output_dir / "results.json"
        pipeline.export_results(str(json_file), format='json')
        print(f"结果已导出到JSON: {json_file}")
        
        # CSV格式
        csv_file = output_dir / "results.csv"
        pipeline.export_results(str(csv_file), format='csv')
        print(f"结果已导出到CSV: {csv_file}")
        
        # 显示统计信息
        stats = pipeline.get_statistics()
        print("\n分析统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"导出失败: {str(e)}")


def main():
    """主函数"""
    print("情绪分析管道演示程序")
    print("技术路线: STT -> Normalization -> NLTK -> Classification")
    print()
    
    try:
        # 1. 文本情绪分析演示
        demo_text_emotion_analysis()
        
        # 2. 音频情绪分析演示
        demo_audio_emotion_analysis()
        
        # 3. 模型训练演示
        demo_model_training()
        
        # 4. 批量分析演示
        demo_batch_analysis()
        
        # 5. 结果导出演示
        demo_export_results()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"\n演示过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 