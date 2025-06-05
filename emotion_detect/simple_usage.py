#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单使用示例
展示如何使用统一情绪识别接口
"""

def simple_emotion_detection():
    """最简单的情绪检测示例"""
    print("=== 简单情绪检测示例 ===")
    
    try:
        # 使用简化版检测器，避免复杂依赖
        from simple_unified_detector import SimpleUnifiedDetector
        
        # 初始化检测器
        detector = SimpleUnifiedDetector()
        
        # 检查状态
        status = detector.get_status()
        print("模块状态:")
        for module, ready in status['modules_ready'].items():
            print(f"  {module}: {'✅' if ready else '❌'}")
        
        if not status['ready']:
            print("⚠️ 系统未准备就绪，只有NLP模块可用")
            print("这是由于音频和视觉模块的依赖包未完全安装")
            print("但您仍然可以使用文本情绪分析功能")
        
        # 文本情绪分析
        print("\n1. 文本情绪分析:")
        test_texts = [
            "今天天气真好，我很开心！",
            "这个产品质量太差了，我很生气。",
            "我对考试结果感到失望。",
            "哇，这个消息太令人震惊了！",
            "我很害怕明天的面试。"
        ]
        
        for text in test_texts:
            result = detector.analyze_text(text)
            if result['success']:
                print(f"  文本: {text}")
                print(f"  情绪: {result['emotion']} (置信度: {result['confidence']:.3f})")
                print()
            else:
                print(f"  文本: {text}")
                print(f"  ❌ 分析失败: {result['error']}")
                print()
        
        # 批量分析统计
        print("2. 批量分析统计:")
        results = detector.batch_analyze_texts(test_texts)
        successful_results = [r for r in results if r['success']]
        
        if successful_results:
            emotions = [r['emotion'] for r in successful_results]
            from collections import Counter
            emotion_counts = Counter(emotions)
            
            print("情绪分布:")
            for emotion, count in emotion_counts.most_common():
                percentage = (count / len(successful_results)) * 100
                print(f"  {emotion}: {count}次 ({percentage:.1f}%)")
            
            avg_confidence = sum(r['confidence'] for r in successful_results) / len(successful_results)
            print(f"平均置信度: {avg_confidence:.3f}")
        
        print("\n✅ 文本分析演示完成！")
        print("\n💡 提示: 要使用完整的多模态功能，请安装以下依赖:")
        print("   pip install pyaudio pydub SpeechRecognition")
        print("   pip install opencv-python tensorflow")
        print("   pip install nltk")
        
    except ImportError as e:
        print(f"模块导入失败: {e}")
        print("请确保基本依赖已安装")
    except Exception as e:
        print(f"检测过程中发生错误: {e}")


def quick_text_analysis():
    """快速文本分析"""
    print("=== 快速文本分析 ===")
    
    try:
        from nlp.emotion_classifier_simple import SimpleEmotionClassifier
        
        classifier = SimpleEmotionClassifier()
        
        print("💡 输入文本进行情绪分析，输入'quit'退出")
        
        while True:
            text = input("\n请输入要分析的文本: ").strip()
            if text.lower() == 'quit':
                break
            
            if not text:
                print("请输入有效文本")
                continue
            
            emotion, confidence, scores = classifier.predict_emotion(text)
            print(f"🎯 情绪: {emotion}")
            print(f"📊 置信度: {confidence:.3f}")
            
            # 显示前3个情绪得分
            top_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"📈 前3个情绪: {', '.join([f'{e}({s:.2f})' for e, s in top_emotions])}")
    
    except Exception as e:
        print(f"文本分析失败: {e}")


def check_dependencies():
    """检查依赖包状态"""
    print("=== 依赖包检查 ===")
    
    dependencies = {
        'nltk': '自然语言处理',
        'numpy': '数值计算',
        'pandas': '数据处理', 
        'scikit-learn': '机器学习',
        'pyaudio': '音频录制',
        'pydub': '音频处理',
        'SpeechRecognition': '语音识别',
        'opencv-python': '计算机视觉',
        'tensorflow': '深度学习'
    }
    
    print("检查Python包安装状态:")
    for package, description in dependencies.items():
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package} - {description}")
        except ImportError:
            print(f"  ❌ {package} - {description} (未安装)")
    
    print("\n安装缺失的包:")
    print("pip install nltk numpy pandas scikit-learn")
    print("pip install pyaudio pydub SpeechRecognition") 
    print("pip install opencv-python tensorflow")


if __name__ == "__main__":
    print("选择功能:")
    print("1. 简单情绪检测演示")
    print("2. 快速文本情绪分析")
    print("3. 检查依赖包状态")
    
    choice = input("请选择 (1-3): ").strip()
    
    if choice == '1':
        simple_emotion_detection()
    elif choice == '2':
        quick_text_analysis()
    elif choice == '3':
        check_dependencies()
    else:
        print("无效选择") 