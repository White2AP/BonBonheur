#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版统一情绪识别器
只使用基本依赖，避免复杂的模块导入问题
"""

import time
import logging
from typing import Dict, Optional
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleUnifiedDetector:
    """简化版统一情绪识别器"""
    
    def __init__(self):
        """初始化简化检测器"""
        self.nlp_classifier = None
        self.modules_ready = {
            'nlp': False,
            'audio': False,
            'vision': False
        }
        
        self._init_nlp_module()
    
    def _init_nlp_module(self):
        """初始化NLP模块"""
        try:
            import sys
            import os
            
            # 添加nlp目录到路径
            nlp_path = os.path.join(os.path.dirname(__file__), 'nlp')
            if nlp_path not in sys.path:
                sys.path.append(nlp_path)
            
            from emotion_classifier_simple import SimpleEmotionClassifier
            self.nlp_classifier = SimpleEmotionClassifier()
            self.modules_ready['nlp'] = True
            logger.info("✅ NLP模块初始化成功")
            
        except Exception as e:
            logger.warning(f"⚠️ NLP模块初始化失败: {e}")
            self.modules_ready['nlp'] = False
    
    def analyze_text(self, text: str) -> Dict:
        """分析文本情绪"""
        if not self.modules_ready['nlp'] or not self.nlp_classifier:
            return {'success': False, 'error': 'NLP模块未准备就绪'}
        
        try:
            emotion, confidence, scores = self.nlp_classifier.predict_emotion(text)
            
            return {
                'success': True,
                'emotion': emotion,
                'confidence': confidence,
                'scores': dict(scores),
                'text': text,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"文本分析失败: {e}")
            return {'success': False, 'error': f'文本分析失败: {str(e)}'}
    
    def batch_analyze_texts(self, texts: list) -> list:
        """批量分析文本"""
        results = []
        for text in texts:
            result = self.analyze_text(text)
            results.append(result)
        return results
    
    def get_status(self) -> Dict:
        """获取检测器状态"""
        return {
            'modules_ready': self.modules_ready,
            'available_functions': [
                'analyze_text',
                'batch_analyze_texts'
            ],
            'ready': self.modules_ready['nlp']
        }
    
    def demo_text_analysis(self):
        """演示文本分析功能"""
        print("=== 简化版情绪识别演示 ===")
        
        # 检查状态
        status = self.get_status()
        print(f"NLP模块状态: {'✅ 就绪' if status['modules_ready']['nlp'] else '❌ 未就绪'}")
        
        if not status['ready']:
            print("系统未准备就绪，无法进行演示")
            return
        
        # 测试文本
        test_texts = [
            "今天天气真好，我很开心！",
            "这个产品质量太差了，我很生气。", 
            "我对考试结果感到失望。",
            "哇，这个消息太令人震惊了！",
            "我很害怕明天的面试。",
            "这件事让我感到很悲伤。",
            "今天的会议很普通。",
            "我对这个结果非常满意！"
        ]
        
        print("\n📝 文本情绪分析结果:")
        print("-" * 60)
        
        for i, text in enumerate(test_texts, 1):
            result = self.analyze_text(text)
            
            if result['success']:
                print(f"{i}. 文本: {text}")
                print(f"   情绪: {result['emotion']}")
                print(f"   置信度: {result['confidence']:.3f}")
                
                # 显示前3个情绪得分
                top_emotions = sorted(result['scores'].items(), 
                                    key=lambda x: x[1], reverse=True)[:3]
                print(f"   前3情绪: {', '.join([f'{e}({s:.2f})' for e, s in top_emotions])}")
                print()
            else:
                print(f"{i}. 文本: {text}")
                print(f"   ❌ 分析失败: {result['error']}")
                print()
        
        # 统计分析
        results = self.batch_analyze_texts(test_texts)
        successful_results = [r for r in results if r['success']]
        
        if successful_results:
            emotions = [r['emotion'] for r in successful_results]
            from collections import Counter
            emotion_counts = Counter(emotions)
            
            print("📊 情绪分布统计:")
            for emotion, count in emotion_counts.most_common():
                percentage = (count / len(successful_results)) * 100
                print(f"   {emotion}: {count}次 ({percentage:.1f}%)")
            
            avg_confidence = sum(r['confidence'] for r in successful_results) / len(successful_results)
            print(f"\n📈 平均置信度: {avg_confidence:.3f}")
        
        print("\n✅ 演示完成！")


def demo_simple_detector():
    """演示简化版检测器"""
    detector = SimpleUnifiedDetector()
    
    print("选择功能:")
    print("1. 文本情绪分析演示")
    print("2. 交互式文本分析")
    print("3. 检查系统状态")
    
    try:
        choice = input("请选择 (1-3): ").strip()
        
        if choice == '1':
            detector.demo_text_analysis()
            
        elif choice == '2':
            print("\n=== 交互式文本分析 ===")
            print("输入 'quit' 退出")
            
            while True:
                text = input("\n请输入要分析的文本: ").strip()
                if text.lower() == 'quit':
                    break
                
                if not text:
                    print("请输入有效文本")
                    continue
                
                result = detector.analyze_text(text)
                
                if result['success']:
                    print(f"🎯 情绪: {result['emotion']}")
                    print(f"📊 置信度: {result['confidence']:.3f}")
                    
                    # 显示详细得分
                    print("📈 详细得分:")
                    for emotion, score in sorted(result['scores'].items(), 
                                               key=lambda x: x[1], reverse=True):
                        print(f"   {emotion}: {score:.3f}")
                else:
                    print(f"❌ 分析失败: {result['error']}")
            
            print("交互式分析结束")
            
        elif choice == '3':
            status = detector.get_status()
            print("\n=== 系统状态 ===")
            print(f"整体状态: {'✅ 就绪' if status['ready'] else '❌ 未就绪'}")
            print("模块状态:")
            for module, ready in status['modules_ready'].items():
                print(f"  {module}: {'✅' if ready else '❌'}")
            print(f"可用功能: {', '.join(status['available_functions'])}")
        
        else:
            print("无效选择")
    
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"演示过程中发生错误: {e}")


if __name__ == "__main__":
    demo_simple_detector() 