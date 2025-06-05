#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速情绪识别API
提供最简单易用的情绪识别接口
"""

import time
import logging
from typing import Dict, Optional, Callable
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuickEmotionAPI:
    """快速情绪识别API"""
    
    def __init__(self):
        """初始化快速API"""
        self.detector = None
        self._init_detector()
    
    def _init_detector(self):
        """初始化检测器"""
        try:
            from unified_emotion_detector import UnifiedEmotionDetector
            self.detector = UnifiedEmotionDetector()
            logger.info("✅ 情绪识别API初始化成功")
        except Exception as e:
            logger.error(f"❌ API初始化失败: {e}")
            self.detector = None
    
    def is_ready(self) -> bool:
        """检查API是否准备就绪"""
        if not self.detector:
            return False
        
        status = self.detector.get_status()
        return any(status['modules_ready'].values())
    
    def quick_record_and_analyze(self, duration: int = 5) -> Dict:
        """一键录制并分析情绪"""
        if not self.is_ready():
            return {'success': False, 'error': 'API未准备就绪'}
        
        print(f"🎬 开始录制，时长: {duration}秒")
        print("💡 请对着摄像头说话，系统将分析您的语音和表情")
        
        results = []
        
        def result_handler(result):
            results.append(result)
            print(f"🎯 检测到情绪: {result['final_emotion']} (置信度: {result['confidence']:.2f})")
        
        self.detector.set_result_callback(result_handler)
        
        if not self.detector.start_recording(duration):
            return {'success': False, 'error': '录制启动失败'}
        
        time.sleep(duration + 1)
        self.detector.stop_recording()
        
        if results:
            return self._generate_summary(results)
        else:
            return {'success': False, 'error': '未检测到任何情绪结果'}
    
    def analyze_text(self, text: str) -> Dict:
        """分析文本情绪"""
        if not self.is_ready():
            return {'success': False, 'error': 'API未准备就绪'}
        
        result = self.detector.analyze_text(text)
        
        if 'error' in result:
            return {'success': False, 'error': result['error']}
        
        return {
            'success': True,
            'emotion': result['emotion'],
            'confidence': result['confidence'],
            'text': text,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def analyze_audio_file(self, audio_file: str) -> Dict:
        """
        分析音频文件情绪
        
        Args:
            audio_file: 音频文件路径
            
        Returns:
            分析结果
        """
        if not self.is_ready():
            return {
                'success': False,
                'error': 'API未准备就绪'
            }
        
        result = self.detector.analyze_audio_file(audio_file)
        
        if 'error' in result:
            return {
                'success': False,
                'error': result['error']
            }
        
        return {
            'success': True,
            'emotion': result['emotion'],
            'confidence': result['confidence'],
            'file': audio_file,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def analyze_image_file(self, image_file: str) -> Dict:
        """
        分析图像文件情绪
        
        Args:
            image_file: 图像文件路径
            
        Returns:
            分析结果
        """
        if not self.is_ready():
            return {
                'success': False,
                'error': 'API未准备就绪'
            }
        
        result = self.detector.analyze_image_file(image_file)
        
        if 'error' in result:
            return {
                'success': False,
                'error': result['error']
            }
        
        return {
            'success': True,
            'emotion': result['emotion'],
            'confidence': result['confidence'],
            'file': image_file,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _generate_summary(self, results: list) -> Dict:
        """生成结果摘要"""
        if not results:
            return {'success': False, 'error': '无结果数据'}
        
        emotion_counts = {}
        confidence_scores = []
        
        for result in results:
            emotion = result['final_emotion']
            confidence = result['confidence']
            
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            confidence_scores.append(confidence)
        
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        return {
            'success': True,
            'dominant_emotion': dominant_emotion,
            'average_confidence': avg_confidence,
            'emotion_distribution': emotion_counts,
            'total_detections': len(results),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': f"主要情绪: {dominant_emotion} (平均置信度: {avg_confidence:.2f})"
        }
    
    def get_status(self) -> Dict:
        """获取API状态"""
        if not self.detector:
            return {
                'ready': False,
                'error': '检测器未初始化'
            }
        
        status = self.detector.get_status()
        
        return {
            'ready': self.is_ready(),
            'modules': status['modules_enabled'],
            'modules_ready': status['modules_ready']
        }


# 全局API实例
_api_instance = None

def get_emotion_api() -> QuickEmotionAPI:
    """获取全局API实例"""
    global _api_instance
    if _api_instance is None:
        _api_instance = QuickEmotionAPI()
    return _api_instance


def quick_emotion_check(duration: int = 5) -> Dict:
    """快速情绪检测（最简单的接口）"""
    api = get_emotion_api()
    return api.quick_record_and_analyze(duration)


def analyze_text_emotion(text: str) -> Dict:
    """快速文本情绪分析"""
    api = get_emotion_api()
    return api.analyze_text(text)


def analyze_audio_emotion(audio_file: str) -> Dict:
    """
    快速音频情绪分析
    
    Args:
        audio_file: 音频文件路径
        
    Returns:
        情绪分析结果
    """
    api = get_emotion_api()
    return api.analyze_audio_file(audio_file)


def analyze_image_emotion(image_file: str) -> Dict:
    """
    快速图像情绪分析
    
    Args:
        image_file: 图像文件路径
        
    Returns:
        情绪分析结果
    """
    api = get_emotion_api()
    return api.analyze_image_file(image_file)


def check_api_status() -> Dict:
    """检查API状态"""
    api = get_emotion_api()
    return api.get_status()


def demo_quick_api():
    """演示快速API"""
    print("=" * 50)
    print("快速情绪识别API演示")
    print("=" * 50)
    
    # 检查状态
    status = check_api_status()
    print(f"API状态: {'✅ 就绪' if status['ready'] else '❌ 未就绪'}")
    
    if not status['ready']:
        print("API未准备就绪，请检查模块配置")
        return
    
    print("\n选择功能:")
    print("1. 快速情绪检测（5秒录制）")
    print("2. 文本情绪分析")
    print("3. 音频文件分析")
    print("4. 图像文件分析")
    
    try:
        choice = input("请选择 (1-4): ").strip()
        
        if choice == '1':
            print("\n=== 快速情绪检测 ===")
            result = quick_emotion_check(5)
            
            if result['success']:
                print(f"✅ {result['summary']}")
                print(f"情绪分布: {result['emotion_distribution']}")
                print(f"检测次数: {result['total_detections']}")
            else:
                print(f"❌ 检测失败: {result['error']}")
        
        elif choice == '2':
            text = input("请输入要分析的文本: ")
            result = analyze_text_emotion(text)
            
            if result['success']:
                print(f"✅ 文本情绪: {result['emotion']}")
                print(f"置信度: {result['confidence']:.2f}")
            else:
                print(f"❌ 分析失败: {result['error']}")
        
        elif choice == '3':
            audio_file = input("请输入音频文件路径: ")
            result = analyze_audio_emotion(audio_file)
            
            if result['success']:
                print(f"✅ 音频情绪: {result['emotion']}")
                print(f"置信度: {result['confidence']:.2f}")
            else:
                print(f"❌ 分析失败: {result['error']}")
        
        elif choice == '4':
            image_file = input("请输入图像文件路径: ")
            result = analyze_image_emotion(image_file)
            
            if result['success']:
                print(f"✅ 图像情绪: {result['emotion']}")
                print(f"置信度: {result['confidence']:.2f}")
            else:
                print(f"❌ 分析失败: {result['error']}")
        
        else:
            print("无效选择")
    
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
    
    print("\n演示完成！")


if __name__ == "__main__":
    demo_quick_api() 