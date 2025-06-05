#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一情绪识别接口
整合语音、语言和表情三个模块，提供简单易用的多模态情绪识别服务
技术路线：请求 -> 开始录制音视频 -> 解析 -> 反馈情绪
"""

import cv2
import numpy as np
import threading
import time
import queue
import tempfile
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Callable, Tuple
import logging

# 导入各个模块
from nlp.emotion_pipeline import EmotionAnalysisPipeline
from audio.real_time_audio_emotion import RealTimeAudioEmotionRecognizer
from vision.emotion_detector import VisionEmotionDetector

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedEmotionDetector:
    """
    统一情绪识别器
    整合语音、语言和表情三个模块的多模态情绪识别
    """
    
    def __init__(self, 
                 audio_model_path: str = "audio/models/audio_emotion_model.pkl",
                 vision_model_path: str = "vision/emotion_detection_cnn.h5",
                 enable_audio: bool = True,
                 enable_vision: bool = True,
                 enable_nlp: bool = True):
        """
        初始化统一情绪识别器
        
        Args:
            audio_model_path: 音频模型路径
            vision_model_path: 视觉模型路径
            enable_audio: 是否启用音频识别
            enable_vision: 是否启用视觉识别
            enable_nlp: 是否启用语言识别
        """
        self.enable_audio = enable_audio
        self.enable_vision = enable_vision
        self.enable_nlp = enable_nlp
        
        # 初始化各个模块
        self.audio_detector = None
        self.vision_detector = None
        self.nlp_pipeline = None
        
        self._init_modules(audio_model_path, vision_model_path)
        
        # 录制控制
        self.is_recording = False
        self.recording_thread = None
        self.video_capture = None
        
        # 结果队列
        self.audio_results = queue.Queue()
        self.vision_results = queue.Queue()
        self.nlp_results = queue.Queue()
        self.unified_results = queue.Queue()
        
        # 回调函数
        self.result_callback = None
        
        # 情绪映射（统一不同模块的情绪标签）
        self.emotion_mapping = {
            # 音频模块映射
            'happy': '快乐',
            'sad': '悲伤',
            'angry': '愤怒',
            'fear': '恐惧',
            'surprise': '惊讶',
            'neutral': '中性',
            
            # 视觉模块映射
            'content': '快乐',
            'triste': '悲伤',
            'peur': '恐惧',
            'detester': '愤怒',
            'nature': '中性',
            
            # NLP模块映射（已经是中文）
            '积极': '积极',
            '消极': '消极',
            '愤怒': '愤怒',
            '悲伤': '悲伤',
            '快乐': '快乐',
            '恐惧': '恐惧',
            '惊讶': '惊讶',
            '中性': '中性'
        }
    
    def _init_modules(self, audio_model_path: str, vision_model_path: str):
        """初始化各个识别模块"""
        
        # 初始化音频模块
        if self.enable_audio:
            try:
                self.audio_detector = RealTimeAudioEmotionRecognizer(audio_model_path)
                logger.info("✅ 音频情绪识别模块初始化成功")
            except Exception as e:
                logger.warning(f"⚠️ 音频模块初始化失败: {e}")
                self.enable_audio = False
        
        # 初始化视觉模块
        if self.enable_vision:
            try:
                self.vision_detector = VisionEmotionDetector(vision_model_path)
                if not self.vision_detector.is_ready():
                    raise Exception("视觉模块未准备就绪")
                logger.info("✅ 视觉情绪识别模块初始化成功")
            except Exception as e:
                logger.warning(f"⚠️ 视觉模块初始化失败: {e}")
                self.enable_vision = False
        
        # 初始化NLP模块
        if self.enable_nlp:
            try:
                self.nlp_pipeline = EmotionAnalysisPipeline()
                logger.info("✅ 语言情绪识别模块初始化成功")
            except Exception as e:
                logger.warning(f"⚠️ NLP模块初始化失败: {e}")
                self.enable_nlp = False
    
    def set_result_callback(self, callback: Callable[[Dict], None]):
        """
        设置结果回调函数
        
        Args:
            callback: 回调函数，接收统一的情绪识别结果
        """
        self.result_callback = callback
    
    def start_recording(self, duration: int = 10):
        """
        开始录制音视频并进行情绪识别
        
        Args:
            duration: 录制时长（秒），0表示持续录制直到手动停止
        """
        if self.is_recording:
            logger.warning("⚠️ 已经在录制中")
            return False
        
        logger.info(f"🎬 开始多模态情绪识别录制，时长: {duration}秒" if duration > 0 else "🎬 开始持续多模态情绪识别录制")
        
        self.is_recording = True
        
        # 启动录制线程
        self.recording_thread = threading.Thread(
            target=self._recording_worker, 
            args=(duration,)
        )
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        return True
    
    def stop_recording(self):
        """停止录制"""
        if not self.is_recording:
            logger.warning("⚠️ 没有在录制")
            return False
        
        logger.info("🛑 停止多模态情绪识别录制")
        self.is_recording = False
        
        # 停止各个模块
        if self.enable_audio and self.audio_detector:
            self.audio_detector.stop_recognition()
        
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
        # 等待录制线程结束
        if self.recording_thread:
            self.recording_thread.join(timeout=3)
        
        return True
    
    def _recording_worker(self, duration: int):
        """录制工作线程"""
        start_time = time.time()
        
        # 启动音频识别
        if self.enable_audio and self.audio_detector:
            self.audio_detector.set_emotion_callback(self._audio_callback)
            self.audio_detector.start_recognition()
        
        # 启动视频捕获
        if self.enable_vision:
            try:
                self.video_capture = cv2.VideoCapture(0)
                if not self.video_capture.isOpened():
                    logger.error("❌ 无法打开摄像头")
                    self.enable_vision = False
            except Exception as e:
                logger.error(f"❌ 摄像头初始化失败: {e}")
                self.enable_vision = False
        
        # 主录制循环
        frame_count = 0
        last_vision_time = 0
        vision_interval = 1.0  # 每秒分析一次视觉情绪
        
        while self.is_recording:
            current_time = time.time()
            
            # 检查录制时长
            if duration > 0 and (current_time - start_time) >= duration:
                break
            
            # 处理视频帧
            if self.enable_vision and self.video_capture:
                ret, frame = self.video_capture.read()
                if ret:
                    frame_count += 1
                    
                    # 定期进行视觉情绪分析
                    if current_time - last_vision_time >= vision_interval:
                        self._analyze_vision_frame(frame, current_time)
                        last_vision_time = current_time
            
            # 融合多模态结果
            self._fuse_multimodal_results()
            
            time.sleep(0.1)  # 避免过度占用CPU
        
        # 清理资源
        self.is_recording = False
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
    
    def _audio_callback(self, audio_result: Dict):
        """音频识别结果回调"""
        try:
            # 标准化音频结果
            standardized_result = {
                'modality': 'audio',
                'timestamp': audio_result.get('timestamp', time.time()),
                'emotion': self.emotion_mapping.get(audio_result.get('emotion', 'neutral'), '中性'),
                'confidence': audio_result.get('confidence', 0.0),
                'raw_result': audio_result
            }
            
            self.audio_results.put(standardized_result)
            
        except Exception as e:
            logger.error(f"音频结果处理失败: {e}")
    
    def _analyze_vision_frame(self, frame: np.ndarray, timestamp: float):
        """分析视频帧"""
        try:
            if self.vision_detector:
                result = self.vision_detector.analyze_image(frame)
                
                if result.get('faces_detected', 0) > 0:
                    # 取第一个检测到的人脸结果
                    face_result = result['faces'][0] if result.get('faces') else {}
                    
                    standardized_result = {
                        'modality': 'vision',
                        'timestamp': timestamp,
                        'emotion': self.emotion_mapping.get(face_result.get('emotion', 'nature'), '中性'),
                        'confidence': face_result.get('confidence', 0.0),
                        'raw_result': result
                    }
                    
                    self.vision_results.put(standardized_result)
        
        except Exception as e:
            logger.error(f"视觉分析失败: {e}")
    
    def _fuse_multimodal_results(self):
        """融合多模态结果"""
        try:
            # 收集最近的结果
            recent_results = {
                'audio': [],
                'vision': [],
                'nlp': []
            }
            
            current_time = time.time()
            time_window = 2.0  # 2秒时间窗口
            
            # 收集音频结果
            while not self.audio_results.empty():
                try:
                    result = self.audio_results.get_nowait()
                    if current_time - result['timestamp'] <= time_window:
                        recent_results['audio'].append(result)
                except queue.Empty:
                    break
            
            # 收集视觉结果
            while not self.vision_results.empty():
                try:
                    result = self.vision_results.get_nowait()
                    if current_time - result['timestamp'] <= time_window:
                        recent_results['vision'].append(result)
                except queue.Empty:
                    break
            
            # 收集NLP结果
            while not self.nlp_results.empty():
                try:
                    result = self.nlp_results.get_nowait()
                    if current_time - result['timestamp'] <= time_window:
                        recent_results['nlp'].append(result)
                except queue.Empty:
                    break
            
            # 如果有结果，进行融合
            if any(recent_results.values()):
                fused_result = self._compute_fused_emotion(recent_results)
                
                if fused_result:
                    self.unified_results.put(fused_result)
                    
                    # 调用回调函数
                    if self.result_callback:
                        self.result_callback(fused_result)
        
        except Exception as e:
            logger.error(f"结果融合失败: {e}")
    
    def _compute_fused_emotion(self, recent_results: Dict) -> Optional[Dict]:
        """计算融合后的情绪"""
        try:
            # 情绪投票统计
            emotion_votes = {}
            total_confidence = 0.0
            modality_count = 0
            
            # 权重设置
            modality_weights = {
                'audio': 0.4,
                'vision': 0.4,
                'nlp': 0.2
            }
            
            modality_results = {}
            
            for modality, results in recent_results.items():
                if results:
                    # 取最新的结果
                    latest_result = max(results, key=lambda x: x['timestamp'])
                    emotion = latest_result['emotion']
                    confidence = latest_result['confidence']
                    
                    # 加权投票
                    weight = modality_weights.get(modality, 0.3)
                    weighted_confidence = confidence * weight
                    
                    if emotion in emotion_votes:
                        emotion_votes[emotion] += weighted_confidence
                    else:
                        emotion_votes[emotion] = weighted_confidence
                    
                    total_confidence += weighted_confidence
                    modality_count += 1
                    
                    modality_results[modality] = {
                        'emotion': emotion,
                        'confidence': confidence,
                        'timestamp': latest_result['timestamp']
                    }
            
            if not emotion_votes:
                return None
            
            # 找到得票最高的情绪
            final_emotion = max(emotion_votes, key=emotion_votes.get)
            final_confidence = emotion_votes[final_emotion]
            
            # 归一化置信度
            if total_confidence > 0:
                final_confidence = min(final_confidence / total_confidence * modality_count, 1.0)
            
            return {
                'timestamp': time.time(),
                'final_emotion': final_emotion,
                'confidence': final_confidence,
                'modality_results': modality_results,
                'emotion_votes': emotion_votes,
                'fusion_method': 'weighted_voting'
            }
        
        except Exception as e:
            logger.error(f"情绪融合计算失败: {e}")
            return None
    
    def analyze_text(self, text: str) -> Dict:
        """
        分析文本情绪（单独调用NLP模块）
        
        Args:
            text: 输入文本
            
        Returns:
            分析结果
        """
        if not self.enable_nlp or not self.nlp_pipeline:
            return {'error': 'NLP模块未启用或未初始化'}
        
        try:
            # 使用基于规则的分析
            normalized_text = self.nlp_pipeline.emotion_classifier.normalize_text(text)
            result = self.nlp_pipeline._rule_based_emotion_analysis(normalized_text)
            
            return {
                'modality': 'nlp',
                'timestamp': time.time(),
                'emotion': result['predicted_emotion'],
                'confidence': result['confidence'],
                'raw_result': result
            }
        
        except Exception as e:
            logger.error(f"文本分析失败: {e}")
            return {'error': f'文本分析失败: {str(e)}'}
    
    def analyze_audio_file(self, audio_file: str) -> Dict:
        """
        分析音频文件情绪
        
        Args:
            audio_file: 音频文件路径
            
        Returns:
            分析结果
        """
        if not self.enable_audio or not self.audio_detector:
            return {'error': '音频模块未启用或未初始化'}
        
        try:
            result = self.audio_detector.recognizer.predict_emotion(audio_file)
            
            return {
                'modality': 'audio',
                'timestamp': time.time(),
                'emotion': self.emotion_mapping.get(result.get('emotion', 'neutral'), '中性'),
                'confidence': result.get('confidence', 0.0),
                'raw_result': result
            }
        
        except Exception as e:
            logger.error(f"音频文件分析失败: {e}")
            return {'error': f'音频文件分析失败: {str(e)}'}
    
    def analyze_image_file(self, image_file: str) -> Dict:
        """
        分析图像文件情绪
        
        Args:
            image_file: 图像文件路径
            
        Returns:
            分析结果
        """
        if not self.enable_vision or not self.vision_detector:
            return {'error': '视觉模块未启用或未初始化'}
        
        try:
            image = cv2.imread(image_file)
            if image is None:
                return {'error': '无法读取图像文件'}
            
            result = self.vision_detector.analyze_image(image)
            
            if result.get('faces_detected', 0) > 0:
                face_result = result['faces'][0]
                return {
                    'modality': 'vision',
                    'timestamp': time.time(),
                    'emotion': self.emotion_mapping.get(face_result.get('emotion', 'nature'), '中性'),
                    'confidence': face_result.get('confidence', 0.0),
                    'raw_result': result
                }
            else:
                return {'error': '未检测到人脸'}
        
        except Exception as e:
            logger.error(f"图像分析失败: {e}")
            return {'error': f'图像分析失败: {str(e)}'}
    
    def get_latest_result(self) -> Optional[Dict]:
        """获取最新的融合结果"""
        try:
            return self.unified_results.get_nowait()
        except queue.Empty:
            return None
    
    def get_all_results(self) -> List[Dict]:
        """获取所有融合结果"""
        results = []
        while True:
            try:
                result = self.unified_results.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        return results
    
    def get_status(self) -> Dict:
        """获取检测器状态"""
        return {
            'is_recording': self.is_recording,
            'modules_enabled': {
                'audio': self.enable_audio,
                'vision': self.enable_vision,
                'nlp': self.enable_nlp
            },
            'modules_ready': {
                'audio': self.audio_detector is not None,
                'vision': self.vision_detector is not None and self.vision_detector.is_ready(),
                'nlp': self.nlp_pipeline is not None
            }
        }
    
    def export_results(self, output_file: str, format: str = 'json'):
        """
        导出所有结果
        
        Args:
            output_file: 输出文件路径
            format: 输出格式 ('json', 'csv')
        """
        try:
            all_results = self.get_all_results()
            
            if format == 'json':
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
            
            elif format == 'csv':
                import pandas as pd
                
                # 展平结果
                flattened_results = []
                for result in all_results:
                    flat_result = {
                        'timestamp': result['timestamp'],
                        'final_emotion': result['final_emotion'],
                        'confidence': result['confidence'],
                        'fusion_method': result['fusion_method']
                    }
                    
                    # 添加各模态结果
                    for modality, mod_result in result.get('modality_results', {}).items():
                        flat_result[f'{modality}_emotion'] = mod_result['emotion']
                        flat_result[f'{modality}_confidence'] = mod_result['confidence']
                    
                    flattened_results.append(flat_result)
                
                df = pd.DataFrame(flattened_results)
                df.to_csv(output_file, index=False, encoding='utf-8')
            
            logger.info(f"结果已导出到: {output_file}")
            
        except Exception as e:
            logger.error(f"结果导出失败: {e}")


def demo_unified_detection():
    """演示统一情绪识别"""
    print("=" * 60)
    print("统一多模态情绪识别演示")
    print("=" * 60)
    
    # 初始化检测器
    detector = UnifiedEmotionDetector()
    
    # 检查状态
    status = detector.get_status()
    print("模块状态:")
    for module, enabled in status['modules_enabled'].items():
        ready = status['modules_ready'][module]
        status_text = "✅ 就绪" if enabled and ready else "❌ 未就绪"
        print(f"  {module}: {status_text}")
    
    # 设置结果回调
    def result_callback(result):
        print(f"\n🎯 检测到情绪: {result['final_emotion']}")
        print(f"   置信度: {result['confidence']:.3f}")
        print(f"   时间: {datetime.fromtimestamp(result['timestamp']).strftime('%H:%M:%S')}")
        
        # 显示各模态结果
        for modality, mod_result in result.get('modality_results', {}).items():
            print(f"   {modality}: {mod_result['emotion']} ({mod_result['confidence']:.3f})")
    
    detector.set_result_callback(result_callback)
    
    print("\n选择测试模式:")
    print("1. 实时多模态检测（10秒）")
    print("2. 文本情绪分析")
    print("3. 音频文件分析")
    print("4. 图像文件分析")
    
    try:
        choice = input("请选择 (1-4): ").strip()
        
        if choice == '1':
            print("\n开始实时多模态情绪检测...")
            print("请对着摄像头说话，系统将分析您的语音和表情")
            
            detector.start_recording(duration=10)
            
            # 等待录制完成
            time.sleep(11)
            
            # 获取所有结果
            results = detector.get_all_results()
            print(f"\n检测完成！共获得 {len(results)} 个融合结果")
            
        elif choice == '2':
            text = input("请输入要分析的文本: ")
            result = detector.analyze_text(text)
            
            if 'error' not in result:
                print(f"文本情绪: {result['emotion']}")
                print(f"置信度: {result['confidence']:.3f}")
            else:
                print(f"分析失败: {result['error']}")
        
        elif choice == '3':
            audio_file = input("请输入音频文件路径: ")
            result = detector.analyze_audio_file(audio_file)
            
            if 'error' not in result:
                print(f"音频情绪: {result['emotion']}")
                print(f"置信度: {result['confidence']:.3f}")
            else:
                print(f"分析失败: {result['error']}")
        
        elif choice == '4':
            image_file = input("请输入图像文件路径: ")
            result = detector.analyze_image_file(image_file)
            
            if 'error' not in result:
                print(f"图像情绪: {result['emotion']}")
                print(f"置信度: {result['confidence']:.3f}")
            else:
                print(f"分析失败: {result['error']}")
        
        else:
            print("无效选择")
    
    except KeyboardInterrupt:
        print("\n演示被用户中断")
        detector.stop_recording()
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        detector.stop_recording()
    
    print("\n演示完成！")


if __name__ == "__main__":
    demo_unified_detection() 