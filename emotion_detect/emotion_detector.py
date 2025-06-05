#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉情绪识别模块
基于CNN模型进行人脸表情识别
支持的情绪：angry, detester, peur, content, triste, surprise, nature
"""

import cv2
import numpy as np
import os
import logging
from typing import List, Dict, Optional, Tuple

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisionEmotionDetector:
    """视觉情绪识别器"""
    
    def __init__(self, model_path: str = "emotion_detection_cnn.h5"):
        """
        初始化视觉情绪识别器
        
        Args:
            model_path: 模型文件路径
        """
        self.model_path = model_path
        self.model = None
        self.face_cascade = None
        
        # 情绪标签（与模型训练时的顺序一致）
        self.emotions = ["angry", "detester", "peur", "content", "triste", "surprise", "nature"]
        
        # 初始化人脸检测器
        self._init_face_detector()
        
        # 加载模型
        self._load_model()
    
    def _init_face_detector(self):
        """初始化人脸检测器"""
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            logger.info("✅ 人脸检测器初始化成功")
        except Exception as e:
            logger.error(f"❌ 人脸检测器初始化失败: {e}")
            self.face_cascade = None
    
    def _load_model(self):
        """加载情绪识别模型"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"⚠️ 模型文件不存在: {self.model_path}")
                return
            
            from keras.models import load_model
            self.model = load_model(self.model_path)
            logger.info(f"✅ 情绪识别模型加载成功: {self.model_path}")
            
        except ImportError as e:
            logger.error(f"❌ 无法导入Keras: {e}")
            self.model = None
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            self.model = None
    
    def is_ready(self) -> bool:
        """检查检测器是否准备就绪"""
        return self.model is not None and self.face_cascade is not None
    
    def preprocess_face(self, face_roi: np.ndarray, target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """
        预处理人脸图像
        
        Args:
            face_roi: 人脸区域图像
            target_size: 目标尺寸
            
        Returns:
            预处理后的图像张量
        """
        try:
            # 调整大小
            resized = cv2.resize(face_roi, target_size)
            
            # 归一化
            normalized = resized.astype("float32") / 255.0
            
            # 添加批次和通道维度
            input_tensor = np.expand_dims(normalized, axis=(0, -1))
            
            return input_tensor
            
        except Exception as e:
            logger.error(f"人脸预处理失败: {e}")
            return None
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        检测图像中的人脸
        
        Args:
            image: 输入图像
            
        Returns:
            人脸边界框列表 [(x, y, w, h), ...]
        """
        if self.face_cascade is None:
            return []
        
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 检测人脸
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.3, 
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            return faces.tolist() if len(faces) > 0 else []
            
        except Exception as e:
            logger.error(f"人脸检测失败: {e}")
            return []
    
    def predict_emotion_from_face(self, face_roi: np.ndarray) -> Dict:
        """
        从人脸区域预测情绪
        
        Args:
            face_roi: 人脸区域图像（灰度图）
            
        Returns:
            预测结果字典
        """
        if self.model is None:
            return {
                'emotion': 'nature',
                'confidence': 0.0,
                'error': '模型未加载'
            }
        
        try:
            # 预处理
            input_tensor = self.preprocess_face(face_roi)
            if input_tensor is None:
                return {
                    'emotion': 'nature',
                    'confidence': 0.0,
                    'error': '预处理失败'
                }
            
            # 预测
            predictions = self.model.predict(input_tensor, verbose=0)[0]
            
            # 获取最高概率的情绪
            emotion_idx = np.argmax(predictions)
            emotion = self.emotions[emotion_idx]
            confidence = float(predictions[emotion_idx])
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': {
                    self.emotions[i]: float(prob) 
                    for i, prob in enumerate(predictions)
                },
                'method': 'vision'
            }
            
        except Exception as e:
            logger.error(f"情绪预测失败: {e}")
            return {
                'emotion': 'nature',
                'confidence': 0.0,
                'error': f'预测失败: {str(e)}'
            }
    
    def analyze_image(self, image: np.ndarray) -> Dict:
        """
        分析单张图像的情绪
        
        Args:
            image: 输入图像
            
        Returns:
            分析结果
        """
        if not self.is_ready():
            return {
                'emotion': 'nature',
                'confidence': 0.0,
                'faces_detected': 0,
                'error': '检测器未准备就绪'
            }
        
        # 检测人脸
        faces = self.detect_faces(image)
        
        if not faces:
            return {
                'emotion': 'nature',
                'confidence': 0.0,
                'faces_detected': 0,
                'error': '未检测到人脸'
            }
        
        # 分析每个人脸的情绪
        face_emotions = []
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        for (x, y, w, h) in faces:
            # 提取人脸区域
            face_roi = gray[y:y + h, x:x + w]
            
            # 预测情绪
            emotion_result = self.predict_emotion_from_face(face_roi)
            emotion_result['bbox'] = (x, y, w, h)
            face_emotions.append(emotion_result)
        
        # 如果有多个人脸，选择置信度最高的
        if face_emotions:
            best_result = max(face_emotions, key=lambda x: x.get('confidence', 0))
            
            return {
                'emotion': best_result['emotion'],
                'confidence': best_result['confidence'],
                'faces_detected': len(faces),
                'all_faces': face_emotions,
                'method': 'vision'
            }
        else:
            return {
                'emotion': 'nature',
                'confidence': 0.0,
                'faces_detected': len(faces),
                'error': '人脸情绪分析失败'
            }
    
    def analyze_video_frames(self, frames: List[np.ndarray]) -> Dict:
        """
        分析视频帧序列的情绪
        
        Args:
            frames: 视频帧列表
            
        Returns:
            分析结果
        """
        if not frames:
            return {
                'emotion': 'nature',
                'confidence': 0.0,
                'total_frames': 0,
                'valid_detections': 0,
                'error': '没有输入帧'
            }
        
        emotion_counts = {}
        confidence_scores = []
        valid_detections = 0
        
        for frame in frames:
            result = self.analyze_image(frame)
            
            if result.get('faces_detected', 0) > 0 and 'error' not in result:
                emotion = result['emotion']
                confidence = result['confidence']
                
                # 统计情绪
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                confidence_scores.append(confidence)
                valid_detections += 1
        
        if valid_detections > 0:
            # 找到最频繁的情绪
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            
            return {
                'emotion': dominant_emotion,
                'confidence': avg_confidence,
                'total_frames': len(frames),
                'valid_detections': valid_detections,
                'emotion_distribution': emotion_counts,
                'method': 'vision'
            }
        else:
            return {
                'emotion': 'nature',
                'confidence': 0.0,
                'total_frames': len(frames),
                'valid_detections': 0,
                'error': '所有帧都未检测到有效人脸'
            }
    
    def analyze_video_file(self, video_path: str, max_frames: int = 100) -> Dict:
        """
        分析视频文件的情绪
        
        Args:
            video_path: 视频文件路径
            max_frames: 最大分析帧数
            
        Returns:
            分析结果
        """
        if not os.path.exists(video_path):
            return {
                'emotion': 'nature',
                'confidence': 0.0,
                'error': f'视频文件不存在: {video_path}'
            }
        
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            frame_count = 0
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                frame_count += 1
            
            cap.release()
            
            if not frames:
                return {
                    'emotion': 'nature',
                    'confidence': 0.0,
                    'error': '无法读取视频帧'
                }
            
            return self.analyze_video_frames(frames)
            
        except Exception as e:
            return {
                'emotion': 'nature',
                'confidence': 0.0,
                'error': f'视频分析失败: {str(e)}'
            }
    
    def real_time_detection(self, duration: int = 5) -> Dict:
        """
        实时情绪检测
        
        Args:
            duration: 检测时长（秒）
            
        Returns:
            检测结果
        """
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return {
                    'emotion': 'nature',
                    'confidence': 0.0,
                    'error': '无法打开摄像头'
                }
            
            frames = []
            import time
            start_time = time.time()
            
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if ret:
                    frames.append(frame.copy())
                time.sleep(0.1)  # 控制帧率
            
            cap.release()
            
            if not frames:
                return {
                    'emotion': 'nature',
                    'confidence': 0.0,
                    'error': '未捕获到视频帧'
                }
            
            return self.analyze_video_frames(frames)
            
        except Exception as e:
            return {
                'emotion': 'nature',
                'confidence': 0.0,
                'error': f'实时检测失败: {str(e)}'
            }


def demo_vision_detection():
    """演示视觉情绪检测功能"""
    print("👁️ 视觉情绪识别演示")
    print("=" * 40)
    
    # 初始化检测器
    detector = VisionEmotionDetector()
    
    if not detector.is_ready():
        print("❌ 检测器未准备就绪，请检查模型文件和依赖")
        return
    
    print("✅ 检测器准备就绪")
    print(f"支持的情绪: {detector.emotions}")
    
    print("\n请选择测试模式:")
    print("1. 实时检测（需要摄像头）")
    print("2. 分析视频文件")
    print("3. 退出")
    
    while True:
        choice = input("\n请输入选择 (1-3): ").strip()
        
        if choice == '1':
            print("\n准备开始实时检测，请面对摄像头...")
            input("按回车键开始检测（5秒）...")
            
            result = detector.real_time_detection(duration=5)
            
            if 'error' not in result:
                print(f"\n🎯 检测结果:")
                print(f"   情绪: {result['emotion']}")
                print(f"   置信度: {result['confidence']:.3f}")
                print(f"   有效检测: {result['valid_detections']}/{result['total_frames']}")
                if result.get('emotion_distribution'):
                    print(f"   情绪分布: {result['emotion_distribution']}")
            else:
                print(f"❌ 检测失败: {result['error']}")
        
        elif choice == '2':
            video_path = input("请输入视频文件路径: ").strip()
            if video_path:
                result = detector.analyze_video_file(video_path)
                
                if 'error' not in result:
                    print(f"\n🎯 分析结果:")
                    print(f"   情绪: {result['emotion']}")
                    print(f"   置信度: {result['confidence']:.3f}")
                    print(f"   有效检测: {result['valid_detections']}/{result['total_frames']}")
                else:
                    print(f"❌ 分析失败: {result['error']}")
        
        elif choice == '3':
            print("👋 再见！")
            break
        
        else:
            print("❌ 无效选择，请重新输入")


if __name__ == "__main__":
    demo_vision_detection() 