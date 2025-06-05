#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
实时音频情绪识别模块
支持麦克风实时录音和情绪识别
"""

import numpy as np
import pyaudio
import wave
import threading
import time
import queue
from audio_emotion_recognition import AudioEmotionRecognizer
import tempfile
import os

class RealTimeAudioEmotionRecognizer:
    """实时音频情绪识别器"""

    
    def __init__(self, model_path="models/audio_emotion_model.pkl"):
        """
        初始化实时音频情绪识别器
        
        Args:
            model_path: 训练好的模型路径
        """
        # 加载训练好的模型
        self.recognizer = AudioEmotionRecognizer()
        try:
            self.recognizer.load_model(model_path)
            print("✅ 模型加载成功")
        except FileNotFoundError:
            print("❌ 模型文件不存在，请先训练模型")
            raise
        
        # 音频参数
        self.sample_rate = 22050
        self.chunk_size = 1024
        self.channels = 1
        self.format = pyaudio.paInt16
        self.record_duration = 3  # 录音时长（秒）
        
        # PyAudio对象
        self.audio = pyaudio.PyAudio()
        
        # 控制变量
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # 回调函数
        self.emotion_callback = None
    
    def set_emotion_callback(self, callback):
        """
        设置情绪识别结果回调函数
        
        Args:
            callback: 回调函数，接收识别结果作为参数
        """
        self.emotion_callback = callback
    
    def record_audio_chunk(self):
        """录制一段音频"""
        try:
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            frames = []
            for _ in range(int(self.sample_rate / self.chunk_size * self.record_duration)):
                if not self.is_recording:
                    break
                data = stream.read(self.chunk_size)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            
            if frames:
                # 保存为临时文件
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                
                wf = wave.open(temp_file.name, 'wb')
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                return temp_file.name
            
        except Exception as e:
            print(f"录音错误: {e}")
        
        return None
    
    def process_audio_worker(self):
        """音频处理工作线程"""
        while self.is_recording:
            try:
                # 录制音频
                audio_file = self.record_audio_chunk()
                
                if audio_file:
                    # 识别情绪
                    result = self.recognizer.predict_emotion(audio_file)
                    
                    # 清理临时文件
                    os.unlink(audio_file)
                    
                    if result:
                        # 添加时间戳
                        result['timestamp'] = time.time()
                        
                        # 放入结果队列
                        self.result_queue.put(result)
                        
                        # 调用回调函数
                        if self.emotion_callback:
                            self.emotion_callback(result)
                
            except Exception as e:
                print(f"音频处理错误: {e}")
            
            # 短暂休息
            time.sleep(0.1)
    
    def start_recognition(self):
        """开始实时情绪识别"""
        if self.is_recording:
            print("⚠️ 已经在录音中")
            return
        
        print("🎤 开始实时音频情绪识别...")
        self.is_recording = True
        
        # 启动音频处理线程
        self.process_thread = threading.Thread(target=self.process_audio_worker)
        self.process_thread.daemon = True
        self.process_thread.start()
    
    def stop_recognition(self):
        """停止实时情绪识别"""
        if not self.is_recording:
            print("⚠️ 没有在录音")
            return
        
        print("🛑 停止实时音频情绪识别")
        self.is_recording = False
        
        # 等待线程结束
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=2)
    
    def get_latest_result(self):
        """获取最新的识别结果"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_all_results(self):
        """获取所有识别结果"""
        results = []
        while True:
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        return results
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'audio'):
            self.audio.terminate()

class EmotionMonitor:
    """情绪监控器 - 用于统计和分析情绪变化"""
    
    def __init__(self, window_size=10):
        """
        初始化情绪监控器
        
        Args:
            window_size: 滑动窗口大小
        """
        self.window_size = window_size
        self.emotion_history = []
        self.confidence_history = []
        self.timestamp_history = []
    
    def add_result(self, result):
        """添加识别结果"""
        self.emotion_history.append(result['emotion'])
        self.confidence_history.append(result['confidence'])
        self.timestamp_history.append(result['timestamp'])
        
        # 保持窗口大小
        if len(self.emotion_history) > self.window_size:
            self.emotion_history.pop(0)
            self.confidence_history.pop(0)
            self.timestamp_history.pop(0)
    
    def get_dominant_emotion(self):
        """获取主导情绪"""
        if not self.emotion_history:
            return None
        
        # 统计情绪频次
        emotion_counts = {}
        for emotion in self.emotion_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # 返回最频繁的情绪
        return max(emotion_counts, key=emotion_counts.get)
    
    def get_average_confidence(self):
        """获取平均置信度"""
        if not self.confidence_history:
            return 0.0
        return np.mean(self.confidence_history)
    
    def get_emotion_stability(self):
        """获取情绪稳定性（变化频率）"""
        if len(self.emotion_history) < 2:
            return 1.0
        
        changes = 0
        for i in range(1, len(self.emotion_history)):
            if self.emotion_history[i] != self.emotion_history[i-1]:
                changes += 1
        
        return 1.0 - (changes / (len(self.emotion_history) - 1))
    
    def get_statistics(self):
        """获取统计信息"""
        if not self.emotion_history:
            return None
        
        return {
            'dominant_emotion': self.get_dominant_emotion(),
            'average_confidence': self.get_average_confidence(),
            'emotion_stability': self.get_emotion_stability(),
            'total_samples': len(self.emotion_history),
            'emotion_distribution': {
                emotion: self.emotion_history.count(emotion) / len(self.emotion_history)
                for emotion in set(self.emotion_history)
            }
        }

def demo_real_time_recognition():
    """演示实时音频情绪识别"""
    print("🎵 实时音频情绪识别演示")
    print("=" * 50)
    
    try:
        # 创建实时识别器
        recognizer = RealTimeAudioEmotionRecognizer()
        
        # 创建情绪监控器
        monitor = EmotionMonitor()
        
        # 设置回调函数
        def emotion_callback(result):
            monitor.add_result(result)
            print(f"🎭 检测到情绪: {result['emotion']} (置信度: {result['confidence']:.3f})")
            
            # 每5个结果显示一次统计
            if len(monitor.emotion_history) % 5 == 0:
                stats = monitor.get_statistics()
                if stats:
                    print(f"📊 主导情绪: {stats['dominant_emotion']}")
                    print(f"📊 平均置信度: {stats['average_confidence']:.3f}")
                    print(f"📊 情绪稳定性: {stats['emotion_stability']:.3f}")
                    print("-" * 30)
        
        recognizer.set_emotion_callback(emotion_callback)
        
        # 开始识别
        recognizer.start_recognition()
        
        print("🎤 请对着麦克风说话...")
        print("按 Enter 键停止识别")
        
        # 等待用户输入
        input()
        
        # 停止识别
        recognizer.stop_recognition()
        
        # 显示最终统计
        final_stats = monitor.get_statistics()
        if final_stats:
            print("\n📈 最终统计结果:")
            print(f"  主导情绪: {final_stats['dominant_emotion']}")
            print(f"  平均置信度: {final_stats['average_confidence']:.3f}")
            print(f"  情绪稳定性: {final_stats['emotion_stability']:.3f}")
            print(f"  总样本数: {final_stats['total_samples']}")
            print("  情绪分布:")
            for emotion, ratio in final_stats['emotion_distribution'].items():
                print(f"    {emotion}: {ratio:.2%}")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")

if __name__ == "__main__":
    demo_real_time_recognition() 