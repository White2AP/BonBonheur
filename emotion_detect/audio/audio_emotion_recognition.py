#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
音频情绪识别系统
基于MFCC特征和机器学习模型进行音频情绪分类
支持的情绪：angry, happy, neutral, sad
"""

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

class AudioEmotionRecognizer:
    """音频情绪识别器"""
    
    def __init__(self, dataset_path=None):
        """
        初始化音频情绪识别器
        
        Args:
            dataset_path: 数据集路径（可选，仅训练时需要）
        """
        self.dataset_path = dataset_path or "Dataset/RAVDESS_1s_4categories"
        self.emotions = ['angry', 'happy', 'neutral', 'sad']
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        
        # MFCC参数
        self.n_mfcc = 13
        self.n_fft = 2048
        self.hop_length = 512
        self.sample_rate = 22050
        
        # 语音转文本（简单实现）
        self.speech_to_text_enabled = False
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.speech_to_text_enabled = True
        except ImportError:
            self.recognizer = None
    
    def extract_mfcc_features(self, audio_path, duration=None):
        """
        提取MFCC特征
        
        Args:
            audio_path: 音频文件路径
            duration: 音频时长（秒），None表示使用完整音频
            
        Returns:
            numpy.ndarray: MFCC特征向量
        """
        try:
            # 加载音频文件
            y, sr = librosa.load(audio_path, duration=duration, sr=self.sample_rate)
            
            # 如果音频太短，进行填充
            if len(y) < self.sample_rate * 0.5:  # 少于0.5秒
                y = np.pad(y, (0, int(self.sample_rate * 0.5) - len(y)), mode='constant')
            
            # 提取MFCC特征
            mfccs = librosa.feature.mfcc(
                y=y, 
                sr=sr, 
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # 计算统计特征
            mfcc_features = []
            
            # 对每个MFCC系数计算统计量
            for i in range(self.n_mfcc):
                mfcc_coeff = mfccs[i]
                mfcc_features.extend([
                    np.mean(mfcc_coeff),      # 均值
                    np.std(mfcc_coeff),       # 标准差
                    np.max(mfcc_coeff),       # 最大值
                    np.min(mfcc_coeff),       # 最小值
                    np.median(mfcc_coeff),    # 中位数
                ])
            
            # 添加其他音频特征
            # 零交叉率
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            mfcc_features.extend([
                np.mean(zcr),
                np.std(zcr)
            ])
            
            # 谱质心
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            mfcc_features.extend([
                np.mean(spectral_centroids),
                np.std(spectral_centroids)
            ])
            
            # 谱带宽
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            mfcc_features.extend([
                np.mean(spectral_bandwidth),
                np.std(spectral_bandwidth)
            ])
            
            # 谱滚降
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            mfcc_features.extend([
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff)
            ])
            
            # RMS能量
            rms = librosa.feature.rms(y=y)[0]
            mfcc_features.extend([
                np.mean(rms),
                np.std(rms)
            ])
            
            return np.array(mfcc_features)
            
        except Exception as e:
            print(f"提取特征时出错 {audio_path}: {e}")
            return None
    
    def speech_to_text(self, audio_path):
        """
        语音转文本
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            str: 识别的文本，失败返回空字符串
        """
        if not self.speech_to_text_enabled:
            return ""
        
        try:
            import speech_recognition as sr
            
            # 使用speech_recognition库
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio, language='zh-CN')
                return text
        except Exception as e:
            # 语音转文本失败，返回空字符串
            return ""
    
    def load_dataset(self):
        """
        加载数据集并提取特征
        
        Returns:
            tuple: (特征矩阵, 标签数组)
        """
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            raise ValueError(f"数据集路径不存在: {self.dataset_path}")
            
        print("🎵 开始加载数据集并提取特征...")
        
        features = []
        labels = []
        file_paths = []
        
        for emotion in self.emotions:
            emotion_path = os.path.join(self.dataset_path, emotion)
            if not os.path.exists(emotion_path):
                print(f"⚠️ 警告: 找不到情绪文件夹 {emotion_path}")
                continue
                
            print(f"📁 处理 {emotion} 情绪...")
            emotion_files = [f for f in os.listdir(emotion_path) if f.endswith(('.wav', '.mp3'))]
            
            for i, file_name in enumerate(emotion_files):
                file_path = os.path.join(emotion_path, file_name)
                
                # 提取特征
                feature_vector = self.extract_mfcc_features(file_path)
                
                if feature_vector is not None:
                    features.append(feature_vector)
                    labels.append(emotion)
                    file_paths.append(file_path)
                
                # 显示进度
                if (i + 1) % 50 == 0:
                    print(f"  已处理 {i + 1}/{len(emotion_files)} 个文件")
            
            print(f"  ✅ {emotion} 完成，共 {len([l for l in labels if l == emotion])} 个样本")
        
        if not features:
            raise ValueError("没有成功提取到任何特征，请检查数据集路径和文件格式")
        
        features = np.array(features)
        labels = np.array(labels)
        
        print(f"\n📊 数据集加载完成:")
        print(f"  总样本数: {len(features)}")
        print(f"  特征维度: {features.shape[1]}")
        print(f"  情绪分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        return features, labels, file_paths
    
    def prepare_data(self, features, labels, test_size=0.2, random_state=42):
        """
        准备训练和测试数据
        
        Args:
            features: 特征矩阵
            labels: 标签数组
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            tuple: 训练和测试数据
        """
        print("🔄 准备训练和测试数据...")
        
        # 标签编码
        self.label_encoder = LabelEncoder()
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels_encoded, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels_encoded
        )
        
        # 特征标准化
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 保存特征列名
        self.feature_columns = [f'feature_{i}' for i in range(features.shape[1])]
        
        print(f"  训练集: {X_train_scaled.shape[0]} 样本")
        print(f"  测试集: {X_test_scaled.shape[0]} 样本")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """
        训练多个模型并选择最佳模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            
        Returns:
            dict: 训练结果
        """
        print("🤖 开始训练模型...")
        
        # 定义模型
        models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'MLP': MLPClassifier(random_state=42, max_iter=1000)
        }
        
        # 定义参数网格
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            },
            'GradientBoosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            },
            'MLP': {
                'hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
                'alpha': [0.0001, 0.001],
                'learning_rate': ['constant', 'adaptive']
            }
        }
        
        results = {}
        best_score = 0
        best_model_name = None
        
        for model_name, model in models.items():
            print(f"\n🔧 训练 {model_name}...")
            
            # 网格搜索
            grid_search = GridSearchCV(
                model, 
                param_grids[model_name],
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            # 交叉验证评分
            cv_scores = cross_val_score(
                grid_search.best_estimator_, 
                X_train, y_train, 
                cv=5, 
                scoring='accuracy'
            )
            
            results[model_name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"  最佳参数: {grid_search.best_params_}")
            print(f"  交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # 更新最佳模型
            if cv_scores.mean() > best_score:
                best_score = cv_scores.mean()
                best_model_name = model_name
        
        # 选择最佳模型
        self.model = results[best_model_name]['model']
        print(f"\n🏆 最佳模型: {best_model_name} (准确率: {best_score:.4f})")
        
        return results
    
    def evaluate_model(self, X_test, y_test):
        """
        评估模型性能
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
            
        Returns:
            dict: 评估结果
        """
        print("\n📊 评估模型性能...")
        
        # 预测
        y_pred = self.model.predict(X_test)
        
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        
        # 分类报告
        emotion_names = self.label_encoder.classes_
        report = classification_report(
            y_test, y_pred, 
            target_names=emotion_names,
            output_dict=True
        )
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"测试集准确率: {accuracy:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=emotion_names))
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'emotion_names': emotion_names
        }
    
    def plot_confusion_matrix(self, cm, emotion_names, save_path=None):
        """
        绘制混淆矩阵
        
        Args:
            cm: 混淆矩阵
            emotion_names: 情绪名称
            save_path: 保存路径
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=emotion_names,
            yticklabels=emotion_names
        )
        plt.title('音频情绪识别混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存到: {save_path}")
        
        plt.show()
    
    def predict_emotion(self, audio_path):
        """
        预测单个音频文件的情绪
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            dict: 预测结果，包含emotion, confidence, transcription等字段
        """
        if self.model is None:
            raise ValueError("模型尚未加载，请先调用 load_model() 方法")
        
        # 提取特征
        features = self.extract_mfcc_features(audio_path)
        if features is None:
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'transcription': '',
                'error': '特征提取失败'
            }
        
        try:
            # 标准化
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # 预测
            prediction = self.model.predict(features_scaled)[0]
            
            # 获取概率（如果模型支持）
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)[0]
                confidence = float(max(probabilities))
            else:
                # 如果模型不支持概率预测，使用默认置信度
                confidence = 0.8
            
            # 解码标签
            emotion = self.label_encoder.inverse_transform([prediction])[0]
            
            # 语音转文本
            transcription = self.speech_to_text(audio_path)
            
            # 构建结果
            result = {
                'emotion': emotion,
                'confidence': confidence,
                'transcription': transcription,
                'raw_emotion': emotion,
                'method': 'audio'
            }
            
            return result
            
        except Exception as e:
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'transcription': '',
                'error': f'预测失败: {str(e)}'
            }
    
    def save_model(self, model_path="models/audio_emotion_model.pkl"):
        """
        保存训练好的模型
        
        Args:
            model_path: 模型保存路径
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'emotions': self.emotions,
            'mfcc_params': {
                'n_mfcc': self.n_mfcc,
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'sample_rate': self.sample_rate
            }
        }
        
        joblib.dump(model_data, model_path)
        print(f"✅ 模型已保存到: {model_path}")
    
    def load_model(self, model_path="models/audio_emotion_model.pkl"):
        """
        加载训练好的模型
        
        Args:
            model_path: 模型文件路径
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        self.emotions = model_data['emotions']
        
        # 加载MFCC参数
        mfcc_params = model_data['mfcc_params']
        self.n_mfcc = mfcc_params['n_mfcc']
        self.n_fft = mfcc_params['n_fft']
        self.hop_length = mfcc_params['hop_length']
        self.sample_rate = mfcc_params['sample_rate']
        
        print(f"✅ 模型已从 {model_path} 加载")
    
    def train(self, test_size=0.2, save_model=True):
        """
        完整的训练流程
        
        Args:
            test_size: 测试集比例
            save_model: 是否保存模型
            
        Returns:
            dict: 训练和评估结果
        """
        print("🚀 开始音频情绪识别模型训练...")
        
        # 1. 加载数据集
        features, labels, file_paths = self.load_dataset()
        
        # 2. 准备数据
        X_train, X_test, y_train, y_test = self.prepare_data(
            features, labels, test_size=test_size
        )
        
        # 3. 训练模型
        training_results = self.train_models(X_train, y_train)
        
        # 4. 评估模型
        evaluation_results = self.evaluate_model(X_test, y_test)
        
        # 5. 绘制混淆矩阵
        self.plot_confusion_matrix(
            evaluation_results['confusion_matrix'],
            evaluation_results['emotion_names'],
            save_path="models/confusion_matrix.png"
        )
        
        # 6. 保存模型
        if save_model:
            self.save_model()
        
        print("\n🎉 训练完成！")
        
        return {
            'training_results': training_results,
            'evaluation_results': evaluation_results
        }

def main():
    """主函数 - 演示完整的训练和预测流程"""
    print("🎵 音频情绪识别系统")
    print("=" * 50)
    
    # 创建识别器
    recognizer = AudioEmotionRecognizer()
    
    try:
        # 训练模型
        results = recognizer.train()
        
        # 显示结果摘要
        accuracy = results['evaluation_results']['accuracy']
        print(f"\n📈 最终模型准确率: {accuracy:.4f}")
        
        # 演示预测功能
        print("\n🔮 预测演示:")
        dataset_path = recognizer.dataset_path
        
        # 从每个情绪类别中选择一个文件进行预测演示
        for emotion in recognizer.emotions:
            emotion_path = os.path.join(dataset_path, emotion)
            if os.path.exists(emotion_path):
                files = [f for f in os.listdir(emotion_path) if f.endswith(('.wav', '.mp3'))]
                if files:
                    test_file = os.path.join(emotion_path, files[0])
                    result = recognizer.predict_emotion(test_file)
                    if result:
                        print(f"  文件: {files[0]}")
                        print(f"  真实情绪: {emotion}")
                        print(f"  预测情绪: {result['emotion']}")
                        print(f"  置信度: {result['confidence']:.4f}")
                        print()
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        print("请检查数据集路径和文件格式")

if __name__ == "__main__":
    main() 