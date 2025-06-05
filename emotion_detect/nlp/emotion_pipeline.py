import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd

from .stt_processor import STTProcessor
from .emotion_classifier import EmotionClassifier

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionAnalysisPipeline:
    """
    完整的情绪分析管道
    技术路线：STT -> Normalization -> NLTK -> Output (Classification)
    """
    
    def __init__(self, 
                 stt_engine: str = 'google',
                 stt_language: str = 'zh-CN',
                 classifier_model: str = 'naive_bayes'):
        """
        初始化情绪分析管道
        
        Args:
            stt_engine: STT引擎类型
            stt_language: STT识别语言
            classifier_model: 分类器模型类型
        """
        self.stt_processor = STTProcessor(engine=stt_engine, language=stt_language)
        self.emotion_classifier = EmotionClassifier(model_type=classifier_model)
        
        # 管道配置
        self.config = {
            'stt_engine': stt_engine,
            'stt_language': stt_language,
            'classifier_model': classifier_model,
            'enable_preprocessing': True,
            'save_intermediate_results': True,
            'output_format': 'detailed'  # 'simple' or 'detailed'
        }
        
        # 结果存储
        self.results_history = []
        
        logger.info("情绪分析管道初始化完成")
    
    def analyze_audio_file(self, audio_file: str, save_results: bool = True) -> Dict:
        """
        分析音频文件的情绪
        
        Args:
            audio_file: 音频文件路径
            save_results: 是否保存结果到历史记录
            
        Returns:
            完整的分析结果
        """
        logger.info(f"开始分析音频文件: {audio_file}")
        
        # 初始化结果结构
        result = {
            'timestamp': datetime.now().isoformat(),
            'audio_file': audio_file,
            'pipeline_config': self.config.copy(),
            'stt_result': {},
            'emotion_analysis': {},
            'success': False,
            'error': None
        }
        
        try:
            # 步骤1: 获取音频信息
            audio_info = self.stt_processor.get_audio_info(audio_file)
            result['audio_info'] = audio_info
            logger.info(f"音频信息: 时长{audio_info.get('duration', 0):.2f}秒")
            
            # 步骤2: STT - 语音转文本
            logger.info("执行STT转录...")
            stt_result = self.stt_processor.transcribe_audio(
                audio_file, 
                preprocess=self.config['enable_preprocessing']
            )
            result['stt_result'] = stt_result
            
            if not stt_result['success']:
                result['error'] = f"STT转录失败: {stt_result['error']}"
                return result
            
            transcribed_text = stt_result['text']
            logger.info(f"STT转录成功: {transcribed_text[:100]}...")
            
            # 步骤3: 文本标准化
            logger.info("执行文本标准化...")
            normalized_text = self.emotion_classifier.normalize_text(transcribed_text)
            result['normalized_text'] = normalized_text
            
            # 步骤4: NLTK情绪分析
            logger.info("执行情绪分析...")
            if self.emotion_classifier.model is None:
                # 如果模型未训练，使用基于规则的分析
                emotion_result = self._rule_based_emotion_analysis(normalized_text)
            else:
                # 使用训练好的模型
                emotion_result = self.emotion_classifier.analyze_text_detailed(normalized_text)
            
            result['emotion_analysis'] = emotion_result
            result['success'] = True
            
            # 生成简化结果
            result['summary'] = self._generate_summary(result)
            
            logger.info(f"情绪分析完成: {emotion_result.get('predicted_emotion', '未知')}")
            
        except Exception as e:
            result['error'] = f"管道处理异常: {str(e)}"
            logger.error(f"管道处理异常: {str(e)}")
        
        # 保存结果
        if save_results:
            self.results_history.append(result)
        
        return result
    
    def analyze_realtime(self, duration: int = 5) -> Dict:
        """
        实时语音情绪分析
        
        Args:
            duration: 录音时长（秒）
            
        Returns:
            分析结果
        """
        logger.info(f"开始实时语音情绪分析，录音时长: {duration}秒")
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'mode': 'realtime',
            'duration': duration,
            'pipeline_config': self.config.copy(),
            'stt_result': {},
            'emotion_analysis': {},
            'success': False,
            'error': None
        }
        
        try:
            # STT实时识别
            stt_result = self.stt_processor.transcribe_realtime(duration)
            result['stt_result'] = stt_result
            
            if not stt_result['success']:
                result['error'] = f"实时STT失败: {stt_result['error']}"
                return result
            
            # 情绪分析
            transcribed_text = stt_result['text']
            normalized_text = self.emotion_classifier.normalize_text(transcribed_text)
            result['normalized_text'] = normalized_text
            
            if self.emotion_classifier.model is None:
                emotion_result = self._rule_based_emotion_analysis(normalized_text)
            else:
                emotion_result = self.emotion_classifier.analyze_text_detailed(normalized_text)
            
            result['emotion_analysis'] = emotion_result
            result['success'] = True
            result['summary'] = self._generate_summary(result)
            
            logger.info(f"实时情绪分析完成: {emotion_result.get('predicted_emotion', '未知')}")
            
        except Exception as e:
            result['error'] = f"实时分析异常: {str(e)}"
            logger.error(f"实时分析异常: {str(e)}")
        
        self.results_history.append(result)
        return result
    
    def batch_analyze(self, audio_files: List[str]) -> List[Dict]:
        """
        批量分析音频文件
        
        Args:
            audio_files: 音频文件路径列表
            
        Returns:
            分析结果列表
        """
        logger.info(f"开始批量分析 {len(audio_files)} 个音频文件")
        
        results = []
        for i, audio_file in enumerate(audio_files):
            logger.info(f"处理第 {i+1}/{len(audio_files)} 个文件")
            result = self.analyze_audio_file(audio_file, save_results=False)
            results.append(result)
        
        # 批量保存结果
        self.results_history.extend(results)
        
        # 生成批量分析报告
        batch_report = self._generate_batch_report(results)
        
        return results, batch_report
    
    def _rule_based_emotion_analysis(self, text: str) -> Dict:
        """
        基于规则的情绪分析（当没有训练模型时使用）
        
        Args:
            text: 输入文本
            
        Returns:
            情绪分析结果
        """
        # 使用VADER和关键词匹配进行基础情绪分析
        features = self.emotion_classifier.extract_features(text)
        vader_scores = features
        
        # 基于VADER分数判断情绪
        compound = vader_scores.get('compound', 0)
        pos = vader_scores.get('pos', 0)
        neg = vader_scores.get('neg', 0)
        neu = vader_scores.get('neu', 0)
        
        # 简单的规则分类
        if compound >= 0.5:
            predicted_emotion = '积极'
            confidence = min(compound, 0.95)
        elif compound <= -0.5:
            predicted_emotion = '消极'
            confidence = min(abs(compound), 0.95)
        elif neg > 0.3:
            predicted_emotion = '愤怒'
            confidence = neg
        else:
            predicted_emotion = '中性'
            confidence = neu
        
        # 关键词分析
        keyword_analysis = {}
        text_lower = text.lower()
        for emotion_type, keywords in self.emotion_classifier.emotion_keywords.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            keyword_analysis[emotion_type] = found_keywords
        
        return {
            'predicted_emotion': predicted_emotion,
            'confidence': float(confidence),
            'probability_distribution': {
                '积极': float(pos),
                '消极': float(neg),
                '中性': float(neu),
                '愤怒': float(neg * 0.8),
                '悲伤': float(neg * 0.6),
                '快乐': float(pos * 0.8),
                '恐惧': float(neg * 0.4),
                '惊讶': float(abs(compound) * 0.3)
            },
            'vader_analysis': vader_scores,
            'keyword_analysis': keyword_analysis,
            'text_features': features,
            'method': 'rule_based'
        }
    
    def _generate_summary(self, result: Dict) -> Dict:
        """生成结果摘要"""
        if not result['success']:
            return {'status': 'failed', 'error': result['error']}
        
        emotion_analysis = result['emotion_analysis']
        stt_result = result['stt_result']
        
        return {
            'status': 'success',
            'transcribed_text': stt_result.get('text', ''),
            'predicted_emotion': emotion_analysis.get('predicted_emotion', '未知'),
            'confidence': emotion_analysis.get('confidence', 0.0),
            'top_emotions': self._get_top_emotions(emotion_analysis.get('probability_distribution', {})),
            'sentiment_score': emotion_analysis.get('vader_analysis', {}).get('compound', 0.0),
            'text_length': len(stt_result.get('text', '')),
            'processing_time': datetime.now().isoformat()
        }
    
    def _get_top_emotions(self, prob_dist: Dict, top_k: int = 3) -> List[Tuple[str, float]]:
        """获取概率最高的前K个情绪"""
        if not prob_dist:
            return []
        
        sorted_emotions = sorted(prob_dist.items(), key=lambda x: x[1], reverse=True)
        return sorted_emotions[:top_k]
    
    def _generate_batch_report(self, results: List[Dict]) -> Dict:
        """生成批量分析报告"""
        total_files = len(results)
        successful_analyses = sum(1 for r in results if r['success'])
        failed_analyses = total_files - successful_analyses
        
        # 统计情绪分布
        emotion_counts = {}
        confidence_scores = []
        
        for result in results:
            if result['success']:
                emotion = result['emotion_analysis'].get('predicted_emotion', '未知')
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                confidence_scores.append(result['emotion_analysis'].get('confidence', 0.0))
        
        return {
            'total_files': total_files,
            'successful_analyses': successful_analyses,
            'failed_analyses': failed_analyses,
            'success_rate': successful_analyses / total_files if total_files > 0 else 0,
            'emotion_distribution': emotion_counts,
            'average_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def train_classifier(self, training_data: List[Dict]):
        """
        训练情绪分类器
        
        Args:
            training_data: 训练数据，格式: [{'text': '文本', 'emotion': '情绪标签'}, ...]
        """
        logger.info(f"开始训练情绪分类器，训练样本数: {len(training_data)}")
        
        # 准备训练数据
        texts = [item['text'] for item in training_data]
        
        # 情绪标签映射
        emotion_to_id = {v: k for k, v in self.emotion_classifier.emotion_labels.items()}
        labels = []
        
        for item in training_data:
            emotion = item['emotion']
            if emotion in emotion_to_id:
                labels.append(emotion_to_id[emotion])
            else:
                labels.append(0)  # 默认为中性
        
        # 训练模型
        training_results = self.emotion_classifier.train(texts, labels)
        
        logger.info("情绪分类器训练完成")
        return training_results
    
    def save_model(self, model_path: str):
        """保存训练好的模型"""
        self.emotion_classifier.save_model(model_path)
    
    def load_model(self, model_path: str):
        """加载训练好的模型"""
        self.emotion_classifier.load_model(model_path)
    
    def export_results(self, output_file: str, format: str = 'json'):
        """
        导出分析结果
        
        Args:
            output_file: 输出文件路径
            format: 输出格式 ('json', 'csv', 'excel')
        """
        if not self.results_history:
            logger.warning("没有分析结果可导出")
            return
        
        if format == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results_history, f, ensure_ascii=False, indent=2)
        
        elif format == 'csv':
            # 转换为DataFrame
            flattened_results = []
            for result in self.results_history:
                if result['success']:
                    flat_result = {
                        'timestamp': result['timestamp'],
                        'audio_file': result.get('audio_file', ''),
                        'transcribed_text': result['stt_result'].get('text', ''),
                        'predicted_emotion': result['emotion_analysis'].get('predicted_emotion', ''),
                        'confidence': result['emotion_analysis'].get('confidence', 0.0),
                        'sentiment_score': result['emotion_analysis'].get('vader_analysis', {}).get('compound', 0.0)
                    }
                    flattened_results.append(flat_result)
            
            df = pd.DataFrame(flattened_results)
            df.to_csv(output_file, index=False, encoding='utf-8')
        
        elif format == 'excel':
            flattened_results = []
            for result in self.results_history:
                if result['success']:
                    flat_result = {
                        'timestamp': result['timestamp'],
                        'audio_file': result.get('audio_file', ''),
                        'transcribed_text': result['stt_result'].get('text', ''),
                        'predicted_emotion': result['emotion_analysis'].get('predicted_emotion', ''),
                        'confidence': result['emotion_analysis'].get('confidence', 0.0),
                        'sentiment_score': result['emotion_analysis'].get('vader_analysis', {}).get('compound', 0.0)
                    }
                    flattened_results.append(flat_result)
            
            df = pd.DataFrame(flattened_results)
            df.to_excel(output_file, index=False)
        
        logger.info(f"结果已导出到: {output_file}")
    
    def get_statistics(self) -> Dict:
        """获取分析统计信息"""
        if not self.results_history:
            return {'message': '暂无分析结果'}
        
        successful_results = [r for r in self.results_history if r['success']]
        
        if not successful_results:
            return {'message': '暂无成功的分析结果'}
        
        # 情绪分布统计
        emotion_counts = {}
        confidence_scores = []
        
        for result in successful_results:
            emotion = result['emotion_analysis'].get('predicted_emotion', '未知')
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            confidence_scores.append(result['emotion_analysis'].get('confidence', 0.0))
        
        return {
            'total_analyses': len(self.results_history),
            'successful_analyses': len(successful_results),
            'success_rate': len(successful_results) / len(self.results_history),
            'emotion_distribution': emotion_counts,
            'average_confidence': sum(confidence_scores) / len(confidence_scores),
            'confidence_range': {
                'min': min(confidence_scores),
                'max': max(confidence_scores)
            }
        }
    
    def configure_pipeline(self, **kwargs):
        """配置管道参数"""
        self.config.update(kwargs)
        logger.info(f"管道配置已更新: {kwargs}")
    
    def reset_history(self):
        """清空历史记录"""
        self.results_history.clear()
        logger.info("历史记录已清空") 