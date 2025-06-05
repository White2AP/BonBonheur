#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版情绪分类器
不依赖语音识别库，专注于文本情绪分析
"""

import re
import string
from typing import Dict, List, Tuple
from collections import Counter


class SimpleEmotionClassifier:
    """
    简化版情绪分类器
    基于规则和关键词匹配的情绪分析
    """
    
    def __init__(self):
        """初始化情绪分类器"""
        
        # 情绪标签映射
        self.emotion_labels = {
            0: '中性',
            1: '积极', 
            2: '消极',
            3: '愤怒',
            4: '悲伤',
            5: '快乐',
            6: '恐惧',
            7: '惊讶'
        }
        
        # 情绪关键词词典
        self.emotion_keywords = {
            '积极': [
                '好', '棒', '优秀', '完美', '喜欢', '爱', '开心', '高兴', '满意', '赞', '不错', '很棒',
                'good', 'great', 'excellent', 'perfect', 'love', 'like', 'happy', 'satisfied', 'awesome'
            ],
            '消极': [
                '坏', '糟糕', '讨厌', '不好', '失望', '难过', '差', '烂', '垃圾',
                'bad', 'terrible', 'hate', 'disappointed', 'sad', 'awful', 'horrible'
            ],
            '愤怒': [
                '生气', '愤怒', '气愤', '恼火', '愤慨', '气死', '烦死', '讨厌死',
                'angry', 'furious', 'mad', 'rage', 'annoyed', 'pissed', 'irritated'
            ],
            '悲伤': [
                '伤心', '难过', '悲伤', '沮丧', '痛苦', '心痛', '难受', '郁闷',
                'sad', 'depressed', 'sorrow', 'grief', 'miserable', 'heartbroken'
            ],
            '快乐': [
                '快乐', '开心', '高兴', '兴奋', '愉快', '欢乐', '喜悦', '乐',
                'happy', 'joyful', 'excited', 'cheerful', 'delighted', 'thrilled'
            ],
            '恐惧': [
                '害怕', '恐惧', '担心', '焦虑', '紧张', '不安', '恐慌', '怕',
                'afraid', 'scared', 'fear', 'anxious', 'worried', 'nervous', 'panic'
            ],
            '惊讶': [
                '惊讶', '震惊', '意外', '吃惊', '惊奇', '没想到', '想不到',
                'surprised', 'shocked', 'amazed', 'astonished', 'unexpected', 'wow'
            ]
        }
        
        # 情绪强度词
        self.intensity_words = {
            'high': ['非常', '极其', '特别', '超级', '太', '很', 'very', 'extremely', 'super', 'really'],
            'medium': ['比较', '还', '挺', '蛮', 'quite', 'pretty', 'rather'],
            'low': ['有点', '稍微', '略', 'a bit', 'slightly', 'somewhat']
        }
        
        # 否定词
        self.negation_words = ['不', '没', '无', '非', '未', '别', '勿', 'not', 'no', 'never', 'none']
        
        # 标点符号情绪权重
        self.punctuation_weights = {
            '!': 1.2,
            '！': 1.2,
            '?': 0.8,
            '？': 0.8,
            '...': 0.9,
            '。': 1.0
        }
    
    def normalize_text(self, text: str) -> str:
        """
        文本标准化处理
        
        Args:
            text: 输入文本
            
        Returns:
            标准化后的文本
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 转换为小写
        text = text.lower()
        
        # 移除URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # 移除邮箱
        text = re.sub(r'\S+@\S+', '', text)
        
        # 移除HTML标签
        text = re.sub(r'<.*?>', '', text)
        
        # 保留中文、英文、数字和基本标点
        text = re.sub(r'[^\w\s\u4e00-\u9fff!！?？.。,，;；:：]', ' ', text)
        
        # 移除多余的空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features(self, text: str) -> Dict:
        """
        提取文本特征
        
        Args:
            text: 输入文本
            
        Returns:
            特征字典
        """
        features = {}
        
        # 基本统计特征
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len([s for s in re.split(r'[.。!！?？]', text) if s.strip()])
        
        # 情绪词汇特征
        text_lower = text.lower()
        for emotion, keywords in self.emotion_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            features[f'{emotion}_keywords'] = count
        
        # 强度词特征
        for intensity, words in self.intensity_words.items():
            count = sum(1 for word in words if word in text_lower)
            features[f'{intensity}_intensity'] = count
        
        # 否定词特征
        negation_count = sum(1 for word in self.negation_words if word in text_lower)
        features['negation_count'] = negation_count
        
        # 标点符号特征
        features['exclamation_count'] = text.count('!') + text.count('！')
        features['question_count'] = text.count('?') + text.count('？')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        return features
    
    def calculate_emotion_scores(self, text: str) -> Dict[str, float]:
        """
        计算各情绪的得分
        
        Args:
            text: 输入文本
            
        Returns:
            情绪得分字典
        """
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            score = 0.0
            
            # 基础关键词匹配得分
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1.0
                    
                    # 检查强度词修饰
                    for intensity, intensity_words in self.intensity_words.items():
                        for intensity_word in intensity_words:
                            if intensity_word in text_lower:
                                if intensity == 'high':
                                    score += 0.5
                                elif intensity == 'medium':
                                    score += 0.3
                                elif intensity == 'low':
                                    score += 0.1
            
            # 检查否定词影响
            negation_count = sum(1 for word in self.negation_words if word in text_lower)
            if negation_count > 0:
                score *= (1 - 0.3 * negation_count)  # 否定词降低得分
            
            # 标点符号影响
            for punct, weight in self.punctuation_weights.items():
                if punct in text:
                    score *= weight
            
            emotion_scores[emotion] = max(0.0, score)  # 确保得分非负
        
        return emotion_scores
    
    def predict_emotion(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        预测文本情绪
        
        Args:
            text: 输入文本
            
        Returns:
            (预测的情绪标签, 置信度, 所有情绪得分)
        """
        # 标准化文本
        normalized_text = self.normalize_text(text)
        
        if not normalized_text:
            return '中性', 0.0, {'中性': 1.0}
        
        # 计算情绪得分
        emotion_scores = self.calculate_emotion_scores(normalized_text)
        
        # 添加中性得分（基于文本长度和情绪词密度）
        total_emotion_score = sum(emotion_scores.values())
        word_count = len(normalized_text.split())
        
        if total_emotion_score == 0 or word_count == 0:
            neutral_score = 1.0
        else:
            emotion_density = total_emotion_score / word_count
            neutral_score = max(0.1, 1.0 - emotion_density)
        
        emotion_scores['中性'] = neutral_score
        
        # 归一化得分
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            normalized_scores = {k: v/total_score for k, v in emotion_scores.items()}
        else:
            normalized_scores = {'中性': 1.0}
        
        # 找到最高得分的情绪
        predicted_emotion = max(normalized_scores, key=normalized_scores.get)
        confidence = normalized_scores[predicted_emotion]
        
        return predicted_emotion, confidence, normalized_scores
    
    def analyze_text_detailed(self, text: str) -> Dict:
        """
        详细分析文本情绪
        
        Args:
            text: 输入文本
            
        Returns:
            详细分析结果
        """
        # 基本预测
        emotion, confidence, emotion_scores = self.predict_emotion(text)
        
        # 提取特征
        features = self.extract_features(text)
        
        # 关键词分析
        keyword_analysis = {}
        text_lower = text.lower()
        for emotion_type, keywords in self.emotion_keywords.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            keyword_analysis[emotion_type] = found_keywords
        
        return {
            'predicted_emotion': emotion,
            'confidence': confidence,
            'emotion_scores': emotion_scores,
            'keyword_analysis': keyword_analysis,
            'text_features': features,
            'normalized_text': self.normalize_text(text),
            'method': 'rule_based_simple'
        }
    
    def batch_predict(self, texts: List[str]) -> List[Dict]:
        """
        批量预测文本情绪
        
        Args:
            texts: 文本列表
            
        Returns:
            预测结果列表
        """
        results = []
        for text in texts:
            result = self.analyze_text_detailed(text)
            results.append(result)
        return results


def main():
    """演示简化版情绪分类器"""
    print("简化版情绪分类器演示")
    print("=" * 50)
    
    classifier = SimpleEmotionClassifier()
    
    # 测试文本
    test_texts = [
        "今天天气真好，我很开心！",
        "这个产品质量太差了，我很生气。",
        "我对考试结果感到失望和难过。",
        "哇，这个消息太令人震惊了！",
        "我对明天的面试感到很紧张和担心。",
        "这部电影还不错，没什么特别的感觉。",
        "我爱这个地方，这里让我感到非常快乐！",
        "这种情况让我感到恐惧和不安。"
    ]
    
    print("正在分析以下文本的情绪：")
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. {text}")
        
        result = classifier.analyze_text_detailed(text)
        
        print(f"   预测情绪: {result['predicted_emotion']}")
        print(f"   置信度: {result['confidence']:.3f}")
        
        # 显示前3个最可能的情绪
        top_emotions = sorted(result['emotion_scores'].items(), 
                            key=lambda x: x[1], reverse=True)[:3]
        print(f"   前3个情绪: {', '.join([f'{e}({s:.3f})' for e, s in top_emotions])}")
        
        # 显示找到的关键词
        found_keywords = []
        for emotion_type, keywords in result['keyword_analysis'].items():
            if keywords:
                found_keywords.extend(keywords)
        if found_keywords:
            print(f"   关键词: {', '.join(found_keywords[:5])}")
    
    print("\n" + "=" * 50)
    print("演示完成！")


if __name__ == "__main__":
    main() 