import nltk
import re
import string
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer


class EmotionClassifier:
    """
    基于NLTK的情绪分类器
    支持多种情绪类别：积极、消极、中性、愤怒、悲伤、快乐、恐惧、惊讶
    """
    
    def __init__(self, model_type: str = 'naive_bayes'):
        """
        初始化情绪分类器
        
        Args:
            model_type: 模型类型 ('naive_bayes', 'logistic', 'svm', 'random_forest')
        """
        self.model_type = model_type
        self.model = None
        self.vectorizer = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.sia = SentimentIntensityAnalyzer()
        
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
            '积极': ['好', '棒', '优秀', '完美', '喜欢', '爱', '开心', '高兴', '满意', 'good', 'great', 'excellent', 'perfect', 'love', 'like', 'happy', 'satisfied'],
            '消极': ['坏', '糟糕', '讨厌', '不好', '失望', '难过', 'bad', 'terrible', 'hate', 'disappointed', 'sad'],
            '愤怒': ['生气', '愤怒', '气愤', '恼火', '愤慨', 'angry', 'furious', 'mad', 'rage', 'annoyed'],
            '悲伤': ['伤心', '难过', '悲伤', '沮丧', '痛苦', 'sad', 'depressed', 'sorrow', 'grief', 'miserable'],
            '快乐': ['快乐', '开心', '高兴', '兴奋', '愉快', 'happy', 'joyful', 'excited', 'cheerful', 'delighted'],
            '恐惧': ['害怕', '恐惧', '担心', '焦虑', '紧张', 'afraid', 'scared', 'fear', 'anxious', 'worried'],
            '惊讶': ['惊讶', '震惊', '意外', '吃惊', 'surprised', 'shocked', 'amazed', 'astonished', 'unexpected']
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化机器学习模型"""
        if self.model_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=1.0)
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == 'svm':
            self.model = SVC(kernel='linear', random_state=42, probability=True)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        # 初始化TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
    
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
        
        # 移除特殊字符，保留字母、数字和空格
        text = re.sub(r'[^a-zA-Z0-9\s\u4e00-\u9fff]', ' ', text)
        
        # 移除多余的空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        文本预处理：分词、去停用词、词形还原
        
        Args:
            text: 输入文本
            
        Returns:
            处理后的词汇列表
        """
        # 标准化文本
        text = self.normalize_text(text)
        
        if not text:
            return []
        
        # 分词
        tokens = word_tokenize(text)
        
        # 过滤停用词和标点符号
        tokens = [token for token in tokens 
                 if token not in self.stop_words 
                 and token not in string.punctuation
                 and len(token) > 1]
        
        # 词形还原
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
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
        features['sentence_count'] = len(sent_tokenize(text))
        
        # 情绪词汇特征
        text_lower = text.lower()
        for emotion, keywords in self.emotion_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            features[f'{emotion}_keywords'] = count
        
        # VADER情感分析特征
        vader_scores = self.sia.polarity_scores(text)
        features.update(vader_scores)
        
        # 标点符号特征
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        return features
    
    def train(self, texts: List[str], labels: List[int], test_size: float = 0.2):
        """
        训练情绪分类模型
        
        Args:
            texts: 训练文本列表
            labels: 对应的标签列表
            test_size: 测试集比例
        """
        if len(texts) != len(labels):
            raise ValueError("文本和标签数量不匹配")
        
        # 预处理文本
        processed_texts = []
        for text in texts:
            tokens = self.preprocess_text(text)
            processed_texts.append(' '.join(tokens))
        
        # 向量化
        X = self.vectorizer.fit_transform(processed_texts)
        y = np.array(labels)
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 评估模型
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"训练集准确率: {train_score:.4f}")
        print(f"测试集准确率: {test_score:.4f}")
        
        # 交叉验证
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        print(f"交叉验证平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # 预测测试集
        y_pred = self.model.predict(X_test)
        
        # 打印分类报告
        print("\n分类报告:")
        target_names = [self.emotion_labels[i] for i in sorted(set(labels))]
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'cv_scores': cv_scores,
            'classification_report': classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        }
    
    def predict(self, text: str) -> Tuple[str, float, Dict]:
        """
        预测单个文本的情绪
        
        Args:
            text: 输入文本
            
        Returns:
            (预测的情绪标签, 置信度, 所有类别的概率分布)
        """
        if not self.model or not self.vectorizer:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        # 预处理文本
        tokens = self.preprocess_text(text)
        processed_text = ' '.join(tokens)
        
        # 向量化
        X = self.vectorizer.transform([processed_text])
        
        # 预测
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # 获取所有类别的概率
        prob_dict = {}
        for i, prob in enumerate(probabilities):
            if i in self.emotion_labels:
                prob_dict[self.emotion_labels[i]] = float(prob)
        
        emotion_label = self.emotion_labels.get(prediction, '未知')
        confidence = float(max(probabilities))
        
        return emotion_label, confidence, prob_dict
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float, Dict]]:
        """
        批量预测文本情绪
        
        Args:
            texts: 文本列表
            
        Returns:
            预测结果列表
        """
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
    
    def save_model(self, filepath: str):
        """
        保存训练好的模型
        
        Args:
            filepath: 保存路径
        """
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'model_type': self.model_type,
            'emotion_labels': self.emotion_labels
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """
        加载训练好的模型
        
        Args:
            filepath: 模型文件路径
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.model_type = model_data['model_type']
        self.emotion_labels = model_data['emotion_labels']
        
        print(f"模型已从 {filepath} 加载")
    
    def analyze_text_detailed(self, text: str) -> Dict:
        """
        详细分析文本情绪
        
        Args:
            text: 输入文本
            
        Returns:
            详细分析结果
        """
        # 基本预测
        emotion, confidence, prob_dist = self.predict(text)
        
        # 提取特征
        features = self.extract_features(text)
        
        # VADER分析
        vader_scores = self.sia.polarity_scores(text)
        
        # 关键词分析
        keyword_analysis = {}
        text_lower = text.lower()
        for emotion_type, keywords in self.emotion_keywords.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            keyword_analysis[emotion_type] = found_keywords
        
        return {
            'predicted_emotion': emotion,
            'confidence': confidence,
            'probability_distribution': prob_dist,
            'vader_analysis': vader_scores,
            'keyword_analysis': keyword_analysis,
            'text_features': features,
            'normalized_text': self.normalize_text(text)
        } 