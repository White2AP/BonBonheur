#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练数据生成器
用于生成情绪分类的训练样本数据
"""

import random
import json
from typing import List, Dict
from pathlib import Path


class TrainingDataGenerator:
    """训练数据生成器"""
    
    def __init__(self):
        """初始化训练数据生成器"""
        
        # 情绪类别和对应的样本模板
        self.emotion_templates = {
            '积极': [
                "今天{event}，我感到非常{positive_adj}！",
                "这{thing}真的很{positive_adj}，让我很{positive_feeling}。",
                "我对{event}感到很{positive_feeling}和满意。",
                "{positive_adj}的{thing}让我心情{positive_feeling}。",
                "太{positive_adj}了！这让我感到{positive_feeling}。",
                "我{positive_verb}这个{thing}，它让我{positive_feeling}。",
                "这是一个{positive_adj}的{event}，我很{positive_feeling}。",
                "我为{event}感到{positive_feeling}和自豪。"
            ],
            
            '消极': [
                "这{thing}让我感到很{negative_adj}。",
                "我对{event}感到{negative_feeling}和失望。",
                "{negative_adj}的{thing}让我心情{negative_feeling}。",
                "这种{situation}真的很{negative_adj}。",
                "我{negative_verb}这个{thing}，它让我{negative_feeling}。",
                "太{negative_adj}了，这让我感到{negative_feeling}。",
                "我对{event}的结果感到{negative_feeling}。",
                "这{thing}的质量{negative_adj}，让我很{negative_feeling}。"
            ],
            
            '愤怒': [
                "这{thing}让我非常{angry_adj}！",
                "我对{event}感到极其{angry_feeling}！",
                "太{angry_adj}了！我{angry_verb}这种{situation}！",
                "这种{behavior}让我{angry_feeling}得要命！",
                "我{angry_verb}这个{thing}，它让我{angry_feeling}！",
                "这{situation}真是{angry_adj}，我很{angry_feeling}！",
                "我对{event}感到{angry_feeling}和不满！",
                "这种{behavior}让我{angry_feeling}，太{angry_adj}了！"
            ],
            
            '悲伤': [
                "我对{event}感到很{sad_feeling}。",
                "这{thing}让我感到{sad_adj}和{sad_feeling}。",
                "我为{event}感到{sad_feeling}，心情很{sad_adj}。",
                "这种{situation}让我{sad_feeling}，我感到很{sad_adj}。",
                "我{sad_verb}这个{thing}，它让我{sad_feeling}。",
                "太{sad_adj}了，这让我感到{sad_feeling}。",
                "我对{event}的结果感到{sad_feeling}和沮丧。",
                "这{thing}让我心情{sad_adj}，感到{sad_feeling}。"
            ],
            
            '快乐': [
                "我今天感到非常{happy_adj}！",
                "这{thing}让我{happy_feeling}得不得了！",
                "我对{event}感到{happy_feeling}和兴奋！",
                "太{happy_adj}了！我{happy_verb}这个{thing}！",
                "这种{situation}让我感到{happy_feeling}！",
                "我{happy_verb}这{thing}，它让我{happy_feeling}！",
                "这是{happy_adj}的一天，我很{happy_feeling}！",
                "我为{event}感到{happy_feeling}和高兴！"
            ],
            
            '恐惧': [
                "我对{event}感到{fear_feeling}和不安。",
                "这{thing}让我感到{fear_adj}。",
                "我{fear_verb}这种{situation}，它让我{fear_feeling}。",
                "这{situation}真的很{fear_adj}，让我{fear_feeling}。",
                "我对{event}感到{fear_feeling}和紧张。",
                "这{thing}让我{fear_feeling}，我很{fear_adj}。",
                "我{fear_verb}会发生{event}，这让我{fear_feeling}。",
                "这种{situation}让我感到{fear_feeling}和焦虑。"
            ],
            
            '惊讶': [
                "哇！这{thing}真是太{surprise_adj}了！",
                "我对{event}感到非常{surprise_feeling}！",
                "这{thing}让我{surprise_feeling}，太{surprise_adj}了！",
                "真是{surprise_adj}！我没想到{event}！",
                "这个{thing}让我感到{surprise_feeling}和震惊！",
                "太{surprise_adj}了！这{event}真是出乎意料！",
                "我对{event}感到{surprise_feeling}，这真是{surprise_adj}！",
                "这{situation}让我{surprise_feeling}，完全没想到！"
            ],
            
            '中性': [
                "这{thing}还可以，没什么特别的。",
                "我对{event}没有特别的感觉。",
                "这{thing}很普通，一般般。",
                "我觉得{event}还行，不好不坏。",
                "这{situation}比较平常，没什么特殊的。",
                "我对{thing}的看法比较中立。",
                "这{event}很正常，没什么特别的。",
                "我觉得{thing}还可以，不算好也不算坏。"
            ]
        }
        
        # 词汇库
        self.vocabulary = {
            'positive_adj': ['棒', '好', '优秀', '完美', '出色', '精彩', '美好', '令人满意'],
            'positive_feeling': ['开心', '高兴', '愉快', '满意', '兴奋', '快乐', '欣喜', '舒心'],
            'positive_verb': ['喜欢', '爱', '欣赏', '享受', '赞美', '称赞'],
            
            'negative_adj': ['糟糕', '差', '不好', '令人失望', '糟糕透了', '很差'],
            'negative_feeling': ['失望', '难过', '不满', '郁闷', '沮丧', '不开心'],
            'negative_verb': ['讨厌', '不喜欢', '厌恶', '反感'],
            
            'angry_adj': ['气人', '可恶', '讨厌', '愤怒', '恼人', '令人愤慨'],
            'angry_feeling': ['生气', '愤怒', '气愤', '恼火', '愤慨', '暴怒'],
            'angry_verb': ['讨厌', '痛恨', '厌恶', '反感'],
            
            'sad_adj': ['难过', '悲伤', '沮丧', '忧郁', '伤心', '痛苦'],
            'sad_feeling': ['难过', '悲伤', '伤心', '痛苦', '忧伤', '沮丧'],
            'sad_verb': ['为...难过', '为...伤心', '为...痛苦'],
            
            'happy_adj': ['开心', '快乐', '愉快', '高兴', '兴奋', '欢乐'],
            'happy_feeling': ['开心', '快乐', '高兴', '愉快', '兴奋', '欢乐'],
            'happy_verb': ['喜欢', '爱', '享受', '欣赏'],
            
            'fear_adj': ['害怕', '恐惧', '紧张', '担心', '焦虑', '不安'],
            'fear_feeling': ['害怕', '恐惧', '担心', '紧张', '焦虑', '不安'],
            'fear_verb': ['害怕', '担心', '恐惧', '忧虑'],
            
            'surprise_adj': ['惊讶', '震惊', '意外', '出乎意料', '令人惊讶', '不可思议'],
            'surprise_feeling': ['惊讶', '震惊', '意外', '吃惊', '惊奇'],
            
            'event': ['考试', '面试', '会议', '约会', '旅行', '工作', '学习', '比赛', '表演', '聚会'],
            'thing': ['电影', '书', '产品', '服务', '食物', '音乐', '游戏', '课程', '活动', '经历'],
            'situation': ['情况', '状况', '环境', '氛围', '场面', '局面'],
            'behavior': ['行为', '做法', '态度', '表现', '举动']
        }
        
        # 英文样本模板
        self.english_templates = {
            '积极': [
                "I feel {positive_adj} about {event}!",
                "This {thing} is {positive_adj} and makes me {positive_feeling}.",
                "I'm {positive_feeling} with {event}.",
                "What a {positive_adj} {thing}! I feel {positive_feeling}.",
                "I {positive_verb} this {thing}, it's {positive_adj}!",
                "This {event} makes me feel {positive_feeling}.",
                "I'm so {positive_feeling} about {event}!",
                "This is a {positive_adj} experience!"
            ],
            
            '消极': [
                "I feel {negative_adj} about {event}.",
                "This {thing} is {negative_adj} and disappointing.",
                "I'm {negative_feeling} with {event}.",
                "What a {negative_adj} {thing}! I feel {negative_feeling}.",
                "I {negative_verb} this {thing}, it's {negative_adj}.",
                "This {event} makes me feel {negative_feeling}.",
                "I'm so {negative_feeling} about {event}.",
                "This is a {negative_adj} experience."
            ],
            
            '愤怒': [
                "I'm {angry_feeling} about {event}!",
                "This {thing} makes me {angry_feeling}!",
                "I {angry_verb} this {situation}!",
                "This is {angry_adj}! I'm so {angry_feeling}!",
                "I'm furious about {event}!",
                "This {behavior} makes me {angry_feeling}!",
                "I'm {angry_feeling} and frustrated!",
                "This is absolutely {angry_adj}!"
            ],
            
            '悲伤': [
                "I feel {sad_feeling} about {event}.",
                "This {thing} makes me {sad_feeling}.",
                "I'm {sad_feeling} and {sad_adj}.",
                "This {situation} is {sad_adj}.",
                "I feel {sad_feeling} and disappointed.",
                "This makes me {sad_feeling}.",
                "I'm {sad_feeling} about {event}.",
                "This is a {sad_adj} situation."
            ],
            
            '快乐': [
                "I'm {happy_feeling} about {event}!",
                "This {thing} makes me {happy_feeling}!",
                "I feel {happy_adj} and excited!",
                "What a {happy_adj} {thing}!",
                "I'm so {happy_feeling}!",
                "This {event} brings me {happy_feeling}!",
                "I {happy_verb} this {thing}!",
                "This is {happy_adj}!"
            ],
            
            '恐惧': [
                "I'm {fear_feeling} about {event}.",
                "This {thing} makes me {fear_feeling}.",
                "I feel {fear_adj} and nervous.",
                "This {situation} is {fear_adj}.",
                "I'm {fear_feeling} of {event}.",
                "This makes me {fear_feeling}.",
                "I feel {fear_adj} about {event}.",
                "This is a {fear_adj} situation."
            ],
            
            '惊讶': [
                "Wow! This {thing} is {surprise_adj}!",
                "I'm {surprise_feeling} by {event}!",
                "This is {surprise_adj}!",
                "What a {surprise_adj} {thing}!",
                "I'm {surprise_feeling} and amazed!",
                "This {event} is unexpected!",
                "I'm {surprise_feeling} by this {thing}!",
                "This is absolutely {surprise_adj}!"
            ],
            
            '中性': [
                "This {thing} is okay.",
                "I have no particular feeling about {event}.",
                "This {thing} is average.",
                "The {event} is normal.",
                "This {situation} is ordinary.",
                "I'm neutral about {thing}.",
                "This {event} is typical.",
                "The {thing} is fine, nothing special."
            ]
        }
        
        # 英文词汇库
        self.english_vocabulary = {
            'positive_adj': ['great', 'excellent', 'wonderful', 'amazing', 'fantastic', 'perfect', 'outstanding'],
            'positive_feeling': ['happy', 'excited', 'pleased', 'satisfied', 'delighted', 'joyful'],
            'positive_verb': ['love', 'like', 'enjoy', 'appreciate', 'adore'],
            
            'negative_adj': ['bad', 'terrible', 'awful', 'disappointing', 'poor', 'horrible'],
            'negative_feeling': ['disappointed', 'upset', 'sad', 'frustrated', 'unhappy'],
            'negative_verb': ['hate', 'dislike', 'despise'],
            
            'angry_adj': ['annoying', 'infuriating', 'outrageous', 'maddening'],
            'angry_feeling': ['angry', 'furious', 'mad', 'enraged', 'livid'],
            'angry_verb': ['hate', 'despise', 'loathe'],
            
            'sad_adj': ['sad', 'depressing', 'heartbreaking', 'tragic'],
            'sad_feeling': ['sad', 'depressed', 'heartbroken', 'miserable', 'sorrowful'],
            
            'happy_adj': ['happy', 'joyful', 'cheerful', 'delightful'],
            'happy_feeling': ['happy', 'joyful', 'excited', 'cheerful', 'elated'],
            'happy_verb': ['love', 'enjoy', 'adore'],
            
            'fear_adj': ['scary', 'frightening', 'terrifying', 'worrying'],
            'fear_feeling': ['afraid', 'scared', 'worried', 'anxious', 'nervous'],
            
            'surprise_adj': ['surprising', 'amazing', 'shocking', 'unexpected', 'incredible'],
            'surprise_feeling': ['surprised', 'shocked', 'amazed', 'astonished'],
            
            'event': ['exam', 'interview', 'meeting', 'date', 'trip', 'work', 'study', 'game', 'party'],
            'thing': ['movie', 'book', 'product', 'service', 'food', 'music', 'game', 'course', 'experience'],
            'situation': ['situation', 'condition', 'environment', 'atmosphere', 'circumstance'],
            'behavior': ['behavior', 'action', 'attitude', 'performance']
        }
    
    def generate_sample(self, emotion: str, language: str = 'chinese') -> Dict:
        """
        生成单个样本
        
        Args:
            emotion: 情绪类别
            language: 语言 ('chinese' 或 'english')
            
        Returns:
            样本字典
        """
        if language == 'chinese':
            templates = self.emotion_templates.get(emotion, [])
            vocab = self.vocabulary
        else:
            templates = self.english_templates.get(emotion, [])
            vocab = self.english_vocabulary
        
        if not templates:
            return {'text': '', 'emotion': emotion}
        
        # 随机选择模板
        template = random.choice(templates)
        
        # 填充模板
        text = template
        for placeholder in vocab:
            if '{' + placeholder + '}' in text:
                replacement = random.choice(vocab[placeholder])
                text = text.replace('{' + placeholder + '}', replacement)
        
        return {
            'text': text,
            'emotion': emotion,
            'language': language
        }
    
    def generate_sample_data(self, 
                           samples_per_emotion: int = 100,
                           emotions: List[str] = None,
                           language: str = 'chinese') -> List[Dict]:
        """
        生成训练样本数据
        
        Args:
            samples_per_emotion: 每个情绪类别的样本数
            emotions: 要生成的情绪类别列表
            language: 语言
            
        Returns:
            训练样本列表
        """
        if emotions is None:
            emotions = list(self.emotion_templates.keys())
        
        samples = []
        for emotion in emotions:
            for _ in range(samples_per_emotion):
                sample = self.generate_sample(emotion, language)
                if sample['text']:  # 只添加非空样本
                    samples.append(sample)
        
        # 打乱样本顺序
        random.shuffle(samples)
        
        return samples
    
    def generate_mixed_language_data(self, 
                                   samples_per_emotion: int = 50) -> List[Dict]:
        """
        生成中英文混合的训练数据
        
        Args:
            samples_per_emotion: 每个情绪类别每种语言的样本数
            
        Returns:
            混合语言训练样本列表
        """
        chinese_samples = self.generate_sample_data(
            samples_per_emotion, language='chinese'
        )
        english_samples = self.generate_sample_data(
            samples_per_emotion, language='english'
        )
        
        all_samples = chinese_samples + english_samples
        random.shuffle(all_samples)
        
        return all_samples
    
    def save_training_data(self, 
                          samples: List[Dict], 
                          filepath: str):
        """
        保存训练数据到文件
        
        Args:
            samples: 训练样本列表
            filepath: 保存路径
        """
        # 确保目录存在
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        print(f"训练数据已保存到: {filepath}")
        print(f"总样本数: {len(samples)}")
        
        # 统计各情绪类别的样本数
        emotion_counts = {}
        for sample in samples:
            emotion = sample['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print("各情绪类别样本数:")
        for emotion, count in emotion_counts.items():
            print(f"  {emotion}: {count}")
    
    def load_training_data(self, filepath: str) -> List[Dict]:
        """
        从文件加载训练数据
        
        Args:
            filepath: 文件路径
            
        Returns:
            训练样本列表
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        print(f"从 {filepath} 加载了 {len(samples)} 个训练样本")
        return samples
    
    def add_custom_templates(self, 
                           emotion: str, 
                           templates: List[str],
                           language: str = 'chinese'):
        """
        添加自定义模板
        
        Args:
            emotion: 情绪类别
            templates: 模板列表
            language: 语言
        """
        if language == 'chinese':
            if emotion not in self.emotion_templates:
                self.emotion_templates[emotion] = []
            self.emotion_templates[emotion].extend(templates)
        else:
            if emotion not in self.english_templates:
                self.english_templates[emotion] = []
            self.english_templates[emotion].extend(templates)
        
        print(f"已为 {emotion} 添加 {len(templates)} 个{language}模板")
    
    def add_custom_vocabulary(self, 
                            category: str, 
                            words: List[str],
                            language: str = 'chinese'):
        """
        添加自定义词汇
        
        Args:
            category: 词汇类别
            words: 词汇列表
            language: 语言
        """
        if language == 'chinese':
            if category not in self.vocabulary:
                self.vocabulary[category] = []
            self.vocabulary[category].extend(words)
        else:
            if category not in self.english_vocabulary:
                self.english_vocabulary[category] = []
            self.english_vocabulary[category].extend(words)
        
        print(f"已为 {category} 添加 {len(words)} 个{language}词汇")


def main():
    """演示训练数据生成"""
    generator = TrainingDataGenerator()
    
    # 生成中文训练数据
    print("生成中文训练数据...")
    chinese_data = generator.generate_sample_data(samples_per_emotion=20)
    
    # 生成英文训练数据
    print("生成英文训练数据...")
    english_data = generator.generate_sample_data(samples_per_emotion=20, language='english')
    
    # 生成混合语言数据
    print("生成混合语言训练数据...")
    mixed_data = generator.generate_mixed_language_data(samples_per_emotion=10)
    
    # 显示样本
    print("\n中文样本示例:")
    for i, sample in enumerate(chinese_data[:5]):
        print(f"{i+1}. [{sample['emotion']}] {sample['text']}")
    
    print("\n英文样本示例:")
    for i, sample in enumerate(english_data[:5]):
        print(f"{i+1}. [{sample['emotion']}] {sample['text']}")
    
    # 保存数据
    output_dir = Path(__file__).parent / "data"
    generator.save_training_data(chinese_data, str(output_dir / "chinese_training_data.json"))
    generator.save_training_data(english_data, str(output_dir / "english_training_data.json"))
    generator.save_training_data(mixed_data, str(output_dir / "mixed_training_data.json"))


if __name__ == "__main__":
    main() 