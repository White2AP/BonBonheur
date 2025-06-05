#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
音频特征分析工具
用于分析MFCC特征和其他音频特征的分布和重要性
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_classif
from audio_emotion_recognition import AudioEmotionRecognizer
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AudioFeatureAnalyzer:
    """音频特征分析器"""
    
    def __init__(self, dataset_path="Dataset/RAVDESS_1s_4categories"):
        """
        初始化音频特征分析器
        
        Args:
            dataset_path: 数据集路径
        """
        self.dataset_path = dataset_path
        self.emotions = ['angry', 'happy', 'neutral', 'sad']
        self.recognizer = AudioEmotionRecognizer(dataset_path)
        
        # 特征名称
        self.feature_names = self._generate_feature_names()
    
    def _generate_feature_names(self):
        """生成特征名称"""
        feature_names = []
        
        # MFCC特征名称
        for i in range(13):  # 13个MFCC系数
            feature_names.extend([
                f'MFCC_{i+1}_mean',
                f'MFCC_{i+1}_std',
                f'MFCC_{i+1}_max',
                f'MFCC_{i+1}_min',
                f'MFCC_{i+1}_median'
            ])
        
        # 其他特征名称
        feature_names.extend([
            'ZCR_mean', 'ZCR_std',
            'SpectralCentroid_mean', 'SpectralCentroid_std',
            'SpectralBandwidth_mean', 'SpectralBandwidth_std',
            'SpectralRolloff_mean', 'SpectralRolloff_std',
            'RMS_mean', 'RMS_std'
        ])
        
        return feature_names
    
    def load_and_analyze_features(self):
        """加载数据并分析特征"""
        print("🔍 开始特征分析...")
        
        # 加载数据
        features, labels, file_paths = self.recognizer.load_dataset()
        
        # 创建DataFrame
        df = pd.DataFrame(features, columns=self.feature_names)
        df['emotion'] = labels
        df['file_path'] = file_paths
        
        print(f"📊 数据集信息:")
        print(f"  样本数: {len(df)}")
        print(f"  特征数: {len(self.feature_names)}")
        print(f"  情绪分布: {df['emotion'].value_counts().to_dict()}")
        
        return df
    
    def plot_feature_distributions(self, df, save_path="analysis"):
        """绘制特征分布图"""
        print("📈 绘制特征分布图...")
        
        os.makedirs(save_path, exist_ok=True)
        
        # 选择一些重要的MFCC特征进行可视化
        important_features = [
            'MFCC_1_mean', 'MFCC_2_mean', 'MFCC_3_mean',
            'ZCR_mean', 'SpectralCentroid_mean', 'RMS_mean'
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, feature in enumerate(important_features):
            for emotion in self.emotions:
                emotion_data = df[df['emotion'] == emotion][feature]
                axes[i].hist(emotion_data, alpha=0.6, label=emotion, bins=20)
            
            axes[i].set_title(f'{feature} 分布')
            axes[i].set_xlabel('特征值')
            axes[i].set_ylabel('频次')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/feature_distributions.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 特征分布图已保存到: {save_path}/feature_distributions.png")
    
    def plot_correlation_matrix(self, df, save_path="analysis"):
        """绘制特征相关性矩阵"""
        print("🔗 绘制特征相关性矩阵...")
        
        os.makedirs(save_path, exist_ok=True)
        
        # 计算相关性矩阵（只包含数值特征）
        numeric_features = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_features].corr()
        
        # 绘制热力图
        plt.figure(figsize=(20, 16))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(
            correlation_matrix, 
            mask=mask,
            annot=False, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f'
        )
        plt.title('音频特征相关性矩阵')
        plt.tight_layout()
        plt.savefig(f"{save_path}/correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 相关性矩阵已保存到: {save_path}/correlation_matrix.png")
        
        return correlation_matrix
    
    def analyze_feature_importance(self, df, save_path="analysis"):
        """分析特征重要性"""
        print("⭐ 分析特征重要性...")
        
        os.makedirs(save_path, exist_ok=True)
        
        # 准备数据
        X = df[self.feature_names].values
        y = df['emotion'].values
        
        # 使用F检验选择重要特征
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)
        
        # 获取特征重要性分数
        feature_scores = selector.scores_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_score': feature_scores
        }).sort_values('importance_score', ascending=False)
        
        # 绘制特征重要性图
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(20)
        
        sns.barplot(data=top_features, y='feature', x='importance_score')
        plt.title('Top 20 重要特征 (F-score)')
        plt.xlabel('重要性分数')
        plt.ylabel('特征名称')
        plt.tight_layout()
        plt.savefig(f"{save_path}/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 特征重要性图已保存到: {save_path}/feature_importance.png")
        
        # 保存特征重要性数据
        feature_importance_df.to_csv(f"{save_path}/feature_importance.csv", index=False)
        print(f"✅ 特征重要性数据已保存到: {save_path}/feature_importance.csv")
        
        return feature_importance_df
    
    def plot_pca_analysis(self, df, save_path="analysis"):
        """PCA降维分析"""
        print("🔄 进行PCA降维分析...")
        
        os.makedirs(save_path, exist_ok=True)
        
        # 准备数据
        X = df[self.feature_names].values
        y = df['emotion'].values
        
        # 标准化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA降维
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # 绘制解释方差比例
        plt.figure(figsize=(12, 5))
        
        # 子图1: 累积解释方差
        plt.subplot(1, 2, 1)
        cumsum_ratio = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(range(1, len(cumsum_ratio) + 1), cumsum_ratio, 'bo-')
        plt.axhline(y=0.95, color='r', linestyle='--', label='95%解释方差')
        plt.xlabel('主成分数量')
        plt.ylabel('累积解释方差比例')
        plt.title('PCA累积解释方差')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 前两个主成分的散点图
        plt.subplot(1, 2, 2)
        colors = ['red', 'blue', 'green', 'orange']
        for i, emotion in enumerate(self.emotions):
            mask = y == emotion
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=colors[i], label=emotion, alpha=0.6)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} 解释方差)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} 解释方差)')
        plt.title('前两个主成分散点图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/pca_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ PCA分析图已保存到: {save_path}/pca_analysis.png")
        
        # 找到95%解释方差所需的主成分数量
        n_components_95 = np.argmax(cumsum_ratio >= 0.95) + 1
        print(f"📊 95%解释方差需要 {n_components_95} 个主成分")
        
        return pca, X_pca
    
    def plot_tsne_visualization(self, df, save_path="analysis"):
        """t-SNE可视化"""
        print("🎨 进行t-SNE可视化...")
        
        os.makedirs(save_path, exist_ok=True)
        
        # 准备数据
        X = df[self.feature_names].values
        y = df['emotion'].values
        
        # 标准化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 先用PCA降维到50维（加速t-SNE）
        pca = PCA(n_components=50)
        X_pca = pca.fit_transform(X_scaled)
        
        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_pca)
        
        # 绘制t-SNE结果
        plt.figure(figsize=(10, 8))
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, emotion in enumerate(self.emotions):
            mask = y == emotion
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=colors[i], label=emotion, alpha=0.6, s=50)
        
        plt.xlabel('t-SNE 维度 1')
        plt.ylabel('t-SNE 维度 2')
        plt.title('音频特征 t-SNE 可视化')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/tsne_visualization.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ t-SNE可视化已保存到: {save_path}/tsne_visualization.png")
        
        return X_tsne
    
    def analyze_emotion_characteristics(self, df, save_path="analysis"):
        """分析各情绪的特征特点"""
        print("🎭 分析各情绪的特征特点...")
        
        os.makedirs(save_path, exist_ok=True)
        
        # 计算各情绪的特征统计
        emotion_stats = df.groupby('emotion')[self.feature_names].agg(['mean', 'std'])
        
        # 选择几个重要特征进行对比
        important_features = [
            'MFCC_1_mean', 'MFCC_2_mean', 'MFCC_3_mean',
            'ZCR_mean', 'SpectralCentroid_mean', 'RMS_mean'
        ]
        
        # 绘制雷达图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), subplot_kw=dict(projection='polar'))
        axes = axes.ravel()
        
        for i, emotion in enumerate(self.emotions):
            ax = axes[i]
            
            # 获取该情绪的特征均值并标准化
            emotion_data = df[df['emotion'] == emotion][important_features].mean()
            
            # 标准化到0-1范围
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            all_data = df[important_features].values
            scaler.fit(all_data)
            normalized_data = scaler.transform(emotion_data.values.reshape(1, -1))[0]
            
            # 绘制雷达图
            angles = np.linspace(0, 2 * np.pi, len(important_features), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))  # 闭合图形
            values = np.concatenate((normalized_data, [normalized_data[0]]))
            
            ax.plot(angles, values, 'o-', linewidth=2, label=emotion)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(important_features, fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_title(f'{emotion.upper()} 情绪特征', fontsize=12, fontweight='bold')
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/emotion_characteristics.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 情绪特征对比图已保存到: {save_path}/emotion_characteristics.png")
        
        # 保存统计数据
        emotion_stats.to_csv(f"{save_path}/emotion_statistics.csv")
        print(f"✅ 情绪统计数据已保存到: {save_path}/emotion_statistics.csv")
        
        return emotion_stats
    
    def generate_analysis_report(self, df, save_path="analysis"):
        """生成分析报告"""
        print("📋 生成分析报告...")
        
        os.makedirs(save_path, exist_ok=True)
        
        report = []
        report.append("# 音频情绪识别特征分析报告\n")
        report.append(f"生成时间: {pd.Timestamp.now()}\n\n")
        
        # 数据集概览
        report.append("## 1. 数据集概览\n")
        report.append(f"- 总样本数: {len(df)}\n")
        report.append(f"- 特征维度: {len(self.feature_names)}\n")
        report.append(f"- 情绪类别: {', '.join(self.emotions)}\n")
        
        emotion_counts = df['emotion'].value_counts()
        report.append("\n### 情绪分布:\n")
        for emotion, count in emotion_counts.items():
            percentage = count / len(df) * 100
            report.append(f"- {emotion}: {count} 样本 ({percentage:.1f}%)\n")
        
        # 特征统计
        report.append("\n## 2. 特征统计\n")
        numeric_features = df[self.feature_names]
        report.append(f"- 特征均值范围: {numeric_features.mean().min():.4f} ~ {numeric_features.mean().max():.4f}\n")
        report.append(f"- 特征标准差范围: {numeric_features.std().min():.4f} ~ {numeric_features.std().max():.4f}\n")
        
        # 缺失值检查
        missing_values = df.isnull().sum().sum()
        report.append(f"- 缺失值总数: {missing_values}\n")
        
        # 特征重要性
        feature_importance = self.analyze_feature_importance(df, save_path)
        top_5_features = feature_importance.head(5)
        report.append("\n## 3. Top 5 重要特征\n")
        for idx, row in top_5_features.iterrows():
            report.append(f"- {row['feature']}: {row['importance_score']:.2f}\n")
        
        # 情绪特征分析
        report.append("\n## 4. 各情绪特征特点\n")
        emotion_stats = df.groupby('emotion')[self.feature_names].mean()
        
        for emotion in self.emotions:
            report.append(f"\n### {emotion.upper()}:\n")
            emotion_features = emotion_stats.loc[emotion]
            top_features = emotion_features.nlargest(3)
            report.append("主要特征:\n")
            for feature, value in top_features.items():
                report.append(f"- {feature}: {value:.4f}\n")
        
        # 建议
        report.append("\n## 5. 建议\n")
        report.append("- 考虑使用特征选择方法减少特征维度\n")
        report.append("- 可以尝试深度学习方法提取更高级的特征\n")
        report.append("- 建议增加数据增强技术提高模型泛化能力\n")
        
        # 保存报告
        with open(f"{save_path}/analysis_report.md", 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        print(f"✅ 分析报告已保存到: {save_path}/analysis_report.md")
        
        return ''.join(report)
    
    def run_complete_analysis(self):
        """运行完整的特征分析流程"""
        print("🚀 开始完整的音频特征分析...")
        print("=" * 60)
        
        # 1. 加载数据
        df = self.load_and_analyze_features()
        
        # 2. 特征分布分析
        self.plot_feature_distributions(df)
        
        # 3. 相关性分析
        self.plot_correlation_matrix(df)
        
        # 4. 特征重要性分析
        self.analyze_feature_importance(df)
        
        # 5. PCA分析
        self.plot_pca_analysis(df)
        
        # 6. t-SNE可视化
        self.plot_tsne_visualization(df)
        
        # 7. 情绪特征分析
        self.analyze_emotion_characteristics(df)
        
        # 8. 生成报告
        self.generate_analysis_report(df)
        
        print("\n🎉 特征分析完成！")
        print("📁 所有结果已保存到 'analysis' 文件夹")

def main():
    """主函数"""
    analyzer = AudioFeatureAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 