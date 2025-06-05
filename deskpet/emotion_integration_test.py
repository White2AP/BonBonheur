#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情绪检测集成测试脚本
测试桌宠的情绪感知功能
"""

import tkinter as tk
from main import DesktopPet

def test_emotion_detection():
    """测试情绪检测功能"""
    print("=" * 60)
    print("桌宠情绪检测集成测试")
    print("=" * 60)
    
    print("✅ 新增功能:")
    print("1. 情绪检测系统集成")
    print("2. 聊天时自动分析用户情绪")
    print("3. 根据心情调整日程安排策略")
    print("4. 番茄钟结束后情绪预测专注度")
    print("5. 取消功能升级弹窗")
    
    print("\n🎭 情绪感知功能:")
    print("- 心情好：保持正常日程安排")
    print("- 心情差：减少40%的日程安排")
    print("- 中性：稍微减少20%的日程安排")
    
    print("\n📊 专注度预测:")
    print("- 积极情绪：85%专注度")
    print("- 快乐情绪：90%专注度")
    print("- 消极情绪：45%专注度")
    print("- 悲伤情绪：40%专注度")
    
    print("\n🧪 测试步骤:")
    print("1. 启动桌宠")
    print("2. 点击'Bonbonheur聊天'")
    print("3. 发送不同情绪的消息测试情绪分析")
    print("4. 点击'智能日程管理'测试情绪感知优化")
    print("5. 启动番茄钟测试情绪预测专注度")
    
    print("\n💬 测试消息示例:")
    print("- 积极：'今天天气真好，我很开心！'")
    print("- 消极：'工作压力太大了，感觉很累'")
    print("- 悲伤：'最近遇到一些困难，心情不太好'")
    print("- 兴奋：'刚收到好消息，太棒了！'")
    
    print("\n🚀 启动桌宠...")
    
    try:
        root = tk.Tk()
        pet = DesktopPet(root)
        
        # 测试情绪检测系统
        if pet.emotion_detector:
            print("✅ 情绪检测系统初始化成功")
            
            # 测试几个示例
            test_texts = [
                "今天天气真好，我很开心！",
                "工作压力太大了，感觉很累",
                "最近遇到一些困难，心情不太好",
                "刚收到好消息，太棒了！"
            ]
            
            print("\n🧪 情绪分析测试:")
            for text in test_texts:
                emotion, confidence = pet.analyze_user_emotion(text)
                adjustment = pet.get_mood_adjustment_factor()
                predicted_focus = pet.predict_focus_from_emotion(emotion, confidence)
                
                print(f"文本: {text}")
                print(f"  情绪: {emotion} (置信度: {confidence:.2f})")
                print(f"  日程调整因子: {adjustment:.2f}")
                print(f"  预测专注度: {predicted_focus:.2f}")
                print()
        else:
            print("⚠️ 情绪检测系统初始化失败")
        
        print("✅ 桌宠启动成功！开始测试情绪感知功能")
        root.mainloop()
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_emotion_detection() 