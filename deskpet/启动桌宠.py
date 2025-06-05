#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
桌宠智能助手 - 快速启动脚本
集成情绪感知功能
"""

import tkinter as tk
from main import DesktopPet

def main():
    print("🎊 桌宠智能助手 - 情绪感知版")
    print("=" * 50)
    print("✅ 新功能:")
    print("  🎭 智能情绪检测")
    print("  📊 心情感知日程调整")
    print("  🍅 情绪预测专注度")
    print("  💬 情绪感知聊天")
    print("=" * 50)
    print("🚀 正在启动桌宠...")
    
    try:
        root = tk.Tk()
        pet = DesktopPet(root)
        
        if pet.emotion_detector:
            print("✅ 情绪检测系统已就绪")
        else:
            print("⚠️ 情绪检测系统未启用")
        
        print("✅ 桌宠启动成功！")
        print("\n💡 使用提示:")
        print("  - 点击桌宠打开菜单")
        print("  - 聊天时会自动分析情绪")
        print("  - 日程优化会根据心情调整")
        print("  - 番茄钟结束后预测专注度")
        
        root.mainloop()
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        input("按回车键退出...")

if __name__ == "__main__":
    main() 