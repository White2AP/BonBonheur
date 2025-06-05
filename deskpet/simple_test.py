#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单菜单功能测试
"""

import tkinter as tk
from main import DesktopPet

def main():
    print("=" * 50)
    print("桌宠菜单功能测试")
    print("=" * 50)
    
    print("✅ 新的菜单管理功能:")
    print("1. 点击桌宠时，旧菜单自动关闭，新菜单打开")
    print("2. 菜单会在2秒后自动关闭")
    print("3. 次级窗口（聊天等）不会自动关闭")
    
    print("\n🎯 测试方法:")
    print("- 多次点击桌宠，观察菜单是否正确切换")
    print("- 等待2秒观察菜单是否自动关闭")
    print("- 测试聊天功能是否保持开启")
    
    print("\n🚀 启动桌宠...")
    
    try:
        root = tk.Tk()
        pet = DesktopPet(root)
        print("✅ 桌宠启动成功！点击桌宠测试菜单功能")
        root.mainloop()
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    main() 