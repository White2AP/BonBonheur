#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能日历应用启动脚本
一键启动带有智能时间优化功能的日历应用
"""

import sys
import os

def main():
    """启动智能日历应用"""
    print("🚀 启动智能日历应用...")
    print("=" * 50)
    print("功能特性:")
    print("  ✅ 单一日程和循环日程支持")
    print("  ✅ 完整的番茄钟功能")
    print("  ✅ 智能时间优化 ⭐ 新增")
    print("  ✅ 专注度监测与分析")
    print("  ✅ 多模态识别系统接口")
    print("=" * 50)
    
    try:
        # 导入主应用
        from calendar_app_with_pomodoro import main as run_calendar
        
        # 启动应用
        print("正在启动应用...")
        run_calendar()
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保所有依赖已正确安装:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 