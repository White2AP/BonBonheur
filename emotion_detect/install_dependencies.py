#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依赖包安装脚本
帮助用户安装情绪识别系统所需的依赖包
"""

import subprocess
import sys
import os

def check_package(package_name):
    """检查包是否已安装"""
    try:
        __import__(package_name.replace('-', '_'))
        return True
    except ImportError:
        return False

def install_package(package_name):
    """安装单个包"""
    try:
        print(f"正在安装 {package_name}...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', package_name
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {package_name} 安装成功")
            return True
        else:
            print(f"❌ {package_name} 安装失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {package_name} 安装出错: {e}")
        return False

def install_nltk_data():
    """安装NLTK数据"""
    try:
        import nltk
        print("正在下载NLTK数据...")
        
        # 下载必要的NLTK数据
        nltk_downloads = [
            'punkt',
            'stopwords', 
            'vader_lexicon',
            'wordnet',
            'averaged_perceptron_tagger'
        ]
        
        for item in nltk_downloads:
            try:
                nltk.download(item, quiet=True)
                print(f"✅ NLTK {item} 下载成功")
            except Exception as e:
                print(f"⚠️ NLTK {item} 下载失败: {e}")
        
        return True
    except Exception as e:
        print(f"❌ NLTK数据下载失败: {e}")
        return False

def main():
    """主安装流程"""
    print("=" * 60)
    print("情绪识别系统 - 依赖包安装脚本")
    print("=" * 60)
    
    # 基础依赖包
    basic_packages = [
        'numpy',
        'pandas', 
        'scikit-learn',
        'nltk'
    ]
    
    # 音频处理包
    audio_packages = [
        'pyaudio',
        'pydub',
        'SpeechRecognition'
    ]
    
    # 视觉处理包
    vision_packages = [
        'opencv-python',
        'tensorflow'
    ]
    
    print("选择安装模式:")
    print("1. 基础模式 (只安装NLP功能)")
    print("2. 完整模式 (安装所有功能)")
    print("3. 自定义模式 (选择性安装)")
    print("4. 检查当前状态")
    
    choice = input("请选择 (1-4): ").strip()
    
    if choice == '1':
        print("\n=== 基础模式安装 ===")
        packages_to_install = basic_packages
        
    elif choice == '2':
        print("\n=== 完整模式安装 ===")
        packages_to_install = basic_packages + audio_packages + vision_packages
        
    elif choice == '3':
        print("\n=== 自定义模式安装 ===")
        packages_to_install = []
        
        print("选择要安装的功能模块:")
        if input("安装NLP模块? (y/n): ").lower() == 'y':
            packages_to_install.extend(basic_packages)
        
        if input("安装音频模块? (y/n): ").lower() == 'y':
            packages_to_install.extend(audio_packages)
        
        if input("安装视觉模块? (y/n): ").lower() == 'y':
            packages_to_install.extend(vision_packages)
    
    elif choice == '4':
        print("\n=== 当前依赖状态 ===")
        all_packages = basic_packages + audio_packages + vision_packages
        
        for package in all_packages:
            status = "✅ 已安装" if check_package(package) else "❌ 未安装"
            print(f"{package}: {status}")
        
        return
    
    else:
        print("无效选择")
        return
    
    if not packages_to_install:
        print("没有选择要安装的包")
        return
    
    # 开始安装
    print(f"\n准备安装 {len(packages_to_install)} 个包...")
    print("包列表:", ', '.join(packages_to_install))
    
    if input("确认安装? (y/n): ").lower() != 'y':
        print("安装已取消")
        return
    
    # 升级pip
    print("\n正在升级pip...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                  capture_output=True)
    
    # 安装包
    success_count = 0
    failed_packages = []
    
    for package in packages_to_install:
        if check_package(package):
            print(f"⏭️ {package} 已安装，跳过")
            success_count += 1
        else:
            if install_package(package):
                success_count += 1
            else:
                failed_packages.append(package)
    
    # 安装NLTK数据
    if 'nltk' in packages_to_install and check_package('nltk'):
        print("\n正在安装NLTK数据...")
        install_nltk_data()
    
    # 安装结果
    print("\n" + "=" * 60)
    print("安装完成!")
    print(f"成功: {success_count}/{len(packages_to_install)} 个包")
    
    if failed_packages:
        print(f"失败的包: {', '.join(failed_packages)}")
        print("\n手动安装失败的包:")
        for package in failed_packages:
            print(f"pip install {package}")
    
    print("\n测试安装结果:")
    print("python simple_usage.py")
    
    # 特殊提示
    if 'pyaudio' in failed_packages:
        print("\n💡 PyAudio安装提示:")
        print("Windows: 下载wheel文件 https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")
        print("macOS: brew install portaudio && pip install pyaudio")
        print("Linux: sudo apt-get install python3-pyaudio")
    
    if 'tensorflow' in failed_packages:
        print("\n💡 TensorFlow安装提示:")
        print("如果安装失败，尝试: pip install tensorflow-cpu")

if __name__ == "__main__":
    main() 