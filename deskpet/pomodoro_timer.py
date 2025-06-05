#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
番茄钟计时器模块
支持日程提醒和专注计时功能
"""

import datetime
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, Optional
import pygame
import os

class PomodoroTimer:
    """番茄钟计时器"""
    
    def __init__(self, parent_window=None):
        self.parent_window = parent_window
        self.timer_window = None
        self.is_running = False
        self.is_paused = False
        self.remaining_time = 0
        self.total_time = 0
        self.timer_thread = None
        self.completion_callback = None
        self.emotion_predict_callback = None  # 情绪预测回调
        
        # 初始化音频
        try:
            pygame.mixer.init()
            self.sound_enabled = True
        except:
            self.sound_enabled = False
            print("音频初始化失败，将使用静默模式")
    
    def show_schedule_reminder(self, schedule_name: str, start_time: datetime.datetime, 
                             duration_minutes: int, on_start_callback: Callable = None) -> bool:
        """显示日程提醒对话框"""
        current_time = datetime.datetime.now()
        
        # 检查是否到了开始时间（允许5分钟的提前提醒）
        time_diff = (start_time - current_time).total_seconds()
        
        if -300 <= time_diff <= 300:  # 前后5分钟内
            result = messagebox.askyesno(
                "日程提醒", 
                f"要开始「{schedule_name}」了吗？\n\n"
                f"开始时间: {start_time.strftime('%H:%M')}\n"
                f"预计时长: {duration_minutes}分钟\n\n"
                f"点击「是」开始番茄钟专注计时",
                icon='question'
            )
            
            if result and on_start_callback:
                on_start_callback()
                self.start_pomodoro(schedule_name, duration_minutes)
                return True
        
        return False
    
    def start_pomodoro(self, schedule_name: str, duration_minutes: int, 
                      completion_callback: Callable = None):
        """开始番茄钟计时"""
        if self.is_running:
            messagebox.showwarning("警告", "已有计时器在运行中")
            return
        
        self.completion_callback = completion_callback
        self.total_time = duration_minutes * 60  # 转换为秒
        self.remaining_time = self.total_time
        self.is_running = True
        self.is_paused = False
        
        # 创建计时器窗口
        self.create_timer_window(schedule_name)
        
        # 启动计时线程
        self.timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self.timer_thread.start()
    
    def create_timer_window(self, schedule_name: str):
        """创建计时器窗口"""
        self.timer_window = tk.Toplevel(self.parent_window)
        self.timer_window.title(f"番茄钟 - {schedule_name}")
        self.timer_window.geometry("400x300")
        self.timer_window.resizable(False, False)
        
        # 设置窗口置顶
        self.timer_window.attributes('-topmost', True)
        
        # 主框架
        main_frame = ttk.Frame(self.timer_window, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 日程名称
        self.schedule_label = ttk.Label(
            main_frame, 
            text=f"专注进行中: {schedule_name}", 
            font=("Arial", 14, "bold")
        )
        self.schedule_label.pack(pady=(0, 20))
        
        # 时间显示
        self.time_label = ttk.Label(
            main_frame, 
            text="25:00", 
            font=("Arial", 36, "bold"),
            foreground="red"
        )
        self.time_label.pack(pady=20)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main_frame, 
            variable=self.progress_var, 
            maximum=100,
            length=300
        )
        self.progress_bar.pack(pady=20)
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        # 暂停/继续按钮
        self.pause_button = ttk.Button(
            button_frame, 
            text="暂停", 
            command=self.toggle_pause,
            width=10
        )
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        # 停止按钮
        self.stop_button = ttk.Button(
            button_frame, 
            text="停止", 
            command=self.stop_timer,
            width=10
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # 延长按钮
        self.extend_button = ttk.Button(
            button_frame, 
            text="+5分钟", 
            command=self.extend_timer,
            width=10
        )
        self.extend_button.pack(side=tk.LEFT, padx=5)
        
        # 状态标签
        self.status_label = ttk.Label(main_frame, text="专注中...", font=("Arial", 10))
        self.status_label.pack(pady=(10, 0))
        
        # 关闭窗口事件
        self.timer_window.protocol("WM_DELETE_WINDOW", self.on_window_close)
    
    def _timer_loop(self):
        """计时器主循环"""
        while self.is_running and self.remaining_time > 0:
            if not self.is_paused:
                self.remaining_time -= 1
                self._update_display()
            time.sleep(1)
        
        if self.is_running and self.remaining_time <= 0:
            self._timer_completed()
    
    def _update_display(self):
        """更新显示"""
        if self.timer_window and self.timer_window.winfo_exists():
            try:
                # 更新时间显示
                minutes = self.remaining_time // 60
                seconds = self.remaining_time % 60
                time_text = f"{minutes:02d}:{seconds:02d}"
                self.time_label.config(text=time_text)
                
                # 更新进度条
                progress = ((self.total_time - self.remaining_time) / self.total_time) * 100
                self.progress_var.set(progress)
                
                # 更新颜色
                if self.remaining_time <= 300:  # 最后5分钟
                    self.time_label.config(foreground="red")
                elif self.remaining_time <= 600:  # 最后10分钟
                    self.time_label.config(foreground="orange")
                else:
                    self.time_label.config(foreground="green")
                
            except tk.TclError:
                # 窗口已关闭
                self.is_running = False
    
    def _timer_completed(self):
        """计时完成"""
        self.is_running = False
        
        # 播放提醒音
        self._play_completion_sound()
        
        # 调用情绪预测回调（在显示对话框之前）
        if hasattr(self, 'emotion_predict_callback') and self.emotion_predict_callback:
            try:
                self.emotion_predict_callback()
            except Exception as e:
                print(f"情绪预测回调失败: {e}")
        
        # 显示完成对话框
        if self.timer_window and self.timer_window.winfo_exists():
            self.timer_window.attributes('-topmost', True)
            messagebox.showinfo(
                "番茄钟完成", 
                "🍅 专注时间结束！\n\n恭喜您完成了这个番茄钟！\n已根据您的情绪状态预测专注度。",
                parent=self.timer_window
            )
        
        # 调用完成回调
        if self.completion_callback:
            self.completion_callback()
        
        # 关闭窗口
        if self.timer_window:
            self.timer_window.destroy()
    
    def _play_completion_sound(self):
        """播放完成提醒音"""
        if not self.sound_enabled:
            return
        
        try:
            # 创建简单的提醒音
            import numpy as np
            
            # 生成提醒音频
            sample_rate = 22050
            duration = 1.0
            frequency = 800
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            wave = np.sin(2 * np.pi * frequency * t) * 0.3
            
            # 转换为pygame可用的格式
            sound_array = (wave * 32767).astype(np.int16)
            sound = pygame.sndarray.make_sound(sound_array)
            sound.play()
            
        except Exception as e:
            print(f"播放提醒音失败: {e}")
    
    def toggle_pause(self):
        """切换暂停/继续状态"""
        if not self.is_running:
            return
        
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            self.pause_button.config(text="继续")
            self.status_label.config(text="已暂停")
        else:
            self.pause_button.config(text="暂停")
            self.status_label.config(text="专注中...")
    
    def stop_timer(self):
        """停止计时器"""
        if messagebox.askyesno("确认", "确定要停止当前的番茄钟吗？", parent=self.timer_window):
            self.is_running = False
            if self.timer_window:
                self.timer_window.destroy()
    
    def extend_timer(self):
        """延长计时器5分钟"""
        self.remaining_time += 300  # 5分钟
        self.total_time += 300
        self.status_label.config(text="已延长5分钟")
        
        # 3秒后恢复状态显示
        self.timer_window.after(3000, lambda: self.status_label.config(text="专注中..."))
    
    def on_window_close(self):
        """窗口关闭事件"""
        if self.is_running:
            if messagebox.askyesno("确认", "计时器正在运行中，确定要关闭吗？", parent=self.timer_window):
                self.is_running = False
                self.timer_window.destroy()
        else:
            self.timer_window.destroy()

    def set_emotion_predict_callback(self, callback):
        """设置情绪预测回调函数"""
        self.emotion_predict_callback = callback

class ScheduleReminder:
    """日程提醒管理器"""
    
    def __init__(self, calendar_manager, pomodoro_timer):
        self.calendar_manager = calendar_manager
        self.pomodoro_timer = pomodoro_timer
        self.reminder_thread = None
        self.is_running = False
        self.checked_schedules = set()  # 已检查过的日程ID
    
    def start_reminder_service(self):
        """启动提醒服务"""
        if self.is_running:
            return
        
        self.is_running = True
        self.reminder_thread = threading.Thread(target=self._reminder_loop, daemon=True)
        self.reminder_thread.start()
        print("日程提醒服务已启动")
    
    def stop_reminder_service(self):
        """停止提醒服务"""
        self.is_running = False
        print("日程提醒服务已停止")
    
    def _reminder_loop(self):
        """提醒服务主循环"""
        while self.is_running:
            try:
                self._check_upcoming_schedules()
                time.sleep(60)  # 每分钟检查一次
            except Exception as e:
                print(f"提醒服务错误: {e}")
                time.sleep(60)
    
    def _check_upcoming_schedules(self):
        """检查即将开始的日程"""
        today = datetime.date.today()
        schedules = self.calendar_manager.get_schedules_by_date(today)
        
        current_time = datetime.datetime.now()
        
        for schedule in schedules:
            # 跳过已完成的日程
            if hasattr(schedule, 'completed') and schedule.completed:
                continue
            
            # 跳过已检查过的日程
            schedule_key = f"{schedule.schedule_id}_{schedule.start_time.isoformat()}"
            if schedule_key in self.checked_schedules:
                continue
            
            # 检查是否到了提醒时间（提前5分钟）
            time_diff = (schedule.start_time - current_time).total_seconds()
            
            if -300 <= time_diff <= 300:  # 前后5分钟内
                self.checked_schedules.add(schedule_key)
                
                # 计算持续时间（分钟）
                duration_minutes = int((schedule.end_time - schedule.start_time).total_seconds() / 60)
                
                # 显示提醒
                def on_start():
                    def mark_completed():
                        schedule.completed = True
                        # 保存到文件
                        year, month = schedule.start_time.year, schedule.start_time.month
                        self.calendar_manager.save_month_schedules(year, month)
                    
                    return mark_completed
                
                self.pomodoro_timer.show_schedule_reminder(
                    schedule.name,
                    schedule.start_time,
                    duration_minutes,
                    on_start()
                ) 