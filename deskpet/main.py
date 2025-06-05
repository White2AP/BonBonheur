import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import datetime
from dateutil import parser
import json
import os
import threading
import requests
from tkcalendar import Calendar
from googletrans import Translator
# 导入高级日程系统
from calendar_app_with_pomodoro import PomodoroCalendarGUI
# 导入情绪检测系统
from simple_unified_detector import SimpleUnifiedDetector


class DesktopPet:
    def __init__(self, root):
        self.root = root
        self.root.title("智能桌宠助手")

        # 透明背景设置
        self.root.config(bg='white')  # 必须先设置背景色
        self.root.attributes('-transparentcolor', 'white')  # 关键透明设置
        self.root.wm_attributes('-topmost', True)
        self.root.overrideredirect(True)

        # 放大桌宠尺寸
        self.pet_size = 500
        self.x = root.winfo_screenwidth() - self.pet_size - 50
        self.y = root.winfo_screenheight() - self.pet_size - 50
        self.root.geometry(f"{self.pet_size}x{self.pet_size}+{self.x}+{self.y}")

        # 加载透明GIF/
        self.gif_path = "animate.gif"
        self.load_gif_frames()

        # 透明标签显示
        self.pet_label = tk.Label(self.root, bg='white', bd=0)
        self.pet_label.pack()
        self.animate_gif(0)

        # 事件绑定
        self.root.bind("<Button-1>", self.show_menu)
        self.root.bind("<B1-Motion>", self.drag_pet)

        # 任务数据初始化
        self.task_data = []
        self.load_tasks()

        # DeepSeek API配置
        self.DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
        self.API_KEY = "sk-07fadfcbaab14562810a87a80a83d759"  # 已使用您提供的API密钥
        self.chat_history = []
        
        # 用于跟踪次级窗口的变量
        self.secondary_windows = []
        self.current_menu = None  # 跟踪当前打开的菜单
        
        # 初始化情绪检测系统
        self.emotion_detector = None
        self.init_emotion_detector()
        
        # 情绪状态跟踪
        self.current_mood = "中性"
        self.mood_confidence = 0.5
        self.daily_mood_history = []

    def init_emotion_detector(self):
        """初始化情绪检测系统"""
        try:
            self.emotion_detector = SimpleUnifiedDetector()
            print("✅ 情绪检测系统初始化成功")
        except Exception as e:
            print(f"⚠️ 情绪检测系统初始化失败: {e}")
            self.emotion_detector = None

    def analyze_user_emotion(self, text=""):
        """分析用户情绪"""
        if not self.emotion_detector:
            return "中性", 0.5
        
        try:
            if text:
                # 分析文本情绪
                result = self.emotion_detector.analyze_text(text)
                emotion = result.get('emotion', '中性')
                confidence = result.get('confidence', 0.5)
            else:
                # 如果没有文本，使用默认值
                emotion = "中性"
                confidence = 0.5
            
            # 更新当前情绪状态
            self.current_mood = emotion
            self.mood_confidence = confidence
            
            # 记录到历史
            self.daily_mood_history.append({
                'time': datetime.datetime.now(),
                'emotion': emotion,
                'confidence': confidence,
                'text': text
            })
            
            return emotion, confidence
            
        except Exception as e:
            print(f"情绪分析失败: {e}")
            return "中性", 0.5

    def get_mood_adjustment_factor(self):
        """根据当前心情获取日程调整因子"""
        if self.current_mood in ["消极", "悲伤", "恐惧", "愤怒"]:
            # 心情不好时，减少日程安排
            return 0.6  # 减少40%的日程
        elif self.current_mood in ["积极", "快乐", "兴奋"]:
            # 心情好时，保持正常日程
            return 1.0
        else:
            # 中性情绪，稍微减少一些
            return 0.8

    def predict_focus_from_emotion(self, emotion, confidence):
        """根据情绪预测专注度"""
        emotion_focus_map = {
            "积极": 0.85,
            "快乐": 0.90,
            "中性": 0.70,
            "消极": 0.45,
            "悲伤": 0.40,
            "恐惧": 0.35,
            "愤怒": 0.30,
            "惊讶": 0.60,
            "兴奋": 0.75
        }
        
        base_focus = emotion_focus_map.get(emotion, 0.70)
        # 置信度影响预测准确性
        adjusted_focus = base_focus * confidence + 0.5 * (1 - confidence)
        return min(max(adjusted_focus, 0.1), 1.0)

    def animate_gif(self, frame_index):
        """播放GIF动画"""
        if hasattr(self, 'gif_frames') and self.gif_frames:
            frame = frame_index % len(self.gif_frames)
            self.pet_label.config(image=self.gif_frames[frame])
            self.root.after(100, lambda: self.animate_gif(frame_index + 1))

    def animate_gif(self, frame_index):
        """播放GIF动画（减慢版）"""
        if hasattr(self, 'gif_frames') and self.gif_frames:
            frame = frame_index % len(self.gif_frames)
            self.pet_label.config(image=self.gif_frames[frame])
            # 将100ms改为更大的值（单位：毫秒）
            self.root.after(300, lambda: self.animate_gif(frame_index + 1))  # 从100改为300
    
    def drag_pet(self, event):
        """拖动桌宠"""
        x = self.root.winfo_x() + (event.x - self.pet_size // 2)
        y = self.root.winfo_y() + (event.y - self.pet_size // 2)
        self.root.geometry(f"+{x}+{y}")

    def setup_menu_auto_close(self, menu):
        """为主菜单设置2秒后自动关闭"""
        def auto_close():
            try:
                if menu and menu.winfo_exists():
                    menu.destroy()
                    if self.current_menu == menu:
                        self.current_menu = None
            except:
                pass
        
        # 2秒后自动关闭
        self.root.after(2000, auto_close)

    def close_program(self):
        """关闭整个程序"""
        if messagebox.askyesno("确认退出", "确定要关闭桌宠程序吗？"):
            # 关闭所有次级窗口
            for window in self.secondary_windows[:]:
                self.safe_close_window(window)
            # 关闭主程序
            self.root.quit()
            self.root.destroy()

    def show_menu(self, event):
        """显示紧贴桌宠的菜单（像素级精准定位）"""
        # 先关闭旧菜单
        if self.current_menu and self.current_menu.winfo_exists():
            try:
                self.current_menu.destroy()
            except:
                pass
        
        menu = tk.Toplevel(self.root)
        menu.title("功能菜单")
        self.current_menu = menu  # 记录当前菜单

        # 获取鼠标点击位置（相对于屏幕）
        click_x = self.root.winfo_pointerx()
        click_y = self.root.winfo_pointery()

        # 计算菜单弹出位置（智能避让）
        menu_width = 200
        menu_height = 180  # 增加高度以容纳新按钮
        pet_rect = {
            'left': self.root.winfo_x(),
            'right': self.root.winfo_x() + self.pet_size,
            'top': self.root.winfo_y(),
            'bottom': self.root.winfo_y() + self.pet_size
        }

        # 智能定位策略（优先右侧，次选左侧）
        if (click_x + menu_width) < self.root.winfo_screenwidth():
            menu_x = click_x + 5  # 鼠标右侧5像素
        else:
            menu_x = click_x - menu_width - 5  # 鼠标左侧5像素

        # 垂直对齐（优先下方，次选上方）
        if (click_y + menu_height) < self.root.winfo_screenheight():
            menu_y = click_y + 5  # 鼠标下方5像素
        else:
            menu_y = click_y - menu_height - 5  # 鼠标上方5像素

        # 最终定位（带边界保护）
        menu_x = max(0, min(menu_x, self.root.winfo_screenwidth() - menu_width))
        menu_y = max(0, min(menu_y, self.root.winfo_screenheight() - menu_height))

        menu.geometry(f"{menu_width}x{menu_height}+{int(menu_x)}+{int(menu_y)}")
        menu.attributes('-topmost', True)
        menu.overrideredirect(True)
        menu.configure(
            bg="#f0f0f0",
            bd=1,
            relief=tk.RAISED,
            highlightthickness=1,
            highlightbackground="#c0c0c0"
        )

        # 添加菜单阴影效果（Windows系统）
        try:
            from ctypes import windll
            windll.user32.SetWindowPos(
                menu.winfo_id(), -1,
                int(menu_x), int(menu_y),
                menu_width, menu_height, 0x40
            )
        except:
            pass

        # 优化菜单按钮样式
        style = ttk.Style()
        style.configure("TightMenu.TButton",
                        font=('Microsoft YaHei', 9),
                        padding=2,
                        relief=tk.FLAT)

        ttk.Button(menu, text="智能日程管理", style="TightMenu.TButton",
                   command=lambda: self.create_advanced_schedule_window(menu)).pack(
            fill=tk.X, padx=2, pady=1)
        ttk.Button(menu, text="Bonbonheur聊天^^", style="TightMenu.TButton",
                   command=lambda: self.create_chat_window(menu)).pack(
            fill=tk.X, padx=2, pady=1)
        ttk.Button(menu, text="Discuter en français", style="TightMenu.TButton",
                   command=lambda: self.create_french_chat_window(menu)).pack(
            fill=tk.X, padx=2, pady=1)
        ttk.Button(menu, text="关闭菜单", style="TightMenu.TButton",
                   command=lambda: self.close_current_menu()).pack(
            fill=tk.X, padx=2, pady=1)
        
        # 添加分隔线
        separator = tk.Frame(menu, height=1, bg="#c0c0c0")
        separator.pack(fill=tk.X, padx=2, pady=2)
        
        # 添加关闭程序按钮
        ttk.Button(menu, text="关闭程序", style="TightMenu.TButton",
                   command=self.close_program).pack(
            fill=tk.X, padx=2, pady=1)

        # 将菜单添加到次级窗口列表并设置自动关闭
        self.secondary_windows.append(menu)
        self.setup_menu_auto_close(menu)

    def close_current_menu(self):
        """关闭当前菜单"""
        if self.current_menu and self.current_menu.winfo_exists():
            try:
                self.current_menu.destroy()
                self.current_menu = None
            except:
                pass

    def safe_close_window(self, window):
        """安全关闭窗口"""
        try:
            if window and window.winfo_exists():
                window.destroy()
                if window in self.secondary_windows:
                    self.secondary_windows.remove(window)
        except:
            pass

    def load_gif_frames(self):
        """加载透明GIF帧"""
        self.gif_frames = []
        try:
            gif = Image.open(self.gif_path)
            for frame_index in range(gif.n_frames):
                gif.seek(frame_index)
                frame = gif.convert("RGBA")
                frame = frame.resize((self.pet_size, self.pet_size), Image.LANCZOS)
                self.gif_frames.append(ImageTk.PhotoImage(frame))
            print(f"成功加载GIF: {len(self.gif_frames)}帧")
        except Exception as e:
            print(f"GIF加载失败: {e}")
            blank = Image.new("RGBA", (self.pet_size, self.pet_size), (0, 0, 0, 0))
            self.gif_frames = [ImageTk.PhotoImage(blank)]
            print("使用透明占位图像")

    def create_advanced_schedule_window(self, menu):
        """创建高级日程管理窗口（集成番茄钟功能和情绪检测）"""
        menu.destroy()
        
        # 检查是否已经有日程窗口打开
        for window in self.secondary_windows:
            if hasattr(window, 'title') and window.winfo_exists():
                try:
                    if "日历" in window.title():
                        window.lift()
                        return
                except:
                    pass
        
        try:
            # 创建高级日程管理应用
            calendar_app = PomodoroCalendarGUI()
            calendar_window = calendar_app.root
            
            # 集成情绪检测到日程系统
            self.integrate_emotion_to_calendar(calendar_app)
            
            # 将窗口添加到次级窗口列表（不设置自动关闭）
            self.secondary_windows.append(calendar_window)
            
            # 修改窗口标题以便识别
            calendar_window.title("智能日历应用 - 支持情绪感知和番茄钟")
            
            # 设置窗口关闭事件
            def on_calendar_close():
                if calendar_window in self.secondary_windows:
                    self.secondary_windows.remove(calendar_window)
                calendar_app.reminder_service.stop_reminder_service()
                calendar_window.destroy()
            
            calendar_window.protocol("WM_DELETE_WINDOW", on_calendar_close)
            
            # 不再显示功能升级弹窗，直接静默启动
            print("✅ 智能日程管理系统已启动（集成情绪感知功能）")
            
        except Exception as e:
            messagebox.showerror("错误", f"无法启动高级日程系统: {str(e)}")
            print(f"日程系统启动错误: {e}")

    def integrate_emotion_to_calendar(self, calendar_app):
        """将情绪检测集成到日程应用中"""
        try:
            # 保存原始方法的引用
            original_time_range_optimize = calendar_app.show_time_range_optimization
            
            # 重写时间范围优化方法，加入情绪调整
            def emotion_aware_time_range_optimize():
                # 分析当前情绪（可以从聊天记录中获取最近的文本）
                recent_text = self.get_recent_chat_text()
                emotion, confidence = self.analyze_user_emotion(recent_text)
                
                # 获取调整因子
                adjustment_factor = self.get_mood_adjustment_factor()
                
                print(f"🎭 当前情绪: {emotion} (置信度: {confidence:.2f})")
                print(f"📊 日程调整因子: {adjustment_factor:.2f}")
                
                # 显示情绪感知提示
                mood_message = self.get_mood_message(emotion, adjustment_factor)
                
                # 创建情绪感知的时间范围优化对话框
                emotion_range_dialog = EmotionAwareTimeRangeDialog(
                    calendar_app.root, 
                    calendar_app.calendar_manager, 
                    calendar_app.smart_optimizer, 
                    calendar_app.optimizer_gui,
                    emotion,
                    confidence,
                    adjustment_factor,
                    mood_message
                )
                emotion_range_dialog.show()
            
            # 为番茄钟设置情绪预测回调
            def emotion_predict_focus():
                # 分析当前情绪
                recent_text = self.get_recent_chat_text()
                emotion, confidence = self.analyze_user_emotion(recent_text)
                
                # 预测专注度
                predicted_focus = self.predict_focus_from_emotion(emotion, confidence)
                
                print(f"🎭 番茄钟结束 - 情绪: {emotion}, 预测专注度: {predicted_focus:.2f}")
                
                # 如果有智能优化器，更新专注度记录
                if hasattr(calendar_app, 'smart_optimizer'):
                    # 创建一个模拟的专注会话记录
                    now = datetime.datetime.now()
                    try:
                        calendar_app.smart_optimizer.record_focus_session(
                            "番茄钟任务", now - datetime.timedelta(minutes=25), now, predicted_focus
                        )
                        print(f"📝 已记录情绪预测的专注度: {predicted_focus:.2f}")
                    except Exception as e:
                        print(f"记录专注度失败: {e}")
                
                # 显示情绪分析结果
                emotion_message = f"🎭 情绪分析结果：\n情绪：{emotion}\n置信度：{confidence:.2f}\n预测专注度：{predicted_focus:.2f}"
                messagebox.showinfo("情绪感知分析", emotion_message)
            
            # 替换时间范围优化方法（保持原有的快速优化今日不变）
            calendar_app.show_time_range_optimization = emotion_aware_time_range_optimize
            
            # 为番茄钟设置情绪预测回调
            if hasattr(calendar_app, 'pomodoro_timer'):
                calendar_app.pomodoro_timer.set_emotion_predict_callback(emotion_predict_focus)
                print("✅ 番茄钟情绪预测回调已设置")
            
            print("✅ 情绪检测已集成到时间范围优化系统")
            
        except Exception as e:
            print(f"⚠️ 情绪集成失败: {e}")

    def get_recent_chat_text(self):
        """获取最近的聊天文本用于情绪分析"""
        if self.chat_history:
            # 获取最近的用户消息
            for message in reversed(self.chat_history):
                if message.get('role') == 'user':
                    return message.get('content', '')
        return ""

    def get_mood_message(self, emotion, adjustment_factor):
        """根据情绪生成相应的提示消息"""
        if emotion in ["消极", "悲伤", "恐惧", "愤怒"]:
            return f"😔 检测到您心情欠佳({emotion})，已为您减少今日日程安排，建议适当休息。"
        elif emotion in ["积极", "快乐", "兴奋"]:
            return f"😊 检测到您心情不错({emotion})，保持当前的日程安排，加油！"
        else:
            return f"😐 当前情绪状态：{emotion}，已适当调整日程安排。"

    def create_chat_window(self, menu):
        """创建聊天窗口"""
        menu.destroy()
        if hasattr(self, 'chat_window') and self.chat_window and self.chat_window.winfo_exists():
            self.chat_window.lift()
            return

        self.chat_window = tk.Toplevel(self.root)
        self.chat_window.title("Bonbonheur")
        self.chat_window.geometry("500x600")

        # 聊天显示区
        self.chat_display = scrolledtext.ScrolledText(
            self.chat_window,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.chat_display.tag_config("assistant", foreground="#4a90e2", font=("Arial", 10, "bold"))
        self.chat_display.tag_config("user", foreground="#e67e22", font=("Arial", 10, "bold"))
        self.chat_display.tag_config("system", foreground="#ff0000", font=("Arial", 10, "bold"))

        # 输入区
        input_frame = tk.Frame(self.chat_window)
        input_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.user_input = tk.Entry(input_frame, font=("Arial", 10))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.user_input.bind("<Return>", lambda e: self.send_chat_message())

        tk.Button(input_frame, text="发送", command=self.send_chat_message, bg="#4a90e2", fg="white").pack(
            side=tk.RIGHT)

        # 检查API密钥是否有效
        if self.API_KEY == "your_api_key_here" or not self.API_KEY:
            self.add_chat_message("system", "⚠️ 错误：API密钥未配置！")
            self.add_chat_message("system", "请在代码中设置有效的DeepSeek API密钥")
        else:
            self.add_chat_message("assistant", "Bonjour!我是你的Bonbonheur小猫，和我一起聊天吧~")

        # 将窗口添加到次级窗口列表（不设置自动关闭）
        self.secondary_windows.append(self.chat_window)

    def send_chat_message(self, event=None):
        """发送消息到DeepSeek API"""
        message = self.user_input.get().strip()
        if not message:
            return

        # 分析用户情绪
        emotion, confidence = self.analyze_user_emotion(message)
        print(f"🎭 用户情绪分析: {emotion} (置信度: {confidence:.2f})")

        self.add_chat_message("user", message)
        self.user_input.delete(0, tk.END)

        # 检查API密钥
        if self.API_KEY == "your_api_key_here" or not self.API_KEY:
            self.add_chat_message("assistant", "抱歉，聊天功能不可用，请检查API密钥配置")
            return

        # 调用DeepSeek API
        threading.Thread(target=self.call_deepseek_api, args=(message,), daemon=True).start()

        # 添加"思考中"提示
        self.add_chat_message("assistant", "思考中...")

    def call_deepseek_api(self, message):
        """调用DeepSeek API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.API_KEY}",
                "Content-Type": "application/json"
            }

            # 维护对话上下文
            self.chat_history.append({"role": "user", "content": message})

            # 限制历史记录长度（5轮对话）
            if len(self.chat_history) > 5:
                self.chat_history = self.chat_history[-5:]

            # 添加系统提示，包含情绪感知
            system_prompt = f"""你是智能桌宠助手，回复要简洁有帮助，帮助用户管理日程和解答问题。
当前用户情绪状态：{self.current_mood}（置信度：{self.mood_confidence:.2f}）
请根据用户的情绪状态调整回复的语气和建议。如果用户心情不好，要更加关怀和鼓励。"""

            messages = [{
                "role": "system",
                "content": system_prompt
            }] + self.chat_history

            data = {
                "model": "deepseek-chat",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024
            }

            response = requests.post(
                self.DEEPSEEK_API_URL,
                headers=headers,
                json=data,
                timeout=30
            )

            # 专门处理401未授权错误
            if response.status_code == 401:
                raise PermissionError("API密钥无效，请检查密钥是否正确")

            response.raise_for_status()

            result = response.json()
            ai_reply = result['choices'][0]['message']['content']
            self.chat_history.append({"role": "assistant", "content": ai_reply})

            # 更新显示
            self.root.after(0, lambda: self.finalize_ai_response(ai_reply))

        except Exception as e:
            error_msg = f"API调用失败: {str(e)}"
            self.root.after(0, lambda: self.finalize_ai_response(error_msg))

    def finalize_ai_response(self, ai_reply):
        """更新聊天显示"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("end-3l", "end-1c")
        self.add_chat_message("assistant", ai_reply)
        self.chat_display.config(state=tk.DISABLED)

    def add_chat_message(self, role, message):
        """添加消息到聊天窗口"""
        self.chat_display.config(state=tk.NORMAL)
        if role == "assistant":
            self.chat_display.insert(tk.END, "助手: ", "assistant")
        elif role == "user":
            self.chat_display.insert(tk.END, "你: ", "user")
        elif role == "system":
            self.chat_display.insert(tk.END, "系统: ", "system")

        self.chat_display.insert(tk.END, f"{message}\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def create_french_chat_window(self, menu):
        """创建法语聊天窗口"""
        menu.destroy()
        if hasattr(self, 'french_chat_window') and self.french_chat_window and self.french_chat_window.winfo_exists():
            self.french_chat_window.lift()
            return

        self.french_chat_window = tk.Toplevel(self.root)
        self.french_chat_window.title("Discuter en français")
        self.french_chat_window.geometry("500x600")

        # 聊天显示区
        self.french_chat_display = scrolledtext.ScrolledText(
            self.french_chat_window,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.french_chat_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.french_chat_display.tag_config("assistant", foreground="#4a90e2", font=("Arial", 10, "bold"))
        self.french_chat_display.tag_config("user", foreground="#e67e22", font=("Arial", 10, "bold"))
        self.french_chat_display.tag_config("translation", foreground="#27ae60", font=("Arial", 9, "italic"))

        # 输入区
        input_frame = tk.Frame(self.french_chat_window)
        input_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.french_user_input = tk.Entry(input_frame, font=("Arial", 10))
        self.french_user_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.french_user_input.bind("<Return>", lambda e: self.send_french_message())

        tk.Button(input_frame, text="Envoyer", command=self.send_french_message, bg="#4a90e2", fg="white").pack(
            side=tk.RIGHT)

        # 初始化翻译器
        self.translator = Translator()
        
        # 添加欢迎消息
        self.add_french_chat_message("assistant", "Bonjour! Je suis là pour discuter avec toi en français. Comment vas-tu? 🌟")
        self.add_french_chat_message("translation", "(你好！我在这里和你用法语聊天。你好吗？)")

        # 将窗口添加到次级窗口列表（不设置自动关闭）
        self.secondary_windows.append(self.french_chat_window)

    def send_french_message(self):
        """处理法语聊天消息"""
        message = self.french_user_input.get().strip()
        if not message:
            return

        self.french_user_input.delete(0, tk.END)
        
        # 启动翻译和响应线程
        threading.Thread(target=self.process_french_chat, args=(message,), daemon=True).start()

    def process_french_chat(self, message):
        """处理法语聊天的翻译和响应"""
        try:
            # 使用DeepSeek API进行翻译
            headers = {
                "Authorization": f"Bearer {self.API_KEY}",
                "Content-Type": "application/json"
            }

            # 首先翻译用户输入
            translation_messages = [{
                "role": "system",
                "content": "You are a professional translator. Translate the following text to French. Only output the translation, nothing else."
            }, {
                "role": "user",
                "content": message
            }]

            translation_data = {
                "model": "deepseek-chat",
                "messages": translation_messages,
                "temperature": 0.3,
                "max_tokens": 100
            }

            # 翻译请求
            translation_response = requests.post(
                self.DEEPSEEK_API_URL,
                headers=headers,
                json=translation_data,
                timeout=30
            )

            translation_response.raise_for_status()
            translation_result = translation_response.json()
            user_message_fr = translation_result['choices'][0]['message']['content']

            # 显示用户消息及其翻译
            self.root.after(0, lambda: self.add_french_chat_message("user", user_message_fr))
            if message != user_message_fr:  # 如果原文不是法语
                self.root.after(0, lambda: self.add_french_chat_message("translation", f"(原文: {message})"))

            # 使用DeepSeek API生成法语回复
            chat_messages = [{
                "role": "system",
                "content": "Tu es un ami français chaleureux et sympathique. Réponds de manière simple, informelle et réconfortante. "
                          "Utilise un langage familier et des expressions courantes. Garde tes réponses courtes (2-3 phrases maximum). "
                          "Ajoute des émojis appropriés. Montre de l'empathie et de la bienveillance."
            }, {
                "role": "user",
                "content": user_message_fr
            }]

            chat_data = {
                "model": "deepseek-chat",
                "messages": chat_messages,
                "temperature": 0.7,
                "max_tokens": 150
            }

            chat_response = requests.post(
                self.DEEPSEEK_API_URL,
                headers=headers,
                json=chat_data,
                timeout=30
            )

            chat_response.raise_for_status()
            chat_result = chat_response.json()
            french_reply = chat_result['choices'][0]['message']['content']

            # 如果原文不是法语，使用DeepSeek翻译回原语言
            if message != user_message_fr:
                back_translation_messages = [{
                    "role": "system",
                    "content": "You are a professional translator. Translate the following French text to Chinese. Only output the translation, nothing else."
                }, {
                    "role": "user",
                    "content": french_reply
                }]

                back_translation_data = {
                    "model": "deepseek-chat",
                    "messages": back_translation_messages,
                    "temperature": 0.3,
                    "max_tokens": 100
                }

                back_translation_response = requests.post(
                    self.DEEPSEEK_API_URL,
                    headers=headers,
                    json=back_translation_data,
                    timeout=30
                )

                back_translation_response.raise_for_status()
                back_translation_result = back_translation_response.json()
                translation = back_translation_result['choices'][0]['message']['content']

                self.root.after(0, lambda: self.add_french_chat_message("assistant", french_reply))
                self.root.after(0, lambda: self.add_french_chat_message("translation", f"({translation})"))
            else:
                self.root.after(0, lambda: self.add_french_chat_message("assistant", french_reply))

        except Exception as e:
            error_msg = f"Désolé, une erreur s'est produite: {str(e)}"
            self.root.after(0, lambda: self.add_french_chat_message("assistant", error_msg))

    def add_french_chat_message(self, role, message):
        """添加消息到法语聊天窗口"""
        self.french_chat_display.config(state=tk.NORMAL)
        if role == "assistant":
            self.french_chat_display.insert(tk.END, "🐱: ", "assistant")
        elif role == "user":
            self.french_chat_display.insert(tk.END, "👤: ", "user")
        
        if role in ["assistant", "user"]:
            self.french_chat_display.insert(tk.END, f"{message}\n", role)
        else:  # translation
            self.french_chat_display.insert(tk.END, f"{message}\n\n", "translation")
            
        self.french_chat_display.see(tk.END)
        self.french_chat_display.config(state=tk.DISABLED)

    def save_tasks(self):
        """保存任务到文件（兼容性方法）"""
        try:
            with open("tasks.json", "w", encoding="utf-8") as f:
                json.dump(self.task_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存任务失败: {e}")

    def load_tasks(self):
        """从文件加载任务（兼容性方法）"""
        try:
            if os.path.exists("tasks.json"):
                with open("tasks.json", "r", encoding="utf-8") as f:
                    self.task_data = json.load(f)
                    print(f"已加载{len(self.task_data)}条任务")
        except Exception as e:
            print(f"加载任务失败: {e}")
            self.task_data = []


class EmotionAwareTimeRangeDialog:
    """情绪感知的时间范围优化对话框"""
    
    def __init__(self, parent, calendar_manager, smart_optimizer, optimizer_gui, emotion, confidence, adjustment_factor, mood_message):
        self.parent = parent
        self.calendar_manager = calendar_manager
        self.smart_optimizer = smart_optimizer
        self.optimizer_gui = optimizer_gui
        self.emotion = emotion
        self.confidence = confidence
        self.adjustment_factor = adjustment_factor
        self.mood_message = mood_message
        self.dialog = None
        
    def show(self):
        """显示情绪感知的时间范围选择对话框"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("情绪感知智能优化")
        self.dialog.geometry("450x400")
        self.dialog.resizable(False, False)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # 居中显示
        self.dialog.geometry("+%d+%d" % (
            self.parent.winfo_rootx() + 50,
            self.parent.winfo_rooty() + 50
        ))
        
        # 主框架
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(main_frame, text="🎭 情绪感知智能优化", font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 15))
        
        # 情绪状态显示
        emotion_frame = ttk.LabelFrame(main_frame, text="当前情绪状态", padding=10)
        emotion_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(emotion_frame, text=f"情绪: {self.emotion}", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(emotion_frame, text=f"置信度: {self.confidence:.2f}").pack(anchor=tk.W)
        ttk.Label(emotion_frame, text=f"日程调整因子: {self.adjustment_factor:.2f}").pack(anchor=tk.W)
        
        # 情绪建议
        advice_frame = ttk.LabelFrame(main_frame, text="情绪建议", padding=10)
        advice_frame.pack(fill=tk.X, pady=(0, 15))
        
        advice_label = ttk.Label(advice_frame, text=self.mood_message, wraplength=350, justify=tk.LEFT)
        advice_label.pack(anchor=tk.W)
        
        # 快速选择按钮
        quick_frame = ttk.LabelFrame(main_frame, text="选择优化时间范围", padding=10)
        quick_frame.pack(fill=tk.X, pady=(0, 15))
        
        quick_buttons = [
            ("今日", self.optimize_today),
            ("本周", self.optimize_this_week),
            ("下周", self.optimize_next_week),
            ("本月", self.optimize_this_month)
        ]
        
        for i, (text, command) in enumerate(quick_buttons):
            row = i // 2
            col = i % 2
            btn = ttk.Button(quick_frame, text=text, command=command, width=12)
            btn.grid(row=row, column=col, padx=5, pady=5)
        
        # 自定义范围
        custom_frame = ttk.LabelFrame(main_frame, text="自定义范围", padding=10)
        custom_frame.pack(fill=tk.X, pady=(0, 15))
        
        # 开始日期
        ttk.Label(custom_frame, text="开始日期:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.start_date_var = tk.StringVar(value=datetime.datetime.now().strftime("%Y-%m-%d"))
        start_date_entry = ttk.Entry(custom_frame, textvariable=self.start_date_var, width=15)
        start_date_entry.grid(row=0, column=1, padx=(10, 0), pady=5)
        
        # 结束日期
        ttk.Label(custom_frame, text="结束日期:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.end_date_var = tk.StringVar(value=(datetime.datetime.now() + datetime.timedelta(days=7)).strftime("%Y-%m-%d"))
        end_date_entry = ttk.Entry(custom_frame, textvariable=self.end_date_var, width=15)
        end_date_entry.grid(row=1, column=1, padx=(10, 0), pady=5)
        
        # 自定义优化按钮
        custom_btn = ttk.Button(custom_frame, text="优化自定义范围", command=self.optimize_custom_range)
        custom_btn.grid(row=2, column=0, columnspan=2, pady=10)
        
        # 关闭按钮
        close_btn = ttk.Button(main_frame, text="关闭", command=self.dialog.destroy)
        close_btn.pack(pady=10)
        
    def optimize_today(self):
        """优化今日"""
        today = datetime.date.today()
        self._perform_emotion_optimization(today, today, "今日")
        
    def optimize_this_week(self):
        """优化本周"""
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=today.weekday())
        end_date = start_date + datetime.timedelta(days=6)
        self._perform_emotion_optimization(start_date, end_date, "本周")
        
    def optimize_next_week(self):
        """优化下周"""
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=today.weekday()) + datetime.timedelta(days=7)
        end_date = start_date + datetime.timedelta(days=6)
        self._perform_emotion_optimization(start_date, end_date, "下周")
        
    def optimize_this_month(self):
        """优化本月"""
        today = datetime.date.today()
        start_date = today.replace(day=1)
        next_month = start_date.replace(month=start_date.month + 1) if start_date.month < 12 else start_date.replace(year=start_date.year + 1, month=1)
        end_date = next_month - datetime.timedelta(days=1)
        self._perform_emotion_optimization(start_date, end_date, "本月")
        
    def optimize_custom_range(self):
        """优化自定义范围"""
        try:
            start_date = datetime.datetime.strptime(self.start_date_var.get(), "%Y-%m-%d").date()
            end_date = datetime.datetime.strptime(self.end_date_var.get(), "%Y-%m-%d").date()
            
            if start_date > end_date:
                messagebox.showerror("错误", "开始日期不能晚于结束日期")
                return
                
            if (end_date - start_date).days > 90:
                messagebox.showwarning("警告", "时间范围过长，建议不超过90天")
                return
                
            range_text = f"{start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}"
            self._perform_emotion_optimization(start_date, end_date, range_text)
            
        except ValueError:
            messagebox.showerror("错误", "日期格式错误，请使用 YYYY-MM-DD 格式")
            
    def _perform_emotion_optimization(self, start_date, end_date, range_description):
        """执行情绪感知的时间范围优化"""
        try:
            # 收集时间范围内的所有可用时间段
            all_available_slots = []
            current_date = start_date
            
            while current_date <= end_date:
                daily_slots = self.smart_optimizer.find_available_time_slots(self.calendar_manager, current_date)
                all_available_slots.extend(daily_slots)
                current_date += datetime.timedelta(days=1)
            
            if not all_available_slots:
                messagebox.showinfo("信息", f"{range_description}没有可用的时间段进行优化")
                return
            
            # 根据情绪调整时间段数量
            original_count = len(all_available_slots)
            adjusted_count = int(original_count * self.adjustment_factor)
            
            if adjusted_count < original_count:
                # 保留最重要的时间段
                all_available_slots = all_available_slots[:adjusted_count]
                print(f"🎭 根据情绪调整：从{original_count}个时间段减少到{adjusted_count}个")
            
            # 分配时间
            allocated_slots = self.smart_optimizer.allocate_time_slots(all_available_slots)
            
            # 显示情绪感知优化结果
            self.show_emotion_optimization_result(start_date, end_date, range_description, allocated_slots, original_count, adjusted_count)
            
            # 关闭对话框
            self.dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("错误", f"优化过程中发生错误: {str(e)}")
            
    def show_emotion_optimization_result(self, start_date, end_date, range_description, allocated_slots, original_count, adjusted_count):
        """显示情绪感知优化结果"""
        result_dialog = tk.Toplevel(self.parent)
        result_dialog.title(f"情绪感知优化结果 - {range_description}")
        result_dialog.geometry("650x600")
        result_dialog.transient(self.parent)
        
        # 主框架
        main_frame = ttk.Frame(result_dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(main_frame, text=f"🎭 情绪感知优化结果 - {range_description}", font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 15))
        
        # 情绪调整信息
        emotion_info_frame = ttk.LabelFrame(main_frame, text="情绪调整信息", padding=10)
        emotion_info_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(emotion_info_frame, text=f"当前情绪: {self.emotion} (置信度: {self.confidence:.2f})").pack(anchor=tk.W)
        ttk.Label(emotion_info_frame, text=f"调整因子: {self.adjustment_factor:.2f}").pack(anchor=tk.W)
        ttk.Label(emotion_info_frame, text=f"原始时间段: {original_count} → 调整后: {adjusted_count}").pack(anchor=tk.W)
        
        # 统计信息
        stats_frame = ttk.LabelFrame(main_frame, text="优化统计", padding=10)
        stats_frame.pack(fill=tk.X, pady=(0, 15))
        
        total_slots = len(allocated_slots)
        total_hours = sum(slot.duration_hours for slot in allocated_slots)
        
        ttk.Label(stats_frame, text=f"优化时间段数量: {total_slots}").pack(anchor=tk.W)
        ttk.Label(stats_frame, text=f"总优化时间: {total_hours:.1f} 小时").pack(anchor=tk.W)
        ttk.Label(stats_frame, text=f"时间范围: {start_date} 至 {end_date}").pack(anchor=tk.W)
        
        # 结果列表
        list_frame = ttk.LabelFrame(main_frame, text="优化安排详情", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # 创建树形视图
        columns = ("日期", "时间", "项目", "预期专注度", "优先级")
        tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=10)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 填充数据
        for slot in allocated_slots:
            date_str = slot.start_time.strftime("%Y-%m-%d")
            time_str = f"{slot.start_time.strftime('%H:%M')}-{slot.end_time.strftime('%H:%M')}"
            project_name = slot.assigned_project if slot.assigned_project else "未分配"
            focus_score = f"{slot.focus_score:.2f}" if slot.assigned_project else "N/A"
            
            # 查找项目优先级
            priority = "N/A"
            if slot.assigned_project:
                for goal in self.smart_optimizer.project_goals:
                    if goal.name == slot.assigned_project:
                        priority = f"{goal.priority:.1f}"
                        break
            
            tree.insert("", tk.END, values=(date_str, time_str, project_name, focus_score, priority))
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        # 应用优化按钮
        apply_btn = ttk.Button(button_frame, text="应用优化结果", 
                              command=lambda: self.apply_emotion_optimization_result(allocated_slots, result_dialog))
        apply_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # 关闭按钮
        close_btn = ttk.Button(button_frame, text="关闭", command=result_dialog.destroy)
        close_btn.pack(side=tk.RIGHT)
        
    def apply_emotion_optimization_result(self, allocated_slots, result_dialog):
        """应用情绪感知优化结果到日历"""
        if messagebox.askyesno("确认", "确定要将情绪感知优化结果添加到日历中吗？"):
            added_count = 0
            
            for slot in allocated_slots:
                if slot.assigned_project:
                    # 创建日程描述，包含情绪信息
                    schedule_text = f"[情绪感知] {slot.assigned_project} {slot.start_time.strftime('%H:%M')}-{slot.end_time.strftime('%H:%M')}"
                    
                    # 添加到日历
                    schedule_id = self.calendar_manager.add_schedule_from_text(
                        schedule_text, 
                        slot.start_time.date()
                    )
                    
                    if schedule_id:
                        added_count += 1
            
            messagebox.showinfo("成功", f"已成功添加 {added_count} 个情绪感知优化日程到日历中")
            result_dialog.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    pet = DesktopPet(root)
    root.mainloop()