import calendar
import datetime
import json
import uuid
import re
import os
from typing import Dict, List, Optional
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import jieba
from recurring_schedule import RecurringSchedule, RecurrenceType, RecurringNLPProcessor
from pomodoro_timer import PomodoroTimer, ScheduleReminder
from smart_time_optimizer import SmartTimeOptimizer, SmartOptimizerGUI

class PomodoroCalendarManager:
    """支持番茄钟的日历管理器"""
    
    def __init__(self, data_dir: str = "calendar_data"):
        self.data_dir = data_dir
        self.schedules: Dict[str, Dict[str, RecurringSchedule]] = {}
        self.recurring_nlp_processor = RecurringNLPProcessor()
        
        # 创建数据目录
        os.makedirs(data_dir, exist_ok=True)
        
        # 加载所有月份的数据
        self.load_all_schedules()
    
    def _get_month_key(self, year: int, month: int) -> str:
        """获取月份键值"""
        return f"{year}-{month:02d}"
    
    def _get_schedules_file(self, year: int, month: int) -> str:
        """获取指定年月的日程文件路径"""
        month_key = self._get_month_key(year, month)
        return os.path.join(self.data_dir, f"schedules_{month_key}.json")
    
    def add_schedule_from_text(self, text: str, selected_date: datetime.date = None) -> Optional[str]:
        """从自然语言添加日程（支持循环日程）"""
        parsed_data = self.recurring_nlp_processor.parse_recurring_schedule(text, selected_date)
        
        if parsed_data:
            schedule_id = str(uuid.uuid4())
            schedule = RecurringSchedule(
                schedule_id=schedule_id,
                name=parsed_data['name'],
                start_time=parsed_data['start_time'],
                end_time=parsed_data['end_time'],
                content=parsed_data['content'],
                recurrence_type=parsed_data.get('recurrence_type', RecurrenceType.NONE),
                recurrence_interval=parsed_data.get('recurrence_interval', 1),
                recurrence_end_date=parsed_data.get('recurrence_end_date'),
                recurrence_count=parsed_data.get('recurrence_count'),
                completed=False
            )
            
            if schedule.is_recurring():
                self._save_recurring_schedule(schedule)
            else:
                year = schedule.start_time.year
                month = schedule.start_time.month
                month_key = self._get_month_key(year, month)
                
                if month_key not in self.schedules:
                    self.schedules[month_key] = {}
                
                self.schedules[month_key][schedule_id] = schedule
                self.save_month_schedules(year, month)
            
            return schedule_id
        
        return None
    
    def _save_recurring_schedule(self, schedule: RecurringSchedule):
        """保存循环日程到所有相关月份"""
        end_date = schedule.recurrence_end_date or datetime.date.today() + datetime.timedelta(days=365)
        occurrences = schedule.generate_occurrences(
            start_date=schedule.start_time.date(),
            end_date=end_date
        )
        
        months_to_save = set()
        for start_time, end_time in occurrences:
            months_to_save.add((start_time.year, start_time.month))
        
        for year, month in months_to_save:
            month_key = self._get_month_key(year, month)
            if month_key not in self.schedules:
                self.schedules[month_key] = {}
            
            self.schedules[month_key][schedule.schedule_id] = schedule
            self.save_month_schedules(year, month)
    
    def get_schedules_by_date(self, date: datetime.date) -> List[RecurringSchedule]:
        """获取指定日期的日程（包括循环日程的实例）"""
        month_key = self._get_month_key(date.year, date.month)
        if month_key not in self.schedules:
            return []
        
        result_schedules = []
        for schedule in self.schedules[month_key].values():
            if schedule.is_recurring():
                occurrences = schedule.generate_occurrences(date, date)
                if occurrences:
                    for start_time, end_time in occurrences:
                        instance = RecurringSchedule(
                            schedule_id=schedule.schedule_id,
                            name=schedule.name,
                            start_time=start_time,
                            end_time=end_time,
                            content=schedule.content,
                            recurrence_type=schedule.recurrence_type,
                            recurrence_interval=schedule.recurrence_interval,
                            recurrence_end_date=schedule.recurrence_end_date,
                            recurrence_count=schedule.recurrence_count,
                            completed=schedule.completed
                        )
                        result_schedules.append(instance)
            else:
                if schedule.start_time.date() == date:
                    result_schedules.append(schedule)
        
        return sorted(result_schedules, key=lambda x: x.start_time)
    
    def mark_schedule_completed(self, schedule_id: str, date: datetime.date):
        """标记日程为已完成"""
        month_key = self._get_month_key(date.year, date.month)
        if month_key in self.schedules and schedule_id in self.schedules[month_key]:
            self.schedules[month_key][schedule_id].completed = True
            self.save_month_schedules(date.year, date.month)
    
    def save_month_schedules(self, year: int, month: int):
        """保存指定月份的日程到文件"""
        month_key = self._get_month_key(year, month)
        schedules_file = self._get_schedules_file(year, month)
        
        if month_key in self.schedules:
            data = {schedule_id: schedule.to_dict() 
                    for schedule_id, schedule in self.schedules[month_key].items()}
        else:
            data = {}
        
        with open(schedules_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_month_schedules(self, year: int, month: int):
        """从文件加载指定月份的日程"""
        month_key = self._get_month_key(year, month)
        schedules_file = self._get_schedules_file(year, month)
        
        if os.path.exists(schedules_file):
            try:
                with open(schedules_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.schedules[month_key] = {
                    schedule_id: RecurringSchedule.from_dict(schedule_data)
                    for schedule_id, schedule_data in data.items()
                }
            except Exception as e:
                print(f"加载{year}年{month}月日程失败: {e}")
                self.schedules[month_key] = {}
        else:
            self.schedules[month_key] = {}
    
    def load_all_schedules(self):
        """加载所有月份的日程"""
        if os.path.exists(self.data_dir):
            for filename in os.listdir(self.data_dir):
                if filename.startswith('schedules_') and filename.endswith('.json'):
                    month_key = filename[10:-5]
                    try:
                        year, month = map(int, month_key.split('-'))
                        self.load_month_schedules(year, month)
                    except ValueError:
                        continue
    
    def delete_schedule(self, schedule_id: str) -> bool:
        """删除日程"""
        for month_key, month_schedules in self.schedules.items():
            if schedule_id in month_schedules:
                schedule = month_schedules[schedule_id]
                
                if schedule.is_recurring():
                    self._delete_recurring_schedule(schedule_id)
                else:
                    year, month = schedule.start_time.year, schedule.start_time.month
                    del month_schedules[schedule_id]
                    self.save_month_schedules(year, month)
                
                return True
        return False
    
    def _delete_recurring_schedule(self, schedule_id: str):
        """删除循环日程"""
        months_to_update = []
        
        for month_key, month_schedules in self.schedules.items():
            if schedule_id in month_schedules:
                del month_schedules[schedule_id]
                year, month = map(int, month_key.split('-'))
                months_to_update.append((year, month))
        
        for year, month in months_to_update:
            self.save_month_schedules(year, month)

class PomodoroCalendarGUI:
    """集成番茄钟的日历图形界面"""
    
    def __init__(self):
        self.calendar_manager = PomodoroCalendarManager()
        self.root = tk.Tk()
        self.root.title("智能日历应用 - 支持番茄钟和智能优化")
        self.root.geometry("1400x900")
        
        # 番茄钟和提醒服务
        self.pomodoro_timer = PomodoroTimer(self.root)
        self.reminder_service = ScheduleReminder(self.calendar_manager, self.pomodoro_timer)
        
        # 智能时间优化器
        self.smart_optimizer = SmartTimeOptimizer()
        self.optimizer_gui = SmartOptimizerGUI(self.smart_optimizer, self.calendar_manager)
        
        # 当前选中的日期
        self.selected_date = datetime.date.today()
        self.current_year = datetime.datetime.now().year
        self.current_month = datetime.datetime.now().month
        
        # 日历按钮字典
        self.calendar_buttons = {}
        
        self.setup_ui()
        
        # 启动提醒服务
        self.reminder_service.start_reminder_service()
    
    def setup_ui(self):
        """设置用户界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧日历区域
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 日历显示区域
        calendar_frame = ttk.LabelFrame(left_frame, text="日历", padding=10)
        calendar_frame.pack(fill=tk.BOTH, expand=True)
        
        # 年月选择和导航
        nav_frame = ttk.Frame(calendar_frame)
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(nav_frame, text="◀", command=self.prev_month, width=3).pack(side=tk.LEFT)
        
        ttk.Label(nav_frame, text="年份:").pack(side=tk.LEFT, padx=(10, 0))
        self.year_var = tk.StringVar(value=str(self.current_year))
        year_spinbox = ttk.Spinbox(nav_frame, from_=2020, to=2030, textvariable=self.year_var, width=8)
        year_spinbox.pack(side=tk.LEFT, padx=(5, 10))
        year_spinbox.bind('<Return>', self.on_date_change)
        
        ttk.Label(nav_frame, text="月份:").pack(side=tk.LEFT)
        self.month_var = tk.StringVar(value=str(self.current_month))
        month_spinbox = ttk.Spinbox(nav_frame, from_=1, to=12, textvariable=self.month_var, width=8)
        month_spinbox.pack(side=tk.LEFT, padx=(5, 10))
        month_spinbox.bind('<Return>', self.on_date_change)
        
        ttk.Button(nav_frame, text="▶", command=self.next_month, width=3).pack(side=tk.LEFT)
        ttk.Button(nav_frame, text="今天", command=self.go_to_today).pack(side=tk.LEFT, padx=(10, 0))
        
        # 日历网格
        self.calendar_grid_frame = ttk.Frame(calendar_frame)
        self.calendar_grid_frame.pack(fill=tk.BOTH, expand=True)
        
        # 右侧日程管理区域
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 选中日期显示
        date_info_frame = ttk.Frame(right_frame)
        date_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.selected_date_label = ttk.Label(date_info_frame, text="", font=("Arial", 12, "bold"))
        self.selected_date_label.pack()
        
        # 番茄钟控制区域
        pomodoro_frame = ttk.LabelFrame(right_frame, text="番茄钟", padding=10)
        pomodoro_frame.pack(fill=tk.X, pady=(0, 10))
        
        pomodoro_button_frame = ttk.Frame(pomodoro_frame)
        pomodoro_button_frame.pack(fill=tk.X)
        
        ttk.Button(pomodoro_button_frame, text="开始25分钟番茄钟", 
                  command=lambda: self.start_manual_pomodoro(25)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(pomodoro_button_frame, text="开始50分钟专注", 
                  command=lambda: self.start_manual_pomodoro(50)).pack(side=tk.LEFT, padx=(0, 5))
        
        # 智能优化区域
        optimizer_frame = ttk.LabelFrame(right_frame, text="智能时间优化", padding=10)
        optimizer_frame.pack(fill=tk.X, pady=(0, 10))
        
        optimizer_button_frame = ttk.Frame(optimizer_frame)
        optimizer_button_frame.pack(fill=tk.X)
        
        ttk.Button(optimizer_button_frame, text="目标设定与优化", 
                  command=self.show_optimizer_dialog).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(optimizer_button_frame, text="快速优化今日", 
                  command=self.quick_optimize_today).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(optimizer_button_frame, text="专注度统计", 
                  command=self.show_focus_statistics).pack(side=tk.LEFT, padx=(0, 5))
        
        # 日程管理区域
        schedule_frame = ttk.LabelFrame(right_frame, text="日程管理", padding=10)
        schedule_frame.pack(fill=tk.BOTH, expand=True)
        
        # 自然语言输入
        nlp_frame = ttk.Frame(schedule_frame)
        nlp_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(nlp_frame, text="自然语言输入（支持循环日程）:").pack(anchor=tk.W)
        
        example_frame = ttk.Frame(nlp_frame)
        example_frame.pack(fill=tk.X, pady=(5, 5))
        example_text = "示例: '每天早上8点跑步30分钟'"
        ttk.Label(example_frame, text=example_text, font=("Arial", 9), foreground="gray").pack(anchor=tk.W)
        
        self.nlp_entry = ttk.Entry(nlp_frame, width=60)
        self.nlp_entry.pack(fill=tk.X, pady=(5, 5))
        self.nlp_entry.bind('<Return>', lambda e: self.add_schedule_from_nlp())
        
        button_frame = ttk.Frame(nlp_frame)
        button_frame.pack(fill=tk.X)
        ttk.Button(button_frame, text="添加日程", command=self.add_schedule_from_nlp).pack(side=tk.LEFT)
        
        # 日程列表
        list_frame = ttk.Frame(schedule_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        ttk.Label(list_frame, text="当日日程:").pack(anchor=tk.W)
        
        # 创建Treeview
        columns = ('时间', '名称', '类型', '状态', '内容')
        self.schedule_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=12)
        
        self.schedule_tree.heading('时间', text='时间')
        self.schedule_tree.heading('名称', text='名称')
        self.schedule_tree.heading('类型', text='类型')
        self.schedule_tree.heading('状态', text='状态')
        self.schedule_tree.heading('内容', text='内容')
        
        self.schedule_tree.column('时间', width=120)
        self.schedule_tree.column('名称', width=100)
        self.schedule_tree.column('类型', width=80)
        self.schedule_tree.column('状态', width=80)
        self.schedule_tree.column('内容', width=150)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.schedule_tree.yview)
        self.schedule_tree.configure(yscrollcommand=scrollbar.set)
        
        self.schedule_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=(5, 0))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=(5, 0))
        
        # 按钮区域
        button_frame2 = ttk.Frame(schedule_frame)
        button_frame2.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame2, text="开始番茄钟", command=self.start_schedule_pomodoro).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame2, text="标记完成", command=self.mark_schedule_completed).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame2, text="删除日程", command=self.delete_schedule).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame2, text="更新专注度", command=self.update_focus_score).pack(side=tk.LEFT, padx=(0, 5))
        
        # 初始化显示
        self.generate_calendar()
        self.update_selected_date_display()
        self.refresh_schedule_list()
    
    def create_calendar_grid(self):
        """创建日历网格"""
        # 清空现有网格
        for widget in self.calendar_grid_frame.winfo_children():
            widget.destroy()
        
        self.calendar_buttons.clear()
        
        # 星期标题
        weekdays = ['一', '二', '三', '四', '五', '六', '日']
        for i, day in enumerate(weekdays):
            label = ttk.Label(self.calendar_grid_frame, text=day, font=("Arial", 10, "bold"))
            label.grid(row=0, column=i, padx=1, pady=1, sticky="nsew")
        
        # 获取日历数据
        cal = calendar.monthcalendar(self.current_year, self.current_month)
        today = datetime.date.today()
        
        # 创建日期按钮
        for week_num, week in enumerate(cal, 1):
            for day_num, day in enumerate(week):
                if day == 0:
                    label = ttk.Label(self.calendar_grid_frame, text="")
                    label.grid(row=week_num, column=day_num, padx=1, pady=1, sticky="nsew")
                else:
                    date_obj = datetime.date(self.current_year, self.current_month, day)
                    
                    # 检查是否有日程
                    has_schedule = len(self.calendar_manager.get_schedules_by_date(date_obj)) > 0
                    
                    # 设置按钮样式
                    if date_obj < today:
                        style = "Past.TButton"  # 过去的日期用灰色
                    elif date_obj == self.selected_date:
                        style = "Selected.TButton"
                    elif date_obj == today:
                        style = "Today.TButton"
                    elif has_schedule:
                        style = "HasSchedule.TButton"
                    else:
                        style = "TButton"
                    
                    btn = ttk.Button(
                        self.calendar_grid_frame,
                        text=str(day),
                        style=style,
                        command=lambda d=date_obj: self.select_date(d)
                    )
                    btn.grid(row=week_num, column=day_num, padx=1, pady=1, sticky="nsew")
                    self.calendar_buttons[date_obj] = btn
        
        # 配置网格权重
        for i in range(7):
            self.calendar_grid_frame.columnconfigure(i, weight=1)
        for i in range(len(cal) + 1):
            self.calendar_grid_frame.rowconfigure(i, weight=1)
        
        # 配置按钮样式
        style = ttk.Style()
        style.configure("Selected.TButton", background="lightblue")
        style.configure("Today.TButton", background="lightgreen")
        style.configure("HasSchedule.TButton", background="lightyellow")
        style.configure("Past.TButton", background="lightgray", foreground="gray")
    
    def select_date(self, date: datetime.date):
        """选择日期"""
        self.selected_date = date
        self.update_calendar_buttons()
        self.update_selected_date_display()
        self.refresh_schedule_list()
    
    def update_calendar_buttons(self):
        """更新日历按钮样式"""
        today = datetime.date.today()
        
        for date_obj, btn in self.calendar_buttons.items():
            has_schedule = len(self.calendar_manager.get_schedules_by_date(date_obj)) > 0
            
            if date_obj < today:
                btn.configure(style="Past.TButton")
            elif date_obj == self.selected_date:
                btn.configure(style="Selected.TButton")
            elif date_obj == today:
                btn.configure(style="Today.TButton")
            elif has_schedule:
                btn.configure(style="HasSchedule.TButton")
            else:
                btn.configure(style="TButton")
    
    def update_selected_date_display(self):
        """更新选中日期显示"""
        date_str = self.selected_date.strftime("%Y年%m月%d日 (%A)")
        weekdays = {
            'Monday': '星期一', 'Tuesday': '星期二', 'Wednesday': '星期三',
            'Thursday': '星期四', 'Friday': '星期五', 'Saturday': '星期六', 'Sunday': '星期日'
        }
        for en, cn in weekdays.items():
            date_str = date_str.replace(en, cn)
        
        self.selected_date_label.config(text=f"选中日期: {date_str}")
    
    def prev_month(self):
        """上一个月"""
        if self.current_month == 1:
            self.current_month = 12
            self.current_year -= 1
        else:
            self.current_month -= 1
        
        self.year_var.set(str(self.current_year))
        self.month_var.set(str(self.current_month))
        self.generate_calendar()
    
    def next_month(self):
        """下一个月"""
        if self.current_month == 12:
            self.current_month = 1
            self.current_year += 1
        else:
            self.current_month += 1
        
        self.year_var.set(str(self.current_year))
        self.month_var.set(str(self.current_month))
        self.generate_calendar()
    
    def go_to_today(self):
        """跳转到今天"""
        today = datetime.date.today()
        self.current_year = today.year
        self.current_month = today.month
        self.selected_date = today
        
        self.year_var.set(str(self.current_year))
        self.month_var.set(str(self.current_month))
        self.generate_calendar()
    
    def on_date_change(self, event=None):
        """年月改变事件"""
        try:
            self.current_year = int(self.year_var.get())
            self.current_month = int(self.month_var.get())
            self.generate_calendar()
        except ValueError:
            messagebox.showerror("错误", "请输入有效的年份和月份")
    
    def generate_calendar(self):
        """生成并显示日历"""
        try:
            self.calendar_manager.load_month_schedules(self.current_year, self.current_month)
            self.create_calendar_grid()
            
            if self.selected_date.year != self.current_year or self.selected_date.month != self.current_month:
                self.selected_date = datetime.date(self.current_year, self.current_month, 1)
                self.update_selected_date_display()
                self.refresh_schedule_list()
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的年份和月份")
    
    def add_schedule_from_nlp(self):
        """从自然语言添加日程"""
        text = self.nlp_entry.get().strip()
        if not text:
            messagebox.showwarning("警告", "请输入日程描述")
            return
        
        schedule_id = self.calendar_manager.add_schedule_from_text(text, self.selected_date)
        if schedule_id:
            messagebox.showinfo("成功", "日程添加成功！")
            self.nlp_entry.delete(0, tk.END)
            self.refresh_schedule_list()
            self.update_calendar_buttons()
        else:
            messagebox.showerror("错误", "无法解析日程信息，请检查输入格式")
    
    def refresh_schedule_list(self):
        """刷新日程列表"""
        # 清空现有项目
        for item in self.schedule_tree.get_children():
            self.schedule_tree.delete(item)
        
        # 添加选中日期的日程
        schedules = self.calendar_manager.get_schedules_by_date(self.selected_date)
        for schedule in schedules:
            time_str = f"{schedule.start_time.strftime('%H:%M')}-{schedule.end_time.strftime('%H:%M')}"
            
            # 确定日程类型
            if schedule.is_recurring():
                type_str = f"循环({schedule.recurrence_type.value})"
            else:
                type_str = "单次"
            
            # 确定状态
            status_str = "已完成 ⭐" if schedule.completed else "待完成"
            
            self.schedule_tree.insert('', tk.END, values=(
                time_str,
                schedule.name,
                type_str,
                status_str,
                schedule.content
            ), tags=(schedule.schedule_id,))
    
    def start_manual_pomodoro(self, minutes: int):
        """手动开始番茄钟"""
        self.pomodoro_timer.start_pomodoro(f"{minutes}分钟专注", minutes)
    
    def start_schedule_pomodoro(self):
        """为选中的日程开始番茄钟"""
        selected = self.schedule_tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请选择要开始的日程")
            return
        
        schedule_id = self.schedule_tree.item(selected[0])['tags'][0]
        
        # 查找日程
        schedules = self.calendar_manager.get_schedules_by_date(self.selected_date)
        target_schedule = None
        for schedule in schedules:
            if schedule.schedule_id == schedule_id:
                target_schedule = schedule
                break
        
        if not target_schedule:
            messagebox.showerror("错误", "找不到对应的日程")
            return
        
        if target_schedule.completed:
            messagebox.showinfo("提示", "该日程已完成")
            return
        
        # 计算持续时间
        duration_minutes = int((target_schedule.end_time - target_schedule.start_time).total_seconds() / 60)
        
        # 开始番茄钟
        def on_completion():
            self.calendar_manager.mark_schedule_completed(schedule_id, self.selected_date)
            self.refresh_schedule_list()
            self.update_calendar_buttons()
        
        self.pomodoro_timer.start_pomodoro(target_schedule.name, duration_minutes, on_completion)
    
    def mark_schedule_completed(self):
        """手动标记日程为已完成"""
        selected = self.schedule_tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请选择要标记的日程")
            return
        
        schedule_id = self.schedule_tree.item(selected[0])['tags'][0]
        self.calendar_manager.mark_schedule_completed(schedule_id, self.selected_date)
        self.refresh_schedule_list()
        self.update_calendar_buttons()
        messagebox.showinfo("成功", "日程已标记为完成")
    
    def delete_schedule(self):
        """删除日程"""
        selected = self.schedule_tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请选择要删除的日程")
            return
        
        if messagebox.askyesno("确认", "确定要删除选中的日程吗？"):
            schedule_id = self.schedule_tree.item(selected[0])['tags'][0]
            
            if self.calendar_manager.delete_schedule(schedule_id):
                messagebox.showinfo("成功", "日程删除成功")
                self.refresh_schedule_list()
                self.update_calendar_buttons()
            else:
                messagebox.showerror("错误", "删除失败")
    
    def show_optimizer_dialog(self):
        """显示智能优化器对话框"""
        self.optimizer_gui.show_goal_setting_dialog()
    
    def quick_optimize_today(self):
        """快速优化今日时间安排"""
        if not self.smart_optimizer.project_goals:
            messagebox.showwarning("警告", "请先设置项目目标")
            self.show_optimizer_dialog()
            return
        
        today = datetime.date.today()
        
        # 查找可用时间段
        available_slots = self.smart_optimizer.find_available_time_slots(self.calendar_manager, today)
        
        if not available_slots:
            messagebox.showinfo("信息", "今日没有可用的时间段进行优化")
            return
        
        # 分配时间
        allocated_slots = self.smart_optimizer.allocate_time_slots(available_slots)
        
        # 显示优化结果
        self.optimizer_gui.show_optimization_result(today, allocated_slots)
    
    def show_focus_statistics(self):
        """显示专注度统计"""
        self.optimizer_gui.show_statistics()
    
    def update_focus_score(self):
        """更新选中日程的专注度评分（模拟多模态识别系统的输入）"""
        selection = self.schedule_tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请选择要更新专注度的日程")
            return
        
        # 获取选中的日程信息
        item = self.schedule_tree.item(selection[0])
        schedule_name = item['values'][1]  # 名称列
        
        # 弹出对话框输入专注度评分
        score_str = simpledialog.askstring(
            "更新专注度", 
            f"请输入'{schedule_name}'的实际专注度评分 (0.0-1.0):",
            initialvalue="0.8"
        )
        
        if score_str:
            try:
                score = float(score_str)
                if 0.0 <= score <= 1.0:
                    # 查找对应的专注会话并更新
                    for i, session in enumerate(self.smart_optimizer.focus_sessions):
                        if (session.project_name == schedule_name and 
                            session.start_time.date() == self.selected_date and
                            session.actual_focus_score is None):
                            self.smart_optimizer.update_focus_session_result(i, score, True)
                            messagebox.showinfo("成功", f"已更新'{schedule_name}'的专注度评分为{score:.2f}")
                            return
                    
                    # 如果没有找到对应的会话，创建一个新的
                    now = datetime.datetime.now()
                    session = self.smart_optimizer.record_focus_session(
                        schedule_name, now, now + datetime.timedelta(hours=1), 0.8
                    )
                    session_id = len(self.smart_optimizer.focus_sessions) - 1
                    self.smart_optimizer.update_focus_session_result(session_id, score, True)
                    messagebox.showinfo("成功", f"已记录'{schedule_name}'的专注度评分为{score:.2f}")
                else:
                    messagebox.showerror("错误", "专注度评分必须在0.0到1.0之间")
            except ValueError:
                messagebox.showerror("错误", "请输入有效的数字")
    
    def run(self):
        """运行应用"""
        try:
            self.root.mainloop()
        finally:
            # 停止提醒服务
            self.reminder_service.stop_reminder_service()

def main():
    """主函数"""
    app = PomodoroCalendarGUI()
    app.run()

if __name__ == "__main__":
    main() 