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
        
        # 第二行按钮
        optimizer_button_frame2 = ttk.Frame(optimizer_frame)
        optimizer_button_frame2.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(optimizer_button_frame2, text="时间范围优化", 
                  command=self.show_time_range_optimization).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(optimizer_button_frame2, text="专注度统计", 
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
        ttk.Button(button_frame2, text="清空当日", command=self.clear_daily_schedules).pack(side=tk.LEFT, padx=(0, 5))
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
        
        # 直接应用优化结果到日历
        self.apply_quick_optimization_result(today, allocated_slots)
    
    def apply_quick_optimization_result(self, date: datetime.date, allocated_slots):
        """应用快速优化结果到日历"""
        if not messagebox.askyesno("确认应用", f"确定要将优化结果应用到 {date} 的日历中吗？"):
            return
        
        added_count = 0
        failed_count = 0
        
        print(f"🔄 开始应用快速优化结果到日历...")
        
        for slot in allocated_slots:
            if slot.assigned_project:
                # 创建日程文本 - 使用标准格式
                schedule_text = f"{slot.assigned_project} {slot.start_time.strftime('%H:%M')}-{slot.end_time.strftime('%H:%M')}"
                
                # 添加到日历
                schedule_id = self.calendar_manager.add_schedule_from_text(schedule_text, date)
                if schedule_id:
                    added_count += 1
                    print(f"   ✅ {date} {schedule_text}")
                    
                    # 记录专注会话
                    self.smart_optimizer.record_focus_session(
                        slot.assigned_project,
                        slot.start_time,
                        slot.end_time,
                        slot.focus_score
                    )
                else:
                    failed_count += 1
                    print(f"   ❌ {date} {schedule_text}")
        
        # 显示详细结果
        result_message = f"快速优化应用结果:\n✅ 成功: {added_count} 个日程\n"
        if failed_count > 0:
            result_message += f"❌ 失败: {failed_count} 个日程\n"
        result_message += f"📊 成功率: {added_count/(added_count+failed_count)*100:.1f}%" if (added_count+failed_count) > 0 else "N/A"
        
        messagebox.showinfo("应用完成", result_message)
        
        # 强制刷新界面
        self.force_refresh_interface()
        
        print(f"✅ 快速优化结果应用完成: {added_count} 个日程已保存到日历")
    
    def show_time_range_optimization(self):
        """显示时间范围优化对话框"""
        if not self.smart_optimizer.project_goals:
            messagebox.showwarning("警告", "请先设置项目目标")
            self.show_optimizer_dialog()
            return
        
        # 创建时间范围选择对话框
        range_dialog = TimeRangeOptimizeDialog(self.root, self.calendar_manager, self.smart_optimizer, self.optimizer_gui, self)
        range_dialog.show()
    
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
    
    def force_refresh_interface(self):
        """强制刷新整个界面（供外部调用）"""
        try:
            print("🔄 执行强制界面刷新...")
            
            # 刷新日程列表
            self.refresh_schedule_list()
            print("   ✅ 日程列表已刷新")
            
            # 刷新日历按钮
            self.update_calendar_buttons()
            print("   ✅ 日历按钮已刷新")
            
            # 重新生成当前月份的日历
            self.generate_calendar()
            print("   ✅ 日历已重新生成")
            
            # 更新选中日期显示
            self.update_selected_date_display()
            print("   ✅ 日期显示已更新")
            
            print("✅ 界面强制刷新完成")
            return True
            
        except Exception as e:
            print(f"❌ 强制刷新界面失败: {e}")
            return False
    
    def run(self):
        """运行应用"""
        try:
            self.root.mainloop()
        finally:
            # 停止提醒服务
            self.reminder_service.stop_reminder_service()
    
    def clear_daily_schedules(self):
        """清空当日所有日程"""
        if not messagebox.askyesno("确认清空", f"确定要清空 {self.selected_date} 的所有日程吗？\n\n此操作不可撤销！"):
            return
        
        # 获取当日所有日程
        schedules = self.calendar_manager.get_schedules_by_date(self.selected_date)
        
        if not schedules:
            messagebox.showinfo("信息", "当日没有日程需要清空")
            return
        
        # 删除所有日程
        deleted_count = 0
        failed_count = 0
        
        print(f"🗑️ 开始清空 {self.selected_date} 的日程...")
        
        for schedule in schedules:
            try:
                if self.calendar_manager.delete_schedule(schedule.schedule_id):
                    deleted_count += 1
                    print(f"   ✅ 已删除: {schedule.name}")
                else:
                    failed_count += 1
                    print(f"   ❌ 删除失败: {schedule.name}")
            except Exception as e:
                failed_count += 1
                print(f"   ❌ 删除失败: {schedule.name} - {e}")
        
        # 显示结果
        result_message = f"清空结果:\n✅ 成功删除: {deleted_count} 个日程"
        if failed_count > 0:
            result_message += f"\n❌ 删除失败: {failed_count} 个日程"
        
        messagebox.showinfo("清空完成", result_message)
        
        # 刷新界面
        self.refresh_schedule_list()
        self.update_calendar_buttons()
        
        print(f"✅ 日程清空完成: 删除了 {deleted_count} 个日程")

class TimeRangeOptimizeDialog:
    """时间范围优化对话框"""
    
    def __init__(self, parent, calendar_manager, smart_optimizer, optimizer_gui, main_gui=None):
        self.parent = parent
        self.calendar_manager = calendar_manager
        self.smart_optimizer = smart_optimizer
        self.optimizer_gui = optimizer_gui
        self.main_gui = main_gui  # 主界面引用
        self.dialog = None
        
    def show(self):
        """显示时间范围选择对话框"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("智能时间范围优化")
        self.dialog.geometry("400x300")
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
        title_label = ttk.Label(main_frame, text="选择优化时间范围", font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # 快速选择按钮
        quick_frame = ttk.LabelFrame(main_frame, text="快速选择", padding=10)
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
        self.start_date_var = tk.StringVar(value=datetime.date.today().strftime("%Y-%m-%d"))
        start_date_entry = ttk.Entry(custom_frame, textvariable=self.start_date_var, width=15)
        start_date_entry.grid(row=0, column=1, padx=(10, 0), pady=5)
        
        # 结束日期
        ttk.Label(custom_frame, text="结束日期:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.end_date_var = tk.StringVar(value=(datetime.date.today() + datetime.timedelta(days=7)).strftime("%Y-%m-%d"))
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
        self._perform_optimization(today, today, "今日")
        
    def optimize_this_week(self):
        """优化本周"""
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=today.weekday())  # 本周一
        end_date = start_date + datetime.timedelta(days=6)  # 本周日
        self._perform_optimization(start_date, end_date, "本周")
        
    def optimize_next_week(self):
        """优化下周"""
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=today.weekday()) + datetime.timedelta(days=7)  # 下周一
        end_date = start_date + datetime.timedelta(days=6)  # 下周日
        self._perform_optimization(start_date, end_date, "下周")
        
    def optimize_this_month(self):
        """优化本月"""
        today = datetime.date.today()
        start_date = today.replace(day=1)  # 本月第一天
        next_month = start_date.replace(month=start_date.month + 1) if start_date.month < 12 else start_date.replace(year=start_date.year + 1, month=1)
        end_date = next_month - datetime.timedelta(days=1)  # 本月最后一天
        self._perform_optimization(start_date, end_date, "本月")
        
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
            self._perform_optimization(start_date, end_date, range_text)
            
        except ValueError:
            messagebox.showerror("错误", "日期格式错误，请使用 YYYY-MM-DD 格式")
            
    def _perform_optimization(self, start_date, end_date, range_description):
        """执行时间范围优化"""
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
            
            # 分配时间
            allocated_slots = self.smart_optimizer.allocate_time_slots(all_available_slots)
            
            # 统计分配结果
            assigned_slots = [slot for slot in allocated_slots if slot.assigned_project]
            
            if not assigned_slots:
                messagebox.showinfo("信息", f"{range_description}没有找到合适的时间分配方案")
                return
            
            # 询问用户是否直接应用或查看详细结果
            choice = messagebox.askyesnocancel(
                "优化完成", 
                f"时间范围优化完成！\n\n"
                f"📊 优化统计:\n"
                f"   时间范围: {range_description}\n"
                f"   找到 {len(assigned_slots)} 个时间段\n"
                f"   总时长: {sum(slot.duration_hours for slot in assigned_slots):.1f} 小时\n\n"
                f"选择操作:\n"
                f"✅ 是 - 直接应用到日历\n"
                f"📋 否 - 查看详细结果再决定\n"
                f"❌ 取消 - 放弃优化结果"
            )
            
            if choice is True:
                # 直接应用优化结果
                self.apply_range_optimization_directly(allocated_slots, range_description)
            elif choice is False:
                # 显示详细优化结果
                self.show_range_optimization_result(start_date, end_date, range_description, allocated_slots)
            # choice is None 表示取消，不做任何操作
            
            # 关闭对话框
            self.dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("错误", f"优化过程中发生错误: {str(e)}")
    
    def apply_range_optimization_directly(self, allocated_slots, range_description):
        """直接应用时间范围优化结果"""
        added_count = 0
        failed_count = 0
        
        print(f"🔄 开始直接应用{range_description}优化结果到日历...")
        
        for slot in allocated_slots:
            if slot.assigned_project:
                # 创建日程描述
                schedule_text = f"{slot.assigned_project} {slot.start_time.strftime('%H:%M')}-{slot.end_time.strftime('%H:%M')}"
                
                # 添加到日历
                schedule_id = self.calendar_manager.add_schedule_from_text(
                    schedule_text, 
                    slot.start_time.date()
                )
                
                if schedule_id:
                    added_count += 1
                    print(f"   ✅ {slot.start_time.date()} {schedule_text}")
                    
                    # 记录专注会话
                    self.smart_optimizer.record_focus_session(
                        slot.assigned_project,
                        slot.start_time,
                        slot.end_time,
                        slot.focus_score
                    )
                else:
                    failed_count += 1
                    print(f"   ❌ {slot.start_time.date()} {schedule_text}")
        
        # 显示详细结果
        result_message = f"{range_description}优化直接应用结果:\n✅ 成功: {added_count} 个日程\n"
        if failed_count > 0:
            result_message += f"❌ 失败: {failed_count} 个日程\n"
        result_message += f"📊 成功率: {added_count/(added_count+failed_count)*100:.1f}%" if (added_count+failed_count) > 0 else "N/A"
        
        messagebox.showinfo("应用完成", result_message)
        
        # 强制刷新主界面
        self._refresh_main_interface()
        
        print(f"✅ {range_description}优化结果直接应用完成: {added_count} 个日程已保存")
    
    def show_range_optimization_result(self, start_date, end_date, range_description, allocated_slots):
        """显示时间范围优化结果"""
        result_dialog = tk.Toplevel(self.parent)
        result_dialog.title(f"时间优化结果 - {range_description}")
        result_dialog.geometry("700x600")
        result_dialog.transient(self.parent)
        
        # 主框架
        main_frame = ttk.Frame(result_dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(main_frame, text=f"时间优化结果 - {range_description}", font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 15))
        
        # 统计信息
        stats_frame = ttk.LabelFrame(main_frame, text="优化统计", padding=10)
        stats_frame.pack(fill=tk.X, pady=(0, 15))
        
        # 计算统计信息
        total_slots = len(allocated_slots)
        assigned_slots = [slot for slot in allocated_slots if slot.assigned_project]
        total_hours = sum(slot.duration_hours for slot in allocated_slots)
        assigned_hours = sum(slot.duration_hours for slot in assigned_slots)
        
        # 按日期分组统计
        daily_stats = {}
        for slot in assigned_slots:
            date_key = slot.start_time.date()
            if date_key not in daily_stats:
                daily_stats[date_key] = {'count': 0, 'hours': 0, 'projects': set()}
            daily_stats[date_key]['count'] += 1
            daily_stats[date_key]['hours'] += slot.duration_hours
            daily_stats[date_key]['projects'].add(slot.assigned_project)
        
        # 显示总体统计
        ttk.Label(stats_frame, text=f"时间范围: {start_date} 至 {end_date}").pack(anchor=tk.W)
        ttk.Label(stats_frame, text=f"总时间段: {total_slots} 个，已分配: {len(assigned_slots)} 个").pack(anchor=tk.W)
        ttk.Label(stats_frame, text=f"总时间: {total_hours:.1f} 小时，已分配: {assigned_hours:.1f} 小时").pack(anchor=tk.W)
        ttk.Label(stats_frame, text=f"时间利用率: {assigned_hours/total_hours*100:.1f}%" if total_hours > 0 else "时间利用率: 0%").pack(anchor=tk.W)
        
        # 按日期统计
        if daily_stats:
            ttk.Label(stats_frame, text="", font=("Arial", 8)).pack()  # 空行
            ttk.Label(stats_frame, text="📅 每日分配统计:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
            for date_key in sorted(daily_stats.keys()):
                stats = daily_stats[date_key]
                projects_text = ", ".join(list(stats['projects'])[:3])  # 最多显示3个项目
                if len(stats['projects']) > 3:
                    projects_text += "..."
                ttk.Label(stats_frame, text=f"  {date_key}: {stats['count']}个时段, {stats['hours']:.1f}小时 ({projects_text})").pack(anchor=tk.W)
        
        # 结果列表
        list_frame = ttk.LabelFrame(main_frame, text="优化安排详情", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # 创建树形视图
        columns = ("日期", "时间", "项目", "预期专注度", "优先级")
        tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=12)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 按日期排序填充数据
        sorted_slots = sorted(allocated_slots, key=lambda x: (x.start_time.date(), x.start_time.time()))
        
        for slot in sorted_slots:
            if slot.assigned_project:  # 只显示已分配的时间段
                date_str = slot.start_time.strftime("%Y-%m-%d")
                time_str = f"{slot.start_time.strftime('%H:%M')}-{slot.end_time.strftime('%H:%M')}"
                project_name = slot.assigned_project
                focus_score = f"{slot.focus_score:.2f}"
                
                # 查找项目优先级
                priority = "N/A"
                for goal in self.smart_optimizer.project_goals:
                    if goal.name == slot.assigned_project:
                        priority = f"{goal.priority:.2f}"
                        break
                
                tree.insert("", tk.END, values=(date_str, time_str, project_name, focus_score, priority))
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        # 应用优化按钮
        apply_btn = ttk.Button(button_frame, text="应用优化结果", 
                              command=lambda: self.apply_optimization_result(allocated_slots, result_dialog))
        apply_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # 关闭按钮
        close_btn = ttk.Button(button_frame, text="关闭", command=result_dialog.destroy)
        close_btn.pack(side=tk.RIGHT)
        
    def apply_optimization_result(self, allocated_slots, result_dialog):
        """应用优化结果到日历"""
        if messagebox.askyesno("确认", "确定要将优化结果添加到日历中吗？"):
            added_count = 0
            failed_count = 0
            
            print("🔄 开始应用时间范围优化结果到日历...")
            
            for slot in allocated_slots:
                if slot.assigned_project:
                    # 创建日程描述
                    schedule_text = f"{slot.assigned_project} {slot.start_time.strftime('%H:%M')}-{slot.end_time.strftime('%H:%M')}"
                    
                    # 添加到日历
                    schedule_id = self.calendar_manager.add_schedule_from_text(
                        schedule_text, 
                        slot.start_time.date()
                    )
                    
                    if schedule_id:
                        added_count += 1
                        print(f"   ✅ {slot.start_time.date()} {schedule_text}")
                        
                        # 记录专注会话
                        self.smart_optimizer.record_focus_session(
                            slot.assigned_project,
                            slot.start_time,
                            slot.end_time,
                            slot.focus_score
                        )
                    else:
                        failed_count += 1
                        print(f"   ❌ {slot.start_time.date()} {schedule_text}")
            
            # 显示详细结果
            result_message = f"时间范围优化应用结果:\n✅ 成功: {added_count} 个日程\n"
            if failed_count > 0:
                result_message += f"❌ 失败: {failed_count} 个日程\n"
            result_message += f"📊 成功率: {added_count/(added_count+failed_count)*100:.1f}%" if (added_count+failed_count) > 0 else "N/A"
            
            messagebox.showinfo("应用完成", result_message)
            result_dialog.destroy()
            
            # 强制刷新主界面 - 改进版本
            self._refresh_main_interface()
            
            print(f"✅ 时间范围优化结果应用完成: {added_count} 个日程已保存")
    
    def _refresh_main_interface(self):
        """刷新主界面（改进版本）"""
        try:
            print("🔄 开始刷新主界面...")
            
            # 优先使用传入的主界面引用
            main_gui = self.main_gui
            
            # 如果没有主界面引用，尝试查找
            if not main_gui:
                main_gui = self._find_main_gui()
            
            if main_gui:
                print("   📍 找到主界面，开始刷新...")
                
                # 使用主界面的强制刷新方法
                if hasattr(main_gui, 'force_refresh_interface'):
                    success = main_gui.force_refresh_interface()
                    if success:
                        print("✅ 主界面刷新完成")
                        messagebox.showinfo("刷新完成", "日历界面已更新，请查看新添加的日程")
                        return
                
                # 备用方案：逐个调用刷新方法
                print("   🔄 使用备用刷新方案...")
                
                # 刷新日程列表
                if hasattr(main_gui, 'refresh_schedule_list'):
                    main_gui.refresh_schedule_list()
                    print("   ✅ 日程列表已刷新")
                
                # 刷新日历按钮
                if hasattr(main_gui, 'update_calendar_buttons'):
                    main_gui.update_calendar_buttons()
                    print("   ✅ 日历按钮已刷新")
                
                # 重新生成日历（如果需要）
                if hasattr(main_gui, 'generate_calendar'):
                    main_gui.generate_calendar()
                    print("   ✅ 日历已重新生成")
                
                print("✅ 主界面刷新完成")
                messagebox.showinfo("刷新完成", "日历界面已更新，请查看新添加的日程")
                
            else:
                print("⚠️ 未找到主界面，尝试替代方案...")
                
                # 替代方案：提示用户手动刷新
                messagebox.showinfo("提示", 
                    "日程已成功保存到日历！\n\n"
                    "如果没有立即显示，请尝试:\n"
                    "1. 切换到其他月份再切换回来\n"
                    "2. 点击'今天'按钮\n"
                    "3. 重新启动程序")
                
        except Exception as e:
            print(f"⚠️ 刷新主界面时出错: {e}")
            
            # 即使刷新失败，也要告知用户日程已保存
            messagebox.showinfo("保存成功", 
                f"日程已成功保存！\n\n"
                f"如果界面未更新，请手动刷新:\n"
                f"- 切换月份或点击'今天'按钮\n"
                f"- 或重新启动程序查看")
    
    def _find_main_gui(self):
        """查找主界面实例"""
        # 方法1: 通过parent层级查找
        current = self.parent
        while current:
            if hasattr(current, 'refresh_schedule_list') and hasattr(current, 'update_calendar_buttons'):
                return current
            current = getattr(current, 'master', None) or getattr(current, 'parent', None)
        
        # 方法2: 通过parent的子组件查找
        if hasattr(self.parent, 'winfo_children'):
            for widget in self.parent.winfo_children():
                if hasattr(widget, 'refresh_schedule_list') and hasattr(widget, 'update_calendar_buttons'):
                    return widget
        
        # 方法3: 通过parent的master查找
        if hasattr(self.parent, 'master') and hasattr(self.parent.master, 'winfo_children'):
            for widget in self.parent.master.winfo_children():
                if hasattr(widget, 'refresh_schedule_list') and hasattr(widget, 'update_calendar_buttons'):
                    return widget
        
        # 方法4: 通过全局变量查找（如果有的话）
        import tkinter as tk
        for widget in tk._default_root.winfo_children() if tk._default_root else []:
            if hasattr(widget, 'refresh_schedule_list') and hasattr(widget, 'update_calendar_buttons'):
                return widget
        
        return None

def main():
    """主函数"""
    app = PomodoroCalendarGUI()
    app.run()

if __name__ == "__main__":
    main() 