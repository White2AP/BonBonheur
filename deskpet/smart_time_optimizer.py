#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能时间优化模块
支持目标设定、时间分配和专注度优化
"""

import datetime
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

class FocusType(Enum):
    """专注类型枚举"""
    EXTREMELY_FOCUSED = "极为专注"
    MODERATELY_FOCUSED = "较为专注"
    EASILY_DISTRACTED = "容易分神"

@dataclass
class ProjectGoal:
    """项目目标数据类"""
    name: str
    target_score: float
    current_score: float
    _manual_priority: float = None  # 手动设置的优先级（可选）
    
    @property
    def score_gap(self) -> float:
        """分数差距"""
        return max(0, self.target_score - self.current_score)
    
    @property
    def improvement_ratio(self) -> float:
        """改进比例"""
        if self.target_score == 0:
            return 0
        return self.score_gap / self.target_score
    
    @property
    def priority(self) -> float:
        """自动计算的优先级（基于分数差距和改进比例）"""
        if self._manual_priority is not None:
            return self._manual_priority
        
        # 自动计算优先级：分数差距越大，改进比例越高，优先级越高
        # 基础优先级 = 分数差距 * 改进比例
        base_priority = self.score_gap * self.improvement_ratio
        
        # 归一化到0.1-1.0范围
        normalized_priority = max(0.1, min(1.0, base_priority / 100.0 + 0.1))
        
        return normalized_priority
    
    def set_manual_priority(self, priority: float):
        """手动设置优先级（可选）"""
        self._manual_priority = priority
    
    def use_auto_priority(self):
        """使用自动计算的优先级"""
        self._manual_priority = None

@dataclass
class TimeSlot:
    """时间段数据类"""
    start_time: datetime.datetime
    end_time: datetime.datetime
    duration_minutes: int
    is_available: bool = True
    assigned_project: str = None
    focus_score: float = 1.0  # 专注度评分 (0-1)
    
    @property
    def duration_hours(self) -> float:
        """时长（小时）"""
        return self.duration_minutes / 60

@dataclass
class FocusSession:
    """专注会话记录"""
    project_name: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    planned_focus_score: float
    actual_focus_score: float = None  # 来自多模态识别系统
    completed: bool = False

class SmartTimeOptimizer:
    """智能时间优化器"""
    
    def __init__(self, data_dir: str = "calendar_data"):
        self.data_dir = data_dir
        self.goals_file = os.path.join(data_dir, "project_goals.json")
        self.focus_sessions_file = os.path.join(data_dir, "focus_sessions.json")
        self.optimizer_config_file = os.path.join(data_dir, "optimizer_config.json")
        
        # 默认配置
        self.default_focus_duration = 60  # 默认最佳专注时长（分钟）
        self.focus_type = FocusType.MODERATELY_FOCUSED
        self.work_start_hour = 8  # 工作开始时间
        self.work_end_hour = 22   # 工作结束时间
        self.min_slot_duration = 60  # 最小时间段（分钟）
        
        # 专注类型对应的时间段调整系数
        self.focus_type_coefficients = {
            FocusType.EXTREMELY_FOCUSED: {
                'optimal_duration': 90,
                'min_duration': 60,
                'break_ratio': 0.1
            },
            FocusType.MODERATELY_FOCUSED: {
                'optimal_duration': 60,
                'min_duration': 45,
                'break_ratio': 0.15
            },
            FocusType.EASILY_DISTRACTED: {
                'optimal_duration': 45,
                'min_duration': 30,
                'break_ratio': 0.2
            }
        }
        
        self.project_goals: List[ProjectGoal] = []
        self.focus_sessions: List[FocusSession] = []
        
        # 创建数据目录
        os.makedirs(data_dir, exist_ok=True)
        
        # 加载数据
        self.load_data()
    
    def load_data(self):
        """加载所有数据"""
        self.load_project_goals()
        self.load_focus_sessions()
        self.load_optimizer_config()
    
    def save_data(self):
        """保存所有数据"""
        self.save_project_goals()
        self.save_focus_sessions()
        self.save_optimizer_config()
    
    def load_project_goals(self):
        """加载项目目标"""
        if os.path.exists(self.goals_file):
            try:
                with open(self.goals_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.project_goals = []
                for goal_data in data:
                    # 兼容旧数据格式，移除priority字段
                    goal_params = {
                        'name': goal_data['name'],
                        'target_score': goal_data['target_score'],
                        'current_score': goal_data['current_score']
                    }
                    # 如果有手动设置的优先级，保存它
                    manual_priority = goal_data.get('priority') if 'priority' in goal_data else None
                    
                    goal = ProjectGoal(**goal_params)
                    
                    # 如果原来有手动优先级且不是自动计算的，则设置为手动优先级
                    if manual_priority is not None and manual_priority != goal.priority:
                        goal.set_manual_priority(manual_priority)
                    
                    self.project_goals.append(goal)
            except Exception as e:
                print(f"加载项目目标失败: {e}")
                self.project_goals = []
    
    def save_project_goals(self):
        """保存项目目标"""
        data = [
            {
                'name': goal.name,
                'target_score': goal.target_score,
                'current_score': goal.current_score,
                'priority': goal.priority
            }
            for goal in self.project_goals
        ]
        with open(self.goals_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_focus_sessions(self):
        """加载专注会话记录"""
        if os.path.exists(self.focus_sessions_file):
            try:
                with open(self.focus_sessions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.focus_sessions = [
                    FocusSession(
                        project_name=session['project_name'],
                        start_time=datetime.datetime.fromisoformat(session['start_time']),
                        end_time=datetime.datetime.fromisoformat(session['end_time']),
                        planned_focus_score=session['planned_focus_score'],
                        actual_focus_score=session.get('actual_focus_score'),
                        completed=session.get('completed', False)
                    )
                    for session in data
                ]
            except Exception as e:
                print(f"加载专注会话记录失败: {e}")
                self.focus_sessions = []
    
    def save_focus_sessions(self):
        """保存专注会话记录"""
        data = [
            {
                'project_name': session.project_name,
                'start_time': session.start_time.isoformat(),
                'end_time': session.end_time.isoformat(),
                'planned_focus_score': session.planned_focus_score,
                'actual_focus_score': session.actual_focus_score,
                'completed': session.completed
            }
            for session in self.focus_sessions
        ]
        with open(self.focus_sessions_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_optimizer_config(self):
        """加载优化器配置"""
        if os.path.exists(self.optimizer_config_file):
            try:
                with open(self.optimizer_config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.default_focus_duration = config.get('default_focus_duration', 60)
                self.focus_type = FocusType(config.get('focus_type', FocusType.MODERATELY_FOCUSED.value))
                self.work_start_hour = config.get('work_start_hour', 8)
                self.work_end_hour = config.get('work_end_hour', 22)
                self.min_slot_duration = config.get('min_slot_duration', 60)
            except Exception as e:
                print(f"加载优化器配置失败: {e}")
    
    def save_optimizer_config(self):
        """保存优化器配置"""
        config = {
            'default_focus_duration': self.default_focus_duration,
            'focus_type': self.focus_type.value,
            'work_start_hour': self.work_start_hour,
            'work_end_hour': self.work_end_hour,
            'min_slot_duration': self.min_slot_duration
        }
        with open(self.optimizer_config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    def add_project_goal(self, name: str, target_score: float, current_score: float):
        """添加项目目标（优先级自动计算）"""
        goal = ProjectGoal(name, target_score, current_score)
        self.project_goals.append(goal)
        self.save_project_goals()
        return goal
    
    def update_project_goal(self, name: str, target_score: float = None, current_score: float = None):
        """更新项目目标"""
        for goal in self.project_goals:
            if goal.name == name:
                if target_score is not None:
                    goal.target_score = target_score
                if current_score is not None:
                    goal.current_score = current_score
                self.save_project_goals()
                return True
        return False
    
    def remove_project_goal(self, name: str):
        """删除项目目标"""
        self.project_goals = [goal for goal in self.project_goals if goal.name != name]
        self.save_project_goals()
    
    def find_available_time_slots(self, calendar_manager, date: datetime.date) -> List[TimeSlot]:
        """查找指定日期的可用时间段"""
        # 获取当天的所有日程
        schedules = calendar_manager.get_schedules_by_date(date)
        
        # 创建工作时间范围
        work_start = datetime.datetime.combine(date, datetime.time(self.work_start_hour, 0))
        work_end = datetime.datetime.combine(date, datetime.time(self.work_end_hour, 0))
        
        # 收集已占用的时间段
        occupied_slots = []
        for schedule in schedules:
            if schedule.start_time.date() == date:
                occupied_slots.append((schedule.start_time, schedule.end_time))
        
        # 按开始时间排序
        occupied_slots.sort(key=lambda x: x[0])
        
        # 查找空闲时间段
        available_slots = []
        current_time = work_start
        
        for start, end in occupied_slots:
            # 如果当前时间到下一个日程开始之间有空隙
            if current_time < start:
                duration = int((start - current_time).total_seconds() / 60)
                if duration >= self.min_slot_duration:
                    slot = TimeSlot(
                        start_time=current_time,
                        end_time=start,
                        duration_minutes=duration
                    )
                    available_slots.append(slot)
            
            # 更新当前时间到日程结束时间
            current_time = max(current_time, end)
        
        # 检查最后一个时间段到工作结束时间
        if current_time < work_end:
            duration = int((work_end - current_time).total_seconds() / 60)
            if duration >= self.min_slot_duration:
                slot = TimeSlot(
                    start_time=current_time,
                    end_time=work_end,
                    duration_minutes=duration
                )
                available_slots.append(slot)
        
        return available_slots
    
    def calculate_time_weights(self) -> Dict[str, float]:
        """计算各项目的时间分配权重"""
        if not self.project_goals:
            return {}
        
        # 计算总权重（基于分数差距和优先级）
        total_weight = 0
        project_weights = {}
        
        for goal in self.project_goals:
            # 权重 = 分数差距 * 优先级 * 改进比例
            weight = goal.score_gap * goal.priority * (goal.improvement_ratio + 0.1)
            project_weights[goal.name] = weight
            total_weight += weight
        
        # 归一化权重
        if total_weight > 0:
            for project in project_weights:
                project_weights[project] /= total_weight
        
        return project_weights
    
    def allocate_time_slots(self, available_slots: List[TimeSlot]) -> List[TimeSlot]:
        """为可用时间段分配项目"""
        if not self.project_goals or not available_slots:
            return available_slots
        
        # 获取专注类型配置
        focus_config = self.focus_type_coefficients[self.focus_type]
        optimal_duration = focus_config['optimal_duration']
        min_duration = focus_config['min_duration']
        
        # 为每个项目设置单日最大时长限制（每个项目最多1小时）
        max_daily_duration_per_project = self.default_focus_duration  # 默认60分钟
        
        # 按优先级排序项目
        sorted_projects = sorted(self.project_goals, key=lambda g: g.priority, reverse=True)
        
        # 为每个项目分配时间，每个项目单日最多分配max_daily_duration_per_project分钟
        allocated_slots = []
        project_allocated_time = {goal.name: 0 for goal in self.project_goals}  # 跟踪每个项目已分配的时间
        
        # 使用轮询策略，确保所有项目都能得到时间
        project_index = 0  # 当前分配的项目索引
        
        # 第一轮分配：每个项目最多分配基础时长
        for slot in available_slots:
            if slot.duration_minutes < min_duration:
                # 时间段太短，不分配
                allocated_slots.append(slot)
                continue
            
            # 轮询查找需要时间的项目
            best_project = None
            attempts = 0
            
            while attempts < len(sorted_projects):
                current_project = sorted_projects[project_index % len(sorted_projects)]
                
                if project_allocated_time[current_project.name] < max_daily_duration_per_project:
                    best_project = current_project.name
                    project_index = (project_index + 1) % len(sorted_projects)  # 移动到下一个项目
                    break
                
                project_index = (project_index + 1) % len(sorted_projects)
                attempts += 1
            
            if best_project:
                # 计算这个项目还能分配多少时间
                remaining_project_time = max_daily_duration_per_project - project_allocated_time[best_project]
                
                # 计算实际分配时长
                allocated_duration = min(
                    slot.duration_minutes,
                    remaining_project_time,
                    optimal_duration
                )
                
                if allocated_duration >= min_duration:
                    # 分配时间段
                    allocated_slot = TimeSlot(
                        start_time=slot.start_time,
                        end_time=slot.start_time + datetime.timedelta(minutes=allocated_duration),
                        duration_minutes=allocated_duration,
                        assigned_project=best_project,
                        focus_score=self._calculate_focus_score(slot.start_time)
                    )
                    allocated_slots.append(allocated_slot)
                    
                    # 更新项目已分配时间
                    project_allocated_time[best_project] += allocated_duration
                    
                    # 如果还有剩余时间段，创建新的空闲时间段
                    if slot.duration_minutes > allocated_duration:
                        remaining_slot = TimeSlot(
                            start_time=slot.start_time + datetime.timedelta(minutes=allocated_duration),
                            end_time=slot.end_time,
                            duration_minutes=slot.duration_minutes - allocated_duration
                        )
                        allocated_slots.append(remaining_slot)
                else:
                    allocated_slots.append(slot)
            else:
                # 所有项目都已达到单日最大时长，保持空闲
                allocated_slots.append(slot)
        
        # 第二轮分配：如果还有空余时间，允许项目获得额外时间（最多再增加60分钟）
        unassigned_slots = [slot for slot in allocated_slots if not slot.assigned_project and slot.duration_minutes >= min_duration]
        
        if unassigned_slots and sorted_projects:
            print(f"🔄 第二轮分配：发现 {len(unassigned_slots)} 个空余时间段，进行额外分配")
            
            # 移除空余时间段，准备重新分配
            allocated_slots = [slot for slot in allocated_slots if slot.assigned_project or slot.duration_minutes < min_duration]
            
            # 允许每个项目再获得最多60分钟的额外时间
            max_extra_duration = 60
            project_extra_time = {goal.name: 0 for goal in self.project_goals}
            
            project_index = 0  # 重置项目索引
            
            for slot in unassigned_slots:
                # 轮询查找可以获得额外时间的项目
                best_project = None
                attempts = 0
                
                while attempts < len(sorted_projects):
                    current_project = sorted_projects[project_index % len(sorted_projects)]
                    
                    if project_extra_time[current_project.name] < max_extra_duration:
                        best_project = current_project.name
                        project_index = (project_index + 1) % len(sorted_projects)
                        break
                    
                    project_index = (project_index + 1) % len(sorted_projects)
                    attempts += 1
                
                if best_project:
                    # 计算额外分配时长
                    remaining_extra_time = max_extra_duration - project_extra_time[best_project]
                    allocated_duration = min(
                        slot.duration_minutes,
                        remaining_extra_time,
                        optimal_duration
                    )
                    
                    if allocated_duration >= min_duration:
                        # 分配额外时间段
                        allocated_slot = TimeSlot(
                            start_time=slot.start_time,
                            end_time=slot.start_time + datetime.timedelta(minutes=allocated_duration),
                            duration_minutes=allocated_duration,
                            assigned_project=best_project,
                            focus_score=self._calculate_focus_score(slot.start_time)
                        )
                        allocated_slots.append(allocated_slot)
                        
                        # 更新项目额外分配时间
                        project_extra_time[best_project] += allocated_duration
                        project_allocated_time[best_project] += allocated_duration
                        
                        print(f"  ✅ {best_project} 获得额外 {allocated_duration} 分钟")
                        
                        # 如果还有剩余时间段，创建新的空闲时间段
                        if slot.duration_minutes > allocated_duration:
                            remaining_slot = TimeSlot(
                                start_time=slot.start_time + datetime.timedelta(minutes=allocated_duration),
                                end_time=slot.end_time,
                                duration_minutes=slot.duration_minutes - allocated_duration
                            )
                            allocated_slots.append(remaining_slot)
                    else:
                        allocated_slots.append(slot)
                else:
                    # 所有项目都已达到额外时长限制，保持空闲
                    allocated_slots.append(slot)
        
        # 输出分配统计
        print(f"\n📊 时间分配统计:")
        for project_name, total_time in project_allocated_time.items():
            if total_time > 0:
                print(f"  {project_name}: {total_time} 分钟")
        
        return allocated_slots
    
    def _calculate_focus_score(self, start_time: datetime.datetime) -> float:
        """根据时间计算预期专注度评分"""
        hour = start_time.hour
        
        # 基于时间的专注度曲线（一般规律）
        if 8 <= hour <= 10:  # 上午高峰
            base_score = 0.9
        elif 10 <= hour <= 12:  # 上午后期
            base_score = 0.8
        elif 12 <= hour <= 14:  # 午休时间
            base_score = 0.6
        elif 14 <= hour <= 16:  # 下午前期
            base_score = 0.7
        elif 16 <= hour <= 18:  # 下午后期
            base_score = 0.8
        elif 18 <= hour <= 20:  # 晚上前期
            base_score = 0.7
        else:  # 其他时间
            base_score = 0.6
        
        # 根据专注类型调整
        if self.focus_type == FocusType.EXTREMELY_FOCUSED:
            return min(1.0, base_score + 0.1)
        elif self.focus_type == FocusType.EASILY_DISTRACTED:
            return max(0.3, base_score - 0.2)
        else:
            return base_score
    
    def optimize_low_focus_periods(self, calendar_manager) -> List[Dict]:
        """优化专注度过低的时间段"""
        # 分析历史专注会话
        low_focus_sessions = [
            session for session in self.focus_sessions
            if session.actual_focus_score is not None and session.actual_focus_score < 0.6
        ]
        
        if not low_focus_sessions:
            return []
        
        # 分析低专注度的时间模式
        low_focus_hours = {}
        for session in low_focus_sessions:
            hour = session.start_time.hour
            if hour not in low_focus_hours:
                low_focus_hours[hour] = []
            low_focus_hours[hour].append(session.actual_focus_score)
        
        # 计算每小时的平均专注度
        hour_avg_focus = {
            hour: sum(scores) / len(scores)
            for hour, scores in low_focus_hours.items()
        }
        
        # 找出需要优化的时间段
        optimization_suggestions = []
        for hour, avg_focus in hour_avg_focus.items():
            if avg_focus < 0.6:
                suggestions = self._generate_optimization_suggestions(hour, avg_focus)
                optimization_suggestions.extend(suggestions)
        
        return optimization_suggestions
    
    def _generate_optimization_suggestions(self, hour: int, avg_focus: float) -> List[Dict]:
        """生成优化建议"""
        suggestions = []
        
        if avg_focus < 0.4:
            # 严重专注度不足
            suggestions.append({
                'time_period': f"{hour}:00-{hour+1}:00",
                'issue': f"专注度严重不足 ({avg_focus:.2f})",
                'suggestion': "建议安排休息或轻松任务",
                'action': "avoid_important_tasks"
            })
        elif avg_focus < 0.6:
            # 中等专注度不足
            suggestions.append({
                'time_period': f"{hour}:00-{hour+1}:00",
                'issue': f"专注度偏低 ({avg_focus:.2f})",
                'suggestion': "建议缩短任务时长或增加休息",
                'action': "reduce_task_duration"
            })
        
        return suggestions
    
    def record_focus_session(self, project_name: str, start_time: datetime.datetime, 
                           end_time: datetime.datetime, planned_focus_score: float):
        """记录专注会话（等待多模态识别系统填入实际专注度）"""
        session = FocusSession(
            project_name=project_name,
            start_time=start_time,
            end_time=end_time,
            planned_focus_score=planned_focus_score
        )
        self.focus_sessions.append(session)
        self.save_focus_sessions()
        return session
    
    def update_focus_session_result(self, session_id: int, actual_focus_score: float, completed: bool = True):
        """更新专注会话结果（由多模态识别系统调用）"""
        if 0 <= session_id < len(self.focus_sessions):
            self.focus_sessions[session_id].actual_focus_score = actual_focus_score
            self.focus_sessions[session_id].completed = completed
            self.save_focus_sessions()
            return True
        return False
    
    def get_focus_statistics(self) -> Dict:
        """获取专注度统计信息"""
        if not self.focus_sessions:
            return {}
        
        completed_sessions = [s for s in self.focus_sessions if s.actual_focus_score is not None]
        
        if not completed_sessions:
            return {}
        
        total_sessions = len(completed_sessions)
        avg_focus = sum(s.actual_focus_score for s in completed_sessions) / total_sessions
        
        # 按项目统计
        project_stats = {}
        for session in completed_sessions:
            if session.project_name not in project_stats:
                project_stats[session.project_name] = []
            project_stats[session.project_name].append(session.actual_focus_score)
        
        project_avg = {
            project: sum(scores) / len(scores)
            for project, scores in project_stats.items()
        }
        
        return {
            'total_sessions': total_sessions,
            'average_focus': avg_focus,
            'project_averages': project_avg,
            'best_hours': self._get_best_focus_hours(),
            'worst_hours': self._get_worst_focus_hours()
        }
    
    def _get_best_focus_hours(self) -> List[int]:
        """获取专注度最好的时间段"""
        hour_scores = {}
        for session in self.focus_sessions:
            if session.actual_focus_score is not None:
                hour = session.start_time.hour
                if hour not in hour_scores:
                    hour_scores[hour] = []
                hour_scores[hour].append(session.actual_focus_score)
        
        hour_avg = {
            hour: sum(scores) / len(scores)
            for hour, scores in hour_scores.items()
        }
        
        return sorted(hour_avg.keys(), key=lambda h: hour_avg[h], reverse=True)[:3]
    
    def _get_worst_focus_hours(self) -> List[int]:
        """获取专注度最差的时间段"""
        hour_scores = {}
        for session in self.focus_sessions:
            if session.actual_focus_score is not None:
                hour = session.start_time.hour
                if hour not in hour_scores:
                    hour_scores[hour] = []
                hour_scores[hour].append(session.actual_focus_score)
        
        hour_avg = {
            hour: sum(scores) / len(scores)
            for hour, scores in hour_scores.items()
        }
        
        return sorted(hour_avg.keys(), key=lambda h: hour_avg[h])[:3]

class SmartOptimizerGUI:
    """智能优化器图形界面"""
    
    def __init__(self, optimizer: SmartTimeOptimizer, calendar_manager):
        self.optimizer = optimizer
        self.calendar_manager = calendar_manager
        self.window = None
    
    def show_goal_setting_dialog(self):
        """显示目标设定对话框"""
        self.window = tk.Toplevel()
        self.window.title("智能时间优化 - 目标设定")
        self.window.geometry("600x500")
        self.window.transient()
        self.window.grab_set()
        
        # 创建主框架
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # 专注类型设置
        ttk.Label(main_frame, text="专注类型:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.focus_type_var = tk.StringVar(value=self.optimizer.focus_type.value)
        focus_combo = ttk.Combobox(main_frame, textvariable=self.focus_type_var, 
                                  values=[ft.value for ft in FocusType], state="readonly")
        focus_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # 最佳专注时长
        ttk.Label(main_frame, text="最佳专注时长(分钟):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.focus_duration_var = tk.StringVar(value=str(self.optimizer.default_focus_duration))
        ttk.Entry(main_frame, textvariable=self.focus_duration_var).grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # 工作时间设置
        ttk.Label(main_frame, text="工作开始时间:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.work_start_var = tk.StringVar(value=str(self.optimizer.work_start_hour))
        ttk.Entry(main_frame, textvariable=self.work_start_var).grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(main_frame, text="工作结束时间:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.work_end_var = tk.StringVar(value=str(self.optimizer.work_end_hour))
        ttk.Entry(main_frame, textvariable=self.work_end_var).grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # 项目目标设置
        ttk.Label(main_frame, text="项目目标设置:", font=('TkDefaultFont', 10, 'bold')).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(20, 10))
        
        # 项目目标列表
        goals_frame = ttk.Frame(main_frame)
        goals_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        goals_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
        # 创建表格
        columns = ('项目名称', '目标分数', '现状分数', '自动优先级')
        self.goals_tree = ttk.Treeview(goals_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.goals_tree.heading(col, text=col)
            self.goals_tree.column(col, width=120)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(goals_frame, orient=tk.VERTICAL, command=self.goals_tree.yview)
        self.goals_tree.configure(yscrollcommand=scrollbar.set)
        
        self.goals_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="添加项目", command=self.add_project_goal).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="编辑项目", command=self.edit_project_goal).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="删除项目", command=self.delete_project_goal).pack(side=tk.LEFT, padx=5)
        
        # 底部按钮
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=7, column=0, columnspan=2, pady=20)
        
        ttk.Button(bottom_frame, text="保存配置", command=self.save_configuration).pack(side=tk.LEFT, padx=10)
        ttk.Button(bottom_frame, text="一键优化", command=self.run_optimization).pack(side=tk.LEFT, padx=10)
        ttk.Button(bottom_frame, text="查看统计", command=self.show_statistics).pack(side=tk.LEFT, padx=10)
        ttk.Button(bottom_frame, text="关闭", command=self.window.destroy).pack(side=tk.LEFT, padx=10)
        
        # 加载现有目标
        self.refresh_goals_list()
    
    def refresh_goals_list(self):
        """刷新目标列表"""
        for item in self.goals_tree.get_children():
            self.goals_tree.delete(item)
        
        for goal in self.optimizer.project_goals:
            self.goals_tree.insert('', 'end', values=(
                goal.name, 
                f"{goal.target_score:.1f}", 
                f"{goal.current_score:.1f}", 
                f"{goal.priority:.2f}"
            ))
    
    def add_project_goal(self):
        """添加项目目标"""
        dialog = ProjectGoalDialog(self.window, "添加项目目标")
        if dialog.result:
            name, target, current = dialog.result
            self.optimizer.add_project_goal(name, target, current)
            self.refresh_goals_list()
    
    def edit_project_goal(self):
        """编辑项目目标"""
        selection = self.goals_tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请选择要编辑的项目")
            return
        
        item = self.goals_tree.item(selection[0])
        values = item['values']
        
        dialog = ProjectGoalDialog(self.window, "编辑项目目标", 
                                 (values[0], float(values[1]), float(values[2])))
        if dialog.result:
            name, target, current = dialog.result
            self.optimizer.update_project_goal(values[0], target, current)
            # 如果名称改变，需要删除旧的并添加新的
            if name != values[0]:
                self.optimizer.remove_project_goal(values[0])
                self.optimizer.add_project_goal(name, target, current)
            self.refresh_goals_list()
    
    def delete_project_goal(self):
        """删除项目目标"""
        selection = self.goals_tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请选择要删除的项目")
            return
        
        item = self.goals_tree.item(selection[0])
        project_name = item['values'][0]
        
        if messagebox.askyesno("确认", f"确定要删除项目 '{project_name}' 吗？"):
            self.optimizer.remove_project_goal(project_name)
            self.refresh_goals_list()
    
    def save_configuration(self):
        """保存配置"""
        try:
            # 更新配置
            self.optimizer.focus_type = FocusType(self.focus_type_var.get())
            self.optimizer.default_focus_duration = int(self.focus_duration_var.get())
            self.optimizer.work_start_hour = int(self.work_start_var.get())
            self.optimizer.work_end_hour = int(self.work_end_var.get())
            
            # 保存到文件
            self.optimizer.save_data()
            messagebox.showinfo("成功", "配置已保存")
        except ValueError as e:
            messagebox.showerror("错误", f"配置保存失败: {e}")
    
    def run_optimization(self):
        """运行一键优化"""
        if not self.optimizer.project_goals:
            messagebox.showwarning("警告", "请先设置项目目标")
            return
        
        # 选择优化日期
        date_str = simpledialog.askstring("选择日期", "请输入要优化的日期 (YYYY-MM-DD):", 
                                        initialvalue=datetime.date.today().strftime("%Y-%m-%d"))
        if not date_str:
            return
        
        try:
            target_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            messagebox.showerror("错误", "日期格式不正确")
            return
        
        # 查找可用时间段
        available_slots = self.optimizer.find_available_time_slots(self.calendar_manager, target_date)
        
        if not available_slots:
            messagebox.showinfo("信息", "该日期没有可用的时间段")
            return
        
        # 分配时间
        allocated_slots = self.optimizer.allocate_time_slots(available_slots)
        
        # 显示优化结果
        self.show_optimization_result(target_date, allocated_slots)
    
    def show_optimization_result(self, date: datetime.date, allocated_slots: List[TimeSlot]):
        """显示优化结果"""
        # 创建独立的结果窗口，不依赖于可能已关闭的父窗口
        result_window = tk.Toplevel()
        result_window.title(f"优化结果 - {date}")
        result_window.geometry("700x500")
        
        # 居中显示
        result_window.update_idletasks()
        x = (result_window.winfo_screenwidth() // 2) - (700 // 2)
        y = (result_window.winfo_screenheight() // 2) - (500 // 2)
        result_window.geometry(f"700x500+{x}+{y}")
        
        # 创建表格
        columns = ('时间段', '时长', '分配项目', '预期专注度')
        tree = ttk.Treeview(result_window, columns=columns, show='headings')
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)
        
        # 填充数据
        for slot in allocated_slots:
            time_range = f"{slot.start_time.strftime('%H:%M')}-{slot.end_time.strftime('%H:%M')}"
            duration = f"{slot.duration_minutes}分钟"
            project = slot.assigned_project or "空闲"
            focus = f"{slot.focus_score:.2f}" if slot.assigned_project else "-"
            
            tree.insert('', 'end', values=(time_range, duration, project, focus))
        
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 按钮
        button_frame = ttk.Frame(result_window)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="应用到日历", 
                  command=lambda: self.apply_optimization(date, allocated_slots, result_window)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="关闭", command=result_window.destroy).pack(side=tk.LEFT, padx=5)
    
    def apply_optimization(self, date: datetime.date, allocated_slots: List[TimeSlot], window):
        """应用优化结果到日历"""
        if messagebox.askyesno("确认", "确定要将优化结果添加到日历中吗？"):
            added_count = 0
            failed_count = 0
            
            print("🔄 开始应用优化结果到日历...")
            
            for slot in allocated_slots:
                if slot.assigned_project:
                    # 创建日程文本 - 修复格式，使其与时间范围优化一致
                    schedule_text = f"{slot.assigned_project} {slot.start_time.strftime('%H:%M')}-{slot.end_time.strftime('%H:%M')}"
                    
                    # 添加到日历
                    schedule_id = self.calendar_manager.add_schedule_from_text(schedule_text, date)
                    if schedule_id:
                        added_count += 1
                        print(f"   ✅ {date} {schedule_text}")
                        
                        # 记录专注会话
                        self.optimizer.record_focus_session(
                            slot.assigned_project,
                            slot.start_time,
                            slot.end_time,
                            slot.focus_score
                        )
                    else:
                        failed_count += 1
                        print(f"   ❌ {date} {schedule_text}")
            
            # 显示详细结果
            result_message = f"应用结果:\n✅ 成功: {added_count} 个日程\n"
            if failed_count > 0:
                result_message += f"❌ 失败: {failed_count} 个日程\n"
            result_message += f"📊 成功率: {added_count/(added_count+failed_count)*100:.1f}%" if (added_count+failed_count) > 0 else "N/A"
            
            messagebox.showinfo("应用完成", result_message)
            window.destroy()
            
            # 尝试刷新主界面
            self._refresh_main_interface()
            
            print(f"✅ 优化结果应用完成: {added_count} 个日程已保存")
    
    def _refresh_main_interface(self):
        """刷新主界面（尝试找到并刷新主界面）"""
        try:
            print("🔄 尝试刷新主界面...")
            
            # 方法1: 通过tkinter的根窗口查找主界面
            import tkinter as tk
            if tk._default_root:
                # 查找所有顶级窗口
                for widget in tk._default_root.winfo_children():
                    if hasattr(widget, 'refresh_schedule_list') and hasattr(widget, 'update_calendar_buttons'):
                        print("   📍 找到主界面，开始刷新...")
                        
                        # 使用主界面的强制刷新方法
                        if hasattr(widget, 'force_refresh_interface'):
                            success = widget.force_refresh_interface()
                            if success:
                                print("✅ 主界面刷新完成")
                                return
                        
                        # 备用方案：逐个调用刷新方法
                        if hasattr(widget, 'refresh_schedule_list'):
                            widget.refresh_schedule_list()
                            print("   ✅ 日程列表已刷新")
                        
                        if hasattr(widget, 'update_calendar_buttons'):
                            widget.update_calendar_buttons()
                            print("   ✅ 日历按钮已刷新")
                        
                        print("✅ 主界面刷新完成")
                        return
                
                # 方法2: 查找所有Toplevel窗口
                for widget in tk._default_root.winfo_children():
                    if isinstance(widget, tk.Toplevel):
                        for child in widget.winfo_children():
                            if hasattr(child, 'refresh_schedule_list') and hasattr(child, 'update_calendar_buttons'):
                                print("   📍 在Toplevel中找到主界面，开始刷新...")
                                
                                if hasattr(child, 'force_refresh_interface'):
                                    success = child.force_refresh_interface()
                                    if success:
                                        print("✅ 主界面刷新完成")
                                        return
                                
                                if hasattr(child, 'refresh_schedule_list'):
                                    child.refresh_schedule_list()
                                    print("   ✅ 日程列表已刷新")
                                
                                if hasattr(child, 'update_calendar_buttons'):
                                    child.update_calendar_buttons()
                                    print("   ✅ 日历按钮已刷新")
                                
                                print("✅ 主界面刷新完成")
                                return
            
            print("⚠️ 未找到主界面，跳过刷新")
            
        except Exception as e:
            print(f"⚠️ 刷新主界面时出错: {e}")
    
    def show_statistics(self):
        """显示统计信息"""
        stats = self.optimizer.get_focus_statistics()
        
        if not stats:
            messagebox.showinfo("信息", "暂无专注度统计数据")
            return
        
        # 创建独立的统计窗口，不依赖于可能已关闭的父窗口
        stats_window = tk.Toplevel()
        stats_window.title("专注度统计")
        stats_window.geometry("500x400")
        
        # 居中显示
        stats_window.update_idletasks()
        x = (stats_window.winfo_screenwidth() // 2) - (500 // 2)
        y = (stats_window.winfo_screenheight() // 2) - (400 // 2)
        stats_window.geometry(f"500x400+{x}+{y}")
        
        text_widget = tk.Text(stats_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # 显示统计信息
        content = f"""专注度统计报告
        
总会话数: {stats['total_sessions']}
平均专注度: {stats['average_focus']:.2f}

各项目平均专注度:
"""
        for project, avg in stats['project_averages'].items():
            content += f"  {project}: {avg:.2f}\n"
        
        content += f"\n最佳专注时间段: {', '.join(map(str, stats['best_hours']))}点\n"
        content += f"最差专注时间段: {', '.join(map(str, stats['worst_hours']))}点\n"
        
        # 优化建议
        suggestions = self.optimizer.optimize_low_focus_periods(self.calendar_manager)
        if suggestions:
            content += "\n优化建议:\n"
            for suggestion in suggestions:
                content += f"  {suggestion['time_period']}: {suggestion['suggestion']}\n"
        
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)

class ProjectGoalDialog:
    """项目目标对话框"""
    
    def __init__(self, parent, title, initial_values=None):
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("400x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # 居中显示
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (300 // 2)
        self.dialog.geometry(f"400x300+{x}+{y}")
        
        # 创建表单
        frame = ttk.Frame(self.dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 项目名称
        ttk.Label(frame, text="项目名称:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.name_var = tk.StringVar(value=initial_values[0] if initial_values else "")
        ttk.Entry(frame, textvariable=self.name_var, width=30).grid(row=0, column=1, pady=5)
        
        # 目标分数
        ttk.Label(frame, text="目标分数:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.target_var = tk.StringVar(value=str(initial_values[1]) if initial_values else "100")
        ttk.Entry(frame, textvariable=self.target_var, width=30).grid(row=1, column=1, pady=5)
        
        # 现状分数
        ttk.Label(frame, text="现状分数:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.current_var = tk.StringVar(value=str(initial_values[2]) if initial_values else "0")
        ttk.Entry(frame, textvariable=self.current_var, width=30).grid(row=2, column=1, pady=5)
        
        # 按钮
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="确定", command=self.ok_clicked).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="取消", command=self.dialog.destroy).pack(side=tk.LEFT, padx=10)
        
        # 等待对话框关闭
        self.dialog.wait_window()
    
    def ok_clicked(self):
        """确定按钮点击事件"""
        try:
            name = self.name_var.get().strip()
            target = float(self.target_var.get())
            current = float(self.current_var.get())
            
            if not name:
                messagebox.showerror("错误", "项目名称不能为空")
                return
            
            if target < 0 or current < 0:
                messagebox.showerror("错误", "分数必须为非负数")
                return
            
            self.result = (name, target, current)
            self.dialog.destroy()
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字") 