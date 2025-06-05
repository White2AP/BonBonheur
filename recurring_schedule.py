#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
循环日程模块
支持重复日程的创建和管理
"""

import datetime
import uuid
from typing import Dict, List, Optional, Tuple
from enum import Enum

class RecurrenceType(Enum):
    """循环类型枚举"""
    NONE = "none"           # 不循环
    DAILY = "daily"         # 每日
    WEEKLY = "weekly"       # 每周
    MONTHLY = "monthly"     # 每月
    YEARLY = "yearly"       # 每年

class RecurringSchedule:
    """循环日程类"""
    
    def __init__(self, schedule_id: str, name: str, start_time: datetime.datetime,
                 end_time: datetime.datetime, content: str,
                 recurrence_type: RecurrenceType = RecurrenceType.NONE,
                 recurrence_interval: int = 1,
                 recurrence_end_date: datetime.date = None,
                 recurrence_count: int = None,
                 completed: bool = False):
        self.schedule_id = schedule_id
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.content = content
        self.recurrence_type = recurrence_type
        self.recurrence_interval = recurrence_interval  # 间隔（如每2周）
        self.recurrence_end_date = recurrence_end_date  # 结束日期
        self.recurrence_count = recurrence_count        # 重复次数
        self.completed = completed                      # 完成状态
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'schedule_id': self.schedule_id,
            'name': self.name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'content': self.content,
            'recurrence_type': self.recurrence_type.value,
            'recurrence_interval': self.recurrence_interval,
            'recurrence_end_date': self.recurrence_end_date.isoformat() if self.recurrence_end_date else None,
            'recurrence_count': self.recurrence_count,
            'completed': self.completed
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """从字典创建循环日程对象"""
        return cls(
            schedule_id=data['schedule_id'],
            name=data['name'],
            start_time=datetime.datetime.fromisoformat(data['start_time']),
            end_time=datetime.datetime.fromisoformat(data['end_time']),
            content=data['content'],
            recurrence_type=RecurrenceType(data.get('recurrence_type', 'none')),
            recurrence_interval=data.get('recurrence_interval', 1),
            recurrence_end_date=datetime.date.fromisoformat(data['recurrence_end_date']) if data.get('recurrence_end_date') else None,
            recurrence_count=data.get('recurrence_count'),
            completed=data.get('completed', False)
        )
    
    def is_recurring(self) -> bool:
        """判断是否为循环日程"""
        return self.recurrence_type != RecurrenceType.NONE
    
    def generate_occurrences(self, start_date: datetime.date = None, 
                           end_date: datetime.date = None) -> List[Tuple[datetime.datetime, datetime.datetime]]:
        """生成循环日程的所有发生时间"""
        if not self.is_recurring():
            return [(self.start_time, self.end_time)]
        
        occurrences = []
        current_start = self.start_time
        current_end = self.end_time
        count = 0
        
        # 设置查询范围
        if start_date is None:
            start_date = self.start_time.date()
        if end_date is None:
            end_date = self.recurrence_end_date or datetime.date.today() + datetime.timedelta(days=365)
        
        while True:
            # 检查是否在查询范围内
            if current_start.date() >= start_date and current_start.date() <= end_date:
                occurrences.append((current_start, current_end))
            
            # 检查结束条件
            if self.recurrence_end_date and current_start.date() > self.recurrence_end_date:
                break
            if self.recurrence_count and count >= self.recurrence_count - 1:
                break
            if current_start.date() > end_date:
                break
            
            # 计算下一次发生时间
            if self.recurrence_type == RecurrenceType.DAILY:
                current_start += datetime.timedelta(days=self.recurrence_interval)
                current_end += datetime.timedelta(days=self.recurrence_interval)
            elif self.recurrence_type == RecurrenceType.WEEKLY:
                current_start += datetime.timedelta(weeks=self.recurrence_interval)
                current_end += datetime.timedelta(weeks=self.recurrence_interval)
            elif self.recurrence_type == RecurrenceType.MONTHLY:
                # 月份循环需要特殊处理
                next_month = current_start.month + self.recurrence_interval
                next_year = current_start.year
                while next_month > 12:
                    next_month -= 12
                    next_year += 1
                try:
                    current_start = current_start.replace(year=next_year, month=next_month)
                    current_end = current_end.replace(year=next_year, month=next_month)
                except ValueError:
                    # 处理月末日期问题（如1月31日到2月）
                    import calendar
                    last_day = calendar.monthrange(next_year, next_month)[1]
                    day = min(current_start.day, last_day)
                    current_start = current_start.replace(year=next_year, month=next_month, day=day)
                    current_end = current_end.replace(year=next_year, month=next_month, day=day)
            elif self.recurrence_type == RecurrenceType.YEARLY:
                current_start = current_start.replace(year=current_start.year + self.recurrence_interval)
                current_end = current_end.replace(year=current_end.year + self.recurrence_interval)
            
            count += 1
            
            # 防止无限循环
            if count > 1000:
                break
        
        return occurrences

class RecurringNLPProcessor:
    """循环日程自然语言处理器"""
    
    def __init__(self):
        # 循环关键词
        self.recurrence_keywords = {
            '每天': RecurrenceType.DAILY,
            '每日': RecurrenceType.DAILY,
            '每周': RecurrenceType.WEEKLY,
            '每星期': RecurrenceType.WEEKLY,
            '每月': RecurrenceType.MONTHLY,
            '每年': RecurrenceType.YEARLY,
            '周一': (RecurrenceType.WEEKLY, 0),
            '周二': (RecurrenceType.WEEKLY, 1),
            '周三': (RecurrenceType.WEEKLY, 2),
            '周四': (RecurrenceType.WEEKLY, 3),
            '周五': (RecurrenceType.WEEKLY, 4),
            '周六': (RecurrenceType.WEEKLY, 5),
            '周日': (RecurrenceType.WEEKLY, 6),
            '星期一': (RecurrenceType.WEEKLY, 0),
            '星期二': (RecurrenceType.WEEKLY, 1),
            '星期三': (RecurrenceType.WEEKLY, 2),
            '星期四': (RecurrenceType.WEEKLY, 3),
            '星期五': (RecurrenceType.WEEKLY, 4),
            '星期六': (RecurrenceType.WEEKLY, 5),
            '星期日': (RecurrenceType.WEEKLY, 6),
        }
        
        # 时间关键词
        self.time_keywords = {
            '上午': 'AM', '下午': 'PM', '晚上': 'PM', '早上': 'AM'
        }
        
        # 数字映射
        self.number_map = {
            '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
            '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15, '十六': 16, '十七': 17, 
            '十八': 18, '十九': 19, '二十': 20, '二十一': 21, '二十二': 22, '二十三': 23
        }
    
    def parse_recurring_schedule(self, text: str, selected_date: datetime.date = None) -> Optional[Dict]:
        """解析日程（支持单一日程和循环日程）"""
        import re
        import jieba
        
        try:
            words = list(jieba.cut(text))
            
            # 首先检查是否包含循环关键词
            recurrence_info = self._extract_recurrence_info(text, words)
            
            # 提取基本信息
            name = self._extract_schedule_name(text, words)
            
            # 清理内容，去掉时间前缀
            content = self._clean_content(text)
            
            if recurrence_info:
                # 循环日程
                start_time, end_time = self._extract_time_info(text, words, selected_date, recurrence_info)
                end_date = self._extract_end_date(text, words)
                
                if name and start_time:
                    result = {
                        'name': name,
                        'start_time': start_time,
                        'end_time': end_time or start_time + datetime.timedelta(hours=1),
                        'content': content,
                        'recurrence_type': recurrence_info['type'],
                        'recurrence_interval': recurrence_info.get('interval', 1),
                        'recurrence_end_date': end_date,
                        'recurrence_count': None,
                        'completed': False
                    }
                    return result
            else:
                # 单一日程
                start_time, end_time = self._extract_single_schedule_time(text, words, selected_date)
                
                if name and start_time:
                    result = {
                        'name': name,
                        'start_time': start_time,
                        'end_time': end_time or start_time + datetime.timedelta(hours=1),
                        'content': content,
                        'recurrence_type': RecurrenceType.NONE,
                        'recurrence_interval': 1,
                        'recurrence_end_date': None,
                        'recurrence_count': None,
                        'completed': False
                    }
                    return result
            
            return None
        except Exception as e:
            print(f"解析日程错误: {e}")
            return None
    
    def _clean_content(self, text: str) -> str:
        """清理内容，去掉时间前缀，保留核心信息"""
        import re
        
        # 只去掉日期相关的前缀，保留时间段信息
        date_prefixes = [
            '今天', '明天', '后天', '昨天', '前天', '今日', '明日',
            '今早', '明早', '今晚', '明晚', '后早', '后晚', '昨早', '昨晚'
        ]
        
        cleaned_text = text
        
        # 移除日期前缀
        for prefix in date_prefixes:
            if cleaned_text.startswith(prefix):
                cleaned_text = cleaned_text[len(prefix):]
                break
        
        # 清理开头的空格
        cleaned_text = cleaned_text.strip()
        
        # 如果清理后为空，返回原文本
        if not cleaned_text:
            return text
            
        return cleaned_text
    
    def _extract_recurrence_info(self, text: str, words: List[str]) -> Optional[Dict]:
        """提取循环信息"""
        import re
        
        # 检查每周特定日期的模式
        for keyword, info in self.recurrence_keywords.items():
            if keyword in text and isinstance(info, tuple):
                recurrence_type, weekday = info
                return {
                    'type': recurrence_type,
                    'interval': 1,
                    'weekday': weekday
                }
        
        # 检查一般循环模式
        for keyword, recurrence_type in self.recurrence_keywords.items():
            if keyword in text and isinstance(recurrence_type, RecurrenceType):
                return {
                    'type': recurrence_type,
                    'interval': 1
                }
        
        # 检查"每X周"、"每X天"等模式
        interval_match = re.search(r'每(\d+|[一二三四五六七八九十]+)(天|日|周|星期|月|年)', text)
        if interval_match:
            interval_str = interval_match.group(1)
            unit = interval_match.group(2)
            
            interval = int(interval_str) if interval_str.isdigit() else self.number_map.get(interval_str, 1)
            
            if unit in ['天', '日']:
                return {'type': RecurrenceType.DAILY, 'interval': interval}
            elif unit in ['周', '星期']:
                return {'type': RecurrenceType.WEEKLY, 'interval': interval}
            elif unit == '月':
                return {'type': RecurrenceType.MONTHLY, 'interval': interval}
            elif unit == '年':
                return {'type': RecurrenceType.YEARLY, 'interval': interval}
        
        return None
    
    def _extract_schedule_name(self, text: str, words: List[str]) -> str:
        """提取日程名称"""
        import re
        
        # 时间相关的词汇，不应该被识别为日程名称
        time_related_words = [
            '早上', '上午', '下午', '晚上', '中午', '傍晚', '深夜', '凌晨',
            '今天', '明天', '后天', '昨天', '前天', '今日', '明日',
            '今早', '明早', '今晚', '明晚', '后早', '后晚', '昨早', '昨晚',
            '周一', '周二', '周三', '周四', '周五', '周六', '周日',
            '星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日',
            '礼拜一', '礼拜二', '礼拜三', '礼拜四', '礼拜五', '礼拜六', '礼拜日'
        ]
        
        # 特殊处理"上课"、"上班"等合理的"上"字用法（优先检查）
        special_patterns = ['上课', '上班', '上学', '上网', '上楼', '上车']
        for pattern in special_patterns:
            if pattern in text:
                return pattern
        
        # 查找动词和活动名称作为日程名称
        action_words = ['学', '开会', '会议', '约会', '聚餐', '工作', '运动', '购物', '看电影', '旅行', '毛概', '课程', '培训', '跑步', '体检', '复习', '背单词', '起床', '大扫除', '交房租']
        
        # 检查是否有明确的活动词汇
        for word in action_words:
            if word in text:
                # 确保这个词不是时间词汇的一部分
                is_time_related = False
                for time_word in time_related_words:
                    if word in time_word and time_word in text:
                        is_time_related = True
                        break
                
                if not is_time_related:
                    return word
        
        # 如果没有找到动作词，尝试提取引号内容
        quotes_match = re.search(r'["""\'](.*?)["""\']', text)
        if quotes_match:
            return quotes_match.group(1)
        
        # 改进的名称提取逻辑：更彻底地清理时间相关内容
        cleaned_text = text
        
        # 1. 移除时间相关词汇（包括组合词汇）
        for time_word in time_related_words:
            cleaned_text = cleaned_text.replace(time_word, '')
        
        # 2. 移除循环关键词
        recurrence_words = ['每天', '每日', '每周', '每星期', '每月', '每年']
        for rec_word in recurrence_words:
            cleaned_text = cleaned_text.replace(rec_word, '')
        
        # 3. 移除时间表达（更全面的正则表达式）
        time_patterns = [
            r'\d+点\d*分?',           # 8点、8点30分
            r'\d+:\d+',               # 14:30
            r'[一二三四五六七八九十]+点\d*分?',  # 八点、八点三十分
            r'到\d+月',               # 到6月
            r'\d+月前',               # 6月前
            r'[一二三四五六七八九十]+个?小时',   # 三个小时、三小时
            r'\d+分钟',               # 30分钟
            r'\d+小时',               # 2小时
            r'要|前|中旬|底',         # 要、前、中旬、底
            r'\d+月\d*日?',           # 6月15日、6月15
        ]
        
        for pattern in time_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text)
        
        # 4. 特别处理时间前缀组合（如"后早"、"明早"等）
        # 使用更精确的正则表达式来移除这些组合
        time_prefix_patterns = [
            r'(今|明|后|昨|前)(天|日|早|晚)',  # 今天、明早、后早等
            r'(上午|下午|晚上|早上|中午|傍晚|深夜|凌晨)',
            r'(周|星期|礼拜)[一二三四五六七日天]',
        ]
        
        for pattern in time_prefix_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text)
        
        # 5. 清理多余的空格和标点
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        cleaned_text = re.sub(r'[，。！？；：、]', '', cleaned_text)
        
        # 6. 使用jieba分词提取有意义的词汇
        import jieba
        remaining_words = []
        for word in jieba.cut(cleaned_text):
            word = word.strip()
            # 过滤掉单字符、停用词和数字
            if (len(word) > 1 and 
                word not in ['的', '在', '和', '与', '或', '及', '了', '着', '过', '要', '会', '能', '可以', '应该'] and
                not word.isdigit() and
                not re.match(r'^[一二三四五六七八九十]+$', word)):
                remaining_words.append(word)
        
        # 7. 返回第一个有意义的词汇
        if remaining_words:
            return remaining_words[0]
        
        # 8. 如果还是没有找到，尝试从原文本中直接提取动作词
        # 这是最后的备选方案
        for word in jieba.cut(text):
            if (len(word) > 1 and 
                word not in time_related_words and
                word not in ['的', '在', '和', '与', '或', '及', '了', '着', '过', '要', '会', '能', '可以', '应该'] and
                not word.isdigit() and
                not re.match(r'^\d+[点分时]', word)):
                return word
        
        # 如果还是没有找到，返回默认名称
        return '日程'
    
    def _extract_time_info(self, text: str, words: List[str], selected_date: datetime.date = None, 
                          recurrence_info: Dict = None) -> Tuple[datetime.datetime, datetime.datetime]:
        """提取时间信息"""
        import re
        
        now = datetime.datetime.now()
        
        # 确定开始日期
        if selected_date:
            start_date = selected_date
        else:
            start_date = now.date()
        
        # 如果是每周特定日期，调整到下一个该日期
        if recurrence_info and 'weekday' in recurrence_info:
            target_weekday = recurrence_info['weekday']
            days_ahead = target_weekday - start_date.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            start_date = start_date + datetime.timedelta(days=days_ahead)
        
        # 提取时间
        hour = 19  # 默认晚上7点
        minute = 0
        
        # 处理具体时间 (HH:MM格式)
        time_match = re.search(r'(\d{1,2}):(\d{2})', text)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
        else:
            # 处理中文时间
            hour_match = re.search(r'(\d+|[一二三四五六七八九十]+)点', text)
            if hour_match:
                hour_str = hour_match.group(1)
                if hour_str.isdigit():
                    hour = int(hour_str)
                else:
                    hour = self.number_map.get(hour_str, 19)
        
        # 处理上午/下午
        if '下午' in text or '晚上' in text:
            if hour < 12:
                hour += 12
        elif '上午' in text or '早上' in text:
            if hour >= 12:
                hour -= 12
        
        start_time = datetime.datetime.combine(start_date, datetime.time(hour, minute))
        
        # 提取持续时间
        duration = self._extract_duration(text)
        end_time = start_time + duration
        
        return start_time, end_time
    
    def _extract_duration(self, text: str) -> datetime.timedelta:
        """提取持续时间"""
        import re
        
        # 查找"X小时"、"X分钟"等
        hour_match = re.search(r'(\d+|[一二三四五六七八九十]+)(个)?小时', text)
        minute_match = re.search(r'(\d+|[一二三四五六七八九十]+)分钟', text)
        
        hours = 0
        minutes = 0
        
        if hour_match:
            hour_str = hour_match.group(1)
            hours = int(hour_str) if hour_str.isdigit() else self.number_map.get(hour_str, 1)
        
        if minute_match:
            minute_str = minute_match.group(1)
            minutes = int(minute_str) if minute_str.isdigit() else self.number_map.get(minute_str, 0)
        
        # 如果没有明确指定时间，默认1小时
        if hours == 0 and minutes == 0:
            hours = 1
        
        return datetime.timedelta(hours=hours, minutes=minutes)
    
    def _extract_end_date(self, text: str, words: List[str]) -> Optional[datetime.date]:
        """提取结束日期"""
        import re
        
        now = datetime.datetime.now()
        
        # 查找"到X月"、"X月前"、"X月中旬前"等
        month_patterns = [
            r'到(\d+)月',
            r'(\d+)月前',
            r'(\d+)月中旬前',
            r'(\d+)月底前'
        ]
        
        for pattern in month_patterns:
            match = re.search(pattern, text)
            if match:
                month = int(match.group(1))
                year = now.year
                if month < now.month:
                    year += 1
                
                # 根据模式确定具体日期
                if '中旬前' in pattern:
                    day = 15
                elif '底前' in pattern:
                    import calendar
                    day = calendar.monthrange(year, month)[1]
                else:
                    day = 1
                
                try:
                    return datetime.date(year, month, day)
                except ValueError:
                    continue
        
        return None 
    
    def _extract_single_schedule_time(self, text: str, words: List[str], selected_date: datetime.date = None) -> Tuple[datetime.datetime, datetime.datetime]:
        """提取单一日程的时间信息"""
        import re
        
        now = datetime.datetime.now()
        
        # 确定日期
        target_date = self._extract_target_date(text, selected_date or now.date())
        
        # 提取时间
        hour, minute = self._extract_hour_minute(text)
        
        # 处理上午/下午
        hour = self._adjust_hour_for_period(text, hour)
        
        start_time = datetime.datetime.combine(target_date, datetime.time(hour, minute))
        
        # 提取持续时间
        duration = self._extract_duration(text)
        end_time = start_time + duration
        
        return start_time, end_time
    
    def _extract_target_date(self, text: str, default_date: datetime.date) -> datetime.date:
        """提取目标日期"""
        import re
        
        today = datetime.date.today()
        
        # 相对日期关键词（扩展支持更多表达）
        if '今天' in text or '今日' in text or '今早' in text or '今晚' in text:
            return today
        elif '明天' in text or '明日' in text or '明早' in text or '明晚' in text:
            return today + datetime.timedelta(days=1)
        elif '后天' in text or '后早' in text or '后晚' in text:
            return today + datetime.timedelta(days=2)
        elif '大后天' in text:
            return today + datetime.timedelta(days=3)
        elif '昨天' in text or '昨日' in text or '昨早' in text or '昨晚' in text:
            return today - datetime.timedelta(days=1)
        elif '前天' in text:
            return today - datetime.timedelta(days=2)
        
        # 具体日期模式 (MM月DD日)
        date_match = re.search(r'(\d{1,2})月(\d{1,2})日?', text)
        if date_match:
            month = int(date_match.group(1))
            day = int(date_match.group(2))
            year = today.year
            
            # 如果指定的月份已经过去，则认为是明年
            if month < today.month or (month == today.month and day < today.day):
                year += 1
            
            try:
                return datetime.date(year, month, day)
            except ValueError:
                pass
        
        # 星期几模式
        weekday_map = {
            '周一': 0, '周二': 1, '周三': 2, '周四': 3, '周五': 4, '周六': 5, '周日': 6,
            '星期一': 0, '星期二': 1, '星期三': 2, '星期四': 3, '星期五': 4, '星期六': 5, '星期日': 6,
            '礼拜一': 0, '礼拜二': 1, '礼拜三': 2, '礼拜四': 3, '礼拜五': 4, '礼拜六': 5, '礼拜日': 6
        }
        
        for weekday_text, weekday_num in weekday_map.items():
            if weekday_text in text:
                days_ahead = weekday_num - today.weekday()
                if days_ahead <= 0:  # 如果是今天或已经过去，则指下周
                    days_ahead += 7
                return today + datetime.timedelta(days=days_ahead)
        
        # 如果没有找到特定日期，使用默认日期
        return default_date
    
    def _extract_hour_minute(self, text: str) -> Tuple[int, int]:
        """提取小时和分钟"""
        import re
        
        hour = 19  # 默认晚上7点
        minute = 0
        
        # 处理具体时间 (HH:MM格式)
        time_match = re.search(r'(\d{1,2}):(\d{2})', text)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            return hour, minute
        
        # 处理中文时间
        hour_match = re.search(r'(\d+|[一二三四五六七八九十]+)点', text)
        if hour_match:
            hour_str = hour_match.group(1)
            if hour_str.isdigit():
                hour = int(hour_str)
            else:
                hour = self.number_map.get(hour_str, 19)
        
        # 处理分钟
        minute_match = re.search(r'(\d+|[一二三四五六七八九十]+)分', text)
        if minute_match:
            minute_str = minute_match.group(1)
            if minute_str.isdigit():
                minute = int(minute_str)
            else:
                minute = self.number_map.get(minute_str, 0)
        
        return hour, minute
    
    def _adjust_hour_for_period(self, text: str, hour: int) -> int:
        """根据上午/下午调整小时"""
        if '下午' in text or '晚上' in text:
            if hour < 12:
                hour += 12
        elif '上午' in text or '早上' in text:
            if hour >= 12:
                hour -= 12
        elif '中午' in text:
            if hour < 12:
                hour = 12
        elif '凌晨' in text:
            if hour >= 12:
                hour -= 12
        
        return hour 