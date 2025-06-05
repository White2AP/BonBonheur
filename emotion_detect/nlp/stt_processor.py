import speech_recognition as sr
import os
import wave
import audioop
from typing import Optional, Dict, List
import tempfile
from pydub import AudioSegment
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class STTProcessor:
    """
    语音转文本处理器
    支持多种STT引擎：Google、百度、讯飞等
    """
    
    def __init__(self, engine: str = 'google', language: str = 'zh-CN'):
        """
        初始化STT处理器
        
        Args:
            engine: STT引擎 ('google', 'baidu', 'sphinx', 'wit', 'azure')
            language: 语言代码 ('zh-CN', 'en-US', 'ja-JP' 等)
        """
        self.engine = engine
        self.language = language
        self.recognizer = sr.Recognizer()
        
        # 配置识别器参数
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.operation_timeout = None
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.8
        
        # API密钥配置（需要用户自行配置）
        self.api_keys = {
            'baidu': {
                'app_id': os.getenv('BAIDU_APP_ID'),
                'api_key': os.getenv('BAIDU_API_KEY'),
                'secret_key': os.getenv('BAIDU_SECRET_KEY')
            },
            'azure': {
                'key': os.getenv('AZURE_SPEECH_KEY'),
                'region': os.getenv('AZURE_SPEECH_REGION')
            },
            'wit': {
                'key': os.getenv('WIT_AI_KEY')
            }
        }
    
    def preprocess_audio(self, audio_file: str) -> str:
        """
        预处理音频文件
        
        Args:
            audio_file: 音频文件路径
            
        Returns:
            处理后的音频文件路径
        """
        try:
            # 加载音频文件
            audio = AudioSegment.from_file(audio_file)
            
            # 转换为单声道
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # 设置采样率为16kHz
            if audio.frame_rate != 16000:
                audio = audio.set_frame_rate(16000)
            
            # 音量标准化
            audio = audio.normalize()
            
            # 降噪处理（简单的高通滤波）
            audio = audio.high_pass_filter(300)
            
            # 保存处理后的音频
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            audio.export(temp_file.name, format='wav')
            
            logger.info(f"音频预处理完成: {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"音频预处理失败: {str(e)}")
            return audio_file
    
    def transcribe_audio(self, audio_file: str, preprocess: bool = True) -> Dict:
        """
        将音频文件转换为文本
        
        Args:
            audio_file: 音频文件路径
            preprocess: 是否预处理音频
            
        Returns:
            转录结果字典
        """
        result = {
            'success': False,
            'text': '',
            'confidence': 0.0,
            'error': None,
            'engine': self.engine,
            'language': self.language
        }
        
        try:
            # 预处理音频
            if preprocess:
                processed_file = self.preprocess_audio(audio_file)
            else:
                processed_file = audio_file
            
            # 加载音频文件
            with sr.AudioFile(processed_file) as source:
                # 调整环境噪音
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # 录制音频数据
                audio_data = self.recognizer.record(source)
            
            # 根据选择的引擎进行识别
            if self.engine == 'google':
                text = self._recognize_google(audio_data)
            elif self.engine == 'baidu':
                text = self._recognize_baidu(audio_data)
            elif self.engine == 'sphinx':
                text = self._recognize_sphinx(audio_data)
            elif self.engine == 'wit':
                text = self._recognize_wit(audio_data)
            elif self.engine == 'azure':
                text = self._recognize_azure(audio_data)
            else:
                raise ValueError(f"不支持的STT引擎: {self.engine}")
            
            result['success'] = True
            result['text'] = text
            result['confidence'] = 0.9  # 大多数引擎不返回置信度，使用默认值
            
            logger.info(f"STT转录成功: {text[:50]}...")
            
        except sr.UnknownValueError:
            result['error'] = "无法识别音频内容"
            logger.warning("STT无法识别音频内容")
        except sr.RequestError as e:
            result['error'] = f"STT服务请求失败: {str(e)}"
            logger.error(f"STT服务请求失败: {str(e)}")
        except Exception as e:
            result['error'] = f"STT处理异常: {str(e)}"
            logger.error(f"STT处理异常: {str(e)}")
        finally:
            # 清理临时文件
            if preprocess and processed_file != audio_file:
                try:
                    os.unlink(processed_file)
                except:
                    pass
        
        return result
    
    def _recognize_google(self, audio_data) -> str:
        """使用Google STT引擎"""
        return self.recognizer.recognize_google(
            audio_data, 
            language=self.language,
            show_all=False
        )
    
    def _recognize_baidu(self, audio_data) -> str:
        """使用百度STT引擎"""
        if not all(self.api_keys['baidu'].values()):
            raise ValueError("百度STT需要配置API密钥")
        
        return self.recognizer.recognize_baidu(
            audio_data,
            app_id=self.api_keys['baidu']['app_id'],
            api_key=self.api_keys['baidu']['api_key'],
            secret_key=self.api_keys['baidu']['secret_key'],
            language=self.language
        )
    
    def _recognize_sphinx(self, audio_data) -> str:
        """使用Sphinx STT引擎（离线）"""
        return self.recognizer.recognize_sphinx(audio_data, language=self.language)
    
    def _recognize_wit(self, audio_data) -> str:
        """使用Wit.ai STT引擎"""
        if not self.api_keys['wit']['key']:
            raise ValueError("Wit.ai STT需要配置API密钥")
        
        return self.recognizer.recognize_wit(
            audio_data,
            key=self.api_keys['wit']['key']
        )
    
    def _recognize_azure(self, audio_data) -> str:
        """使用Azure STT引擎"""
        if not all([self.api_keys['azure']['key'], self.api_keys['azure']['region']]):
            raise ValueError("Azure STT需要配置API密钥和区域")
        
        return self.recognizer.recognize_azure(
            audio_data,
            key=self.api_keys['azure']['key'],
            location=self.api_keys['azure']['region'],
            language=self.language
        )
    
    def transcribe_realtime(self, duration: int = 5) -> Dict:
        """
        实时语音识别
        
        Args:
            duration: 录音时长（秒）
            
        Returns:
            识别结果
        """
        result = {
            'success': False,
            'text': '',
            'confidence': 0.0,
            'error': None
        }
        
        try:
            with sr.Microphone() as source:
                logger.info("请开始说话...")
                # 调整环境噪音
                self.recognizer.adjust_for_ambient_noise(source)
                # 录制音频
                audio_data = self.recognizer.listen(source, timeout=duration)
            
            logger.info("录音完成，正在识别...")
            
            # 识别音频
            if self.engine == 'google':
                text = self._recognize_google(audio_data)
            elif self.engine == 'baidu':
                text = self._recognize_baidu(audio_data)
            elif self.engine == 'sphinx':
                text = self._recognize_sphinx(audio_data)
            else:
                text = self._recognize_google(audio_data)  # 默认使用Google
            
            result['success'] = True
            result['text'] = text
            result['confidence'] = 0.9
            
            logger.info(f"实时STT识别成功: {text}")
            
        except sr.UnknownValueError:
            result['error'] = "无法识别语音内容"
        except sr.RequestError as e:
            result['error'] = f"STT服务请求失败: {str(e)}"
        except Exception as e:
            result['error'] = f"实时STT处理异常: {str(e)}"
        
        return result
    
    def batch_transcribe(self, audio_files: List[str]) -> List[Dict]:
        """
        批量转录音频文件
        
        Args:
            audio_files: 音频文件路径列表
            
        Returns:
            转录结果列表
        """
        results = []
        for i, audio_file in enumerate(audio_files):
            logger.info(f"正在处理第 {i+1}/{len(audio_files)} 个文件: {audio_file}")
            result = self.transcribe_audio(audio_file)
            result['file_path'] = audio_file
            results.append(result)
        
        return results
    
    def get_audio_info(self, audio_file: str) -> Dict:
        """
        获取音频文件信息
        
        Args:
            audio_file: 音频文件路径
            
        Returns:
            音频信息字典
        """
        try:
            audio = AudioSegment.from_file(audio_file)
            
            return {
                'duration': len(audio) / 1000.0,  # 秒
                'channels': audio.channels,
                'sample_rate': audio.frame_rate,
                'sample_width': audio.sample_width,
                'frame_count': audio.frame_count(),
                'file_size': os.path.getsize(audio_file),
                'format': audio_file.split('.')[-1].lower()
            }
        except Exception as e:
            logger.error(f"获取音频信息失败: {str(e)}")
            return {}
    
    def set_engine(self, engine: str):
        """设置STT引擎"""
        self.engine = engine
        logger.info(f"STT引擎已切换为: {engine}")
    
    def set_language(self, language: str):
        """设置识别语言"""
        self.language = language
        logger.info(f"STT语言已切换为: {language}")
    
    def configure_api_keys(self, engine: str, **kwargs):
        """
        配置API密钥
        
        Args:
            engine: 引擎名称
            **kwargs: API密钥参数
        """
        if engine in self.api_keys:
            self.api_keys[engine].update(kwargs)
            logger.info(f"{engine} API密钥已更新")
        else:
            logger.warning(f"未知的引擎: {engine}") 