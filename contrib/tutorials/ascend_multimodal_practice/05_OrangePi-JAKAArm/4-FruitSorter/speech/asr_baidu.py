import pyaudio
import wave
import os
from aip import AipSpeech

# 从环境变量读取（在 .env 中配置 BAIDU_APP_ID / BAIDU_API_KEY / BAIDU_SECRET_KEY）
APP_ID = os.getenv("BAIDU_APP_ID", "")
API_KEY = os.getenv("BAIDU_API_KEY", "")
SECRET_KEY = os.getenv("BAIDU_SECRET_KEY", "")

if not (APP_ID and API_KEY and SECRET_KEY):
    print("[ASR] 未配置百度语音凭据（BAIDU_APP_ID/BAIDU_API_KEY/BAIDU_SECRET_KEY），"
          "请在 .env 中填写；或改用主方案的华为云 SIS（见 README）")

class BaiduASR:
    def __init__(self):
        self.client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
        self.listening = True
        self.audio = None
        self.stream = None
        
    def init_audio(self):
        """Initialize audio device"""
        if self.audio is None:
            self.audio = pyaudio.PyAudio()
            
    def pause_listening(self):
        """Pause listening"""
        self.listening = False
        print("[ASR] Voice recognition paused")
        
    def resume_listening(self):
        """Resume listening"""
        self.listening = True
        print("[ASR] Voice recognition resumed")
        
    def listen(self):
        """Listen to voice input"""
        if not self.listening:
            return ""
            
        self.init_audio()
        
        # Open audio stream
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )

        print("[ASR] Listening... (Please speak)")
        frames = []
        
        # Record for 4 seconds
        for _ in range(0, int(16000 / 1024 * 4)):
            frames.append(self.stream.read(1024))

        self.stream.stop_stream()
        self.stream.close()

        # Recognize speech
        result = self.client.asr(b"".join(frames), 'pcm', 16000, {
            'dev_pid': 1537
        })

        if "result" in result:
            text = result["result"][0]
            print(f"[ASR] Recognized: {text}")
            return text
        return ""
    
    def cleanup(self):
        """Clean up resources"""
        if self.stream:
            self.stream.close()
        if self.audio:
            self.audio.terminate()