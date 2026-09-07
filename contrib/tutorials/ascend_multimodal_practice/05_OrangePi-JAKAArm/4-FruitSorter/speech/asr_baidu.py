import pyaudio
import wave
from aip import AipSpeech

APP_ID = "119426775"
API_KEY = "APuDVMhunwewuZnDgyWKbOM7"
SECRET_KEY = "dFZS1KBfLkE9mFG2Nmjr2PhKBbaPJ1Q2"

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