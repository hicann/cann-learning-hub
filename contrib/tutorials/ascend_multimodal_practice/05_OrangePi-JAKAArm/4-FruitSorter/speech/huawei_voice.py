# speech/huawei_voice.py
"""华为云语音交互模块（ASR 语音识别 + TTS 语音播报 + 意图解析）。

本模块把"听 / 说 / 理解"三件事封装成一个独立的 VoiceAssistant 类，
供 start.py 调用，作为机械臂分拣系统的语音外壳。

设计原则：
    - 语音功能"外包"在此文件，不改动任何原有功能代码（标定 / 抓取 / 视觉）。
    - 所有外部依赖（华为云 SIS、SpeechRecognition、OpenAI、playsound）都做了
      "可缺失降级"处理：缺哪个就退化哪一部分，不会让整个程序崩溃，方便本地调试。

技术来源：参考 2.华为语音SDK/speech_llms/Speech&OCR.ipynb
    - ASR：华为云一句话识别 sasr（chinese_16k_general）
    - TTS：华为云语音合成 tts（预置音色）
    - 录音：SpeechRecognition + 麦克风，转 16kHz/16bit
    - 意图解析：DeepSeek 官方 API（通过 llm_deepseek.py 模块）

依赖安装：
    pip install SpeechRecognition pyaudio playsound==1.2.2 python-dotenv openai
    华为云 SIS SDK（见 2.华为语音SDK/speech_llms 目录）：
        cd <speech_llms 目录> && python setup.py install --user

配置（在 4-FruitSorter 目录下创建 .env，参考 .env.example）：
    HUAWEI_SIS_AK=你的AccessKey
    HUAWEI_SIS_SK=你的SecretKey
    HUAWEI_SIS_REGION=cn-east-3
    HUAWEI_SIS_PROJECT_ID=你的项目ID
    DEEPSEEK_API_KEY=你的DeepSeekAPIKey（https://platform.deepseek.com/ 获取）
"""
import os
import json
import subprocess

# 加载 .env（缺失 python-dotenv 也不影响：可改用系统环境变量）
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ============ 配置（统一从环境变量读取）============
AK = os.getenv("HUAWEI_SIS_AK", "")
SK = os.getenv("HUAWEI_SIS_SK", "")
REGION = os.getenv("HUAWEI_SIS_REGION", "cn-east-3")
PROJECT_ID = os.getenv("HUAWEI_SIS_PROJECT_ID", "")

TTS_VOICE = os.getenv("HUAWEI_TTS_VOICE", "chinese_xiaoyan_common")

# 支持的水果：中文 -> 英文（与 perception/camera_debug.py 的 FRUIT_MAP、
# yolo_detector 的 class_names 保持一致：cavocado / lemon / pear / mango / persimmon）
FRUITS_CN2EN = {
    "牛油果": "cavocado",
    "柠檬": "lemon",
    "梨": "pear",
    "鸭梨": "pear",
    "芒果": "mango",
    "柿子": "persimmon",
}



class VoiceAssistant:
    """语音助手：listen() 听、speak() 说、parse_intent() 理解。"""

    def __init__(self, record_seconds=5):
        self.record_seconds = record_seconds
        self._sr = None
        self._recognizer = None
        self._mic = None
        self._llm = None
        # ASR / TTS 是否具备凭证
        self._sis_ready = bool(AK and SK and PROJECT_ID)
        if not self._sis_ready:
            print("[VOICE] 未配置华为云 AK/SK/PROJECT_ID，ASR/TTS 将不可用（可在 .env 配置）")

        self._init_mic()
        self._init_llm()

    # ---------------- 初始化 ----------------
    def _init_mic(self):
        try:
            import speech_recognition as sr
            self._sr = sr
            self._recognizer = sr.Recognizer()
            self._mic = sr.Microphone()
            with self._mic as source:
                print("[VOICE] 正在校准环境噪声...")
                self._recognizer.adjust_for_ambient_noise(source, duration=1)
            print("[VOICE] 麦克风初始化完成")
        except Exception as e:
            print(f"[VOICE] 麦克风初始化失败（将退化为键盘输入）: {e}")
            self._recognizer = None
            self._mic = None

    def _init_llm(self):
        """初始化 LLM（DeepSeek），用于意图解析和对话。"""
        try:
            from speech.llm_deepseek import DeepSeekLLM
            self._llm = DeepSeekLLM()
            if self._llm.ready:
                print("[VOICE] DeepSeek LLM 已就绪，意图解析+对话可用")
            else:
                print("[VOICE] DeepSeek 未配置，意图解析使用本地规则（.env 设置 DEEPSEEK_API_KEY）")
        except Exception as e:
            print(f"[VOICE] LLM 模块加载失败，使用本地规则: {e}")
            self._llm = None

    # ---------------- 听：录音 + ASR ----------------
    def listen(self):
        """录制一段语音并返回识别文本。麦克风不可用时退化为键盘输入。"""
        if self._recognizer is None or self._mic is None:
            try:
                return input("[VOICE] 麦克风不可用，请键入指令: ").strip()
            except EOFError:
                return ""

        try:
            with self._mic as source:
                print("[VOICE] 🎤 请说话...")
                audio = self._recognizer.listen(
                    source, timeout=8, phrase_time_limit=self.record_seconds
                )
            audio_data = audio.get_wav_data(convert_rate=16000, convert_width=2)
            wav_path = "voice_command.wav"
            with open(wav_path, "wb") as f:
                f.write(audio_data)
        except self._sr.WaitTimeoutError:
            print("[VOICE] ⏰ 超时：未检测到语音")
            return ""
        except Exception as e:
            print(f"[VOICE] 录音失败: {e}")
            return ""

        return self._asr(wav_path)

    def _asr(self, wav_path):
        """华为云一句话识别。"""
        if not self._sis_ready:
            print("[VOICE] 未配置华为云凭证，无法进行语音识别")
            return ""
        try:
            from huaweicloud_sis.client.asr_client import AsrCustomizationClient
            from huaweicloud_sis.bean.asr_request import AsrCustomShortRequest
            from huaweicloud_sis.bean.sis_config import SisConfig
            from huaweicloud_sis.utils import io_utils

            config = SisConfig()
            config.set_connect_timeout(10)
            config.set_read_timeout(10)
            client = AsrCustomizationClient(AK, SK, REGION, PROJECT_ID, sis_config=config)

            data = io_utils.encode_file(wav_path)
            req = AsrCustomShortRequest("wav", "chinese_16k_general", data)
            req.set_add_punc("no")
            req.set_digit_norm("yes")
            req.set_need_word_info("no")
            result = client.get_short_response(req)

            text = result["result"]["text"].strip()
            print(f"[VOICE] 📝 识别结果: {text}")
            return text
        except Exception as e:
            print(f"[VOICE] 语音识别失败: {e}")
            return ""

    # ---------------- 说：TTS + 播放 ----------------
    def speak(self, text):
        """把文本合成语音并播放。无凭证时仅打印文本。"""
        if not text:
            return
        print(f"[VOICE] 🔊 播报: {text}")
        if not self._sis_ready:
            return
        try:
            from huaweicloud_sis.client.tts_client import TtsCustomizationClient
            from huaweicloud_sis.bean.tts_request import TtsCustomRequest
            from huaweicloud_sis.bean.sis_config import SisConfig

            config = SisConfig()
            config.set_connect_timeout(10)
            config.set_read_timeout(10)
            client = TtsCustomizationClient(AK, SK, REGION, PROJECT_ID, sis_config=config)

            out_path = "voice_reply.wav"
            req = TtsCustomRequest(text)
            req.set_property(TTS_VOICE)
            req.set_audio_format("wav")
            req.set_sample_rate("16000")
            req.set_saved(True)
            req.set_saved_path(out_path)
            client.get_ttsc_response(req)

            self._play(out_path)
        except Exception as e:
            print(f"[VOICE] 语音合成失败: {e}")

    @staticmethod
    def _play(path):
        """跨平台播放音频：优先 playsound，失败回退到 aplay（Linux/香橙派）。"""
        if not os.path.exists(path):
            print(f"[VOICE] 音频文件不存在: {path}")
            return
        try:
            from playsound import playsound
            playsound(path)
            return
        except Exception:
            pass
        try:
            subprocess.run(["aplay", path], check=False)
        except Exception as e:
            print(f"[VOICE] 音频播放失败: {e}")

    # ---------------- 理解：意图解析 ----------------
    def parse_intent(self, text):
        """把识别文本解析成 {'intent':..., 'fruit':..., 'response':...}。"""
        if not text:
            return {"intent": "unknown", "response": "没听清，请再说一次"}

        if self._llm is not None:
            return self._llm.parse_intent(text)

        return self._parse_rule(text)

    def chat(self, text):
        """多轮对话（委托给 DeepSeek LLM）。"""
        if self._llm is not None:
            return self._llm.chat(text)
        return "（大模型未配置）"


    @staticmethod
    def _parse_rule(text):
        """本地关键词规则解析（离线兜底）。"""
        t = text.lower()

        if ("摄像头" in t or "相机" in t) and ("标定" in t or "校准" in t):
            return {"intent": "camera_calib", "response": "开始摄像头标定"}
        if "机械臂" in t and ("标定" in t or "校准" in t):
            return {"intent": "arm_calib", "response": "开始机械臂标定"}
        if "手眼" in t or "仿射" in t or ("矩阵" in t):
            return {"intent": "eyehand_calib", "response": "开始手眼标定"}
        if "退出" in t or "结束" in t or "再见" in t or "关闭系统" in t:
            return {"intent": "quit", "response": "好的，再见"}

        pick_keywords = ["抓", "拿", "取", "要", "给我", "夹"]
        if any(k in t for k in pick_keywords):
            for cn, en in FRUITS_CN2EN.items():
                if cn in t:
                    return {"intent": "pick", "fruit": en, "response": f"开始抓取{cn}"}

        if "摄像头" in t or "检测" in t or "看看" in t:
            return {"intent": "detect", "response": "打开摄像头开始检测"}

        return {"intent": "unknown", "response": "抱歉，没听清，请再说一次"}


# 简单自测（不连接硬件，仅测试意图解析）
if __name__ == "__main__":
    assistant = VoiceAssistant()
    samples = [
        "帮我抓个芒果",
        "我要一个柠檬",
        "摄像头标定",
        "做一下手眼标定",
        "打开摄像头看看",
        "退出吧",
        "今天天气怎么样",
    ]
    print("\n===== 意图解析自测 =====")
    for s in samples:
        print(f"  '{s}' -> {assistant.parse_intent(s)}")
