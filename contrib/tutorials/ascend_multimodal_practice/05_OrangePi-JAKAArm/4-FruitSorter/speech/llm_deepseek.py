# speech/llm_deepseek.py
"""DeepSeek 大语言模型模块（意图解析 + 多轮对话）。

本模块封装了与 DeepSeek 官方 API 的交互，提供两种能力：
    1. parse_intent(text)：把自然语言指令解析成结构化意图 JSON（给机械臂调度用）
    2. chat(text)：多轮自由对话（让机械臂具备"理解"复杂指令的能力）

设计原则：
    - 与 huawei_voice.py 解耦：语音（ASR/TTS）归华为云 SIS，语言理解归 DeepSeek。
    - 可缺失降级：无 DEEPSEEK_API_KEY 时退化为本地关键词规则。
    - 与 start.py 的意图调度兼容：parse_intent 返回 {intent, fruit, response}。

依赖：
    pip install openai python-dotenv

配置（.env）：
    DEEPSEEK_API_KEY=你的Key（https://platform.deepseek.com/ 获取）
    DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
    DEEPSEEK_MODEL=deepseek-v4-flash
"""
import os
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ============ 配置 ============
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

# 支持的水果（与 huawei_voice.py 保持一致）
FRUITS_CN2EN = {
    "牛油果": "cavocado",
    "柠檬": "lemon",
    "梨": "pear",
    "鸭梨": "pear",
    "芒果": "mango",
    "柿子": "persimmon",
}

# 意图解析的系统提示词（与 huawei_voice.py 共享同一套意图定义）
_INTENT_SYSTEM_PROMPT = """你是一台水果分拣机械臂的智能控制中枢。用户会用口语下达指令，
你需要把指令解析成系统能执行的意图，并只输出一个 JSON 对象，不要输出任何多余文字。

支持的意图 intent：
- camera_calib：摄像头九点标定
- arm_calib：机械臂九点标定
- eyehand_calib：手眼标定
- detect：打开摄像头检测水果
- pick：抓取某个水果（必须给出 fruit 字段）
- status：查询当前系统状态
- chat：自由对话（不属于以上指令的日常交流）
- quit：退出系统
- unknown：无法理解

fruit 取值范围（英文）：cavocado(牛油果)、lemon(柠檬)、pear(梨/鸭梨)、mango(芒果)、persimmon(柿子)。

输出 JSON 格式：
{"intent": "意图", "fruit": "可选-水果英文名", "response": "不超过30字的中文回复"}

示例：
指令：帮我抓个芒果   -> {"intent":"pick","fruit":"mango","response":"好的，开始抓取芒果"}
指令：你现在能做什么 -> {"intent":"chat","response":"我能通过语音、视觉和大模型控制机械臂抓取水果"}
指令：退出吧         -> {"intent":"quit","response":"好的，再见"}

当前用户指令："""

# 多轮对话的系统提示词（人设：水果分拣机械臂助手）
_CHAT_SYSTEM_PROMPT = """你是一台水果分拣机械臂的智能助手，搭载在香橙派（昇腾310 NPU）上。
你的能力：
1. 视觉：通过摄像头 + YOLO 模型识别 5 类水果（牛油果、柠檬、梨、芒果、柿子）
2. 语音：通过华为云 SIS 实现语音识别和语音播报
3. 抓取：通过 JAKA 机械臂抓取指定水果并放到目标区域

用户可能会问你各种问题，请简洁友好地回答（不超过100字）。
如果用户想抓水果，提醒他直接说"抓XX"。"""


class DeepSeekLLM:
    """DeepSeek 大语言模型封装：意图解析 + 多轮对话。"""

    def __init__(self):
        self._client = None
        self._chat_history = [{"role": "system", "content": _CHAT_SYSTEM_PROMPT}]
        self._ready = bool(DEEPSEEK_API_KEY)

        if not self._ready:
            print("[LLM] 未配置 DEEPSEEK_API_KEY，意图解析将使用本地规则")
            return
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
            print(f"[LLM] DeepSeek 已就绪（模型: {DEEPSEEK_MODEL}）")
        except Exception as e:
            print(f"[LLM] DeepSeek 初始化失败，退化为本地规则: {e}")
            self._client = None
            self._ready = False

    @property
    def ready(self):
        return self._ready and self._client is not None

    # ---------------- 意图解析（给调度器用）----------------
    def parse_intent(self, text):
        """把自然语言解析成 {intent, fruit, response}。"""
        if not text:
            return {"intent": "unknown", "response": "没听清，请再说一次"}

        if self.ready:
            result = self._parse_intent_llm(text)
            if result is not None:
                return self._normalize(result, text)

        return self._parse_rule(text)

    def _parse_intent_llm(self, text):
        """用 DeepSeek 做意图解析。"""
        try:
            resp = self._client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.1,
                stream=False,
            )
            content = resp.choices[0].message.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start == -1 or end == 0:
                return None
            return json.loads(content[start:end])
        except Exception as e:
            print(f"[LLM] DeepSeek 意图解析失败，回退规则: {e}")
            return None

    # ---------------- 多轮对话（自由交流）----------------
    def chat(self, text):
        """多轮对话，返回助手回复文本。同时维护对话历史。"""
        if not self.ready:
            return "（大模型未配置，无法对话。请在 .env 设置 DEEPSEEK_API_KEY）"

        self._chat_history.append({"role": "user", "content": text})
        # 限制历史长度，避免 token 超限（保留 system + 最近 10 轮）
        if len(self._chat_history) > 21:
            self._chat_history = [self._chat_history[0]] + self._chat_history[-20:]

        try:
            resp = self._client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=self._chat_history,
                temperature=0.7,
                stream=False,
            )
            reply = resp.choices[0].message.content.strip()
            self._chat_history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            print(f"[LLM] 对话失败: {e}")
            return f"对话出错: {e}"

    def reset_chat(self):
        """重置对话历史。"""
        self._chat_history = [{"role": "system", "content": _CHAT_SYSTEM_PROMPT}]

    # ---------------- 工具方法 ----------------
    @staticmethod
    def _normalize(result, text):
        """规整 LLM 输出。"""
        intent = result.get("intent", "unknown")
        fruit = result.get("fruit")
        if fruit:
            fruit = fruit.strip().lower()
            fruit = FRUITS_CN2EN.get(fruit, fruit)
            result["fruit"] = fruit
        result["intent"] = intent
        if not result.get("response"):
            result["response"] = "好的"
        return result

    @staticmethod
    def _parse_rule(text):
        """本地关键词规则（离线兜底）。"""
        t = text.lower()
        if ("摄像头" in t or "相机" in t) and ("标定" in t or "校准" in t):
            return {"intent": "camera_calib", "response": "开始摄像头标定"}
        if "机械臂" in t and ("标定" in t or "校准" in t):
            return {"intent": "arm_calib", "response": "开始机械臂标定"}
        if "手眼" in t or "仿射" in t or "矩阵" in t:
            return {"intent": "eyehand_calib", "response": "开始手眼标定"}
        if "退出" in t or "结束" in t or "再见" in t or "关闭" in t:
            return {"intent": "quit", "response": "好的，再见"}
        pick_keywords = ["抓", "拿", "取", "要", "给我", "夹"]
        if any(k in t for k in pick_keywords):
            for cn, en in FRUITS_CN2EN.items():
                if cn in t:
                    return {"intent": "pick", "fruit": en, "response": f"开始抓取{cn}"}
        if "摄像头" in t or "检测" in t or "看看" in t:
            return {"intent": "detect", "response": "打开摄像头开始检测"}
        if "状态" in t or "能做什么" in t or "功能" in t:
            return {"intent": "chat", "response": "我能通过视觉、语音和大模型控制机械臂抓取水果"}
        return {"intent": "chat", "response": "抱歉没听懂，可以说「抓芒果」「打开摄像头」等"}


# 自测
if __name__ == "__main__":
    llm = DeepSeekLLM()
    print("\n===== 意图解析测试 =====")
    for s in ["帮我抓个芒果", "我要一个柠檬", "摄像头标定", "退出", "你叫什么名字"]:
        print(f"  '{s}' -> {llm.parse_intent(s)}")

    print("\n===== 多轮对话测试 =====")
    if llm.ready:
        for s in ["你好", "你能做什么？", "帮我抓个牛油果"]:
            reply = llm.chat(s)
            print(f"  用户: {s}\n  助手: {reply}\n")
