# speech/llm_parser.py
import json
import os

# 默认使用大模型模式
USE_LLM = True

# 填入你的API Key（从百度AI Studio获取）
ERNIE_BOT_API_KEY = '60d2fa48f1f3f6f7dc8c2b636eac06da97f300ac'

class LLMParser:
    def __init__(self):
        self.bot = None
        
        if USE_LLM:
            try:
                # erniebot 0.5.9 的正确导入方式
                import erniebot
                
                # 设置API Key
                erniebot.api_type = "aistudio"  # 使用AI Studio
                erniebot.access_token = ERNIE_BOT_API_KEY
                
                print("[LLMParser] 初始化文心大模型...")
                
                # 简单测试是否可用
                response = erniebot.ChatCompletion.create(
                    model="ernie-3.5",
                    messages=[{"role": "user", "content": "一句话简单欢迎用户"}],
                    stream=False
                )
                
                print(f"[LLMParser] 文心大模型初始化成功: {response['result']}")
                self.bot = erniebot
                
            except ImportError as e:
                print(f"[LLMParser] erniebot库导入失败: {e}")
                print("请安装: pip install erniebot")
                self.bot = None
            except Exception as e:
                print(f"[LLMParser] 大模型初始化失败: {e}")
                print("将回退到规则模式")
                self.bot = None
        else:
            print("[LLMParser] 使用规则模式")
            self.bot = None
    
    def parse(self, text: str):
        print(f"[LLMParser] 解析文本: '{text}'")
        
        if USE_LLM and self.bot:
            return self._parse_llm(text)
        else:
            return self._parse_rule(text)
    
    # ---------- 规则模式 ----------
    def _parse_rule(self, text: str):
        text = text.lower()
        
        # 手眼标定相关
        if "摄像头" in text and ("标定" in text or "校准" in text):
            return {"intent": "camera_calib"}
        if "相机" in text and ("标定" in text or "校准" in text):
            return {"intent": "camera_calib"}
        if "机械臂" in text and ("标定" in text or "校准" in text):
            return {"intent": "arm_calib"}
        if "手眼" in text or "仿射" in text or "映射" in text:
            return {"intent": "eyehand_calib"}
        
        # 水果抓取 - 支持多种表达方式
        FRUITS = {
            "苹果": "apple",
            "香蕉": "banana",
            "梨": "pear",
            "鸭梨": "pear",
            "橘子": "mandarin",
            "橙子": "mandarin",
            "柠檬": "lemon",
            "芒果": "mango",
            "草莓": "strawberry",
            "牛油果": "avocado",
            "石榴": "pomegranate",
            "柿子": "persimmon"
        }
        
        # 检查是否包含抓取关键词
        pick_keywords = ["抓", "拿", "取", "要", "给我"]
        has_pick_keyword = any(keyword in text for keyword in pick_keywords)
        
        if has_pick_keyword:
            for chinese_name, english_name in FRUITS.items():
                if chinese_name in text:
                    return {"intent": "pick", "fruit": english_name}
        
        # 目标区域标定指令
        if "目标区域" in text or "篮子位置" in text or "篮子标定" in text:
            return {"intent": "calib_target_area"}
        
        return {"intent": "unknown"}
    
    # ---------- 大模型模式 ----------
    def _parse_llm(self, text: str):
        try:
            # 系统提示词
            system_prompt = """你是一个运行在香橙派上的机械臂控制智能体。
你只能通过调用函数来完成任务，禁止编造不存在的动作。

系统支持的函数如下：
- camera_calib(): 进行摄像头九点标定
- arm_calib(): 进行机械臂九点标定
- eyehand_calib(): 计算手眼仿射变换矩阵
- pick_fruit(fruit): 抓取指定水果并放置到篮子（水果参数: apple, banana, pear, mandarin, lemon, mango, strawberry, avocado, pomegranate, persimmon）
- calib_target_area(): 手动标定4个篮子位置

注意：pick_fruit函数会执行完整流程：检测水果 → 抓取 → 放置到第一个篮子 → 复位机械臂

规则：
1. 用户可能普通话不标准、发音不清，但语义明确
2. 输出必须是JSON格式
3. function为函数调用列表（最多一个）
4. response为不超过20字的简短反馈

输出格式：
{
  "function": ["函数名(参数)"],
  "response": "简短回复"
}

示例：
指令：开始摄像头标定
输出：
{
  "function": ["camera_calib()"],
  "response": "开始相机九点标定"
}

指令：抓个苹果
输出：
{
  "function": ["pick_fruit(apple)"],
  "response": "开始抓取苹果"
}

指令：帮我拿个香蕉
输出：
{
  "function": ["pick_fruit(banana)"],
  "response": "开始抓取香蕉"
}

当前指令："""
            
            messages = [
                {"role": "user", "content": system_prompt + text}
            ]
            
            response = self.bot.ChatCompletion.create(
                model="ernie-3.5",
                messages=messages,
                temperature=0.1,  # 低温度，更确定性
                stream=False
            )
            
            result_text = response['result']
            print(f"[LLMParser] 大模型原始返回: {result_text}")
            
            # 提取JSON部分
            try:
                # 尝试找到JSON开始和结束
                start_idx = result_text.find('{')
                end_idx = result_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx != 0:
                    json_str = result_text[start_idx:end_idx]
                    data = json.loads(json_str)
                    
                    # 提取意图
                    if "function" in data and data["function"]:
                        func_call = data["function"][0]
                        
                        # 解析函数调用
                        if func_call.startswith("pick_fruit("):
                            # 提取水果名称
                            fruit = func_call.split("(")[1].split(")")[0].replace("'", "").replace('"', '')
                            return {"intent": "pick", "fruit": fruit}
                        elif func_call.startswith("camera_calib()"):
                            return {"intent": "camera_calib"}
                        elif func_call.startswith("arm_calib()"):
                            return {"intent": "arm_calib"}
                        elif func_call.startswith("eyehand_calib()"):
                            return {"intent": "eyehand_calib"}
                        elif func_call.startswith("calib_target_area()"):
                            return {"intent": "calib_target_area"}
                
            except json.JSONDecodeError as e:
                print(f"[LLMParser] JSON解析失败: {e}, 原始文本: {result_text}")
            
            # 如果JSON解析失败，回退到规则模式
            print("[LLMParser] 回退到规则模式")
            return self._parse_rule(text)
            
        except Exception as e:
            print(f"[LLMParser] 大模型调用失败: {e}")
            # 回退规则模式
            return self._parse_rule(text)

# 测试函数
def test():
    parser = LLMParser()
    
    test_cases = [
        "抓个苹果",
        "帮我拿个香蕉",
        "我要一个梨",
        "给我个草莓",
        "取个芒果",
        "摄像头标定",
        "目标区域标定"
    ]
    
    print("测试大模型解析器:")
    for text in test_cases:
        result = parser.parse(text)
        print(f"  '{text}' -> {result}")

if __name__ == "__main__":
    test()