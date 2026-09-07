#!/usr/bin/env python3
from tools.eyehand_calib import run_eyehand_calib
from tools.calib_camera_auto import run_camera_calib
from tools.calib_arm import run_arm_calib
from control.gripper_pwm import open_gripper, close_gripper
import csv
import time
import os


def print_voice_help():
    """打印语音指令说明"""
    print("\n" + "="*50)
    print("🍎 多模态机械臂控制系统")
    print("（视觉 + 语音 + DeepSeek 大模型）")
    print("="*50)
    print("请对着麦克风说出指令，例如：")
    print("  「摄像头标定」    - 摄像头九点标定")
    print("  「机械臂标定」    - 机械臂九点标定")
    print("  「手眼标定」      - 计算手眼矩阵")
    print("  「打开摄像头」    - 视觉检测并播报当前水果")
    print("  「帮我抓个芒果」  - 语音抓取指定水果")
    print("  「你能做什么」    - 与 DeepSeek 对话（LLM 理解）")
    print("  「退出」          - 退出系统")
    print("="*50)
    print("三种控制方式：")
    print("  👁️  视觉：YOLO 识别 → 自动抓取")
    print("  🎤  语音：ASR 识别 → 意图解析 → 执行")
    print("  🧠  LLM：DeepSeek 对话 → 理解复杂指令")
    print("="*50)


def check_calibration_status():
    """检查标定状态"""
    files_exist = {
        '摄像头标定': os.path.exists('data/cam_point.csv'),
        '机械臂标定': os.path.exists('data/arm_point.csv'),
        '手眼标定': os.path.exists('data/config_relation_matrix.csv')
    }
    
    print("\n📊 标定状态：")
    for name, exists in files_exist.items():
        status = "✅ 完成" if exists else "❌ 未完成"
        print(f"  {name}: {status}")
    
    return all(files_exist.values())


def voice_scan(camera, detector, assistant, duration=6):
    """语音检测：扫描若干秒，播报检测到的水果（复用 camera_debug 的 DetectionCache）。"""
    from perception.camera_debug import DetectionCache, FRUIT_MAP_REVERSE

    if camera is None or detector is None:
        assistant.speak("摄像头或检测器未初始化")
        return

    cache = DetectionCache()
    start = time.time()
    show = bool(os.environ.get('DISPLAY'))
    while time.time() - start < duration:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        try:
            cache.update(detector.detect(frame))
        except Exception as e:
            print(f"[VOICE] 检测失败: {e}")
        if show:
            try:
                import cv2
                cv2.imshow('Voice Scan', frame)
                cv2.waitKey(1)
            except Exception:
                show = False
    if show:
        try:
            import cv2
            cv2.destroyWindow('Voice Scan')
        except Exception:
            pass

    if cache.ready_fruits:
        names = "、".join(FRUIT_MAP_REVERSE.get(f, f) for f in cache.ready_fruits)
        assistant.speak(f"检测到{names}，请说出要抓取哪一个")
    else:
        assistant.speak("没有检测到水果，请把水果放到检测区")


def voice_grab(camera, detector, agent, assistant, target_fruit, timeout=20):
    """语音抓取：持续检测直到确认目标水果，再调用原有抓取流程。"""
    from perception.camera_debug import DetectionCache, FRUIT_MAP_REVERSE

    fruit_cn = FRUIT_MAP_REVERSE.get(target_fruit, target_fruit)

    if agent is None or agent.arm is None:
        assistant.speak("机械臂未初始化，无法抓取")
        return
    if camera is None or detector is None:
        assistant.speak("摄像头或检测器未初始化")
        return

    assistant.speak(f"正在寻找{fruit_cn}")
    cache = DetectionCache()
    start = time.time()
    show = bool(os.environ.get('DISPLAY'))
    position = None

    while time.time() - start < timeout:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        try:
            cache.update(detector.detect(frame))
        except Exception as e:
            print(f"[VOICE] 检测失败: {e}")
        if show:
            try:
                import cv2
                cv2.imshow('Voice Grab', frame)
                cv2.waitKey(1)
            except Exception:
                show = False
        if cache.is_ready(target_fruit):
            position = cache.get_position(target_fruit)
            break

    if show:
        try:
            import cv2
            cv2.destroyWindow('Voice Grab')
        except Exception:
            pass

    if not position:
        assistant.speak(f"没有找到{fruit_cn}，请把它放到检测区后重试")
        return

    assistant.speak(f"找到{fruit_cn}，开始抓取")
    try:
        success = agent.pick_and_place(position, target_fruit)
        if success:
            agent.reset_arm()
            assistant.speak(f"{fruit_cn}抓取完成")
        else:
            assistant.speak(f"{fruit_cn}抓取失败")
    except Exception as e:
        print(f"[ERROR] 抓取过程出错: {e}")
        import traceback
        traceback.print_exc()
        assistant.speak("抓取过程出现错误")


def main():
    print("[SYSTEM] 系统启动中...")
    
    # 检查标定状态
    all_calibrated = check_calibration_status()
    if not all_calibrated:
        print("\n⚠️  部分标定未完成，请先完成标定后再进行抓取操作")

    # 硬件初始化
    arm = None
    try:
        from control.gripper_pwm import init_gripper
        from control.jaka_arm import JakaArm
        
        print("[SYSTEM] 初始化夹爪...")
        init_gripper()
        
        print("[SYSTEM] 初始化机械臂...")
        arm = JakaArm()
            
    except Exception as e:
        print(f"[ERROR] 硬件初始化失败: {e}")
        print("[WARN] 将以无机械臂模式运行")

    # 感知模块
    camera = None
    detector = None
    try:
        from perception.camera import Camera
        from perception.yolo_detector import YoloDetector
        
        print("[SYSTEM] 初始化摄像头...")
        camera = Camera()
        
        print("[SYSTEM] 初始化YOLO检测器...")
        detector = YoloDetector()
        
    except Exception as e:
        print(f"[ERROR] 感知模块初始化失败: {e}")

    # 任务代理
    agent = None
    if camera and detector:
        try:
            from agent.task_agent import TaskAgent
            print("[SYSTEM] 初始化任务代理...")
            agent = TaskAgent(
                camera=camera,
                detector=detector,
                arm=arm,
                gripper=None
            )
        except Exception as e:
            print(f"[ERROR] 任务代理初始化失败: {e}")

    # 语音助手初始化（语音功能外包在 speech/huawei_voice.py）
    assistant = None
    try:
        from speech.huawei_voice import VoiceAssistant
        print("[SYSTEM] 初始化语音助手...")
        assistant = VoiceAssistant()
    except Exception as e:
        print(f"[ERROR] 语音助手初始化失败: {e}")

    if assistant is None:
        print("[ERROR] 语音助手不可用，系统退出")
        return

    print("\n[SYSTEM] 系统准备就绪！")
    print_voice_help()
    assistant.speak("水果分拣系统已就绪，请说出指令")

    # 主循环（语音控制）
    while True:
        try:
            text = assistant.listen()
            if not text:
                continue

            intent_data = assistant.parse_intent(text)
            intent = intent_data.get('intent', 'unknown')
            response = intent_data.get('response', '')
            if response:
                assistant.speak(response)

            if intent == 'camera_calib':
                print("\n[SYSTEM] 开始摄像头标定...")
                if camera:
                    camera.release()
                    time.sleep(1)
                try:
                    run_camera_calib()
                except Exception as e:
                    print(f"[ERROR] 摄像头标定失败: {e}")
                # 重新初始化摄像头
                if camera:
                    try:
                        from perception.camera import Camera
                        camera = Camera()
                    except:
                        pass

            elif intent == 'arm_calib':
                print("\n[SYSTEM] 开始机械臂标定...")
                if arm:
                    try:
                        run_arm_calib(arm)
                    except Exception as e:
                        print(f"[ERROR] 机械臂标定失败: {e}")
                else:
                    print("[ERROR] 机械臂未初始化")
                    assistant.speak("机械臂未初始化")

            elif intent == 'eyehand_calib':
                print("\n[SYSTEM] 开始手眼标定...")
                if not os.path.exists('data/cam_point.csv'):
                    print("[ERROR] 请先完成摄像头标定！")
                    assistant.speak("请先完成摄像头标定")
                elif not os.path.exists('data/arm_point.csv'):
                    print("[ERROR] 请先完成机械臂标定！")
                    assistant.speak("请先完成机械臂标定")
                else:
                    try:
                        run_eyehand_calib()
                        print("[SYSTEM] 手眼标定完成！")
                        assistant.speak("手眼标定完成")
                    except Exception as e:
                        print(f"[ERROR] 手眼标定失败: {e}")

            elif intent == 'detect':
                print("\n[SYSTEM] 打开摄像头检测...")
                voice_scan(camera, detector, assistant)

            elif intent == 'pick':
                fruit = intent_data.get('fruit')
                if fruit:
                    voice_grab(camera, detector, agent, assistant, fruit)
                else:
                    assistant.speak("请说明要抓哪种水果")

            elif intent == 'chat':
                # 🧠 LLM 多轮对话模式：DeepSeek 理解并回复
                reply = assistant.chat(text)
                print(f"[LLM] 🤖 {reply}")
                assistant.speak(reply)

            elif intent == 'status':
                # 查询系统状态
                status_parts = []
                status_parts.append(f"机械臂: {'已连接' if arm else '未连接'}")
                status_parts.append(f"摄像头: {'正常' if camera else '未初始化'}")
                status_parts.append(f"检测器: {'就绪' if detector else '未初始化'}")
                all_calib = check_calibration_status()
                status_parts.append(f"标定: {'全部完成' if all_calib else '未完成'}")
                status_text = "，".join(status_parts)
                print(f"[STATUS] {status_text}")
                assistant.speak(status_text)

            elif intent == 'quit':
                print("\n[SYSTEM] 退出系统")
                assistant.speak("系统退出，再见")
                break

            else:
                # 未识别的指令交给 LLM 对话处理（让大模型尝试理解）
                print(f"[SYSTEM] 未匹配预设意图，交给 DeepSeek 对话: {text}")
                reply = assistant.chat(text)
                print(f"[LLM] 🤖 {reply}")
                assistant.speak(reply)

        except KeyboardInterrupt:
            print("\n[SYSTEM] 用户退出程序")
            break
        except Exception as e:
            print(f"[ERROR] 执行错误: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)


if __name__ == "__main__":
    main()