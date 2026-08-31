import cv2, os, time, argparse
from datetime import datetime

# CLASS_MAP: 按键 -> 类别名
# 顺序必须与 YOLO 训练 names 列表完全一致：
# 0 apple
# 1 pomegranate
# 2 persimmon
# 3 mandarin
# 4 lemon
# 5 avocado
# 6 strawberry
# 7 mango
# 8 pear
# 9 banana

CLASS_MAP = {
    '1': 'apple',
    '2': 'pomegranate',
    '3': 'persimmon',
    '4': 'mandarin',
    '5': 'lemon',
    '6': 'avocado',
    '7': 'strawberry',
    '8': 'mango',
    '9': 'pear',
    '0': 'banana'
}

def ensure_dirs(base='dataset'):
    for split in ['train','val']:
        for d in ['images','labels']:
            path = os.path.join(base, d, split)
            os.makedirs(path, exist_ok=True)

def save_placeholder_label(img_path):
    txt = os.path.splitext(img_path)[0] + '.txt'
    if not os.path.exists(txt):
        open(txt, 'w').close()

def main(device=1, split='train', auto=False, interval=2):
    ensure_dirs()
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Camera open failed (device={device})")

    cur_key = '1'
    print("按数字键(1-9,0)选择类别；'s'保存；'a'自动；'q'退出。")
    auto_mode = auto
    last_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧，退出。")
            break

        cv2.putText(
            frame,
            f"Mode: {'AUTO' if auto_mode else 'MANUAL'}  Class:{CLASS_MAP[cur_key]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("capture", frame)
        key = cv2.waitKey(1) & 0xFF

        # 自动模式触发保存
        if auto_mode and time.time() - last_time >= interval:
            key = ord('s')

        if key != 255:
            char = chr(key)
            if char in CLASS_MAP:
                cur_key = char
                print("切换类别为：", CLASS_MAP[cur_key])

            elif key == ord('s'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cls = CLASS_MAP[cur_key]
                filename = f"{cls}_{ts}.jpg"
                out_path = os.path.join('dataset', 'images', split, filename)
                cv2.imwrite(out_path, frame)
                save_placeholder_label(out_path)
                print("保存：", out_path)
                last_time = time.time()

            elif key == ord('a'):
                auto_mode = not auto_mode
                print("自动模式：", auto_mode)

            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--interval', type=float, default=2.0)
    args = parser.parse_args()

    main(args.device, args.split, args.auto, args.interval)
