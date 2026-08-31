import os

# 关键：解决 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="dataset.yaml",  # 使用修正后的yaml
        epochs=100,
        imgsz=640,
        batch=8,
        workers=0,
        name="fruit_yolo11n_corrected",
        patience=50,
        # 清空缓存，重新读取数据
        cache=False,
    )
    
    print("\n训练完成！")
    print("现在标签映射是正确的：")
    print("  0: apple")
    print("  1: avocado")
    print("  2: banana")
    print("  3: lemon")
    print("  4: mandarin")
    print("  5: mango")
    print("  6: pear")
    print("  7: persimmon")
    print("  8: pomegranate")
    print("  9: strawberry")


if __name__ == '__main__':
    main()