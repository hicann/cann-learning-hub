# file: tools/eyehand_calib.py
import numpy as np
import csv
import os

def run_eyehand_calib():
    """
    计算手眼仿射变换矩阵（参考项目的计算方法）
    """
    print("[EYEHAND] Computing eye-hand affine matrix (参考项目方法)")
    print("=" * 60)
    
    # 确保data目录存在
    os.makedirs("data", exist_ok=True)
    
    # 检查必要文件
    cam_file = "data/cam_point.csv"
    arm_file = "data/arm_point.csv"
    
    if not os.path.exists(cam_file):
        print(f"[ERROR] 摄像头坐标文件不存在: {cam_file}")
        print("请先进行摄像头标定")
        return
    
    if not os.path.exists(arm_file):
        print(f"[ERROR] 机械臂坐标文件不存在: {arm_file}")
        print("请先进行机械臂标定")
        return
    
    try:
        # 读取摄像头像素坐标
        print("读取摄像头坐标...")
        with open(cam_file, "r") as f:
            cam_points = np.array(list(csv.reader(f)), dtype=np.float32)
        
        # 读取机械臂坐标
        print("读取机械臂坐标...")
        with open(arm_file, "r") as f:
            arm_points = np.array(list(csv.reader(f)), dtype=np.float32)
        
        # 验证数据
        if len(cam_points) != 9 or len(arm_points) != 9:
            print(f"[ERROR] 需要9个点，摄像头: {len(cam_points)}, 机械臂: {len(arm_points)}")
            return
        
        print(f"摄像头坐标 (前3个):")
        for i in range(3):
            print(f"  点{i+1}: [{cam_points[i,0]:.0f}, {cam_points[i,1]:.0f}]")
        
        print(f"机械臂坐标 (前3个):")
        for i in range(3):
            print(f"  点{i+1}: [{arm_points[i,0]:.3f}, {arm_points[i,1]:.3f}]")
        
        # 参考项目的计算方法：使用最小二乘法求解仿射变换
        # 构造增广矩阵 [x, y, 1]
        A = np.hstack([cam_points, np.ones((9, 1))])
        
        # 分别求解X和Y的变换
        # 公式: arm_x = a1*cam_x + a2*cam_y + a3
        #       arm_y = b1*cam_x + b2*cam_y + b3
        coeff_x, residuals_x, rank_x, s_x = np.linalg.lstsq(A, arm_points[:, 0], rcond=None)
        coeff_y, residuals_y, rank_y, s_y = np.linalg.lstsq(A, arm_points[:, 1], rcond=None)
        
        # 构建变换矩阵（3x2格式，参考项目格式）
        affine_matrix = np.column_stack((coeff_x, coeff_y))
        
        print("\n计算得到的仿射变换矩阵（参考项目格式 3x2）：")
        print(affine_matrix)
        
        # 保存矩阵为CSV（参考项目格式）
        csv_path = "data/config_relation_matrix.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            for row in affine_matrix:
                writer.writerow(row)
        print(f"\n[SAVED] 仿射矩阵已保存到 {csv_path} (3x2格式)")
        
        # 保存为3x3格式供其他用途
        affine_matrix_3x3 = np.eye(3)
        affine_matrix_3x3[0, :2] = coeff_x[:2]
        affine_matrix_3x3[0, 2] = coeff_x[2]
        affine_matrix_3x3[1, :2] = coeff_y[:2]
        affine_matrix_3x3[1, 2] = coeff_y[2]
        
        npy_path = "data/eyehand_matrix.npy"
        np.save(npy_path, affine_matrix_3x3)
        print(f"[SAVED] 仿射矩阵已保存为 {npy_path} (3x3格式)")
        
        # ===== 验证误差 =====
        print("\n=== 标定精度验证 ===")
        total_error = 0
        max_error = 0
        error_points = []
        
        for i in range(9):
            # 使用3x2矩阵计算（参考项目方法）
            cam_x, cam_y = cam_points[i]
            pred_x = cam_x * affine_matrix[0,0] + cam_y * affine_matrix[1,0] + affine_matrix[2,0]
            pred_y = cam_x * affine_matrix[0,1] + cam_y * affine_matrix[1,1] + affine_matrix[2,1]
            
            real_x, real_y = arm_points[i]
            error = np.sqrt((pred_x - real_x)**2 + (pred_y - real_y)**2)
            
            total_error += error
            max_error = max(max_error, error)
            error_points.append(error)
            
            print(f"Point {i+1}: 预测 [{pred_x:.3f}, {pred_y:.3f}], "
                  f"实际 [{real_x:.3f}, {real_y:.3f}], "
                  f"误差 {error:.3f} mm")
        
        avg_error = total_error / 9
        std_error = np.std(error_points)
        
        print(f"\n误差统计:")
        print(f"  平均误差: {avg_error:.3f} mm")
        print(f"  最大误差: {max_error:.3f} mm")
        print(f"  标准差: {std_error:.3f} mm")
        
        # 根据误差大小评级
        if avg_error < 3.0:
            print("  ✅ 标定结果优秀！ (<3mm)")
        elif avg_error < 10.0:
            print("  ✅ 标定结果良好 (3-10mm)")
        else:
            print("  ❌ 标定结果差 (>10mm)，建议重新标定")
        
        # 显示矩阵的物理意义
        print("\n=== 矩阵参数解释 ===")
        print(f"  变换公式:")
        print(f"    X_arm = {affine_matrix[0,0]:.6f} * X_cam + {affine_matrix[1,0]:.6f} * Y_cam + {affine_matrix[2,0]:.6f}")
        print(f"    Y_arm = {affine_matrix[0,1]:.6f} * X_cam + {affine_matrix[1,1]:.6f} * Y_cam + {affine_matrix[2,1]:.6f}")
        
        # 测试几个点
        print("\n=== 测试点验证 ===")
        test_points = [
            (320, 240),  # 中心
            (100, 100),  # 左上
            (540, 380),  # 右下
        ]
        
        for px, py in test_points:
            pred_x = px * affine_matrix[0,0] + py * affine_matrix[1,0] + affine_matrix[2,0]
            pred_y = px * affine_matrix[0,1] + py * affine_matrix[1,1] + affine_matrix[2,1]
            print(f"  像素({px}, {py}) -> 机械臂({pred_x:.1f}, {pred_y:.1f})")
        
    except Exception as e:
        print(f"[ERROR] 手眼标定失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
    print("手眼标定程序结束")