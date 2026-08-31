# perception/yolo_detector.py
import cv2
import numpy as np
import time

# acllite imports for OM model inference on Orange Pi
from acllite_resource import AclLiteResource
from acllite_model import AclLiteModel
from acllite_imageproc import AclLiteImageProc


def letterbox(img, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
    """Resize and pad image while meeting stride-multiple constraints"""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    if center:
        dw /= 2  # divide padding into 2 sides
        dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)) if center else 0, int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)) if center else 0, int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )  # add border

    return img, ratio, dw, dh


class YoloDetector:
    def __init__(self, model_path="models/best-fruits.om", model_width=640, model_height=640):
        print("[YOLO] Initializing YOLO detector (OM model)...")
        
        # Model parameters
        self.model_path = model_path
        self.model_width = model_width
        self.model_height = model_height
        self.confidence_threshold = 0.4
        
        # ACL resources
        self.resource = None
        self.dvpp = None
        self.model = None
        
        # Preprocessing state
        self.ratio = None
        self.dw = None
        self.dh = None
        self.input_tensor = None
        self.output = None
        
        # Default class names matching training order for best-fruits.om
        self.class_names = {
            0: "cavocado",
            1: "lemon",
            2: "pear",
            3: "mango",
            4: "persimmon"
        }
        
        # Camera warm-up state
        self.camera_warmed_up = False
        self.warm_up_start_time = 0
        
        # Load model
        self._load_model()
        
        if self.model is not None:
            print(f"[YOLO] Detector ready. Classes: {len(self.class_names)}")
        else:
            print("[YOLO] ERROR: Detector initialization failed")

    def _load_model(self):
        """Load OM model using acllite"""
        try:
            print(f"[YOLO] Loading OM model: {self.model_path}")
            
            # Initialize ACL resources
            self.resource = AclLiteResource()
            self.resource.init()
            self.dvpp = AclLiteImageProc(self.resource)
            self.model = AclLiteModel(self.model_path)
            
            print(f"[YOLO] OM Model loaded. Classes: {self.class_names}")
            
        except Exception as e:
            print(f"[YOLO] ERROR: Model load failed: {e}")
            self.model = None

    def set_confidence_threshold(self, threshold):
        """Set confidence threshold"""
        self.confidence_threshold = max(0.1, min(0.99, threshold))
        print(f"[YOLO] Confidence: {self.confidence_threshold}")

    def _preprocess(self, image):
        """Preprocess image for OM model inference"""
        image, self.ratio, self.dw, self.dh = letterbox(image, new_shape=(self.model_width, self.model_height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32) / 255.0
        self.input_tensor = np.expand_dims(image, axis=0)
        return self.input_tensor

    def _infer(self):
        """Run inference on OM model"""
        self.output = self.model.execute([self.input_tensor])
        self.output = np.squeeze(self.output[0])
        return self.output

    def _postprocess(self):
        """Postprocess OM model output, returns list of [xmin, ymin, xmax, ymax, confidence, label]"""
        result = []
        for i in range(self.output.shape[0]):
            # Read class confidence
            confidence = self.output[i][4]
            # Filter by threshold
            if confidence > self.confidence_threshold:
                # Read class index
                label = int(self.output[i][5])
                # Read coordinates and restore to original image
                xmin = int((self.output[i][0] - int(round(self.dw - 0.1))) / self.ratio[0])
                ymin = int((self.output[i][1] - int(round(self.dh - 0.1))) / self.ratio[1])
                xmax = int((self.output[i][2] - int(round(self.dw + 0.1))) / self.ratio[0])
                ymax = int((self.output[i][3] - int(round(self.dh + 0.1))) / self.ratio[1])
                result.append([xmin, ymin, xmax, ymax, confidence, label])
        return result

    def _ensure_camera_warm_up(self, camera):
        """Ensure camera is warmed up before detection"""
        if self.camera_warmed_up:
            return True
            
        print("[YOLO] Camera warm-up...")
        warm_up_frames = 0
        start_time = time.time()
        
        while time.time() - start_time < 2.0:  # Warm up for 2 seconds
            frame = camera.get_frame()
            if frame is not None:
                warm_up_frames += 1
            time.sleep(0.05)  # 20 FPS during warm-up
        
        self.camera_warmed_up = True
        print(f"[YOLO] Warm-up complete: {warm_up_frames} frames")
        return True

    def _detect_single_frame(self, camera, target_fruit, show_display=True):
        """Detect target fruit in a single frame with optional display"""
        frame = camera.get_frame()
        if frame is None:
            print("[YOLO] ERROR: No frame from camera")
            return None
        
        try:
            # Ensure correct image format
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            # Keep a copy for display
            display_frame = frame.copy()
            
            # Run OM model inference
            self._preprocess(frame)
            self._infer()
            results = self._postprocess()
            
            target_detections = []
            best_det = None
            
            if results:
                for xmin, ymin, xmax, ymax, conf, cls_id in results:
                    try:
                        # Calculate center coordinates and size
                        x = (xmin + xmax) / 2.0
                        y = (ymin + ymax) / 2.0
                        w = xmax - xmin
                        h = ymax - ymin
                        
                        # Get class name
                        class_name = self.class_names.get(cls_id, f"class_{cls_id}")
                        
                        # Draw all detections on display frame
                        if show_display:
                            # Choose color: green for target, blue for others
                            if class_name.lower() == target_fruit.lower():
                                color = (0, 255, 0)  # Green for target
                            else:
                                color = (255, 0, 0)  # Blue for others
                            
                            # Draw bounding box
                            cv2.rectangle(display_frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                            # Draw center point
                            cv2.circle(display_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                            # Draw label
                            label_text = f"{class_name} ({conf:.2f})"
                            cv2.putText(display_frame, label_text, (int(xmin), int(ymin) - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Only keep target fruit
                        if class_name.lower() == target_fruit.lower():
                            target_detections.append({
                                "class": class_name,
                                "bbox": [float(x), float(y)],  # Center point (x, y)
                                "conf": float(conf),
                                "width": float(w),
                                "height": float(h),
                                "class_id": cls_id
                            })
                            
                    except Exception as e:
                        print(f"[YOLO] WARNING: Box processing error: {e}")
                        continue
                
                # Return the detection with highest confidence
                if target_detections:
                    best_det = max(target_detections, key=lambda d: d["conf"])
                    print(f"[YOLO] Found {target_fruit} at ({best_det['bbox'][0]:.1f}, {best_det['bbox'][1]:.1f}), conf={best_det['conf']:.3f}")
            
            # Show detection display
            if show_display:
                # Add status text
                status = f"Target: {target_fruit} | Detected: {len(results) if results else 0} objects"
                cv2.putText(display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if best_det:
                    result_text = f"FOUND: {target_fruit} conf={best_det['conf']:.2f}"
                    cv2.putText(display_frame, result_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "NOT FOUND", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display the frame
                cv2.imshow('YOLO Detection', display_frame)
                cv2.waitKey(500)  # Show for 500ms
            
            if best_det is None:
                print(f"[YOLO] No {target_fruit} detected in this frame")
            
            return best_det
            
        except Exception as e:
            print(f"[YOLO] ERROR: Detection failed: {e}")
            return None

    def detect(self, frame):
        """Legacy single frame detection (for backward compatibility)"""
        if self.model is None or frame is None:
            return []
        
        try:
            # Ensure correct image format
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            # Run OM model inference
            self._preprocess(frame)
            self._infer()
            results = self._postprocess()
            
            detections = []
            
            for xmin, ymin, xmax, ymax, conf, cls_id in results:
                try:
                    # Calculate center coordinates and size
                    x = (xmin + xmax) / 2.0
                    y = (ymin + ymax) / 2.0
                    w = xmax - xmin
                    h = ymax - ymin
                    
                    class_name = self.class_names.get(cls_id, f"class_{cls_id}")
                    
                    detections.append({
                        "class": class_name,
                        "bbox": [float(x), float(y)],
                        "conf": float(conf),
                        "width": float(w),
                        "height": float(h),
                        "class_id": cls_id
                    })
                    
                except Exception as e:
                    print(f"[YOLO] WARNING: Box error: {e}")
                    continue
            
            return detections
            
        except Exception as e:
            print(f"[YOLO] ERROR: Detection failed: {e}")
            return []

    def detect_robust(self, camera, target_fruit, max_retries=3, show_display=True):
        """
        Robust detection with retry logic
        Args:
            camera: Camera object
            target_fruit: Fruit name to detect
            max_retries: Maximum retry attempts
            show_display: Whether to show detection display
        Returns:
            detection dict or None if not found
        """
        print(f"[YOLO] Robust detection for: {target_fruit}, max retries: {max_retries}")
        
        # Ensure camera is warmed up
        self._ensure_camera_warm_up(camera)
        
        result = None
        
        # Try detection with retries
        for attempt in range(max_retries):
            print(f"[YOLO] Attempt {attempt+1}/{max_retries}")
            
            # Single frame detection with display
            detection = self._detect_single_frame(camera, target_fruit, show_display=show_display)
            
            if detection is not None:
                # Verify detection quality
                if detection["conf"] >= self.confidence_threshold:
                    print(f"[YOLO] Successful detection on attempt {attempt+1}")
                    print(f"[YOLO] Position: ({detection['bbox'][0]:.1f}, {detection['bbox'][1]:.1f})")
                    result = detection
                    break
                else:
                    print(f"[YOLO] Low confidence: {detection['conf']:.3f} < {self.confidence_threshold}")
            
            # Wait before retry
            if attempt < max_retries - 1:
                print("[YOLO] Waiting 0.5s before retry...")
                time.sleep(0.5)
        
        # Close display window after detection completes
        if show_display:
            cv2.destroyWindow('YOLO Detection')
        
        if result is None:
            print(f"[YOLO] Failed to detect {target_fruit} after {max_retries} attempts")
        
        return result

    def is_ready(self):
        """Check if detector is ready"""
        return self.model is not None
    
    def get_class_names(self):
        """Get class names"""
        return self.class_names

    def release_resource(self):
        """Release ACL resources"""
        if self.resource:
            del self.resource
            self.resource = None
        if self.dvpp:
            del self.dvpp
            self.dvpp = None
        if self.model:
            del self.model
            self.model = None
        print("[YOLO] Resources released")