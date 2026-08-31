import cv2
import os

def _clean_cv2_environment():

    env_vars_to_remove = [
        'OPENCV_VIDEOIO_PRIORITY_LIST',
        'OPENCV_VIDEOIO_DEBUG',
        'GST_DEBUG'
    ]
    for var in env_vars_to_remove:
        if var in os.environ:
            del os.environ[var]
    

    os.environ['OPENCV_VIDEOIO_PRIORITY_V4L2'] = '1'
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

class Camera:
    def __init__(self, index=0):

        _clean_cv2_environment()
        
        print(f"[CAMERA] Initializing...")
        print(f"[CAMERA] Sudo mode: {os.geteuid() == 0}")
        print(f"[CAMERA] DISPLAY: {os.environ.get('DISPLAY', 'not set')}")
        
        self.cap = None
        self._init_success = False
        

        self._init_direct()
        
        if not self._init_success:
            raise RuntimeError("Camera open failed")
    
    def _init_direct(self):

        print("[CAMERA] Using direct method for Orange Pi AI Pro")
        

        try:
            self.cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self._init_success = True
                    print(f"[CAMERA] Method 1 OK: /dev/video0 + V4L2")
                    print(f"[CAMERA] Frame size: {frame.shape}")
                    return
                else:
                    self.cap.release()
                    self.cap = None
        except Exception as e:
            print(f"[CAMERA] Method 1 error: {e}")
        

        try:
            self.cap = cv2.VideoCapture('/dev/video1', cv2.CAP_V4L2)
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self._init_success = True
                    print(f"[CAMERA] Method 2 OK: /dev/video1 + V4L2")
                    return
                else:
                    self.cap.release()
                    self.cap = None
        except Exception as e:
            print(f"[CAMERA] Method 2 error: {e}")
        
        print("[CAMERA] All methods failed")
    
    def get_frame(self):
        if not self._init_success or not self.cap:
            return None
        
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def __del__(self):
        self.release()