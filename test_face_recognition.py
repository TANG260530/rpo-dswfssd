# test_face_recognition.py
from src.face.recognizer import FaceRecognizer
import cv2
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_face_recognition():
    # 初始化人脸识别器
    recognizer = FaceRecognizer()
    
    # 创建结果目录
    if not os.path.exists('recognition_results'):
        os.makedirs('recognition_results')
    
    # 读取之前的测试图像
    frames_dir = 'captured_frames'
    for filename in os.listdir(frames_dir):
        if filename.endswith('.jpg'):
            # 读取图像
            image_path = os.path.join(frames_dir, filename)
            frame = cv2.imread(image_path)
            
            # 进行人脸识别
            is_authorized, face_locations, names = recognizer.recognize(frame)
            
            # 在图像上标记识别结果
            for (top, right, bottom, left), name in zip(face_locations, names):
                # 画框
                cv2.rectangle(frame, (left, top), (right, bottom), 
                            (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)
                
                # 显示名字
                cv2.putText(frame, name, (left, top - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, 
                           (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)
            
            # 保存结果
            result_path = os.path.join('recognition_results', f'recognized_{filename}')
            cv2.imwrite(result_path, frame)
            
            logger.info(f"处理图片 {filename}: {'检测到授权用户' if is_authorized else '未检测到授权用户'}")

if __name__ == "__main__":
    test_face_recognition()