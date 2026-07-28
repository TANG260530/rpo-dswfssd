import cv2
import time
import logging
import threading
import queue
import os
import asyncio
from datetime import datetime
from src.face.recognizer import FaceRecognizer
from src.wake.wake_controller import WakeController
from agents.graph import create_graph

# 配置日志记录
logging.basicConfig(
    level=logging.DEBUG,  # 改为 DEBUG 级别以获取更多信息
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('face_wake.log')  # 同时保存到文件
    ]
)
logger = logging.getLogger(__name__)

# 设置OpenCV的日志级别
opencv_logger = logging.getLogger('opencv')
opencv_logger.setLevel(logging.DEBUG)

class AIWakeSystem:
    def __init__(self):
        # 初始化人脸识别器，使用GPU配置
        self.recognizer = FaceRecognizer(
            model='cnn',  # 使用CNN模型
            use_gpu=True,  # 尝试使用GPU
            tolerance=0.6,  # 匹配容差
            num_jitters=1,  # 特征提取采样次数
            face_size=(640, 480)  # 目标图像大小
        )
        
        self.wake_controller = WakeController(
            time_window=2.0,
            min_detections=4,
            cooldown_period=5.0,
            detection_threshold=0.85
        )
        self.video_url = 'http://10.81.35.91:5000/video_feed'
        self.is_running = False
        self.wake_queue = queue.Queue()
        
        # 初始化 LangGraph
        self.conversation_graph = create_graph()
        
        # 初始化对话状态
        self.conversation_active = False
        self.last_user = None
        
        # 创建必要的目录
        os.makedirs("wake_events", exist_ok=True)
        
        # 视频处理参数
        self.frame_interval = 0.1  # 帧处理间隔（秒）
        self.last_frame_time = 0  # 上一帧处理时间
        self.video_processing_active = True  # 新增视频处理状态控制
        
        # 添加Web界面所需的属性
        self.current_frame = None
        self.face_locations = None
        
    def stop_video_processing(self):
        """停止视频处理"""
        self.video_processing_active = False
        logger.info("视频处理模块已停止")
        
    def restart_video_processing(self):
        """重启视频处理"""
        self.video_processing_active = True
        # 重新创建视频处理线程
        self.video_thread = threading.Thread(target=self._process_video)
        self.video_thread.daemon = True
        self.video_thread.start()
        logger.info("视频处理模块已重启")

    # async def _send_initial_message(self, user: str) -> str:
    #     """发送初始问候消息，支持流式输出"""
    #     try:
    #         # 构建初始问候消息
    #         initial_message = {
    #             "messages": [
    #                 HumanMessage(content=f"我叫{user}，你好！")
    #             ]
    #         }
            
    #         # 配置
    #         config = {
    #             "configurable": {
    #                 "thread_id": user,
    #                 "user_id": user,
    #                 "timestamp": datetime.now().isoformat()
    #             }
    #         }
            
    #         # 流式获取AI响应
            
            
        #     async for chunk in self.conversation_graph.astream(
        #         initial_message,
        #         config=config,
        #         stream_mode='values'
        #     ):
        #         if chunk:
        #             chunk['messages'][-1].pretty_print()
            
           
           
        # except Exception as e:
        #     logger.error(f"发送初始消息错误: {str(e)}")
        #     return "对不起，我现在无法正常回应。"

    async def _handle_conversation(self, user: str, message: str, thread_id: str = None, user_id: str = None) -> str:
        """处理对话消息，支持流式输出"""
        try:
            # 使用用户名作为默认的thread_id和user_id
            thread_id = thread_id or user
            user_id = user_id or user
            
            # 构建配置
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # 构建输入消息
            input_messages = {
                "messages": [message]
            }
            
            # 流式获取AI响应
            
            
            async for chunk in self.conversation_graph.astream(
                input=input_messages,
                config=config,
                stream_mode='values'
            ):
                if chunk:
                    chunk['messages'][-1].pretty_print()
            
            
            
        except Exception as e:
            logger.error(f"处理用户 {user} 的消息时发生错误: {str(e)}", exc_info=True)
            return "非常抱歉，处理您的消息时出现了技术故障，请稍后再试。"

    def start(self):
        """启动系统"""
        self.is_running = True
        self.video_processing_active = True  # 重置视频处理状态
        
        # 创建视频处理线程
        self.video_thread = threading.Thread(target=self._process_video)
        self.video_thread.daemon = True
        self.video_thread.start()
        
        # 创建唤醒处理线程
        self.wake_thread = threading.Thread(target=self._handle_wake_events)
        self.wake_thread.daemon = True
        self.wake_thread.start()
        
        logger.info("AI唤醒系统已启动")

    def stop(self):
        """停止系统"""
        self.is_running = False
        self.video_processing_active = False # 停止视频处理
        if hasattr(self, 'video_thread'):
            self.video_thread.join()
        if hasattr(self, 'wake_thread'):
            self.wake_thread.join()
        logger.info("系统已停止")

    def _process_video(self):
        """视频处理线程"""
        reconnect_delay = 1.0  # 初始重连延迟
        max_reconnect_delay = 30.0  # 最大重连延迟
        
        while self.is_running and self.video_processing_active:
            try:
                logger.info(f"正在连接视频流: {self.video_url}")
                cap = cv2.VideoCapture(self.video_url)
                
                if not cap.isOpened():
                    raise Exception("无法打开视频流")
                
                # 检查视频流参数
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                logger.info(f"视频流参数 - 宽度: {width}, 高度: {height}, FPS: {fps}")
                
                logger.info("视频流连接成功")
                reconnect_delay = 1.0  # 重置重连延迟
                
                while self.is_running and self.video_processing_active:
                    # 控制帧处理频率
                    current_time = time.time()
                    if current_time - self.last_frame_time < self.frame_interval:
                        time.sleep(0.01)  # 短暂休眠以减少CPU使用
                        continue
                    
                    ret, frame = cap.read()
                    if not ret:
                        logger.error("读取视频帧失败")
                        break
                    
                    self.last_frame_time = current_time
                    
                    try:
                        # 保存当前帧用于Web界面显示
                        self.current_frame = frame.copy()
                        
                        # 人脸识别
                        is_authorized, face_locations, names = self.recognizer.recognize(frame)
                        
                        # 保存人脸位置信息用于Web界面显示
                        self.face_locations = face_locations
                        
                        # 只在调试级别记录详细信息
                        if face_locations:
                            logger.debug(f"检测到人脸: 位置={face_locations}, 授权状态={is_authorized}, 识别名称={names}")
                        
                        # 检查唤醒条件
                        should_wake = self.wake_controller.check_wake_condition(
                            is_authorized, face_locations
                        )
                        
                        if should_wake and names:
                            # 发现授权用户，停止打印等待消息
                            print("\r" + " " * 50 + "\r", end="", flush=True)  # 清除等待消息
                            
                            wake_event = {
                                'timestamp': current_time,
                                'user': names[0],
                                'frame': frame.copy()
                            }
                            self.wake_queue.put(wake_event)
                            logger.info(f"检测到授权用户并触发唤醒: {wake_event['user']}")
                            
                            # 停止视频处理
                            self.stop_video_processing()
                            break
                            
                    except Exception as e:
                        logger.error(f"处理视频帧时发生错误: {str(e)}", exc_info=True)
                        self.current_frame = None
                        self.face_locations = None
                
            except Exception as e:
                logger.error(f"视频处理错误: {str(e)}")
                if 'cap' in locals():
                    cap.release()
                
                # 如果视频处理已停止，不再重试连接
                if not self.video_processing_active:
                    break
                    
                # 使用指数退避进行重连
                logger.info(f"将在 {reconnect_delay} 秒后尝试重新连接...")
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                continue
            
            finally:
                if 'cap' in locals():
                    cap.release()
                    
        logger.info("视频处理线程已退出")

    def _handle_wake_events(self):
        """唤醒事件处理线程"""
        
        while self.is_running:
            try:
                # 从队列获取唤醒事件
                wake_event = self.wake_queue.get(timeout=1.0)
                
                # 处理唤醒事件
                asyncio.run(self._process_wake_event(wake_event))
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"唤醒处理错误: {str(e)}")

    async def _process_wake_event(self, wake_event):
        """处理唤醒事件"""
        user = wake_event['user']
        timestamp = wake_event['timestamp']
        frame = wake_event['frame']
        
        # 保存触发时的图像
        image_path = f"wake_events/wake_{int(timestamp)}_{user}.jpg"
        cv2.imwrite(image_path, frame)
        
        logger.info(f"处理唤醒事件 - 用户: {user}, 时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}")
        
        # 设置当前对话状态
        self.conversation_active = True
        self.last_user = user
        
        try:
            # 暂时调整日志级别
            root_logger = logging.getLogger()
            original_level = root_logger.level
            root_logger.setLevel(logging.WARNING)
            
            # 发送初始问候消息
            print(f"\n检测到授权用户: {user}")
            user_input = f"我叫{user}，你好！"
            
            # 开始对话循环
            while self.conversation_active:
                # TODO: 这里添加语音识别模块获取用户输入
                # 现在暂时使用控制台输入模拟
                               
                # 检查是否要结束对话
                if user_input.lower() in ['退出', 'exit', 'quit']:
                    self.conversation_active = False
                    logger.info(f"用户 {user} 结束对话")
                    break
                
                # 处理用户输入并获取AI响应
                await self._handle_conversation(
                    user=user,
                    message=user_input,
                    thread_id=user,
                    user_id=user
                )

                user_input = input(f"\n{user}> ")
                
        except Exception as e:
            logger.error(f"对话循环错误: {str(e)}")
            self.conversation_active = False
        finally:
            # 恢复原始日志级别
            root_logger.setLevel(original_level)
            print("\n对话已结束")
            print("\n系统将在60秒后重新启动人脸识别...")
            
            # 等待60秒
            await asyncio.sleep(60)
            
            # 重新启动视频处理模块
            self.restart_video_processing()
            print("\n人脸识别模块已重新启动，等待下一次唤醒...")

async def main():
    # 启动系统
    system = AIWakeSystem()
    system.start()
    
    try:
        # 保持主线程运行
        while True:
            await asyncio.sleep(1)
            
            # 只在非对话状态且视频处理模块激活时显示等待提示
            if not system.conversation_active and system.video_processing_active:
                print("\r等待人脸识别唤醒...", end="", flush=True)
            
    except KeyboardInterrupt:
        logger.info("收到停止信号")
        system.stop()
        print("\n系统已停止")

if __name__ == "__main__":
    # 设置基本日志级别
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('face_wake.log'),  # 文件日志
            logging.StreamHandler()  # 控制台日志
        ]
    )
    
    # 设置一些模块的日志级别为WARNING，减少输出
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    asyncio.run(main())