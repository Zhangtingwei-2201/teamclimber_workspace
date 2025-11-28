# teamclimber_challenge功能包  
## 完成情况
1、完成了对摄像头话题的订阅  
2、使用opencv完成了对sphere的识别和发布  
3、使用opencv完成了对rect的识别和发布  
4、使用ultralytic训练了一个识别装甲板的YOLO模型  
5、实现在C++中用TensorRT加速推理模型来识别装甲板  
6、额外训练了一个识别rect的YOLO模型作为识别rect的备选方案  
7、基本实现了坐标的PnP解算  
8、优化了rect的识别，用OpenCV库的卡尔曼滤波实现粗糙的动态目标识别  
## 工具环境情况
1、C++、ros2、opencv版本均符合官方要求  
2、YOLO模型以YOLO11n.pt为基础训练  
3、数据库来源于网上搜集与自己采集  
4、TensorRT版本：10.14.1.48，CUDA版本：12.2，CuDNN版本：9.16.0  
## 节点运行
视觉识别节点：  
source ~/teamclimber_workspace/install/setup.bash  
ros2 launch teamclimber_challenge vision.launch.py  
弹丸击打节点：  
source ~/teamclimber_workspace/install/setup.bash  
ros2 launch teamclimber_challenge shooter.launch.py  
## 技术报告文档 
https://www.yuque.com/codezero-kv7wr/ivc8n1/ouutb92frviycbwf#
