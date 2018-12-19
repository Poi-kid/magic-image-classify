#########################################################


#本工程用于VR魔法游戏中识别玩家绘制的魔法阵的技能类型

#VR客户端中的玩家绘制魔法阵，本工程返回魔法阵识别分类结果

#总共有6种魔法，所以是6分类的图像识别，采用CNN卷积神经网络模型

#训练集很小，每种魔法阵只有50张训练数据，但效果依然不错


#########################################################


#关于接收图片和图片识别


#main.py启动服务器后将监听客户端连接请求

#客户端连接成功后，服务端先将图像识别系统初始化

#然后将进入到接收客户端传来的图像信息和心跳包和控制命令的函数

#当客户端传来图像的线条信息时，绘制一段段线条

#当客户端传来clear命令时，清空已经绘制的线条

#当客户端传来end命令时，结束一幅图像的绘制，并保存到save.png，退出接收图像信息的函数

#退出接收图像信息函数后，图像识别系统识别图像，返回识别结果，并重新进入接收客户端传来的图像信息的函数，继续接收下一张图像


#########################################################


#关于训练


#运行training.py中的run_training函数即可开始训练

#可更改训练参数


#########################################################


#关于模型


#在model.py中定义了2个卷积层+2个池化层+2个全连接层，最后softmax分类输出结果


#########################################################
