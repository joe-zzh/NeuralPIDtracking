import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIDRobotClass import Robot
from TargetPoint import cal_target_index
from BP2 import NeuralPID
import random

# 这一段是设定待跟踪路径，不允许调整。但是可以在这个基础上增加难度。难度增加，并且取得较好的效果，加分项。
x = np.linspace(-2*np.pi, 2*np.pi, 100)
omega = 0.3
alt = 10
y = alt*np.sin(omega*x)
ref = np.column_stack((x, y, np.mod(alt*omega*np.cos(omega*x), 2*np.pi)))
wheellengh = 0.2
Kp=1
Ki=0.1
Kd=5
NeuralPID = NeuralPID(Kp,Ki,Kd)
NeuralPID.kcoef=4
#自定义轨迹
# Kp=1
# Ki=0.12
# Kd=5
# NeuralPID = NeuralPID(Kp,Ki,Kd)
# x = np.linspace(-4*np.pi, 4*np.pi, 100)
# omega = 0.3
# alt = 10
# y = alt*np.sin(omega*x)
# ref = np.column_stack((x, y, np.mod(alt*omega*np.cos(omega*x), 2*np.pi)))
# wheellengh = 0.2
# NeuralPID.kcoef=1.6

all_robx=[]
all_roby=[]
rob = Robot(wheellengh)  # 引用robot类
rob.set_noise(0.01, 0.01)
init_x = random.uniform(-6, 6)
init_y = random.uniform(-10, 10)
init_theta=random.uniform(-np.pi,np.pi)
rob.set(ref[0,0],ref[0,1],0)
# rob.set(init_x,init_y,init_theta)

for i in range(140):
    robot_state = np.zeros(2)
    ind = cal_target_index([rob.x,rob.y],ref[:,[0,1]])
    alpha = np.arctan2(ref[ind, 1]-rob.y, ref[ind, 0]-rob.x)
    l_d = np.linalg.norm(ref[ind,[0,1]]-[rob.x,rob.y])
    theta_e = alpha-rob.orientation
    e_y = l_d*np.sin(theta_e)
    delta_f = NeuralPID.forward(e_y)
    rob.move(delta_f,rob.length)
    all_robx.append(rob.x)
    all_roby.append(rob.y)
    # 调试区
    # print(NeuralPID.kp,NeuralPID.ki,NeuralPID.kd)
    print(NeuralPID.wp,NeuralPID.wi,NeuralPID.wd)
    print(delta_f)
    # print(rob.x,rob.y,rob.orientation)

# 创建一个空白的图像
fig, ax = plt.subplots()
# 初始化路径和机器人位置的绘图对象
path_line, = ax.plot(x, y, label='desired path')
robot_line, = ax.plot([], [], 'ro', label='robot position')



def init():
    robot_line.set_data([], [])
    return robot_line,

def update(frame):
    rob_x = all_robx[frame]
    rob_y = all_roby[frame]
    robot_line.set_data(rob_x, rob_y)
    all_robx_slice = all_robx[:frame+1]
    all_roby_slice = all_roby[:frame+1]
    path_line.set_data(all_robx_slice, all_roby_slice)  # 更新机器人的轨迹
    path_line.set_color('r')  # 更新机器人路径线的颜色为红色
    return robot_line, path_line

ani = FuncAnimation(fig, update, frames=len(all_robx), init_func=init, blit=True)

plt.legend()  # 添加图例
plt.show()

