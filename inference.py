import sys
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt


class DCmotor():
    def __init__(self, Udc, psi_d, Rq, Lq, J, B):
        self.Udc     = Udc      # 直流电源电压，单位：伏特
        self.psi_d   = psi_d    # 电机永磁励磁，单位Wb
        self.Rq      = Rq       # 电枢绕组电阻，单位Ω
        self.Lq      = Lq       # 电感，单位亨利
        self.J       = J        # 转子转动惯量，单位kgm^2
        self.B       = B        # 系统阻尼转矩系数，单位N/m
        self.Ih      = 15       # 滞环电流上限，单位A
        self.Il      = 14       # 滞环电流下限，单位A
        self.wl      = 80       # 滞环角速度下限，单位rad/s
        self.wh      = 120      # 滞环角速度上限，单位rad/s
        self.w_width = 2        # 滞环角速度宽度，单位rad/s

        self.choppingMap = [[1,-1,0],[1,-1,0],[0,0,0]] # 关断or打开or保持的决策矩阵.纵坐标是转速，横坐标是电流


    def judgeChopping(self, epoch):
        # 根据上一个epoch判断此时导管是开是关
        # 0 --> 关
        # 1 --> 开
        # -1 --> 保持
        ret = -1

        # 当前时刻应该设置转速为
        set_w = self.wl

        # 判断此时在周期内的哪个阶段wl或wh
        period = 0.4 / self.step
        now = epoch % period
        if now < period / 2:
            # 前半周期
            set_w = self.wl
        elif now >= period / 2:
            # 后半周期
            set_w = self.wh

        # 设置基于转速滞环宽度的上下界
        this_w_Upper = set_w + self.w_width
        this_w_Lower = set_w - self.w_width


        # 转速标志. == 0 说明过小, == 1 说明正常，== 2 说明过大
        w_flag = 0 

        # 电流标志. == 0 说明过小, == 1 说明正常，== 2 说明过大
        I_flag = 0

        if self.x[epoch-1][1] >= this_w_Lower and self.x[epoch-1][1] <= this_w_Upper:
            w_flag = 1
        elif self.x[epoch-1][1] < this_w_Lower:
            w_flag = 0
        else:
            w_flag = 2

        if self.x[epoch-1][0] >= self.Il and self.x[epoch-1][0] <= self.Ih:
            I_flag = 1
        elif self.x[epoch-1][0] < self.Il:
            I_flag = 0
        else:
            I_flag = 2

        ret = self.choppingMap[w_flag][I_flag]


        return ret


    def Runge_Kutta(self, epoch):

        chopping = self.judgeChopping(epoch)
        if chopping == -1:
            # 保持上一个时刻的开关状态
            chopping = self.chopping_State[epoch-1]

        # 把当前时刻的开关状态存入开关状态列表
        self.chopping_State[epoch] = chopping

        this_K = np.zeros((2,1), dtype = float)

        self.K[epoch][0] = np.matmul(self.coefficient, self.x[epoch-1]) + np.array([[ chopping * self.Udc/self.Lq],[0]]) # K1
        this_K += self.K[epoch][0]

        x_for_K2 = self.x[epoch-1] + 0.5 * self.step * self.K[epoch][0]
        self.K[epoch][1] = np.matmul(self.coefficient, x_for_K2) + np.array([[ chopping * self.Udc/self.Lq],[0]]) # K2
        this_K += 2*self.K[epoch][1]

        x_for_K3 = self.x[epoch-1] + 0.5 * self.step * self.K[epoch][1]
        self.K[epoch][2] = np.matmul(self.coefficient, x_for_K3) + np.array([[ chopping * self.Udc/self.Lq],[0]]) # K3
        this_K += 2*self.K[epoch][2]

        x_for_K4 = self.x[epoch-1] + self.step * self.K[epoch][2]
        self.K[epoch][3] = np.matmul(self.coefficient, x_for_K4) + np.array([[ chopping * self.Udc/self.Lq],[0]]) # K4
        this_K += self.K[epoch][3]

        # forward
        self.x[epoch] = self.x[epoch-1] + self.step / 6 * this_K

        # 如果此时刻iq < 0，且chopping = 0，则应置Uq = w * psi_d， iq = 0, 且需要改变电机状态方程组，只剩下
        # pw = -B/J * w
        if self.x[epoch][0] < 0 and chopping == 0:
            self.x[epoch][0] = 0
            this_K = 0.
            K_1 = -1 * self.B / self.J * self.x[epoch-1][1]
            K_2 = -1 * self.B / self.J * (self.x[epoch-1][1] + self.step / 2 * K_1)
            K_3 = -1 * self.B / self.J * (self.x[epoch-1][1] + self.step / 2 * K_2)
            K_4 = -1 * self.B / self.J * (self.x[epoch-1][1] + self.step * K_3)
            this_K =  (K_1 + 2*K_2 + 2*K_3 + K_4)
            self.x[epoch][1] = self.x[epoch-1][1] + self.step / 6 * this_K


    def Run(self, step, run_time):
        self.run_time = run_time # 运行总时长
        self.step = step    # 步长,单位秒
        self.num_epochs = int(self.run_time / self.step)  # 迭代次数

        self.chopping_State = [1 for _ in range(self.num_epochs + 1)]    # 初始化开关状态列表.-1,保持
                                                                    #                   0,关断
                                                                    #                   1,开启

        x_shape = (self.num_epochs+1, 2, 1) # 定义储存[[iq],[w]]的矩阵形状
        self.x = np.zeros(x_shape, dtype = np.float64) # 将所有iq,w的初始值设为0
        K_shape = (self.num_epochs+1, 4, 2, 1)   # 定义储存[[K1],[K2],[K3],[K4]]的矩阵形状
        self.K = np.zeros(K_shape, dtype = np.float64)   # 将所有步骤的K初始值设为0

        self.coefficient = np.array([[-self.Rq/self.Lq, -self.psi_d/self.Lq], [self.psi_d/self.J, -self.B/self.J]],dtype = np.float64)

        # 正式运行
        # 迭代num_epochs次
        for i in range(1, self.num_epochs+1):
            if(i % 2000 == 0):
                print("{} / {}".format(i/2000,self.num_epochs/2000))
            self.Runge_Kutta(i)

    def draw(self):
        # 把初始时刻到num_epochs次迭代后的图绘制出来
        
        # time -- current
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(0,self.num_epochs+1) * self.step, self.x[:,0].reshape(len(self.x)), color = 'r')
        plt.plot(np.arange(0,self.num_epochs+1) * self.step, 15*np.ones((len(self.x))), 'g--')
        plt.plot(np.arange(0,self.num_epochs+1) * self.step, 14*np.ones((len(self.x))), 'g--')
        plt.xlabel('Seconds (s)')
        plt.ylabel('Current (A)')
        plt.title("Seconds -- Current")


        # time -- rotating velocity
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(0,self.num_epochs+1) * self.step, self.x[:,1].reshape(len(self.x)))
        plt.plot(np.arange(0,self.num_epochs+1) * self.step, 122*np.ones((len(self.x))), 'k--')
        plt.plot(np.arange(0,self.num_epochs+1) * self.step, 118*np.ones((len(self.x))), 'k--')
        plt.plot(np.arange(0,self.num_epochs+1) * self.step, 82*np.ones((len(self.x))), 'y--')
        plt.plot(np.arange(0,self.num_epochs+1) * self.step, 78*np.ones((len(self.x))), 'y--')
        plt.xlabel('Seconds (s)')
        plt.ylabel('Rotating Velocity (rad/s)')
        plt.title("Seconds -- Rotating Velocity")

        plt.show()

    def test(self, step, run_time):
        self.run_time = run_time # 运行总时长
        self.step = step    # 步长,单位秒
        self.num_epochs = int(self.run_time / self.step)  # 迭代次数
        print(str(self.num_epochs))


if __name__ == "__main__":
    mymotor = DCmotor(200,1,0.5,0.05,0.002,0.1)
    
    mymotor.Run(0.0001,1.2)

    mymotor.draw()






        # 迭代次数 --- 转速
        # plt.plot(np.arange(0,self.num_epochs+1) * self.step, self.x[:,0].reshape(len(self.x)))

        # # 激活第一个 subplot
        # plt.subplot(2,  1,  1)  
        # # 绘制第一个图像 : 迭代次数 -- 电流
        # plt.plot(np.arange(0,self.num_epochs+1) * self.step, self.x[:,0].reshape(len(self.x)))
        # plt.title('epoch -- current')  

        # # 将第二个 subplot 激活
        # plt.subplot(2,  1,  2)
        # # 绘制第二个图像 : 迭代次数 -- 转速
        # plt.plot(np.arange(0,self.num_epochs+1) * self.step, self.x[:,1].reshape(len(self.x)))
        # plt.title('epoch -- Rotate')  
        # # 展示图像