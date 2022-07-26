'''
@Course Name: Modeling and analysis of motor system
@Project Name:DC motor simulation
@Author: Lu Xuancun 3190103641
@Date: 2022-04-07
@LastEditTime: 2022-04-08
@Description: Done!
'''

# Import related libraries
import numpy as np
import matplotlib

matplotlib.rcParams['backend'] = 'SVG'
import matplotlib.pyplot as plt


# define a class for DC motor
class DCmotor():
    def __init__(self, U_dc, Psi_f, R_q, L_q, J, B, I_h, I_l, Omega_h, Omega_l, Width, Epochmax, Step):

        # 定义永磁直流电机以及控制系统的各个参数

        self.U_dc = U_dc  # 直流电源(V)
        self.Psi_f = Psi_f  # 电机永磁励磁(Wb)
        self.R_q = R_q  # 电枢绕组电阻(ohm)
        self.L_q = L_q  # 电感(H)
        self.J = J  # 转子转动惯量(kgm^2)
        self.B = B  # 系统阻尼转矩系数 Nm/(rad/s)
        self.I_h = I_h  # 电流上限(A)
        self.I_l = I_l  # 电流下限(A)
        self.Omega_h = Omega_h  # 高转速(rad/s)
        self.Omega_l = Omega_l  # 低转速(rad/s)
        self.Width = Width  # 滞环宽度(rad/s)

        # 可调参数宏定义
        self.epochmax = Epochmax  # 迭代次数
        self.step = Step  # 步长

        # 定义中间变量列表
        self.I_q = []  # 电流(A)
        self.U_q = []  # 电压(V)
        self.Omega = []  # 转速(rad/s)
        self.I_PWM = []  # 电流控制的PWM开关 1：开 0：关
        self.Omega_PWM = []  # 转速控制的PWM开关 1：开 0：关
        self.PWM = []  # 最终合成IGBT开关 1：开 0：关
        self.PID = []
        self.sum_error = 0
        self.KP = 0.00001
        self.KI = 0.0001
        self.value = []

    #   初始化函数
    #   为中间变量的列表根据迭代次数赋值
    def Init(self):
        for i in range(self.epochmax + 1):
            self.I_q.append(0)
            self.Omega.append(0)
            self.I_PWM.append(1)
            self.Omega_PWM.append(1)
            self.PWM.append(1)
            self.PID.append(1)
            self.U_q.append(self.U_dc)
            self.value.append(0)

    #   方程组如下
    #   p(I_q)   = (U_q-Omega*Psi_f-R_q*I_q)/L_q
    #   p(Omega) = (Psi_f*I_q-B*Omega)/J

    #   定义电磁方程
    def Electromagnetic(self, I_q, Omega, epoch):
        return (self.U_q[epoch] - Omega * self.Psi_f - self.R_q * I_q) / self.L_q

    #   定义机械方程
    def Machinery(self, I_q, Omega):
        return (self.Psi_f * I_q - self.B * Omega) / self.J

    #   定义四阶龙格-库塔算法
    #   传入步长和迭代次数参数，得到每次迭代的I_q和Omega
    def Runge_Kutta(self, step, epoch):
        # 当I_q<=0时，I_q = 0
        if (self.I_q[epoch] <= 0 and self.PWM[epoch] == 0):
            self.I_q[epoch] = 0
        K1 = DCmotor.Electromagnetic(self, self.I_q[epoch], self.Omega[epoch], epoch)
        L1 = DCmotor.Machinery(self, self.I_q[epoch], self.Omega[epoch])
        K2 = DCmotor.Electromagnetic(self, self.I_q[epoch] + K1 * step / 2, self.Omega[epoch] + L1 * step / 2, epoch)
        L2 = DCmotor.Machinery(self, self.I_q[epoch] + K1 * step / 2, self.Omega[epoch] + L1 * step / 2)
        K3 = DCmotor.Electromagnetic(self, self.I_q[epoch] + K2 * step / 2, self.Omega[epoch] + L2 * step / 2, epoch)
        L3 = DCmotor.Machinery(self, self.I_q[epoch] + K2 * step / 2, self.Omega[epoch] + L2 * step / 2)
        K4 = DCmotor.Electromagnetic(self, self.I_q[epoch] + K3 * step, self.Omega[epoch] + L3 * step, epoch)
        L4 = DCmotor.Machinery(self, self.I_q[epoch] + K3 * step, self.Omega[epoch] + L3 * step)

        self.I_q[epoch + 1] = self.I_q[epoch] + (K1 + 2 * K2 + 2 * K3 + K4) * step / 6
        self.Omega[epoch + 1] = self.Omega[epoch] + (L1 + 2 * L2 + 2 * L3 + L4) * step / 6

    #   定义PWM控制函数
    # 根据电流和转速来控制PWM开关 1：开 0：关

    def PWM_Control(self, epoch):
        # 如果大于电流上限则关，小于电流下限则开，中间与上一个状态相同
        if self.I_q[epoch] > self.I_h:
            self.I_PWM[epoch] = 0
        elif self.I_q[epoch] < self.I_l:
            self.I_PWM[epoch] = self.I_PWM[epoch - 1]
        else:
            self.I_PWM[epoch] = 1
        # 如果在0-0.2s之间则为80rad/s,在0.2-0.4s之间则为120rad/s
        if (self.step * epoch % 0.4) > 0.2:
            # 如果大于转速上限则关，小于转速下限则开，中间与上一个状态相同
            if self.Omega[epoch] > (self.Omega_h + self.Width):
                self.Omega_PWM[epoch] = 0
            elif self.Omega[epoch] < (self.Omega_h - self.Width):
                self.Omega_PWM[epoch] = 1
            else:
                self.Omega_PWM[epoch] = self.Omega_PWM[epoch - 1]

        if (self.step * epoch % 0.4) < 0.2:
            if self.Omega[epoch] > (self.Omega_l + self.Width):
                self.Omega_PWM[epoch] = 0
            elif self.Omega[epoch] < (self.Omega_l - self.Width):
                self.Omega_PWM[epoch] = 1
            else:
                self.Omega_PWM[epoch] = self.Omega_PWM[epoch - 1]

            error = self.Omega_l - self.Omega[epoch]
            self.sum_error = error + self.sum_error
            self.value[epoch] = self.KP * error + self.KI * self.sum_error
            if (self.value[epoch] > 1.35 and self.Omega[epoch]>self.Omega[epoch-1]):
                    self.PID[epoch] = 0
                    self.Omega_PWM[epoch] = self.Omega_PWM[epoch]*self.PID[epoch]
            elif (self.value[epoch] < 1.3 and self.Omega[epoch] < self.Omega[epoch-1]):
                    self.PID[epoch] = 1
                    self.Omega_PWM[epoch] = self.Omega_PWM[epoch] * self.PID[epoch]

        # 合成最后控制IGBT的PWM信号
        self.PWM[epoch] = self.Omega_PWM[epoch] * self.I_PWM[epoch]
        # IGBT关，U_q = 0;IGBT开，U_q = U_dc;
        if (self.PWM[epoch] == 1):
            self.U_q[epoch] = self.U_dc
        else:
            self.U_q[epoch] = 0
        return self.PWM[epoch]

    #   画出转速电流曲线图
    def draw(self):
        # 绘制电流曲线
        plt.figure(dpi=1000, figsize=(24, 8))
        plt.plot(np.arange(0, self.epochmax + 1) * self.step, self.I_q)
        plt.plot(np.arange(0, self.epochmax + 1) * self.step, self.I_h * np.ones(len(self.I_q)), 'g--')
        plt.plot(np.arange(0, self.epochmax + 1) * self.step, self.I_l * np.ones(len(self.I_q)), 'g--')

        # 绘制转速曲线

        plt.plot(np.arange(0, self.epochmax + 1) * self.step, self.Omega)
        plt.plot(np.arange(0, self.epochmax + 1) * self.step, (self.Omega_h + self.Width) * np.ones((len(self.Omega))),
                 'k--')
        plt.plot(np.arange(0, self.epochmax + 1) * self.step, (self.Omega_h - self.Width) * np.ones((len(self.Omega))),
                 'k--')
        plt.plot(np.arange(0, self.epochmax + 1) * self.step, (self.Omega_l + self.Width) * np.ones((len(self.Omega))),
                 'y--')
        plt.plot(np.arange(0, self.epochmax + 1) * self.step, (self.Omega_l - self.Width) * np.ones((len(self.Omega))),
                 'y--')
        plt.xlabel('Time (s)')
        plt.ylabel('Omega (rad/s) I_q (A)')
        plt.title("Time -- Omega and I_q")

        # 保存为矢量图
        plt.savefig('motorsimulink.svg', format='svg')

        # 画出IGBT开关图
        plt.figure(dpi=1000, figsize=(24, 8))
        plt.plot(np.arange(0, self.epochmax + 1) * self.step, self.PWM, linewidth=0.25)
        plt.xlabel('Time (s)')
        plt.ylabel('0/1')
        plt.title("Time -- PWM")

        # 保存为矢量图
        plt.savefig('motorsimulink_PWM.svg', format='svg')


if __name__ == '__main__':
    motorsimulink = DCmotor(200, 1, 0.5, 0.05, 0.002, 0.1, 15, 14.5, 120, 80, 0, 10000, 0.00005)
    motorsimulink.Init()
    for epoch in range(motorsimulink.epochmax):
        motorsimulink.PWM_Control(epoch)
        motorsimulink.Runge_Kutta(motorsimulink.step, epoch)
        print(epoch, epoch * motorsimulink.step, motorsimulink.I_q[epoch], motorsimulink.Omega[epoch],motorsimulink.value[epoch],motorsimulink.PWM[epoch])
    motorsimulink.draw()
