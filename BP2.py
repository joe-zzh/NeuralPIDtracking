import torch

class NeuralPID:
    def __init__(self,kp,ki,kd):
        self.kcoef = 1.3# 输出比例系数
        self.kp = 0.1 # 积分系数
        self.ki = 0.1# 比例系数
        self.kd = 0.1 # 微分系数
        self.wp = kp # 积分学习系数
        self.wi = ki# 比例学习系数
        self.wd = kd # 微分学习系数
        self.lasterror = 0  # 上一次误差
        self.preerror = 0  # 前一次误差
        self.result = 0  # 控制结果
        self.deadband = 0.001  # 死区
        self.max=torch.pi/4.0
        self.min=-torch.pi/4.0
        self.x = torch.zeros(3) #输入-误差项
        self.w = torch.zeros(3) #权重
        # 初始化权重参数
    def forward(self, error):
        deltaResult = 0

        # 死区处理
        if abs(error) < self.deadband:
            deltaResult = 0
        else:
            
            # 计算误差项
            self.x[0] = error - self.lasterror
            self.x[1] = error 
            self.x[2] = error - self.lasterror * 2 + self.preerror
            #
            sabs = abs(self.wp) + abs(self.wi) + abs(self.wd)

            self.w[0] = self.wp / sabs
            self.w[1] = self.wi / sabs
            self.w[2] = self.wd / sabs
            deltaResult = (self.w[0] * self.x[0] + self.w[1] * self.x[1] + self.w[2] * self.x[2]) * self.kcoef
            # print(deltaResult)
            # print(self.wp,self.wi,self.wd)
        self.result = self.result + deltaResult
        
        # 单神经元学习
        self.neural_learning_rules(error, self.result, self.x)

        self.preerror = self.lasterror
        self.lasterror = error
        # print(self.result)
        if self.result>self.max:
            self.result=self.max
        elif self.result<self.min:
            self.result=self.min
        return self.result
    def neural_learning_rules(self, zk, uk, xi):
        self.wi = self.wi + self.ki * zk * uk * xi[0]
        self.wp = self.wp + self.kp * zk * uk * xi[1]
        self.wd = self.wd + self.kd * zk * uk * xi[2]


