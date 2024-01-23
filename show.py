import matplotlib.pyplot as plt
import torch

# 您的tensor数据
points = torch.tensor([
    [   0.4762,    0.9082],
        [ -31.5831,  -39.9315],
        [ -52.1844,  -71.0071],
        [ -60.4211,  -83.4873],
        [ -69.3365, -100.6035],
        [ -80.3101, -110.0435],
        [ -86.7765, -120.0472],
        [ -93.2574, -127.8565],
        [ -98.5992, -135.3300]
])

# 提取x和y坐标
x = points[:, 0].tolist()
y = points[:, 1].tolist()

# 绘制散点图
plt.scatter(x, y)

# 显示图形
plt.show()