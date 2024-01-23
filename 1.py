import math
import random
import tkinter as tk
import matplotlib.pyplot as plt
import csv
from tkinter import Label

# 创建窗口
root = tk.Tk()
root.attributes('-fullscreen', True)  # 全屏显示

label_n = Label(root, text="n: 0", font=("Helvetica", 16))
label_n.pack()

csv_file_path = "mouse_data.csv"

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# 设置小球的初始位置
ball1_pos = (screen_width/2, screen_height/2)
ball2_pos = (ball1_pos[0] + random.randint(-200, 200),ball1_pos[1] + random.randint(-200, 200))

# 设置小球的半径
ball_radius = 20

# 设置鼠标记录状态
recording = False
mouse_path = []

n=0

# 鼠标移动事件处理函数
def motion(event):
    global recording, mouse_path, n
    if recording:
        mouse_path.append((event.x, event.y))

# 鼠标点击事件处理函数
def mouse_click(event):
    global recording, mouse_path, ball2_pos, n

    if event.x >= ball1_pos[0] - ball_radius and event.x <= ball1_pos[0] + ball_radius and event.y >= ball1_pos[1] - ball_radius and event.y <= ball1_pos[1] + ball_radius:
        recording = True
        mouse_path = [(event.x, event.y)]
    elif event.x >= ball2_pos[0] - ball_radius and event.x <= ball2_pos[0] + ball_radius and event.y >= ball2_pos[1] - ball_radius and event.y <= ball2_pos[1] + ball_radius:
        recording = False
        canvas.delete("ball2")
        #visualize_path(mouse_path)  # 可视化鼠标轨迹
        save_to_csv(mouse_path)
        n = n+1
        if n == 100:
            root.destroy()
        label_n.config(text=f"n: {n}")
        mouse_path = []

        # 重新生成第二个小球的位置
        ball2_pos = (ball1_pos[0] + random.randint(-200, 200), ball1_pos[1] + random.randint(-200, 200))
        
        # 绘制新的第二个小球
        canvas.create_oval(ball2_pos[0]-ball_radius, ball2_pos[1]-ball_radius, ball2_pos[0]+ball_radius, ball2_pos[1]+ball_radius, fill="blue", tags="ball2")


# 键盘事件处理函数
def key(event):
    if event.keysym == "Escape":
        root.destroy()

def save_to_csv(path):
    # 将路径坐标转换为相对于起点的坐标
    x_rel = [px - path[0][0] for px, py in path]
    y_rel = [-(py - path[0][1]) for px, py in path]

    # 计算每个点相对于起点的距离，用于z轴表示
    distances = [math.sqrt((x_rel[0] - px)**2 + (y_rel[0] - py)**2) for px, py in zip(x_rel, y_rel)]

    # 选择10个关键点
    key_points_indices = [int(i) for i in range(0, len(path), max(1, len(path)//10))]
    key_points_x = [x_rel[i] for i in key_points_indices]
    key_points_y = [y_rel[i] for i in key_points_indices]
    key_points_distances = [distances[i] for i in key_points_indices]

    # 打开 CSV 文件进行写操作
    with open(csv_file_path, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # 写入一行数据
        csv_writer.writerow([f"{key_points_x[-1]},{key_points_y[-1]}"] + [f"{key_points_x[i]},{key_points_y[i]}" for i in range(0,10)])


def visualize_path(path):
    # 将路径坐标转换为相对于起点的坐标
    x_rel = [px - path[0][0] for px, py in path]
    y_rel = [-(py - path[0][1]) for px, py in path]

    # 计算每个点相对于起点的距离，用于z轴表示
    distances = [math.sqrt((x_rel[0] - px)**2 + (y_rel[0] - py)**2) for px, py in zip(x_rel, y_rel)]

    # 选择10个关键点
    key_points_indices = [int(i) for i in range(0, len(path), max(1, len(path)//10))]
    key_points_x = [x_rel[i] for i in key_points_indices]
    key_points_y = [y_rel[i] for i in key_points_indices]
    key_points_distances = [distances[i] for i in key_points_indices]

    # 使用z轴信息，通过颜色表示距离的远近
    plt.scatter(key_points_x, key_points_y, c=key_points_distances, cmap='viridis', marker='o', s=50)

    # 在关键点位置添加文本标签，显示终点到起点的距离
    
    plt.text(key_points_x[-1], key_points_y[-1], f'Distance to Origin: {key_points_distances[-1]:.2f}', ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.5))

    # 添加颜色条，表示z轴信息
    plt.colorbar(label='Distance to Endpoint')

    plt.show()

# 绘制小球
canvas = tk.Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
canvas.pack()
canvas.create_oval(ball1_pos[0]-ball_radius, ball1_pos[1]-ball_radius, ball1_pos[0]+ball_radius, ball1_pos[1]+ball_radius, fill="red")
canvas.create_oval(ball2_pos[0]-ball_radius, ball2_pos[1]-ball_radius, ball2_pos[0]+ball_radius, ball2_pos[1]+ball_radius, fill="blue", tags="ball2")

# 绑定鼠标事件
canvas.bind('<Motion>', motion)
canvas.bind('<Button-1>', mouse_click)

# 绑定键盘事件
root.bind('<Key>', key)

# 运行窗口
root.mainloop()
