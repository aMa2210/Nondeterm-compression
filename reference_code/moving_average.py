import numpy as np
import matplotlib.pyplot as plt

# def moving_average(data, smooth_interval=2):

#     if(smooth_interval > len(data)):
#         print("Smooth interval > lenght of data")
#         return
    
#     sum = 0
#     new_data = np.zeros(len(data))

#     for i in range (smooth_interval):
#         for j in range(smooth_interval):
#             sum += data[i+j]
#         average = sum/smooth_interval
#         new_data[i] = average
#         sum = 0

#     for i in range(len(data)-smooth_interval):
#         for j in range(smooth_interval):
#             sum += data[i+j]
#         average = sum/smooth_interval
#         new_data[i+smooth_interval] = average
#         sum = 0
#     return new_data

def moving_average(data, smooth_interval=10):
    data = np.array(data)
    length = len(data)
    new_data = np.zeros(length)
    
    # 计算左右半径 (例如窗口 10，半径就是 5)
    radius = smooth_interval // 2
    
    for i in range(length):
        # 1. 确定窗口的起始和结束位置
        # max(0, ...) 确保不超出左边界
        start_index = max(0, i - radius)
        
        # min(length, ...) 确保不超出右边界
        # 注意：切片是左闭右开，所以要 +1 (或者视逻辑调整，这里为了对称简单处理)
        end_index = min(length, i + radius + 1)
        
        # 2. 取出这一段数据
        window_slice = data[start_index : end_index]
        
        # 3. 求平均并赋值给当前位置 i (而不是 i + interval)
        new_data[i] = np.mean(window_slice)
        
    return new_data
