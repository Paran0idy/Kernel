import subprocess
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

# 解析命令行参数
parser = argparse.ArgumentParser(description='CUDA Kernels Performance Analysis')
parser.add_argument('kernel', type=str, help='The name of the CUDA kernel to execute')
args = parser.parse_args()

# 定义项目和构建目录
project_dir = "./"  # 项目目录，请根据实际情况修改
build_dir = os.path.join(project_dir, "build")

# 创建构建目录，如果不存在
if not os.path.exists(build_dir):
    os.makedirs(build_dir)

# 切换到构建目录
os.chdir(build_dir)

# 运行CMake和Make命令
try:
    # 配置项目
    subprocess.run(["cmake", ".."], check=True)
    # 构建项目
    subprocess.run(["make"], check=True)
    print("Build completed successfully.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while trying to build the project: {e}")
    exit(1)  # 如果构建失败，退出脚本

# 根据命令行参数执行指定的CUDA程序
executable_path = f"./{args.kernel}"  # 通过命令行参数指定的CUDA可执行文件

try:
    # 执行CUDA程序
    subprocess.run([executable_path], check=True)
    print(f"Execution of {args.kernel} completed.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the CUDA program {args.kernel}: {e}")
    exit(1)

# 文件路径
file_path = '../performance.txt'  # 假设你的文件名为 performance_data.txt

# 初始化空列表来存储读取的数据
kernels = []
execution_times = []
tflops = []

# 读取文件
with open(file_path, 'r') as file:
    for line in file:
        # 分割每行数据
        data = line.split()  # 假设数据是空格分隔的
        # 添加数据到对应的列表中
        kernels.append(data[2])  # Kernel 名称
        execution_times.append(float(data[0]))  # 执行时间
        tflops.append(float(data[1]))  # TFLOPS

# 创建图表
fig, ax1 = plt.subplots(figsize=(14, 8))

# 创建双轴图表
ax2 = ax1.twinx()
index = np.arange(len(kernels))
bar_width = 0.4

# 绘制执行时间条形图
ax1.bar(index, execution_times, bar_width, label='Execution Time (ms)', color='b')

# 绘制TFLOPS条形图
ax2.bar(index + bar_width, tflops, bar_width, label='TFLOPS', color='g')

# 设置图表标题和标签
ax1.set_xlabel('Kernel Version')
ax1.set_ylabel('Execution Time (ms)', color='b')
ax2.set_ylabel('TFLOPS', color='g')
ax1.set_title('CUDA Kernels Performance Analysis')
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(kernels)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

fig.tight_layout()

# 显示图表
# plt.show()

plt.savefig('../performance.png')  # 修改这里的路径到你想保存的位置

# 检查txt文件是否存在
if os.path.exists(file_path):
    # 删除文件
    os.remove(file_path)
else:
    print(f"File {file_path} does not exist.")
