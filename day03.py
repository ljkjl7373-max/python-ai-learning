# 第 1 步：导入需要的库
from pathlib import Path                         # 处理文件路径、创建文件夹
import matplotlib.pyplot as plt                  # 画图
from sklearn.datasets import load_digits         # 加载手写数字数据集

# 设置中文字体，避免标题里的中文显示成方框（Windows 自带微软雅黑/黑体）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False   # 让负号 '-' 正常显示，不变成方块

# 第 2 步：加载手写数字数据集
# load_digits() 自带 1797 张 8x8 的灰度手写数字图片（数字 0~9）
digits = load_digits()

# 第 3 步：打印三部分的形状（shape = 各维度的大小）
#   images: 每张图片的 8x8 像素矩阵，形状是 (1797, 8, 8)
#   data:   每张图片展开成一行的 64 个像素，形状是 (1797, 64)
#   target: 每张图片对应的真实数字（标签），形状是 (1797,)
print("images 形状:", digits.images.shape)
print("data   形状:", digits.data.shape)
print("target 形状:", digits.target.shape)

# 第 4 步：创建 2 行 5 列的画布，一共 10 个小格子
fig, axes = plt.subplots(2, 5, figsize=(10, 4))

# 第 5 步：把前 10 张图片一张张画进格子里
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap="gray")      # 显示第 i 张图片，cmap="gray" 表示灰度
    ax.set_title(f"标签: {digits.target[i]}")      # 标题写上这张图的真实数字
    ax.axis("off")                                 # 关掉坐标轴，画面更干净

# 第 6 步：用 Path 在“当前脚本所在的文件夹”旁创建 outputs 文件夹
#   __file__ 是当前脚本文件；.resolve().parent 是它所在的文件夹（绝对路径）
#   mkdir(exist_ok=True)：如果文件夹已存在就不报错，不存在就创建
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(exist_ok=True)

# 第 7 步：把画好的图保存到 outputs/samples_10.png
fig.tight_layout()                          # 自动调整间距，避免标题重叠
save_path = output_dir / "samples_10.png"
plt.savefig(save_path, dpi=150)             # dpi=150 让图片更清晰
plt.close(fig)                              # 关闭画布，释放内存（好习惯）

# 第 8 步：打印保存位置，方便你找到文件
print("完成！图片已保存到:", save_path)
