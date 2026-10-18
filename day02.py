from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# 读取手写数字图片
digits = load_digits()
print("图片数组形状：", digits.images.shape)
print("特征表形状：", digits.data.shape)
print("标签数组形状：", digits.target.shape)
print("前10个正确标签：", digits.target[:10])

print("图片形状：", digits.images.shape)
print("前10张图片的正确数字：", digits.target[:10])
print("第一张图片的像素：")
print(digits.images[0])

print("第一张图片展开后的64个数：")
print(digits.data[0])

print("第一张图片的正确标签：", digits.target[0])

# 创建保存图片的文件夹
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(exist_ok=True)

# 展示20张图片，排成4行5列
fig, axes = plt.subplots(2, 5, figsize=(10, 4))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap="gray")
    ax.set_title(f"Label: {digits.target[i]}")
    ax.axis("off")

fig.tight_layout()
fig.savefig(output_dir / "samples_10.png", dpi=150)
plt.close(fig)

print("完成！图片保存在：", output_dir / "samples_.png")