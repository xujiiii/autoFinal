#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_landscape(csv_path: Path, output_path: Path | None = None, title: str | None = None):
    # 1. 读取 CSV 文件
    df = pd.read_csv(csv_path)

    # 检查核心列是否存在 (处理可能的列名类型)
    required_cols = ["0", "1"]
    for col in required_cols:
        if col not in df.columns and int(col) in df.columns:
            # 防止 pandas 解析整数列名导致匹配不上
            df.rename(columns={int(col): col}, inplace=True)

    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV 文件必须包含列 '0' 和 '1'，当前包含列: {list(df.columns)}")

    # 2. 设置画板样式
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 3. 绘制散点图
    scatter = ax.scatter(
        df["0"],
        df["1"],
        c="#2b5c8f",        # 优雅的深蓝色
        alpha=0.6,          # 半透明度，防止多点重叠遮挡
        edgecolors="none",
        s=25                # 散点大小
    )

    # 4. 美化图表细节
    ax.set_xlabel("$z0$", fontsize=12, fontweight="bold")
    ax.set_ylabel("$z1$", fontsize=12, fontweight="bold")
    
    chart_title = title if title else f"Latent Space Landscape ({csv_path.name})"
    ax.set_title(chart_title, fontsize=14, pad=12)

    ax.grid(True, linestyle="--", alpha=0.3, color="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # 5. 保存或展示
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"图像已成功保存至: {output_path}")
    else:
        # 如果未指定输出路径，自动命名并在当前目录下保存
        default_out = csv_path.parent / f"{csv_path.stem}_plot.png"
        plt.savefig(default_out, dpi=300)
        print(f"图像已保存至默认路径: {default_out}")


def main():
    parser = argparse.ArgumentParser(description="绘制 Autoencoder 2D 隐空间 (z0, z1) 散点图")
    parser.add_argument("--file", type=Path, help="输入的 CSV 文件路径 (例如 landscape_encoded_train_coordinates.csv)")
    parser.add_argument("-o", "--out", type=Path, default=None, help="输出图片的保存路径 (如 output.png)")
    parser.add_argument("-t", "--title", type=str, default=None, help="自定义图表标题")

    args = parser.parse_args()

    plot_landscape(args.file, args.out, args.title)


if __name__ == "__main__":
    main()