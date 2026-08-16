#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_landscape(csv_path: Path, error_csv: Path, output_path: Path | None = None, title: str | None = None):
    # 1. 读取 CSV 文件
    df = pd.read_csv(csv_path)
    df_error = pd.read_csv(error_csv)
    
    df["error"] = df_error["training_error"]
    
    # 检查核心列是否存在
    required_cols = ["0", "1"]
    for col in required_cols:
        if col not in df.columns and int(col) in df.columns:
            df.rename(columns={int(col): col}, inplace=True)

    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV 文件必须包含列 '0' 和 '1'，当前包含列: {list(df.columns)}")

    # 2. 设置画板样式
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    # -------------------------------------------------------------
    # 🌟 使用真实 error 进行颜色映射 (绝对不产生超纲数值)
    # -------------------------------------------------------------
    # 将 error 较大的点放在上面绘制 (sort_values)，防止低 error 的点把高 error 点遮挡
    df_sorted = df.sort_values(by="error", ascending=True)

    # 绘制散点图：颜色直接绑定真实的 error 列，完全不进行任何插值计算
    scatter = ax.scatter(
        df_sorted["0"],
        df_sorted["1"],
        c=df_sorted["error"],   # 严格使用原始数据 error
        cmap="viridis",         # 可选: 'plasma', 'cividis', 'YlOrRd'
        vmin=df["error"].min(), # 明确显示范围下限为真实最小值
        vmax=df["error"].max(), # 明确显示范围上限为真实最大值
        alpha=0.8,              # 保持适度透明感
        edgecolors="none",
        s=30
    )

    # -------------------------------------------------------------
    # 🌟 添加 Colorbar 展示真实 error 范围
    # -------------------------------------------------------------
    cbar = fig.colorbar(scatter, ax=ax, pad=0.03)
    cbar.set_label("Reconstruction Error (RMSD)", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    # 5. 美化图表细节
    ax.set_xlabel("Latent vector z0", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latent vector z1", fontsize=12, fontweight="bold")
    
    chart_title = title if title else f"Latent Space Landscape ({csv_path.name})"
    ax.set_title(chart_title, fontsize=14, pad=12, fontweight="bold")

    ax.grid(True, linestyle="--", alpha=0.3, color="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # 6. 保存或展示
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"Image saved at: {output_path}")
    else:
        default_out = csv_path.parent / f"{csv_path.stem}_plot.png"
        plt.savefig(default_out, dpi=300)
        print(f"Image saved at: {default_out}")


def main():
    parser = argparse.ArgumentParser(description="绘制 Autoencoder 2D 隐空间 (z0, z1) 散点图")
    parser.add_argument("--file", type=Path, help="输入的 CSV 文件路径 (例如 landscape_encoded_train_coordinates.csv)")
    parser.add_argument("-o", "--out", type=Path, default=None, help="输出图片的保存路径 (如 output.png)")
    parser.add_argument("--error", type=Path, default=None)
    parser.add_argument("-t", "--title", type=str, default=None, help="自定义图表标题")

    args = parser.parse_args()

    plot_landscape(args.file, args.error,args.out, args.title)


if __name__ == "__main__":
    main()