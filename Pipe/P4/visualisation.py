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
    df = pd.read_csv(csv_path)
    df_error = pd.read_csv(error_csv)
    
    df["error"] = df_error["training_error"]
    
    required_cols = ["0", "1"]
    for col in required_cols:
        if col not in df.columns and int(col) in df.columns:
            df.rename(columns={int(col): col}, inplace=True)

    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must have '0' and '1'columns, all columns now: {list(df.columns)}")


    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    df_sorted = df.sort_values(by="error", ascending=True)
    scatter = ax.scatter(
        df_sorted["0"],
        df_sorted["1"],
        c=df_sorted["error"],   
        cmap="viridis",         
        vmin=df["error"].min(), 
        vmax=df["error"].max(), 
        alpha=0.8,              
        edgecolors="none",
        s=30
    )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.03)
    cbar.set_label("Reconstruction Error (RMSD)", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

 
    ax.set_xlabel("Latent vector z0", fontsize=14, fontweight="bold")
    ax.set_ylabel("Latent vector z1", fontsize=14, fontweight="bold")
    
    chart_title = title if title else f"Latent Space Landscape ({csv_path.name})"
    ax.set_title(chart_title, fontsize=14, pad=12, fontweight="bold")

    ax.grid(True, linestyle="--", alpha=0.3, color="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()


    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"Image saved at: {output_path}")
    else:
        default_out = csv_path.parent / f"{csv_path.stem}_plot.png"
        plt.savefig(default_out, dpi=300)
        print(f"Image saved at: {default_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, help="Input file like landscape_encoded_train_coordinates.csv)")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Location for saving")
    parser.add_argument("--error", type=Path, default=None)
    parser.add_argument("-t", "--title", type=str, default=None, help="Title of the plot")
    args = parser.parse_args()
    plot_landscape(args.file, args.error,args.out, args.title)


if __name__ == "__main__":
    main()