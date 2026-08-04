import argparse
import re
import shutil
from pathlib import Path
import pickle
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    args = ap.parse_args()

    f3_out = Path(args.folder)
    

    # 读取并载入 pkl 文件
    with open(f"{f3_out}/final.pkl", 'rb') as f:
        data = pickle.load(f)

    print(f"Best loss is {data['best']}, total epochs is {data['epochs']}")    
    
    best_name=f"checkpoint_epoch*_loss{data['best']}.ckpt"
    
    ckpt_files = list(f3_out.glob(best_name))
    if not ckpt_files:
        print(f"No matching checkpoint files found in {f3_out}")
        return

    # 正则提取 loss 数字，排除末尾 .ckpt 的句点
    best_ckpt = min(
        ckpt_files,
        key=lambda p: float(re.search(r"loss([0-9]+\.?[0-9]*)", p.stem).group(1)),
    )

    target_path = f3_out / "best.ckpt"
    shutil.copy2(best_ckpt, target_path)
    print(f"Copied {best_ckpt.name} to {target_path}")


if __name__ == "__main__":
    main()