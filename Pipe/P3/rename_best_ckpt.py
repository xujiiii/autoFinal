import argparse
import re
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    args = ap.parse_args()

    f3_out = Path(args.folder)

    ckpt_files = list(f3_out.glob("checkpoint_epoch*_loss*.ckpt"))
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