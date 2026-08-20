from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from molearn.data import PDBData
from molearn.models.small_foldingnet import Small_AutoEncoder



def load_combined_pdb(path: Path) -> np.ndarray:
    """Return (n_models, n_atoms, 3) for all CA atoms."""
    models = []
    cur = []
    with path.open() as f:
        for line in f:
            if line.startswith("MODEL"):
                cur = []
            elif line.startswith("ATOM") and line[12:16].strip() == "CA":
                cur.append([float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54])])
            elif line.startswith("ENDMDL") and cur:
                models.append(cur)
                cur = []
    return np.asarray(models, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined-pdb", required=True, type=Path)
    ap.add_argument("--manifest-csv", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--train-idx", required=True, type=Path)
    ap.add_argument("--test-idx", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    data = PDBData()
    data.import_pdb(filename=str(args.combined_pdb))
    data.fix_terminal()
    data.atomselect(atoms=["CA"])
    data.prepare_dataset()
    print(f"molearn standardisation: mean={data.mean:.4f}, std={data.std:.4f}")
    coords = data.dataset.numpy()           
    n, n_atoms, _ = coords.shape
    print(f"Loaded {n} models × {n_atoms} CAs (via PDBData)")

    manifest = pd.read_csv(args.manifest_csv, keep_default_na=False)
    if len(manifest) != n:
        raise SystemExit(f"manifest has {len(manifest)} rows, "
                         f"PDB has {n} models — order mismatch?")

    train_idx = np.loadtxt(args.train_idx, dtype=int)
    test_idx = np.loadtxt(args.test_idx, dtype=int)
    membership = np.array(["train"] * n, dtype=object)
    membership[test_idx] = "test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Small_AutoEncoder(out_points=n_atoms).to(device)
    state = torch.load(str(args.checkpoint), map_location=device,
                       weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        net.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "state_dict" in state:
        net.load_state_dict(state["state_dict"])
    else:
        net.load_state_dict(state)
    net.eval()
    print(f"Loaded checkpoint {args.checkpoint}")

    X = torch.from_numpy(coords).to(device)
    Z = []
    with torch.no_grad():
        for i in range(0, n, 128):
            batch = X[i:i + 128].float()
            z = net.encode(batch)
            Z.append(z.cpu().numpy().reshape(z.shape[0], -1))
    Z = np.concatenate(Z, axis=0)
    print(f"Latent shape: {Z.shape}")

    df = manifest.copy()
    df["idx"] = np.arange(n)
    df["z0"] = Z[:, 0]
    df["z1"] = Z[:, 1] if Z.shape[1] > 1 else 0.0
    df["membership"] = membership
    df.to_csv(args.out / "latent_with_labels.csv", index=False)
    print(f"Wrote v9_latent_with_labels.csv ({len(df)} rows)")

    # Legacy "landscape" CSVs for the existing kincore-overlay plotter.
    def write_legacy(indices, name):
        sub = df.loc[indices, ["z0", "z1", "idx", "chain_key"]].copy()
        sub.columns = ["0", "1", "train_index", "pdb_filename"]
        sub["pdb_filename"] = sub["pdb_filename"].astype(str) + ".pdb"
        sub.to_csv(args.out / name, index=False)
    write_legacy(train_idx, "landscape_encoded_train_coordinates.csv")
    write_legacy(test_idx, "landscape_encoded_test_coordinates.csv")
    print("Wrote landscape_encoded_{train,test}_coordinates.csv")


if __name__ == "__main__":
    main()
