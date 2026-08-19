from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ca_spline import read_backbone, process_chain, write_ca_pdb

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "HID": "H", "HIE": "H", "HIP": "H",
}

REGEX_PROFILES = {
    "v9_addendum_latent": {
        "dfg": re.compile(r"D[FYL]G"),
        "anchor": re.compile(r"[AYSTP][LP]E"),
        "min_len": 14, "max_len": 40, "target_len": 27,
    },
    "v9_addendum_litsearch": {
        "dfg": re.compile(r"D[FYLW]G"),
        "anchor": re.compile(r"[APSTV][LIPWV]E"),
        "min_len": 14, "max_len": 40, "target_len": 27,
    },
    "v9_addendum_comprehensive": {
        "dfg": re.compile(r"D[FYLW]G"),
        "anchor": re.compile(r"[AYSTPKDVNCHL][LP]E"),
        "min_len": 14, "max_len": 40, "target_len": 27,
    },
}


def pdb_chain_seq_and_resids(pdb_path: Path, chain_id: str):
    seq, resids, seen = [], [], set()
    with pdb_path.open() as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain_id:
                continue
            if line[12:16].strip() != "CA":
                continue
            rn = line[17:20].strip()
            if rn not in THREE_TO_ONE:
                continue
            try:
                resi = int(line[22:26])
            except ValueError:
                continue
            if resi in seen:
                continue
            seen.add(resi)
            seq.append(THREE_TO_ONE[rn])
            resids.append(resi)
    return "".join(seq), resids


def find_loop_motifs(seq: str, resids: list[int], source: str,
                     expected_anchor: str = ""):
    profile = REGEX_PROFILES[source]
    dfgs = [m.start() for m in profile["dfg"].finditer(seq)]
    anchors = [m.start() for m in profile["anchor"].finditer(seq)]
    candidates = []
    for d in dfgs:
        for a in anchors:
            gap = a - d
            if not (profile["min_len"] <= gap <= profile["max_len"]):
                continue
            e_idx = a + 2
            if e_idx >= len(resids):
                continue
            motif = seq[a:a + 3]
            score = abs(gap - profile["target_len"])
            candidates.append((score, d, a, motif, gap))
    if not candidates:
        return None
    if expected_anchor:
        motif_hits = [c for c in candidates if c[3] == expected_anchor]
        if motif_hits:
            candidates = motif_hits
    _, d_idx, a_idx, motif, gap = min(candidates, key=lambda x: x[0])
    e_idx = a_idx + 2
    return resids[d_idx], resids[e_idx], motif, gap + 1


def load_ref_flank(ref_pdb: Path, ref_chain: str, ref_dfg: int,
                   ref_ape: int, flank: int):
    ref_bb = read_backbone(ref_pdb, ref_chain)
    if ref_bb is None:
        raise SystemExit(f"Could not read reference {ref_pdb}")
    ref_specs, ref_flank = [], []
    for r in range(ref_dfg - flank, ref_dfg):
        if r in ref_bb and "CA" in ref_bb[r]:
            ref_specs.append(("dfg", r - ref_dfg))
            ref_flank.append(ref_bb[r]["CA"])
    for r in range(ref_ape + 1, ref_ape + flank + 1):
        if r in ref_bb and "CA" in ref_bb[r]:
            ref_specs.append(("ape", r - ref_ape))
            ref_flank.append(ref_bb[r]["CA"])
    return ref_specs, np.asarray(ref_flank, dtype=np.float32)


def spline_one_chain(pdb_path: Path, chain_id: str, dfg: int, ape_e: int,
                     ref_specs, ref_flank, ape_offset_to_e: int, **kwargs):
    return process_chain(
        pdb_path, chain_id, dfg, ape_e,
        ref_flank, ref_specs,
        ape_offset_to_e=ape_offset_to_e, **kwargs,
    )


def build_v9_row(row, pdb_dir: Path, ref_specs, ref_flank,
                 ape_offset_to_e: int, **kwargs):
    chain_key = str(row["chain_key"]).upper()
    pdb_path = pdb_dir / f"{chain_key[:4]}.pdb"
    chain_id = chain_key[4:]
    dfg = int(row["dfg_resi"])
    ape = int(row["ape_resi"])
    if not pdb_path.exists():
        return None, {"chain_key": chain_key, "status": "no_pdb_file"}
    try:
        r = spline_one_chain(pdb_path, chain_id, dfg, ape,
                             ref_specs, ref_flank, ape_offset_to_e, **kwargs)
    except Exception as e:
        return None, {"chain_key": chain_key, "status": f"exception:{e}"}
    if r["status"] != "ok":
        return None, {"chain_key": chain_key, **{
            k: v for k, v in r.items() if k not in ("ca", "anchors")}}
    manifest = {
        "chain_key": chain_key,
        "pdb": chain_key[:4],
        "chain": chain_id,
        "gene": row.get("gene", ""),
        "group": row.get("group", ""),
        "dfg_spatial": row.get("dfg_spatial", ""),
        "dihedral": row.get("dihedral", ""),
        "ligand_type": row.get("ligand_type", ""),
        "anchor_motif": "",
        "loop_length": int(row.get("expected_loop", r["expected"])),
        "flank_rmsd": r["flank_rmsd"],
        "source": "v9_indist",
    }
    return r["ca"], manifest


def build_addendum_row(row, pdb_dir: Path, ref_specs, ref_flank,
                       ape_offset_to_e: int, **kwargs):
    chain_key = str(row["chain_key"]).upper()
    source = str(row["source"])
    pdb_path = pdb_dir / f"{chain_key[:4]}.pdb"
    chain_id = chain_key[4:]
    if not pdb_path.exists():
        return None, {"chain_key": chain_key, "status": "no_pdb_file"}
    seq, resids = pdb_chain_seq_and_resids(pdb_path, chain_id)
    expected = str(row.get("anchor_motif", "") or "").strip()
    loop = find_loop_motifs(seq, resids, source, expected_anchor=expected)
    if loop is None:
        return None, {"chain_key": chain_key, "status": "no_motif_pair"}
    dfg, ape, motif, loop_len = loop
    try:
        r = spline_one_chain(pdb_path, chain_id, dfg, ape,
                             ref_specs, ref_flank, ape_offset_to_e, **kwargs)
    except Exception as e:
        return None, {"chain_key": chain_key, "status": f"exception:{e}"}
    if r["status"] != "ok":
        return None, {"chain_key": chain_key, **{
            k: v for k, v in r.items() if k not in ("ca", "anchors")}}
    ca = r["ca"].astype(np.float32)
    # ca = ca - ca.mean(axis=0)  # match published addendum PDB convention
    manifest = {
        "chain_key": chain_key,
        "pdb": chain_key[:4],
        "chain": chain_id,
        "gene": row.get("gene", ""),
        "group": row.get("group", ""),
        "dfg_spatial": row.get("dfg_spatial", ""),
        "dihedral": row.get("dihedral", ""),
        "ligand_type": row.get("ligand_type", ""),
        "anchor_motif": motif,
        "loop_length": int(row.get("loop_length", loop_len)),
        "flank_rmsd": r["flank_rmsd"],
        "source": source,
    }
    return ca, manifest


def write_combined(models: list[np.ndarray], out_pdb: Path):
    with out_pdb.open("w") as handle:
        for i, coords in enumerate(models):
            write_ca_pdb(handle, coords, i)
        handle.write("END\n")


def parse_models(path: Path):
    models, cur = [], []
    with path.open() as f:
        for line in f:
            if line.startswith("MODEL"):
                cur = []
            elif line.startswith("ENDMDL"):
                if cur:
                    models.append(np.array(cur, dtype=np.float64))
            elif line.startswith("ATOM"):
                cur.append([float(line[30:38]), float(line[38:46]),
                            float(line[46:54])])
    return models


def compare_pdbs(built_pdb: Path, ref_pdb: Path, atol: float = 1e-3):
    built, ref = parse_models(built_pdb), parse_models(ref_pdb)
    if len(built) != len(ref):
        return {"n_built": len(built), "n_ref": len(ref),
                "max_diff": None, "n_bad": -1}
    max_diff, n_bad = 0.0, 0
    for a, b in zip(built, ref):
        d = np.abs(a - b).max() if a.shape == b.shape else float("inf")
        max_diff = max(max_diff, d)
        if d > atol:
            n_bad += 1
    return {"n_built": len(built), "n_ref": len(ref),
            "max_diff": max_diff, "n_bad": n_bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-base", required=True, type=Path)
    ap.add_argument("--addendum-merged-csv", required=True, type=Path)
    ap.add_argument("--full-pdb-dir", required=True, type=Path)
    ap.add_argument("--ref-pdb", required=True, type=Path)
    ap.add_argument("--ref-chain", default="C")
    ap.add_argument("--ref-dfg", type=int, default=594)
    ap.add_argument("--ref-ape", type=int, default=623)
    ap.add_argument("--ape-offset-to-e", type=int, default=2)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--reference-pdb", type=Path, default=None)
    ap.add_argument("--reference-manifest", type=Path, default=None)
    ap.add_argument("--n-loop-points", type=int, default=27)
    ap.add_argument("--flank", type=int, default=40)
    ap.add_argument("--min-flank-frac", type=float, default=0.7)
    ap.add_argument("--min-loop-frac", type=float, default=0.7)
    ap.add_argument("--flank-rmsd-max", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N v9 + M addendum chains for testing")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ref_specs, ref_flank = load_ref_flank(
        args.ref_pdb, args.ref_chain, args.ref_dfg, args.ref_ape, args.flank)
    print(f"Reference flank atoms: {len(ref_specs)}/{2 * args.flank}")

    kwargs = dict(
        flank=args.flank,
        min_flank_frac=args.min_flank_frac,
        min_loop_frac=args.min_loop_frac,
        flank_rmsd_max=args.flank_rmsd_max,
        n_loop_points=args.n_loop_points,
    )

    m9 = pd.read_csv(args.manifest_base, keep_default_na=False)
    add = pd.read_csv(args.addendum_merged_csv, keep_default_na=False)
    if args.limit:
        m9 = m9.head(args.limit)
        add = add.head(max(0, args.limit - len(m9)))

    models, manifest_rows, failures = [], [], []
    for _, row in tqdm(m9.iterrows(), total=len(m9), desc="v9_indist"):
        ca, info = build_v9_row(row, args.full_pdb_dir, ref_specs, ref_flank,
                                args.ape_offset_to_e, **kwargs)
        if ca is None:
            failures.append(info)
            continue
        models.append(ca)
        manifest_rows.append(info)

    for _, row in tqdm(add.iterrows(), total=len(add), desc="addendum"):
        ca, info = build_addendum_row(row, args.full_pdb_dir, ref_specs,
                                      ref_flank, args.ape_offset_to_e, **kwargs)
        if ca is None:
            failures.append(info)
            continue
        models.append(ca)
        manifest_rows.append(info)

    for i, r in enumerate(manifest_rows):
        r["model_idx"] = i

    out_pdb = args.out / "combined_ca.pdb"
    write_combined(models, out_pdb)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(args.out / "manifest.csv", index=False)
    pd.DataFrame(failures).to_csv(args.out / "failures_.csv", index=False)

    print(f"\nWrote {out_pdb} ({len(models)} models, {len(failures)} failures)")
    print(manifest["source"].value_counts().to_string())

    if args.reference_pdb and args.reference_pdb.exists():
        cmp = compare_pdbs(out_pdb, args.reference_pdb)
        print(f"\nPDB compare: {cmp}")
    if args.reference_manifest and args.reference_manifest.exists():
        refm = pd.read_csv(args.reference_manifest, keep_default_na=False)
        if args.limit:
            refm = refm.head(len(manifest))
        keys_ok = manifest["chain_key"].tolist() == refm["chain_key"].tolist()
        print(f"Manifest key order match: {keys_ok}")
        if keys_ok and len(manifest) == len(refm):
            fr = np.abs(manifest["flank_rmsd"].astype(float) -
                        refm["flank_rmsd"].astype(float))
            print(f"flank_rmsd max abs diff: {fr.max():.6f}")


if __name__ == "__main__":
    main()
