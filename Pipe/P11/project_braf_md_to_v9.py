"""Project Clayton/Shen (2025) BRAF MD trajectories onto the v9 latent.

Pipeline per (topology, trajectory) pair:

  1. Load topology (prmtop) + trajectory (any AMBER-readable format:
     .nc, .ncdf, .mdcrd, .dcd, .xtc) via MDAnalysis.
  2. For each chain in the topology (monomer = 1 chain, dimer = 2 chains):
       a. Read the chain's residue-resolved sequence.
       b. Find the DFG and APE motifs by regex (D[FYL]G + [AS]PE,
          loop length in [14, 40]) — same logic as
          ``extract_loops_motif_anchored.py``.
       c. Identify the 80 flank Cα residues (40 N-terminal of DFG-D,
          40 C-terminal of APE-E).
  3. For each frame (stride ``--stride-ns``):
       a. Pull flank Cα coords for the chain.
       b. Kabsch-align flanks to BRAF 6UAN chain C flanks
          (loaded from ``--ref-pdb``).
       c. Apply the same rigid transform to loop Cα.
       d. Cubic-spline-resample loop Cα to 27 points (the
          ``spline_ca_arclen`` function ported from
          meyresearch/BRAF, identical to build_v9_ca_spline.py).
       e. Centre to centroid (the v9 train pipeline subtracts the
          per-chain centroid).
       f. Pass through the trained v9 FoldingNet encoder → (z0, z1).
  4. Emit a CSV row per frame×chain:
       system, replicate, chain, frame_ns, z0, z1,
       flank_rmsd, n_loop_present

Usage (one system at a time, easily parallelised across systems):

  python project_braf_md_to_v9.py \
      --prmtop      braf_clayton_md/prmtop/BrafMonomer_apo.prmtop  \
      --trajectory  braf_clayton_md/trajectories/BrafMonomer_apo/rep1.nc \
      --system      BrafMonomer_apo \
      --replicate   rep1 \
      --ref-pdb     manuscript_draft/data/refs/6UAN.pdb \
      --ref-chain   C \
      --ref-dfg     594 --ref-ape 623 --ape-offset-to-e 2 \
      --checkpoint  /home/edina/kinase_v4_training/v9_release/v9_ckpt.pt \
      --stride-ns   1.0 \
      --out         manuscript_draft/data/v9_md_projection/BrafMonomer_apo_rep1.csv

Notes
-----
* ``--stride-ns`` is interpreted via the trajectory's reported ``dt``.
  Many AMBER NetCDF files have dt = 1 ps (frames at 1 ps); for a 5 µs
  trajectory at 1 ps stride you get 5 000 000 frames.  Default
  ``stride-ns = 1.0`` keeps one frame per ns → 5 000 frames/replicate.
* The flank-RMSD cutoff is set to 8 Å (vs 5 Å for static PDB) to
  permit modest flank distortion during dynamics; per-frame
  flank_rmsd values are stored so this can be tightened post-hoc.
* If the topology has chains A and B, both are projected and tagged.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from tqdm import tqdm

# Re-use the existing pipeline's helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v9_ca_spline import (
    spline_ca_arclen, kabsch, read_backbone, BACKBONE
)


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # AMBER protonation variants
    "HID": "H", "HIE": "H", "HIP": "H",
    "ASH": "D", "GLH": "E", "LYN": "K", "CYX": "C", "CYM": "C",
}
DFG_RE = re.compile(r"D[FYL]G")
APE_RE = re.compile(r"[AS]PE")


def find_loop_motifs(seq: str, min_len: int = 14, max_len: int = 40,
                     target_len: int = 27) -> tuple[int, int] | None:
    """Return (dfg_idx, ape_idx) in the sequence's 0-based index, or None.

    Choose the (DFG, APE) pair whose APE_idx - DFG_idx is in
    [min_len, max_len] and closest to ``target_len``.
    """
    dfgs = [m.start() for m in DFG_RE.finditer(seq)]
    apes = [m.start() for m in APE_RE.finditer(seq)]
    best, best_gap = None, None
    for d in dfgs:
        for a in apes:
            gap = a - d
            if not (min_len <= gap <= max_len):
                continue
            score = abs(gap - target_len)
            if best is None or score < best_gap:
                best, best_gap = (d, a), score
    return best


def find_all_loop_motifs(seq: str, min_len: int = 14, max_len: int = 40,
                         target_len: int = 27, min_kinase_span: int = 200
                         ) -> list[tuple[int, int]]:
    """Find ALL DFG/APE motif pairs in the sequence, one per kinase domain.

    Used when the topology is a multimer (e.g. BRAF dimer) under a single
    MDAnalysis chain.  Greedy: pick the best pair, mark its span as
    consumed, then search the remaining sequence (regions ≥ ``min_kinase_span``
    long) for the next pair.

    Returns list of (dfg_idx, ape_idx) tuples in sequence order.
    """
    pairs: list[tuple[int, int]] = []
    consumed = [False] * len(seq)

    while True:
        # Build sub-regions of unconsumed sequence.
        regions = []
        i = 0
        while i < len(seq):
            while i < len(seq) and consumed[i]:
                i += 1
            start = i
            while i < len(seq) and not consumed[i]:
                i += 1
            if i - start >= min_kinase_span:
                regions.append((start, i))
        # Try to find a motif pair in any region.
        found_any = False
        for rs, re_ in regions:
            sub = seq[rs:re_]
            m = find_loop_motifs(sub, min_len, max_len, target_len)
            if m is not None:
                d, a = m
                pairs.append((rs + d, rs + a))
                # Mark the kinase domain "footprint" as consumed: DFG-40
                # to APE+40 (the flank-spanning range) so the next search
                # cannot recycle the same flank residues.
                lo = max(rs, rs + d - 80)
                hi = min(re_, rs + a + 80)
                for k in range(lo, hi):
                    consumed[k] = True
                found_any = True
                break
        if not found_any:
            break
    pairs.sort(key=lambda p: p[0])
    return pairs


def load_ref_flank(ref_pdb: Path, ref_chain: str,
                   ref_dfg: int, ref_ape: int, flank: int):
    """Return (flank_coords (n,3), flank_specs [('dfg'/'ape', off), ...])."""
    bb = read_backbone(ref_pdb, ref_chain)
    if bb is None:
        raise SystemExit(f"Could not read flank from {ref_pdb}:{ref_chain}")
    specs, coords = [], []
    for r in range(ref_dfg - flank, ref_dfg):
        if r in bb and "CA" in bb[r]:
            specs.append(("dfg", r - ref_dfg))
            coords.append(bb[r]["CA"])
    for r in range(ref_ape + 1, ref_ape + flank + 1):
        if r in bb and "CA" in bb[r]:
            specs.append(("ape", r - ref_ape))
            coords.append(bb[r]["CA"])
    return np.asarray(coords, dtype=np.float32), specs


def chain_seq_and_resids(u, segid_or_chain) -> tuple[str, list[int]]:
    """Return (one-letter sequence, list of MDAnalysis residue indices)
    for the given chain / segment.  Falls back to unique segid if
    ``segid_or_chain`` is empty.
    """
    has_chainID = hasattr(u._topology, "chainIDs")
    if segid_or_chain:
        if has_chainID:
            sel = u.select_atoms(
                f"protein and name CA and (segid {segid_or_chain} "
                f"or chainID {segid_or_chain})"
            )
        else:
            sel = u.select_atoms(
                f"protein and name CA and segid {segid_or_chain}"
            )
    else:
        sel = u.select_atoms("protein and name CA")
    seq_chars, resids = [], []
    for atom in sel:
        rn = atom.resname.upper()
        if rn in THREE_TO_ONE:
            seq_chars.append(THREE_TO_ONE[rn])
            resids.append(int(atom.resid))
    return "".join(seq_chars), resids


def detect_chains(u) -> list[str]:
    """Best-effort chain enumeration for an AMBER prmtop.  AMBER drops
    chain IDs, so segments / consecutive-resid-jumps are used."""
    segids = sorted({s.segid for s in u.segments
                     if s.atoms.select_atoms("protein and name CA").n_atoms >= 200})
    if len(segids) >= 1 and any(s for s in segids):
        return segids
    # Fall back: detect chain breaks by resid discontinuity.
    cas = u.select_atoms("protein and name CA")
    if cas.n_atoms == 0:
        return []
    breaks = []
    prev = None
    start = 0
    for i, a in enumerate(cas):
        if prev is not None and a.resid - prev > 30:
            breaks.append((start, i))
            start = i
        prev = a.resid
    breaks.append((start, len(cas)))
    # Synthesize segids "1", "2", ...
    if len(breaks) > 1:
        for j, (s, e) in enumerate(breaks, start=1):
            for a in cas[s:e]:
                a.segment.segid = str(j)
        return [str(j + 1) for j in range(len(breaks))]
    return [""]


# --------------------------------------------------------------- model load


def load_v9_encoder(ckpt_path: Path, n_atoms: int = 27, device=None):
    from molearn.models.small_foldingnet import Small_AutoEncoder
    net = Small_AutoEncoder(out_points=n_atoms).to(device)
    state = torch.load(str(ckpt_path), map_location=device,
                       weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        net.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "state_dict" in state:
        net.load_state_dict(state["state_dict"])
    else:
        net.load_state_dict(state)
    net.eval()
    return net


# --------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prmtop", required=True, type=Path)
    ap.add_argument("--trajectory", required=True, type=Path)
    ap.add_argument("--system", required=True,
                    help="Tag e.g. BrafMonomer_apo")
    ap.add_argument("--replicate", default="rep1",
                    help="Replicate tag, if the deposit has multiple.")
    ap.add_argument("--ref-pdb", required=True, type=Path,
                    help="BRAF 6UAN PDB used for flank alignment.")
    ap.add_argument("--ref-chain", default="C")
    ap.add_argument("--ref-dfg", type=int, default=594)
    ap.add_argument("--ref-ape", type=int, default=623,
                    help="Reference APE-E (not motif start) resi.")
    ap.add_argument("--ape-offset-to-e", type=int, default=2)
    ap.add_argument("--flank", type=int, default=40)
    ap.add_argument("--min-flank-frac", type=float, default=0.7)
    ap.add_argument("--min-loop-frac", type=float, default=0.7)
    ap.add_argument("--flank-rmsd-max", type=float, default=8.0,
                    help="MD-side cutoff in Å; static pipeline uses 5.")
    ap.add_argument("--n-loop-points", type=int, default=27)
    ap.add_argument("--stride-ns", type=float, default=1.0,
                    help="Frames per ns to keep; 1.0 = one frame/ns.")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit-frames", type=int, default=0)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ---- MDAnalysis universe ----
    try:
        import MDAnalysis as mda
    except ImportError:
        raise SystemExit("MDAnalysis is required: pip install MDAnalysis")
    print(f"Loading {args.prmtop.name} + {args.trajectory.name}")
    u = mda.Universe(str(args.prmtop), str(args.trajectory))
    n_frames = u.trajectory.n_frames
    dt_ps = float(u.trajectory.dt)  # ps per frame
    print(f"  {n_frames} frames at dt={dt_ps} ps "
          f"({n_frames * dt_ps / 1000:.1f} ns total)")

    # ---- chains ----
    chains = detect_chains(u)
    print(f"  detected chains: {chains}")

    # ---- per-chain motif + flank/loop residue lists ----
    # For each detected chain (segid), find ALL DFG/APE motif pairs in
    # its sequence -- this handles dimers / multimers that live under a
    # single AMBER segid by their two-kinase-domain sequence content.
    chain_info = []
    for c in chains:
        seq, resids = chain_seq_and_resids(u, c)
        if len(seq) < 200:
            print(f"  [skip] chain {c!r}: only {len(seq)} CAs")
            continue
        all_pairs = find_all_loop_motifs(seq)
        if not all_pairs:
            print(f"  [skip] chain {c!r}: no DFG/APE motif pair found")
            continue
        if len(all_pairs) > 1:
            print(f"  chain {c!r}: {len(all_pairs)} kinase domains found")
        for sub_idx, (dfg_idx, ape_idx) in enumerate(all_pairs):
            sub_tag = c if len(all_pairs) == 1 else f"{c}.{sub_idx + 1}"
            dfg_resi = resids[dfg_idx]
            ape_e_idx = ape_idx + args.ape_offset_to_e
            if ape_e_idx >= len(resids):
                print(f"  [skip] {sub_tag}: APE-E index past end")
                continue
            ape_e_resi = resids[ape_e_idx]
            loop_resis = resids[dfg_idx:ape_e_idx + 1]
            n_flank_start = max(0, dfg_idx - args.flank)
            n_flank_end = dfg_idx
            c_flank_start = ape_e_idx + 1
            c_flank_end = min(len(resids), ape_e_idx + 1 + args.flank)
            flank_resis = (resids[n_flank_start:n_flank_end]
                           + resids[c_flank_start:c_flank_end])
            flank_specs = (
                [("dfg", r - dfg_resi)
                 for r in resids[n_flank_start:n_flank_end]]
                + [("ape", r - ape_e_resi)
                   for r in resids[c_flank_start:c_flank_end]]
            )
            chain_info.append({
                "chain": sub_tag,
                "segid": c,
                "dfg_idx": dfg_idx, "ape_e_idx": ape_e_idx,
                "dfg_resi": dfg_resi, "ape_e_resi": ape_e_resi,
                "loop_resis": loop_resis,
                "flank_resis": flank_resis,
                "flank_specs": flank_specs,
            })
            print(f"  {sub_tag}: DFG at seq[{dfg_idx}]/resi {dfg_resi}, "
                  f"APE-E at seq[{ape_e_idx}]/resi {ape_e_resi}, "
                  f"loop length {len(loop_resis)}, "
                  f"flank atoms {len(flank_resis)}")

    if not chain_info:
        raise SystemExit("No usable chain found.")

    # ---- reference flank (one set, applied to every chain) ----
    ref_flank_full, ref_specs_full = load_ref_flank(
        args.ref_pdb, args.ref_chain, args.ref_dfg, args.ref_ape, args.flank
    )
    ref_lookup = {tuple(s): xyz for s, xyz in zip(ref_specs_full, ref_flank_full)}

    # ---- v9 encoder ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_v9_encoder(args.checkpoint, n_atoms=args.n_loop_points,
                          device=device)
    print(f"  encoder loaded onto {device}")

    # ---- stride ----
    step = max(1, int(round(args.stride_ns * 1000.0 / dt_ps)))
    frame_indices = list(range(0, n_frames, step))
    if args.limit_frames:
        frame_indices = frame_indices[:args.limit_frames]
    print(f"  striding every {step} frames ({args.stride_ns} ns) "
          f"-> {len(frame_indices)} frames")

    # Pre-build per-chain MDAnalysis selections (one CA selection per
    # chain for flank + loop) keyed by resid.
    has_chainID = hasattr(u._topology, "chainIDs")
    def select_ca(chain_id, resi):
        if chain_id:
            if has_chainID:
                sel_str = (f"protein and name CA and resid {resi} and "
                           f"(segid {chain_id} or chainID {chain_id})")
            else:
                sel_str = (f"protein and name CA and resid {resi} "
                           f"and segid {chain_id}")
        else:
            sel_str = f"protein and name CA and resid {resi}"
        a = u.select_atoms(sel_str)
        return a[0] if a.n_atoms >= 1 else None

    chain_sel = {}
    for ci in chain_info:
        # Atom selection must use the underlying segid (the sub-tagged
        # chain id like "SYSTEM.1" won't match), but the per-frame loop
        # also needs to disambiguate residues that appear in BOTH kinase
        # domains.  We do that by intersecting with explicit resid lists
        # below, which are unique per sub-chain because resids are
        # globally unique within the prmtop.
        sel_id = ci.get("segid", ci["chain"])
        flank_atoms = [select_ca(sel_id, r) for r in ci["flank_resis"]]
        loop_atoms = [select_ca(sel_id, r) for r in ci["loop_resis"]]
        # Filter Nones (rare; MDAnalysis should hit every residue from the
        # same topology).
        ci_flank = [(s, a) for s, a in zip(ci["flank_specs"], flank_atoms)
                    if a is not None and tuple(s) in ref_lookup]
        ci_loop = [a for a in loop_atoms if a is not None]
        ref_match = np.asarray(
            [ref_lookup[tuple(s)] for s, _ in ci_flank], dtype=np.float32
        )
        chain_sel[ci["chain"]] = {
            "flank_specs_match": [s for s, _ in ci_flank],
            "flank_atoms": [a for _, a in ci_flank],
            "loop_atoms": ci_loop,
            "ref_flank": ref_match,
            "expected_loop": len(ci["loop_resis"]),
        }
        print(f"  chain {ci['chain']}: matched flank {len(ci_flank)} / "
              f"{len(ci['flank_resis'])}; loop atoms {len(ci_loop)}")

    # ---- iterate frames ----
    rows = []
    batch_coords, batch_meta = [], []

    def flush_batch():
        if not batch_coords:
                return
        # np.stack(batch_coords) 的 shape 是 (B, 27, 3)
        # FoldingNet 要求输入 shape 为 (B, 3, N)，因此必须使用 .transpose(0, 2, 1) 或 .permute(0, 2, 1)
        X = torch.tensor(np.stack(batch_coords), dtype=torch.float32, device=device)
        X = X.permute(0, 2, 1)  # 变为 (B, 3, 27)

        with torch.no_grad():
            z = net.encode(X).cpu().numpy().reshape(X.shape[0], -1)
        for meta, zz in zip(batch_meta, z):
            r = dict(meta)
            r["z0"] = float(zz[0]); r["z1"] = float(zz[1])
            rows.append(r)
        batch_coords.clear(); batch_meta.clear()

    for fi in tqdm(frame_indices, desc="frames"):
        u.trajectory[fi]
        for ci in chain_info:
            sel = chain_sel[ci["chain"]]
            mob = np.array([a.position for a in sel["flank_atoms"]],
                           dtype=np.float32)
            ref = sel["ref_flank"]
            if mob.shape[0] < args.min_flank_frac * len(ci["flank_resis"]):
                continue
            R, t, rmsd = kabsch(mob, ref)
            if rmsd > args.flank_rmsd_max:
                continue
            mc = mob.mean(axis=0); rc = ref.mean(axis=0)
            loop_xyz = np.array([a.position for a in sel["loop_atoms"]],
                                dtype=np.float32)
            if loop_xyz.shape[0] < args.min_loop_frac * sel["expected_loop"]:
                continue
            loop_xyz = (loop_xyz - mc) @ R.T + rc
            spline = spline_ca_arclen(loop_xyz, args.n_loop_points)
            if spline is None:
                continue
            spline = spline - spline.mean(axis=0)  # match train centering
            batch_coords.append(spline.astype(np.float32))
            batch_meta.append({
                "system": args.system,
                "replicate": args.replicate,
                "chain": ci["chain"],
                "frame_ns": fi * dt_ps / 1000.0,
                "flank_rmsd": rmsd,
                "n_loop_present": loop_xyz.shape[0],
            })
            if len(batch_coords) >= 256:
                flush_batch()
    flush_batch()

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} frame-chain rows to {args.out}")
    if len(df):
        for ch, sub in df.groupby("chain"):
            print(f"  chain {ch}: median flank RMSD {sub['flank_rmsd'].median():.2f} Å, "
                  f"z0 mean {sub['z0'].mean():+.2f}, z1 mean {sub['z1'].mean():+.2f}")


if __name__ == "__main__":
    main()
