from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import pandas as pd


def norm_key(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.replace("_", "", regex=False)


def parse_fasta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    name = ""
    seq: list[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    out[name] = "".join(seq)
                name = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
    if name:
        out[name] = "".join(seq)
    return out


def read_chain_residues(pdb_path: Path, chain: str) -> tuple[list[int], list[str]]:
    residues: list[int] = []
    aas: list[str] = []
    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if chain and line[21] != chain:
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[16].strip() not in {"", "A"}:
                continue
            try:
                resi = int(line[22:26])
            except ValueError:
                continue
            if residues and residues[-1] == resi:
                continue
            residues.append(resi)
            aas.append(line[17:20].strip())
    return residues, aas


def write_chain_pdb(src: Path, chain: str, dst: Path) -> bool:
    wrote = False
    with src.open() as handle, dst.open("w") as out:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if chain and line[21] != chain:
                continue
            if line[16].strip() not in {"", "A"}:
                continue
            out.write(line[:16] + " " + line[17:])
            wrote = True
        out.write("END\n")
    if not wrote:
        dst.unlink(missing_ok=True)
    return wrote


def header_to_key(header: str) -> str:
    # FoldMason may emit "4TWPA", "4TWPA_A", or paths.  Our chain-specific
    # files are named CHAINKEY.pdb, so the first token is enough.
    stem = Path(header.split()[0]).stem.upper()
    stem = stem.replace(".PDB", "")
    parts = stem.split("_")
    if len(parts) >= 2 and len(parts[0]) == 5:
        return parts[0]
    return parts[0]


def residue_at_alignment_columns(seq: str, residues: list[int], aas: list[str]) -> dict[int, tuple[int, str]]:
    out: dict[int, tuple[int, str]] = {}
    ptr = 0
    for col, char in enumerate(seq):
        if char == "-":
            continue
        if ptr < len(residues):
            out[col] = (residues[ptr], aas[ptr] if ptr < len(aas) else "")
        ptr += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-csv", required=True, type=Path)
    ap.add_argument("--foldmason60-conserved", required=True, type=Path)
    ap.add_argument("--full-pdb-dir", required=True, type=Path)
    ap.add_argument("--ref-pdb", required=True, type=Path)
    ap.add_argument("--ref-chain", default="C")
    ap.add_argument("--ref-key", default="6UANC")
    ap.add_argument("--ref-dfg", type=int, default=594)
    ap.add_argument("--ref-ape", type=int, default=623)
    ap.add_argument("--foldmason", default="/home/edina/foldmason/bin/foldmason")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    structs = args.out / "chain_pdbs"
    chunks = args.out / "chunks"
    tmp_root = args.out / "tmp"
    structs.mkdir(exist_ok=True)
    chunks.mkdir(exist_ok=True)
    tmp_root.mkdir(exist_ok=True)

    manifest = pd.read_csv(args.manifest_csv, keep_default_na=False)
    manifest["chain_key"] = norm_key(manifest["chain_key"])
    manifest = manifest.drop_duplicates("chain_key").reset_index(drop=True)
    if args.limit:
        manifest = manifest.head(args.limit).copy()

    conserved = pd.read_csv(args.foldmason60_conserved, keep_default_na=False)
    conserved["chain_key"] = norm_key(conserved["chain_key"])
    ref_rows = conserved[conserved["chain_key"].eq(args.ref_key.upper())].copy()
    if ref_rows.empty:
        raise SystemExit(f"Reference {args.ref_key} missing from {args.foldmason60_conserved}")
    ref_rows = ref_rows[~ref_rows["pdb_resi"].astype(int).between(args.ref_dfg, args.ref_ape)].copy()
    braf_resis = sorted(ref_rows["pdb_resi"].astype(int).unique().tolist())
    print(f"BRAF conserved non-loop residues: {len(braf_resis)}")
    (args.out / "braf_conserved_nonloop_residues.txt").write_text(
        "\n".join(str(x) for x in braf_resis) + "\n"
    )

    ref_chain_pdb = structs / f"{args.ref_key.upper()}.pdb"
    if not ref_chain_pdb.exists():
        if not write_chain_pdb(args.ref_pdb, args.ref_chain, ref_chain_pdb):
            raise SystemExit("Could not write reference chain PDB")
    ref_residues, ref_aas = read_chain_residues(ref_chain_pdb, "")
    ref_resi_to_aa = {r: a for r, a in zip(ref_residues, ref_aas)}

    prepared = []
    missing = []
    for _, row in manifest.iterrows():
        key = str(row["chain_key"]).upper()
        pdb = str(row["pdb"]).upper()
        chain = str(row["chain"])
        src = args.full_pdb_dir / f"{pdb}.pdb"
        dst = structs / f"{key}.pdb"
        if not src.exists():
            missing.append({"chain_key": key, "status": "missing_pdb"})
            continue
        if not dst.exists() and not write_chain_pdb(src, chain, dst):
            missing.append({"chain_key": key, "status": "missing_chain"})
            continue
        residues, aas = read_chain_residues(dst, "")
        if len(residues) < 80:
            missing.append({"chain_key": key, "status": "too_few_ca", "n_ca": len(residues)})
            continue
        prepared.append({"chain_key": key, "path": dst, "residues": residues, "aas": aas})
    pd.DataFrame(missing).to_csv(args.out / "prepare_failures.csv", index=False)
    print(f"Prepared chain PDBs: {len(prepared)} / {len(manifest)}")
    print(f"Prepare failures: {len(missing)}")

    all_rows = []
    # Include the reference rows so downstream scripts can annotate BRAF residues.
    for resi in braf_resis:
        all_rows.append(
            {
                "chain_key": args.ref_key.upper(),
                "pdb_id": args.ref_key[:4].upper(),
                "chain": args.ref_chain,
                "msa_column": resi,
                "braf_resi": resi,
                "pdb_resi": resi,
                "aa": ref_resi_to_aa.get(resi, ""),
                "three_di": "",
                "mapping_source": "reference",
            }
        )

    n_chunks = math.ceil(len(prepared) / args.chunk_size)
    for chunk_idx in range(n_chunks):
        chunk_items = prepared[chunk_idx * args.chunk_size : (chunk_idx + 1) * args.chunk_size]
        chunk_dir = chunks / f"chunk_{chunk_idx:04d}"
        chunk_dir.mkdir(exist_ok=True)
        out_prefix = chunk_dir / "msa"
        aa_path = chunk_dir / "msa_aa.fa"
        log_path = chunk_dir / "foldmason.log"
        if not aa_path.exists():
            cmd = [
                args.foldmason,
                "easy-msa",
                "--report-mode",
                "0",
                "--threads",
                str(args.threads),
                "--filter-msa",
                "0",
                str(ref_chain_pdb),
                *[str(item["path"]) for item in chunk_items],
                str(out_prefix),
                str(tmp_root / f"chunk_{chunk_idx:04d}"),
            ]
            print(f"Running chunk {chunk_idx + 1}/{n_chunks}: {len(chunk_items)} chains")
            with log_path.open("w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        else:
            print(f"Skipping completed chunk {chunk_idx + 1}/{n_chunks}")

        msa = parse_fasta(aa_path)
        seq_by_key = {header_to_key(k): v for k, v in msa.items()}
        ref_seq = seq_by_key.get(args.ref_key.upper())
        if ref_seq is None:
            # Sometimes FoldMason appends the single chain ID.  Fall back to
            # any sequence beginning with the reference key.
            matches = [v for k, v in seq_by_key.items() if k.startswith(args.ref_key.upper())]
            ref_seq = matches[0] if matches else None
        if ref_seq is None:
            raise RuntimeError(f"Reference sequence missing in {aa_path}")
        ref_col_map = residue_at_alignment_columns(ref_seq, ref_residues, ref_aas)
        ref_resi_to_col = {resi: col for col, (resi, _) in ref_col_map.items()}

        for item in chunk_items:
            key = item["chain_key"]
            seq = seq_by_key.get(key)
            if seq is None:
                matches = [v for k, v in seq_by_key.items() if k.startswith(key)]
                seq = matches[0] if matches else None
            if seq is None:
                continue
            target_col_map = residue_at_alignment_columns(seq, item["residues"], item["aas"])
            for braf_resi in braf_resis:
                col = ref_resi_to_col.get(braf_resi)
                if col is None:
                    continue
                mapped = target_col_map.get(col)
                if mapped is None:
                    continue
                target_resi, target_aa = mapped
                all_rows.append(
                    {
                        "chain_key": key,
                        "pdb_id": key[:4],
                        "chain": key[4:],
                        "msa_column": braf_resi,
                        "braf_resi": braf_resi,
                        "pdb_resi": target_resi,
                        "aa": target_aa,
                        "three_di": "",
                        "mapping_source": f"foldmason_chunk_{chunk_idx:04d}",
                    }
                )
        if (chunk_idx + 1) % 5 == 0 or chunk_idx + 1 == n_chunks:
            pd.DataFrame(all_rows).to_csv(args.out / "v8_braf_mapped_conserved_residues.partial.csv", index=False)
            print(f"Rows so far: {len(all_rows)}")

    mapped = pd.DataFrame(all_rows)
    mapped.to_csv(args.out / "v8_braf_mapped_conserved_residues.csv", index=False)
    summary = (
        mapped[mapped["chain_key"].ne(args.ref_key.upper())]
        .groupby("chain_key")
        .agg(n_mapped_residues=("braf_resi", "nunique"))
        .reset_index()
    )
    summary.to_csv(args.out / "mapping_coverage_by_chain.csv", index=False)
    print(f"Mapped chains: {len(summary)}")
    print(summary["n_mapped_residues"].describe().to_string())


if __name__ == "__main__":
    main()
