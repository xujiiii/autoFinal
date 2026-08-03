"""Join OncoKB static TSVs to the v9 mutation validation skeleton.

Reads the validation skeleton (one row per gene-mutation pair with
significance metrics) and the OncoKB downloadable TSVs:

  - allActionableVariants.tsv     (the "Actionable Genes" download)
  - allCuratedGenes.tsv           (the "Cancer Genes" download)
  - allCuratedVariants.tsv        (the "Curated Genes / Variants" download)

These files don't need API access; you can grab them as a logged-in
user from:
  https://www.oncokb.org/actionable-genes        → "Download as TSV"
  https://www.oncokb.org/cancer-genes            → "Download as TSV"
  https://www.oncokb.org/curated-genes           → "Download as TSV"

Drop them as-is into ``manuscript_draft/data/oncokb_static/`` and run
this script. It fills the OncoKB placeholder columns of the skeleton:
  oncokb_oncogenic, oncokb_highest_sensitive_level,
  oncokb_highest_resistance_level, oncokb_drug_context,
  oncokb_lookup_status (now = "found_static" / "no_record_static")

Joins by the ``Gene`` × ``Alteration`` columns OncoKB uses. The token
format in the static tables is the standard "V600E" form, so this
matches our skeleton's ``mutation`` column directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_oncokb_static(static_dir: Path) -> pd.DataFrame:
    """Combine Actionable + Curated variants into a single per-(gene, alteration) lookup."""
    frames = []

    actionable = next(static_dir.glob("*ctionable*ariants*.tsv"), None)
    if actionable is None:
        actionable = next(static_dir.glob("*ctionable*.tsv"), None)
    if actionable:
        df = pd.read_csv(actionable, sep="\t", keep_default_na=False)
        df["source_table"] = "actionable"
        frames.append(df)
        print(f"  loaded {actionable.name}: {len(df)} rows, columns={list(df.columns)[:10]}")

    curated_variants = next(static_dir.glob("*urated*ariants*.tsv"), None)
    if curated_variants is None:
        curated_variants = next(static_dir.glob("*ariant*.tsv"), None)
    if curated_variants and (actionable is None or curated_variants != actionable):
        df = pd.read_csv(curated_variants, sep="\t", keep_default_na=False)
        df["source_table"] = "curated_variants"
        frames.append(df)
        print(f"  loaded {curated_variants.name}: {len(df)} rows")

    if not frames:
        raise SystemExit(f"No OncoKB *.tsv files found in {static_dir}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"Combined OncoKB rows: {len(combined)}")
    print(f"Columns: {list(combined.columns)}")
    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeleton-csv", required=True, type=Path)
    ap.add_argument("--oncokb-static-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    sk = pd.read_csv(args.skeleton_csv, keep_default_na=False)
    print(f"Skeleton: {len(sk)} rows")

    onc = load_oncokb_static(args.oncokb_static_dir)
    # Normalise OncoKB column names. The downloadable tables expose
    # different schemas; we accept any of these synonyms.
    col_map = {}
    for src, dst in [
        ("Hugo Symbol", "gene"), ("Gene", "gene"), ("HUGO_SYMBOL", "gene"),
        ("Alteration", "mutation"), ("Alterations", "mutation"), ("Protein Change", "mutation"),
        ("MUTATION", "mutation"), ("Variant", "mutation"),
        ("Oncogenic", "oncokb_oncogenic"),
        ("Highest Sensitive Level", "oncokb_highest_sensitive_level"),
        ("Highest Resistance Level", "oncokb_highest_resistance_level"),
        ("Level", "oncokb_level"),
        ("Drugs", "oncokb_drug_context"),
        ("Drugs (for therapeutic implications only)", "oncokb_drug_context"),
        ("Drugs(s)", "oncokb_drug_context"),
        ("Cancer Type", "oncokb_cancer_type"), ("Cancer Types", "oncokb_cancer_type"),
        ("Tumor Type", "oncokb_cancer_type"),
    ]:
        if src in onc.columns and dst not in col_map.values():
            col_map[src] = dst
    onc = onc.rename(columns=col_map)
    print(f"After rename, OncoKB columns relevant: "
          f"{[c for c in onc.columns if c.startswith('oncokb_') or c in ('gene','mutation')]}")

    # Aggregate per (gene, mutation): collapse multiple rows by joining
    # cancer types / drugs and picking the most-informative oncogenic call.
    grp_keys = ["gene", "mutation"]
    if "gene" not in onc.columns or "mutation" not in onc.columns:
        raise SystemExit("OncoKB tables don't have recognisable gene+mutation columns")
    onc["gene"] = onc["gene"].astype(str).str.upper()
    onc["mutation"] = onc["mutation"].astype(str).str.upper()
    onc["mutation"] = onc["mutation"].str.split(r"\s*,\s*")
    onc = onc.explode("mutation")

    def best_oncogenic(series):
        # OncoKB uses 'Oncogenic', 'Likely Oncogenic', 'Predicted Oncogenic',
        # 'Resistance', 'Likely Neutral', 'Inconclusive', 'Unknown'
        priorities = ["Oncogenic", "Likely Oncogenic", "Predicted Oncogenic",
                      "Resistance", "Likely Neutral", "Inconclusive", "Unknown"]
        for p in priorities:
            for v in series:
                if isinstance(v, str) and v.strip().lower() == p.lower():
                    return p
        return ""

    agg = {}
    for col in ("oncokb_oncogenic",
                "oncokb_highest_sensitive_level",
                "oncokb_highest_resistance_level",
                "oncokb_level"):
        if col in onc.columns:
            agg[col] = (best_oncogenic if col == "oncokb_oncogenic"
                        else lambda s: ";".join(sorted({str(x) for x in s if str(x).strip()})))
    for col in ("oncokb_drug_context", "oncokb_cancer_type"):
        if col in onc.columns:
            agg[col] = lambda s: ";".join(sorted({str(x) for x in s if str(x).strip()}))[:500]
    if not agg:
        raise SystemExit("No OncoKB classification columns found to aggregate.")

    summarised = onc.groupby(grp_keys, as_index=False).agg(agg)
    print(f"Summarised OncoKB per (gene, mutation): {len(summarised)} unique pairs")

    sk["gene"] = sk["gene"].astype(str).str.upper()
    sk["mutation"] = sk["mutation"].astype(str).str.upper()

    # Drop pre-existing placeholder columns we are about to fill.
    for col in summarised.columns:
        if col in ("gene", "mutation"): continue
        if col in sk.columns:
            sk = sk.drop(columns=[col])

    joined = sk.merge(summarised, on=grp_keys, how="left")

    def has_value(value) -> bool:
        return not pd.isna(value) and bool(str(value).strip())

    joined["oncokb_lookup_status"] = joined.apply(
        lambda r: "found_static" if (
            has_value(r.get("oncokb_oncogenic"))
            or has_value(r.get("oncokb_highest_sensitive_level"))
            or has_value(r.get("oncokb_level"))
            or has_value(r.get("oncokb_drug_context"))
        ) else "no_record_static",
        axis=1,
    )
    joined.to_csv(args.out, index=False)
    n_found = (joined["oncokb_lookup_status"] == "found_static").sum()
    print(f"\nWrote {args.out}  ({len(joined)} rows, {n_found} with OncoKB record)")

    # Validation breakdown.
    if "significant_perm_p<0.05" in joined.columns:
        sig = joined[joined["significant_perm_p<0.05"] == True]
        sig_in_oncokb = sig[sig["oncokb_lookup_status"] == "found_static"]
        print(f"\nSignificant (perm p<0.05): {len(sig)}")
        print(f"  of which in OncoKB:                {len(sig_in_oncokb)}")
        if len(sig_in_oncokb):
            print(f"  by oncogenic classification:")
            print(sig_in_oncokb["oncokb_oncogenic"].value_counts().to_string())


if __name__ == "__main__":
    main()
