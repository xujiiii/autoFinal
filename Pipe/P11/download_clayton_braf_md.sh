#!/usr/bin/env bash
# Download the Clayton, Romany & Shen (2025) BRAF MD trajectories from
# Zenodo deposit 14611113.
#
#   AMBER trajectories of dimeric or monomeric BRAF V600E, both apo and
#   in complex with LY3009120 or PHI1.  5 microseconds per system.
#   ~9.9 GB total, CC-BY 4.0.
#
# Usage on coulomb:
#   bash download_clayton_braf_md.sh  /data/student/yuxiz/braf_clayton_md
#
# The target directory will hold:
#   ./prmtop/*.prmtop          (9 topology files, ~30 MB total)
#   ./trajectories.zip          (~9.9 GB)
#   ./trajectories/...          (unzipped trajectories, after step 2)

set -euo pipefail

DEST="${1:-/data/student/yuxiz/braf_clayton_md}"
BASE="https://zenodo.org/api/records/14611113/files"

mkdir -p "${DEST}/prmtop"
cd "${DEST}"

echo "==> Downloading 9 prmtop topology files (small)"
for fname in \
    BrafMonomer_apo.prmtop \
    BrafMonomer_LY.prmtop \
    BrafMonomer_PHI1_chainA.prmtop \
    BrafMonomer_PHI1_chainB.prmtop \
    BrafDimer_apo.prmtop \
    BrafDimer_LY.prmtop \
    BrafDimer_PHI1.prmtop \
    BrafMixedDimer_LY.prmtop \
    BrafMixedDimer_PHI1.prmtop ; do
  if [[ -s "prmtop/${fname}" ]]; then
    echo "  [skip] prmtop/${fname} already present"
  else
    echo "  -> ${fname}"
    wget --no-verbose --show-progress \
         -O "prmtop/${fname}" \
         "${BASE}/${fname}/content"
  fi
done

echo
echo "==> Downloading trajectories.zip (~9.9 GB; may take 20-60 min)"
if [[ -s trajectories.zip ]]; then
  echo "  [skip] trajectories.zip already present (size $(du -h trajectories.zip | cut -f1))"
else
  wget --no-verbose --show-progress \
       -O trajectories.zip \
       "${BASE}/trajectories.zip/content"
fi

echo
echo "==> Unzipping trajectories"
if [[ -d trajectories && -n "$(ls -A trajectories 2>/dev/null)" ]]; then
  echo "  [skip] trajectories/ already populated"
else
  mkdir -p trajectories
  cd trajectories
  unzip -q ../trajectories.zip
  cd ..
fi

echo
echo "==> Summary"
echo "Topology files:"
ls -lh prmtop
echo
echo "Trajectory files (top level):"
ls -lh trajectories | head -50
echo
echo "Total disk usage:"
du -sh "${DEST}"
