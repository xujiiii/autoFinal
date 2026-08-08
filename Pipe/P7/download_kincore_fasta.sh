#!/bin/bash

DEST_DIR=""

while getopts "o:" opt; do
  case $opt in
    o) DEST_DIR="$OPTARG" ;;
    *) echo "用法: $0 -o /path/to/dir" >&2; exit 1 ;;
  esac
done

if [ -z "$DEST_DIR" ]; then
  echo "Error: Please use -o to set the download folder" >&2
  exit 1
fi

wget -P "$DEST_DIR" https://dunbrack.fccc.edu/kincore/static/downloads/fasta_with_labels/PK_labels_PDB.fasta