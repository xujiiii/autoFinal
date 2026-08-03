#!/bin/bash
set -e


if [ -z "$1" ]; then
    echo "Must give the path to download foldmason！"
    exit 1
fi

TARGET_DIR="$1"
mkdir -p "$TARGET_DIR"

ARCH=$(uname -m)
DOWNLOAD_URL=""

if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    DOWNLOAD_URL="https://mmseqs.com/foldmason/foldmason-linux-arm64.tar.gz"
elif [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "amd64" ]; then
    if grep -q -i "avx2" /proc/cpuinfo 2>/dev/null; then
        DOWNLOAD_URL="https://mmseqs.com/foldmason/foldmason-linux-avx2.tar.gz"
    else
        DOWNLOAD_URL="https://mmseqs.com/foldmason/foldmason-linux-sse2.tar.gz"
    fi
else
    DOWNLOAD_URL="https://mmseqs.com/foldmason/foldmason-linux-arm64.tar.gz"
fi

if command -v curl &> /dev/null; then
    curl -sL "$DOWNLOAD_URL" | tar xvzf - -C "$TARGET_DIR"
elif command -v wget &> /dev/null; then
    wget -qO- "$DOWNLOAD_URL" | tar xvzf - -C "$TARGET_DIR"
else
    echo "Error in download"
    exit 1
fi

echo "Successfully download in $TARGET_DIR"