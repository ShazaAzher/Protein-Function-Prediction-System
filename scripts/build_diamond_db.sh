#!/usr/bin/env bash
# Task 1.3 — builds the Diamond reference database from train_sequences.fasta.
set -euo pipefail

TRAIN_FASTA="${1:-data/raw/train_sequences.fasta}"
DB_PATH="${2:-data/processed/train_diamond.dmnd}"

echo "TODO: diamond makedb --in ${TRAIN_FASTA} -d ${DB_PATH}"
