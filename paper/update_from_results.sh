#!/usr/bin/env bash
# Copy the generated tables and figures from results/ into paper/ (run from repo root).
set -e
mkdir -p paper/tables paper/figures
cp results/tables/*.tex paper/tables/
cp results/figures/*.pdf paper/figures/
echo "paper/ updated from results/"
