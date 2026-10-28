#!/bin/bash
# Regenerate every demo of demo/README.md from the stored embeddings (no text encoder needed).
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT=${CKPT:-weights/topx.pt}
OUT=${OUT:-out/demos}
python scripts/generate.py --ckpt "$CKPT" --skeleton demo/bat/skeleton.npz --keep_rest_rotations --text_emb demo/bat/text_emb_1.npy --joint_sem demo/bat/joint_sem.npy --frames 60 --seed 7 --out "$OUT/bat" --name ref_1 --force
python scripts/generate.py --ckpt "$CKPT" --skeleton demo/bear/skeleton.npz --keep_rest_rotations --text_emb demo/bear/text_emb_1.npy --joint_sem demo/bear/joint_sem.npy --frames 200 --seed 17 --out "$OUT/bear" --name ref_1 --force
python scripts/generate.py --ckpt "$CKPT" --skeleton demo/crawling_human/skeleton.npz --keep_rest_rotations --text_emb demo/crawling_human/text_emb_1.npy --joint_sem demo/crawling_human/joint_sem.npy --frames 149 --seed 17 --out "$OUT/crawling_human" --name ref_1 --force
python scripts/generate.py --ckpt "$CKPT" --skeleton demo/mech_striker/skeleton.npz --keep_rest_rotations --text_emb demo/mech_striker/text_emb_1.npy --joint_sem demo/mech_striker/joint_sem.npy --frames 31 --seed 17 --out "$OUT/mech_striker" --name ref_1 --force
python scripts/generate.py --ckpt "$CKPT" --skeleton demo/mech_striker/skeleton.npz --keep_rest_rotations --text_emb demo/mech_striker/text_emb_2.npy --joint_sem demo/mech_striker/joint_sem.npy --frames 26 --seed 17 --out "$OUT/mech_striker" --name ref_2 --force
python scripts/generate.py --ckpt "$CKPT" --skeleton demo/spider/skeleton.npz --keep_rest_rotations --text_emb demo/spider/text_emb_1.npy --joint_sem demo/spider/joint_sem.npy --frames 31 --seed 17 --out "$OUT/spider" --name ref_1 --force
python scripts/generate.py --ckpt "$CKPT" --skeleton demo/stag/skeleton.npz --keep_rest_rotations --text_emb demo/stag/text_emb_1.npy --joint_sem demo/stag/joint_sem.npy --frames 92 --seed 7 --out "$OUT/stag" --name ref_1 --force
python scripts/generate.py --ckpt "$CKPT" --skeleton demo/triceratops/skeleton.npz --keep_rest_rotations --text_emb demo/triceratops/text_emb_1.npy --joint_sem demo/triceratops/joint_sem.npy --frames 61 --seed 7 --out "$OUT/triceratops" --name ref_1 --force
python scripts/generate.py --ckpt "$CKPT" --skeleton demo/triceratops/skeleton.npz --keep_rest_rotations --text_emb demo/triceratops/text_emb_2.npy --joint_sem demo/triceratops/joint_sem.npy --frames 130 --seed 7 --out "$OUT/triceratops" --name ref_2 --force
python scripts/generate.py --ckpt "$CKPT" --skeleton demo/viking_worker/skeleton.npz --keep_rest_rotations --text_emb demo/viking_worker/text_emb_1.npy --joint_sem demo/viking_worker/joint_sem.npy --frames 68 --seed 17 --out "$OUT/viking_worker" --name ref_1 --force
python scripts/generate.py --ckpt "$CKPT" --skeleton demo/viking_worker/skeleton.npz --keep_rest_rotations --text_emb demo/viking_worker/text_emb_2.npy --joint_sem demo/viking_worker/joint_sem.npy --frames 34 --seed 17 --out "$OUT/viking_worker" --name ref_2 --force
python scripts/generate.py --ckpt "$CKPT" --skeleton demo/wolf_lowpoly/skeleton.npz --keep_rest_rotations --text_emb demo/wolf_lowpoly/text_emb_1.npy --joint_sem demo/wolf_lowpoly/joint_sem.npy --frames 113 --seed 7 --out "$OUT/wolf_lowpoly" --name ref_1 --force
