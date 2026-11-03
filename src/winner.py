"""Tahap berikutnya (ablasi, tuning final) WAJIB membaca pemenang dari sini, bukan memilih manual."""
import json


def load_winner(path="results/baseline_winner.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
