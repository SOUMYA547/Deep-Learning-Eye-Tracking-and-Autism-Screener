import argparse
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFilter
import config

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(config.ROOT / "data" / "carette"))
args = ap.parse_args()
src = Path(args.src)
imgs = next((p for p in [src / "Images", src] if (p / "TSImages").exists()), None)
meta = next((p for p in [src / "Metadata", src] if (p / "Metadata_TS.csv").exists()), None)
if imgs is None or meta is None:
    raise SystemExit(f"Cannot find Images/TSImages and Metadata/Metadata_TS.csv under {src}")

parts = pd.read_csv(meta / "Metadata_Participants.csv")
cls = dict(zip(parts["ParticipantID"].astype(str), parts["Class"]))
for c in ("ASD", "TD"):
    shutil.rmtree(config.IMG_DIR / c, ignore_errors=True)
    (config.IMG_DIR / c).mkdir(parents=True, exist_ok=True)


def prep(path):
    """Images are dark, thin lines on black. Brighten, thicken, resize so lines survive down-sizing."""
    a = np.asarray(Image.open(path).convert("RGB"), dtype="float32")
    hi = np.percentile(a[a > 0], 99.5) if (a > 0).any() else 255.0
    a = np.clip(a / max(hi, 1.0) * 255.0, 0, 255).astype("uint8")
    im = Image.fromarray(a).filter(ImageFilter.MaxFilter(5))
    return im.resize((config.IMG_SIZE, config.IMG_SIZE), Image.BILINEAR)


rows = []
for folder, group, label in (("TSImages", "TS", "ASD"), ("TCImages", "TC", "TD")):
    m = pd.read_csv(meta / f"Metadata_{group}.csv")
    for fn, pid in zip(m["Filename"], m["ParticipantID"].astype(str)):
        p = imgs / folder / fn
        if not p.exists():
            continue
        if cls.get(pid, group) != group:
            print("label mismatch for participant", pid)
        out = config.IMG_DIR / label / f"{label}__{pid}__{Path(fn).stem}.png"
        prep(p).save(out)
        rows.append(dict(path=str(out), label=label, participant=pid, source_file=fn, n_points=0))

man = pd.DataFrame(rows)
man.to_csv(config.IMG_DIR / "manifest.csv", index=False)
print("=== SUMMARY ===")
print("images:", len(man), "| children:", man.participant.nunique())
print(man.groupby("label").agg(images=("path", "count"), children=("participant", "nunique")))
print("Next:  python src/train_eye_cnn.py   (add --weights none if pretrained weights cannot download)")
