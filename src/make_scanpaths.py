"""STEP 2 - turn raw gaze CSVs into scanpath images, one image per (file, child).
Run:  python src/make_scanpaths.py
"""
import shutil
import pandas as pd
import config
from eye_utils import (read_table, to_num, detect_participant_col, detect_xy_cols,
                       detect_time_col, normalize_xy, render_scanpath)
from labels import load_label_map, label_for, find_meta_file

meta = find_meta_file()
files = [f for f in sorted(config.RAW_DIR.glob("*.csv")) if f != meta]
if not files:
    raise SystemExit(f"No raw CSV files in {config.RAW_DIR}")
label_map, _, _, _ = load_label_map()
if label_map is None:
    print("WARNING: no metadata labels found; will try to read ASD/TD from participant names.")

for cls in ("ASD", "TD"):
    shutil.rmtree(config.IMG_DIR / cls, ignore_errors=True)
    (config.IMG_DIR / cls).mkdir(parents=True, exist_ok=True)

rows, unlabeled, short = [], set(), 0
for fi, f in enumerate(files, 1):
    df = read_table(f)
    pc, (xc, yc), tc = detect_participant_col(df), detect_xy_cols(df), detect_time_col(df)
    if not (pc and xc and yc):
        print(f"[skip] {f.name}: cannot find participant/x/y columns (columns: {list(df.columns)[:8]}...)")
        continue
    df["_x"], df["_y"] = to_num(df[xc]), to_num(df[yc])
    if tc:
        df["_t"] = to_num(df[tc])
    n_img = 0
    for pid, g in df.groupby(pc):
        lab = label_for(pid, label_map)
        if lab is None:
            unlabeled.add(str(pid))
            continue
        if tc:
            g = g.sort_values("_t")
        nx, ny = normalize_xy(g["_x"].values, g["_y"].values)
        if len(nx) < config.MIN_POINTS:
            short += 1
            continue
        img = render_scanpath(nx, ny)
        safe_pid = "".join(ch if ch.isalnum() else "-" for ch in str(pid))
        out = config.IMG_DIR / lab / f"{lab}__{safe_pid}__{f.stem}.png"
        img.save(out)
        rows.append(dict(path=str(out), label=lab, participant=str(pid), source_file=f.name, n_points=len(nx)))
        n_img += 1
    print(f"[{fi}/{len(files)}] {f.name}: {n_img} images")

if not rows:
    raise SystemExit("No images created. Run inspect_data.py and check labels / columns.")
man = pd.DataFrame(rows)
man.to_csv(config.IMG_DIR / "manifest.csv", index=False)
print("\n=== SUMMARY ===")
print("images:", len(man), "| children:", man.participant.nunique())
print(man.groupby("label").agg(images=("path", "count"), children=("participant", "nunique")))
if unlabeled:
    print(f"WARNING: {len(unlabeled)} participant ids had no label and were skipped, e.g. {sorted(unlabeled)[:5]}")
if short:
    print(f"skipped {short} recordings with < {config.MIN_POINTS} gaze points")
print("Saved:", config.IMG_DIR / "manifest.csv")
print("Next:  python src/train_eye_cnn.py")
