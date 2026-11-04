"""STEP 1 - look at the raw dataset before doing anything else.
Run:  python src/inspect_data.py
"""
import config
from eye_utils import (read_table, to_num, detect_participant_col, detect_xy_cols,
                       detect_time_col, normalize_xy)
from labels import load_label_map, find_meta_file

files = sorted(config.RAW_DIR.glob("*.csv"))
if not files:
    raise SystemExit(f"No CSV files in {config.RAW_DIR}\nCopy the dataset CSVs there first.")

meta = find_meta_file()
print(f"CSV files found: {len(files)}")
print(f"Metadata file  : {meta.name if meta else 'NOT FOUND'}")
label_map, _, id_col, lab_col = load_label_map()
if label_map:
    n_asd = sum(v == "ASD" for v in label_map.values())
    print(f"Metadata cols  : id='{id_col}', label='{lab_col}'")
    print(f"Participants   : {len(label_map)}  (ASD={n_asd}, TD={len(label_map) - n_asd})")
else:
    print(f"Could not read labels from metadata (id col={id_col}, label col={lab_col}).")
    print("-> set META_FILE / META_ID_COL / META_LABEL_COL in src/config.py")

raw = [f for f in files if f != meta]
f0 = raw[0]
df = read_table(f0)
print(f"\n--- first raw file: {f0.name}  shape={df.shape}")
print("columns:", list(df.columns))
pc, (xc, yc), tc = detect_participant_col(df), detect_xy_cols(df), detect_time_col(df)
print(f"detected -> participant='{pc}'  x='{xc}'  y='{yc}'  time='{tc}'")
if pc and xc:
    print("participants in this file:", df[pc].nunique(), "| examples:", list(df[pc].dropna().unique()[:5]))
    nx, ny = normalize_xy(to_num(df[xc]).values, to_num(df[yc]).values)
    print(f"x range (raw): {to_num(df[xc]).min():.1f} .. {to_num(df[xc]).max():.1f}   "
          f"y range (raw): {to_num(df[yc]).min():.1f} .. {to_num(df[yc]).max():.1f}")
    print("-> if raw x goes far above 1920 or y above 1080, edit SCREEN_W / SCREEN_H in src/config.py")
else:
    print("-> auto-detect failed. Set COL_PARTICIPANT / COL_X / COL_Y in src/config.py")
print("\nIf everything above looks right, run:  python src/make_scanpaths.py")
