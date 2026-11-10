"""Quick self-test.  Run from project folder:  python tests/check_pipeline.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
import config

results = []


def check(name, cond, extra=""):
    results.append(bool(cond))
    print(("PASS" if cond else "FAIL"), "-", name, extra)


mf = config.IMG_DIR / "manifest.csv"
check("manifest.csv exists (run prepare_carette.py first)", mf.exists())
if mf.exists():
    man = pd.read_csv(mf)
    check("both classes present", set(man.label) == {"ASD", "TD"}, str(man.label.value_counts().to_dict()))
    check("all image files exist", all(Path(p).exists() for p in man.path))
    per = man.groupby("participant").label.nunique()
    check("no child has two different labels", (per == 1).all())
    print("      images:", len(man), "| children:", man.participant.nunique(),
          "| ASD children:", man[man.label == "ASD"].participant.nunique(),
          "| TD children:", man[man.label == "TD"].participant.nunique())

    y = (man.label == "ASD").astype(int).values
    g = man.participant.astype(str).values
    leak = False
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=config.SEED).split(man, y, g):
        leak |= bool(set(g[tr]) & set(g[te]))
    check("no child appears in both train and test (5 folds)", not leak)

cnn_p, tab_p = config.MODEL_DIR / "asd_eye_cnn.keras", config.MODEL_DIR / "questionnaire_model.joblib"
check("CNN model file exists", cnn_p.exists())
check("questionnaire model file exists", tab_p.exists())

if cnn_p.exists() and mf.exists():
    from tensorflow import keras
    from eye_utils import pil_to_array, predict_image, gradcam
    model = keras.models.load_model(cnn_p)
    for lab in ("ASD", "TD"):
        f = man[man.label == lab].path.iloc[0]
        a = pil_to_array(Image.open(f))
        p = predict_image(model, a)
        check(f"CNN gives probability for a {lab} image", 0 <= p <= 1, f"p={p:.3f}")
    check("Grad-CAM returns a heatmap", gradcam(model, a).ndim == 2)

if tab_p.exists():
    t = joblib.load(tab_p)

    def prob(n_ones):
        row = {f"A{i + 1}_Score": int(i < n_ones) for i in range(10)}
        row.update(age=25, gender=1, jundice=0, austim=0)
        return float(t["model"].predict_proba(pd.DataFrame([row])[t["features"]])[0, 1])

    p0, p9 = prob(0), prob(9)
    check("0 positive answers -> low risk (<0.5)", p0 < 0.5, f"p={p0:.2f}")
    check("9 positive answers -> high risk (>0.5)", p9 > 0.5, f"p={p9:.2f}")

mp = config.OUT_DIR / "eye_cnn_metrics.json"
if mp.exists():
    m = json.load(open(mp))
    acc = m["child_level"]["Accuracy"]
    check("child accuracy not suspiciously high (<0.97 = no leakage sign)", acc < 0.97, f"acc={acc:.3f}")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)