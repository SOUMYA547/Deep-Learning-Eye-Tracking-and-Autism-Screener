"""STEP 3 - train the eye-tracking CNN. Split is by CHILD, never by image.

  python src/train_eye_cnn.py              # one child-level split, saves the model for the app
  python src/train_eye_cnn.py --mode cv    # 5-fold child-level cross-validation (slow, for the report)
Options: --epochs-head 6 --epochs-ft 6 --weights imagenet|none
"""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)
import config
from eye_utils import gradcam, overlay

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["holdout", "cv"], default="holdout")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--epochs-head", type=int, default=6)
ap.add_argument("--epochs-ft", type=int, default=6)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--weights", default="imagenet", help="imagenet or none")
ap.add_argument("--arch", choices=["mobilenet", "small"], default="mobilenet",
                help="small = light CNN trained from scratch (use when no pretrained weights / slow CPU)")
args = ap.parse_args()
WEIGHTS = None if args.weights.lower() == "none" else "imagenet"
tf.keras.utils.set_random_seed(config.SEED)

# ---------------------------------------------------------------- data
man = pd.read_csv(config.IMG_DIR / "manifest.csv")
y = (man.label == "ASD").astype(int).values
groups = man.participant.astype(str).values
X = np.stack([np.asarray(Image.open(p).convert("RGB").resize((config.IMG_SIZE,) * 2), dtype="uint8")
              for p in man.path])
print(f"images={len(X)}  children={len(set(groups))}  ASD images={y.sum()}  TD images={(y == 0).sum()}")
if len(set(groups)) < 10:
    print("WARNING: very few children; results will be unstable.")


def build_model():
    aug = keras.Sequential([layers.RandomTranslation(0.05, 0.05), layers.RandomZoom(0.1)], name="aug")
    shape = (config.IMG_SIZE, config.IMG_SIZE, 3)
    if args.arch == "small":
        bi = keras.Input(shape)
        b = bi
        for f in (16, 32, 64, 96):
            b = layers.Conv2D(f, 3, padding="same", use_bias=False)(b)
            b = layers.BatchNormalization()(b)
            b = layers.ReLU()(b)
            b = layers.MaxPool2D()(b)
        base = keras.Model(bi, layers.Conv2D(96, 3, padding="same", activation="relu")(b), name="backbone")
    else:
        base = keras.applications.MobileNetV2(input_shape=shape, include_top=False, weights=WEIGHTS, name="backbone")
    base.trainable = False
    head = keras.Sequential([layers.GlobalAveragePooling2D(), layers.Dropout(0.3),
                             layers.Dense(1, activation="sigmoid")], name="head")
    inp = keras.Input(shape)
    h = aug(inp)
    h = layers.Rescaling(1 / 127.5, offset=-1, name="rescale")(h)   # inputs scaled to [-1, 1]
    h = base(h, training=False)
    model = keras.Model(inp, head(h), name="asd_eye_cnn")
    return model, base


def train_model(tr, va):
    model, base = build_model()
    cnt = np.bincount(y[tr], minlength=2)
    cw = {0: len(tr) / (2 * max(cnt[0], 1)), 1: len(tr) / (2 * max(cnt[1], 1))}
    mets = ["accuracy", keras.metrics.AUC(name="auc")]
    cb = [keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True)]
    fit = dict(validation_data=(X[va], y[va]), batch_size=args.batch, class_weight=cw, callbacks=cb, verbose=2)
    if WEIGHTS is None or args.arch == "small":   # no pretrained weights: train everything from the start
        base.trainable = True
        model.compile(keras.optimizers.Adam(1e-3), "binary_crossentropy", metrics=mets)
        h = model.fit(X[tr], y[tr], epochs=args.epochs_head + args.epochs_ft, **fit)
        thr = best_threshold(y[va], model.predict(X[va], verbose=0).ravel())
        return model, {k: h.history[k] for k in ("accuracy", "val_accuracy")}, thr
    model.compile(keras.optimizers.Adam(1e-3), "binary_crossentropy", metrics=mets)
    h1 = model.fit(X[tr], y[tr], epochs=args.epochs_head, **fit)
    base.trainable = True
    for l in base.layers[:-30]:
        l.trainable = False
    model.compile(keras.optimizers.Adam(1e-5), "binary_crossentropy", metrics=mets)
    h2 = model.fit(X[tr], y[tr], epochs=args.epochs_ft, **fit)
    thr = best_threshold(y[va], model.predict(X[va], verbose=0).ravel())
    return model, {k: h1.history[k] + h2.history[k] for k in ("accuracy", "val_accuracy")}, thr


def best_threshold(yt, prob):
    """Cut-off that maximises balanced accuracy on VALIDATION children (never on test)."""
    grid = np.unique(np.round(prob, 4))
    if len(set(yt)) < 2 or len(grid) < 2:
        return 0.5
    scores = [(recall_score(yt, prob >= t) + recall_score(1 - yt, prob < t)) / 2 for t in grid]
    return float(grid[int(np.argmax(scores))])


def score(yt, prob, thr=0.5):
    yp = (prob >= thr).astype(int)
    return dict(Accuracy=accuracy_score(yt, yp), Precision=precision_score(yt, yp, zero_division=0),
                Recall=recall_score(yt, yp), F1=f1_score(yt, yp, zero_division=0),
                ROC_AUC=roc_auc_score(yt, prob) if len(set(yt)) > 1 else float("nan"))


def evaluate(model, te, thr):
    prob = model.predict(X[te], verbose=0).ravel()
    img_m = score(y[te], prob, thr)
    df = pd.DataFrame(dict(g=groups[te], y=y[te], p=prob)).groupby("g").agg(y=("y", "first"), p=("p", "mean"))
    child_m = score(df.y.values, df.p.values, thr)           # one vote per child (mean of its images)
    return prob, img_m, child_m, len(df)


def split_groups(idx_pool, n_splits, seed):
    sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return next(sgk.split(idx_pool, y[idx_pool], groups[idx_pool]))


# ---------------------------------------------------------------- CV mode
if args.mode == "cv":
    rows = []
    sgk = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=config.SEED)
    for k, (tr_all, te) in enumerate(sgk.split(X, y, groups), 1):
        a, b = split_groups(tr_all, 4, config.SEED)
        tr, va = tr_all[a], tr_all[b]
        print(f"\n===== fold {k}/{args.folds}: train={len(tr)} val={len(va)} test={len(te)} =====")
        model, _, thr = train_model(tr, va)
        _, im, ch, nch = evaluate(model, te, thr)
        rows.append({"fold": k, "threshold": thr, **{f"img_{m}": v for m, v in im.items()}, **{f"child_{m}": v for m, v in ch.items()}})
        keras.backend.clear_session()
    res = pd.DataFrame(rows)
    res.to_csv(config.OUT_DIR / "cv_results.csv", index=False)
    print("\n=== CROSS-VALIDATION (mean +- std over folds) ===")
    for c in res.columns[1:]:
        print(f"{c:18s} {res[c].mean():.3f} +- {res[c].std():.3f}")
    print("Saved:", config.OUT_DIR / "cv_results.csv")
    raise SystemExit

# ------------------------------------------------------------ holdout mode
all_idx = np.arange(len(X))
a, te = split_groups(all_idx, 5, config.SEED)               # ~20% children -> test
tr_all = all_idx[a]
a2, b2 = split_groups(tr_all, 4, config.SEED)               # ~20% of rest -> validation
tr, va = tr_all[a2], tr_all[b2]
assert not (set(groups[tr]) & set(groups[te])) and not (set(groups[va]) & set(groups[te])), "child leakage!"
print(f"train={len(tr)} val={len(va)} test={len(te)} images | "
      f"children train/val/test = {len(set(groups[tr]))}/{len(set(groups[va]))}/{len(set(groups[te]))}")
if len(set(y[te])) < 2:
    raise SystemExit("Test split has only one class - not enough children.")

model, hist, thr = train_model(tr, va)
prob, img_m, child_m, nch = evaluate(model, te, thr)
print(f"decision threshold chosen on validation children: {thr:.3f}")
print("\n=== TEST RESULTS (children never seen in training, validation-tuned threshold) ===")
print("per-image :", {k: round(float(v), 3) for k, v in img_m.items()})
print(f"per-child ({nch} children):", {k: round(float(v), 3) for k, v in child_m.items()})
json.dump({"threshold": thr, "image_level": img_m, "child_level": child_m, "n_test_images": int(len(te)), "n_test_children": int(nch)},
          open(config.OUT_DIR / "eye_cnn_metrics.json", "w"), indent=2, default=float)

# figures
cm = confusion_matrix(y[te], (prob >= thr).astype(int))
fpr, tpr, _ = roc_curve(y[te], prob)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].imshow(cm, cmap="Blues"); ax[0].set_title("Confusion matrix (0=TD, 1=ASD)")
ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Actual")
for (i, j), v in np.ndenumerate(cm):
    ax[0].text(j, i, int(v), ha="center", va="center", color="red", fontsize=14)
ax[1].plot(fpr, tpr); ax[1].plot([0, 1], [0, 1], "--"); ax[1].set_title(f"ROC (AUC={img_m['ROC_AUC']:.3f})")
ax[2].plot(hist["accuracy"], label="train"); ax[2].plot(hist["val_accuracy"], label="validation")
ax[2].legend(); ax[2].set_title("Accuracy per epoch")
fig.savefig(config.OUT_DIR / "eye_cnn_evaluation.png", dpi=200, bbox_inches="tight"); plt.close(fig)

n = min(8, len(te))
fig, axs = plt.subplots(2, 4, figsize=(14, 7))
for ax, i in zip(axs.ravel(), te[:n]):
    arr = X[i].astype("float32")
    ax.imshow(overlay(arr, gradcam(model, arr))); ax.axis("off")
    ax.set_title(f"true={'ASD' if y[i] else 'TD'}  p={model.predict(arr[None], verbose=0)[0, 0]:.2f}")
fig.savefig(config.OUT_DIR / "eye_cnn_gradcam.png", dpi=200, bbox_inches="tight"); plt.close(fig)

model.save(config.MODEL_DIR / "asd_eye_cnn.keras")
json.dump({"threshold": thr}, open(config.MODEL_DIR / "eye_threshold.json", "w"))
print("Saved model  :", config.MODEL_DIR / "asd_eye_cnn.keras")
print("Saved figures:", config.OUT_DIR)
print("Next:  python src/train_questionnaire.py")
