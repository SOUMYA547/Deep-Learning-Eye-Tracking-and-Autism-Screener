"""Shared helpers: reading gaze CSVs, rendering scanpaths, Grad-CAM, fusion."""
import csv
import re
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from matplotlib import colormaps

import config


# ------------------------------------------------------------------ reading
def read_table(path):
    """Read a CSV with unknown delimiter / encoding."""
    last_err = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                head = f.read(4096)
            try:
                sep = csv.Sniffer().sniff(head, delimiters=",;\t").delimiter
            except csv.Error:
                sep = ","
            return pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err


def to_num(s):
    """Series -> float, accepting decimal commas."""
    if s.dtype == object:
        s = s.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _nospace(c):
    return re.sub(r"[^a-z0-9]+", "", str(c).lower())


def detect_participant_col(df):
    if config.COL_PARTICIPANT and config.COL_PARTICIPANT in df.columns:
        return config.COL_PARTICIPANT
    ns = {c: _nospace(c) for c in df.columns}
    for want in ("participant", "participantname", "participantid", "subject", "subjectid", "username"):
        for c, n in ns.items():
            if n == want:
                return c
    for c, n in ns.items():
        if "participant" in n or "subject" in n:
            return c
    return None


def detect_xy_cols(df):
    if config.COL_X and config.COL_Y and config.COL_X in df.columns and config.COL_Y in df.columns:
        return config.COL_X, config.COL_Y
    ns = {_nospace(c): c for c in df.columns}
    pairs = [("gazepointx", "gazepointy"), ("gazex", "gazey"), ("fixationpointx", "fixationpointy"),
             ("x", "y"), ("posx", "posy"), ("xcoord", "ycoord")]
    for nx, ny in pairs:
        if nx in ns and ny in ns:
            cx, cy = ns[nx], ns[ny]
            if to_num(df[cx]).notna().mean() >= 0.5:
                return cx, cy
    for n, c in ns.items():
        if "gaze" in n and n.endswith("x") and (n[:-1] + "y") in ns:
            return c, ns[n[:-1] + "y"]
    return None, None


def detect_time_col(df):
    if config.COL_TIME and config.COL_TIME in df.columns:
        return config.COL_TIME
    for c in df.columns:
        n = _nospace(c)
        if "timestamp" in n or n in ("time", "t"):
            return c
    return None


def normalize_xy(x, y):
    """Return x, y in 0..1 (screen fractions)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return x, y
    if np.nanpercentile(x, 99) <= 1.5 and np.nanpercentile(y, 99) <= 1.5:
        nx, ny = x, y
    else:
        nx, ny = x / config.SCREEN_W, y / config.SCREEN_H
    return np.clip(nx, 0, 1), np.clip(ny, 0, 1)


# ---------------------------------------------------------------- rendering
def render_scanpath(nx, ny, size=None, max_points=None):
    """Draw a scanpath: line colour goes blue (start) -> red (end)."""
    size = size or config.IMG_SIZE
    max_points = max_points or config.MAX_POINTS
    n = len(nx)
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
        nx, ny, n = nx[idx], ny[idx], max_points
    S = size * 2
    img = Image.new("RGB", (S, S), "white")
    d = ImageDraw.Draw(img)
    px, py = nx * (S - 1), ny * (S - 1)
    for i in range(1, n):
        t = i / max(n - 1, 1)
        col = (int(255 * t), 40, int(255 * (1 - t)))
        d.line([(px[i - 1], py[i - 1]), (px[i], py[i])], fill=col, width=2)
    r = 3
    for i in range(0, n, 5):
        t = i / max(n - 1, 1)
        col = (int(255 * t), 40, int(255 * (1 - t)))
        d.ellipse([px[i] - r, py[i] - r, px[i] + r, py[i] + r], outline=col, width=1)
    return img.resize((size, size), Image.LANCZOS)


def pil_to_array(img):
    img = img.convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE))
    return np.asarray(img, dtype="float32")


# ----------------------------------------------------------- model helpers
def predict_image(model, arr):
    return float(model.predict(arr[None], verbose=0)[0, 0])


def gradcam(model, arr):
    """Grad-CAM heatmap (0..1) showing which image regions drive the prediction."""
    import tensorflow as tf
    x = tf.convert_to_tensor(arr[None], tf.float32)
    x = model.get_layer("rescale")(x)
    base, head = model.get_layer("backbone"), model.get_layer("head")
    with tf.GradientTape() as tape:
        conv = base(x, training=False)
        tape.watch(conv)
        p = head(conv, training=False)[:, 0]
    grads = tape.gradient(p, conv)
    w = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * conv, axis=-1))[0].numpy()
    return cam / (cam.max() + 1e-8)


def overlay(arr, cam, alpha=0.45):
    h, w = arr.shape[:2]
    cam_img = Image.fromarray((cam * 255).astype("uint8")).resize((w, h), Image.BILINEAR)
    heat = colormaps["jet"](np.asarray(cam_img) / 255.0)[..., :3] * 255
    return ((1 - alpha) * arr + alpha * heat).astype("uint8")


def recalibrate(p, thr):
    """Shift probability so the tuned decision threshold maps to 0.5 (keeps ranking)."""
    import math
    eps = 1e-6
    lg = lambda v: math.log(min(max(v, eps), 1 - eps) / (1 - min(max(v, eps), 1 - eps)))
    return 1 / (1 + math.exp(-(lg(p) - lg(thr))))


def fuse(p_img, p_tab, w_img=0.5):
    return w_img * p_img + (1 - w_img) * p_tab


def risk_label(p):
    if p < 0.35:
        return "Low likelihood of ASD traits"
    if p < 0.65:
        return "Moderate - recommend clinical follow-up"
    return "High likelihood of ASD traits - refer for clinical assessment"