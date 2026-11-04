"""STEP 5 - the screener app.  Run:  streamlit run app.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import json
import joblib
import pandas as pd
import streamlit as st
from PIL import Image

import config
from eye_utils import (read_table, to_num, detect_xy_cols, detect_time_col, normalize_xy, render_scanpath,
                       pil_to_array, predict_image, gradcam, overlay, fuse, risk_label, recalibrate)

st.set_page_config(page_title="ASD Eye-Tracking Screener", layout="wide")
st.title("AI Eye-Tracking ASD Screener")
st.caption("Research prototype. Screening support only - NOT a medical diagnosis.")

CNN_PATH = config.MODEL_DIR / "asd_eye_cnn.keras"
TAB_PATH = config.MODEL_DIR / "questionnaire_model.joblib"
if not (CNN_PATH.exists() and TAB_PATH.exists()):
    st.error("Models not found. Run train_eye_cnn.py and train_questionnaire.py first.")
    st.stop()


@st.cache_resource
def load_models():
    from tensorflow import keras
    return keras.models.load_model(CNN_PATH), joblib.load(TAB_PATH)


cnn, tab = load_models()
w_img = st.sidebar.slider("Weight of eye-tracking model", 0.0, 1.0, 0.5, 0.05)

left, right = st.columns(2)
with left:
    st.subheader("1. Eye-tracking input")
    mode = st.radio("Input type", ["Scanpath image (PNG/JPG)", "Raw gaze CSV (one child, one recording)"])
    arr = None
    if mode.startswith("Scanpath"):
        up = st.file_uploader("Upload scanpath image", type=["png", "jpg", "jpeg"])
        if up:
            arr = pil_to_array(Image.open(up))
    else:
        up = st.file_uploader("Upload gaze CSV with X and Y columns", type=["csv"])
        if up:
            df = read_table(up)
            xc, yc = detect_xy_cols(df)
            if xc is None:
                st.error("Could not find X/Y columns. Name them 'x' and 'y'.")
            else:
                tc = detect_time_col(df)
                if tc:
                    df = df.assign(_t=to_num(df[tc])).sort_values("_t")
                nx, ny = normalize_xy(to_num(df[xc]).values, to_num(df[yc]).values)
                if len(nx) < config.MIN_POINTS:
                    st.error(f"Need at least {config.MIN_POINTS} gaze points.")
                else:
                    arr = pil_to_array(render_scanpath(nx, ny))

with right:
    st.subheader("2. Screening questionnaire (AQ-10)")
    st.caption("Tick an item if the answer points toward autism traits (score = 1).")
    cols = st.columns(5)
    scores = [int(cols[i % 5].checkbox(f"A{i + 1}")) for i in range(10)]
    age = st.number_input("Age", 1, 100, 10)
    gender = st.selectbox("Sex", ["m", "f"])
    jaundice = st.selectbox("Born with jaundice", ["no", "yes"])
    family = st.selectbox("Family member with ASD", ["no", "yes"])

if st.button("Analyse", type="primary"):
    row = {f"A{i + 1}_Score": s for i, s in enumerate(scores)}
    row.update(age=age, gender=int(gender == "m"), jundice=int(jaundice == "yes"), austim=int(family == "yes"))
    p_tab = float(tab["model"].predict_proba(pd.DataFrame([row])[tab["features"]])[0, 1])
    c1, c2, c3 = st.columns(3)
    c2.metric("Questionnaire model", f"{p_tab:.0%}")
    if arr is not None:
        p_img = predict_image(cnn, arr)
        thr_file = config.MODEL_DIR / "eye_threshold.json"
        if thr_file.exists():
            p_img = recalibrate(p_img, json.load(open(thr_file))["threshold"])
        final = fuse(p_img, p_tab, w_img)
        c1.metric("Eye-tracking CNN", f"{p_img:.0%}")
        c3.metric("Combined score", f"{final:.0%}")
        st.info(risk_label(final))
        a, b = st.columns(2)
        a.image(arr.astype("uint8"), caption="Scanpath (blue = start, red = end)", use_container_width=True)
        b.image(overlay(arr, gradcam(cnn, arr)), caption="Grad-CAM: regions driving the CNN", use_container_width=True)
    else:
        c3.metric("Combined score", f"{p_tab:.0%}")
        st.warning("No eye-tracking input - questionnaire result only.")
        st.info(risk_label(p_tab))
