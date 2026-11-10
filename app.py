"""ASD Eye-Tracking Screener (research prototype).
Run:  python -m streamlit run app.py
"""
import datetime
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import config
from eye_utils import (read_table, to_num, detect_xy_cols, detect_time_col, normalize_xy, render_scanpath,
                       pil_to_array, predict_image, gradcam, overlay, fuse, risk_label, recalibrate)

st.set_page_config(page_title="ASD Eye-Tracking Screener", page_icon="👁️", layout="wide")

# ------------------------------------------------------------------ style
st.markdown("""
<style>
.hero{padding:1.4rem 1.6rem;border-radius:16px;margin-bottom:1rem;
      background:linear-gradient(120deg,#1f3a93 0%,#3b6fd4 55%,#6a8fe0 100%);color:#fff;}
.hero h1{margin:0;font-size:2rem;color:#fff;}
.hero p{margin:.35rem 0 0 0;opacity:.92;}
.badge{display:inline-block;padding:.45rem 1rem;border-radius:999px;font-weight:700;font-size:1.05rem;}
.b-low{background:rgba(46,160,67,.18);color:#2ea043;border:1px solid #2ea043;}
.b-mid{background:rgba(219,171,9,.18);color:#c99700;border:1px solid #c99700;}
.b-high{background:rgba(218,54,51,.18);color:#da3633;border:1px solid #da3633;}
</style>
""", unsafe_allow_html=True)


def show_img(img, caption=None):
    """st.image that works on old and new Streamlit versions."""
    try:
        st.image(img, caption=caption, width="stretch")
    except Exception:
        st.image(img, caption=caption, use_column_width=True)


# ---------------------------------------------------------------- models
CNN_PATH = config.MODEL_DIR / "asd_eye_cnn.keras"
TAB_PATH = config.MODEL_DIR / "questionnaire_model.joblib"
THR_PATH = config.MODEL_DIR / "eye_threshold.json"
SPLIT_PATH = config.MODEL_DIR / "split_children.json"
MAN_PATH = config.IMG_DIR / "manifest.csv"

st.markdown('<div class="hero"><h1>👁️ AI Eye-Tracking ASD Screener</h1>'
            '<p>Deep-learning analysis of gaze scanpaths + AQ-10 questionnaire, fused into one screening score.</p></div>',
            unsafe_allow_html=True)
st.warning("Research prototype for education. **Screening support only - not a medical diagnosis.** "
           "Always consult a qualified clinician.")

if not (CNN_PATH.exists() and TAB_PATH.exists()):
    st.error("Models not found. Run `python src/train_eye_cnn.py` and `python src/train_questionnaire.py` first.")
    st.stop()


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    from tensorflow import keras
    return keras.models.load_model(CNN_PATH), joblib.load(TAB_PATH)


@st.cache_data
def load_manifest():
    return pd.read_csv(MAN_PATH) if MAN_PATH.exists() else None


cnn, tab = load_models()
manifest = load_manifest()
thr = json.load(open(THR_PATH))["threshold"] if THR_PATH.exists() else None
test_children = set(json.load(open(SPLIT_PATH))["test_children"]) if SPLIT_PATH.exists() else None

if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None


def eye_prob(arr):
    p = predict_image(cnn, arr)
    return recalibrate(p, thr) if thr is not None else p


def badge(p):
    cls = "b-low" if p < 0.35 else ("b-mid" if p < 0.65 else "b-high")
    return f'<span class="badge {cls}">{risk_label(p)}</span>'


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    w_img = st.slider("Weight of eye-tracking model", 0.0, 1.0, 0.5, 0.05,
                      help="Combined = w x eye-tracking + (1 - w) x questionnaire")
    st.caption(f"Eye-tracking: **{w_img:.0%}**  |  Questionnaire: **{1 - w_img:.0%}**")
    alpha = st.slider("Heatmap opacity", 0.1, 0.9, 0.45, 0.05)
    st.divider()
    st.subheader("Model info")
    mp = config.OUT_DIR / "eye_cnn_metrics.json"
    if mp.exists():
        m = json.load(open(mp))
        st.metric("Eye CNN - test ROC-AUC (per child)", f"{m['child_level']['ROC_AUC']:.2f}")
        st.caption(f"{m['n_test_children']} unseen test children")
    st.caption("Eye model: trained on children (Carette scanpaths).  \n"
               "Questionnaire model: trained on adult AQ-10 data (UCI).")
    if st.button("🗑️ Clear history"):
        st.session_state.history = []
        st.session_state.result = None

tab_screen, tab_explore, tab_hist, tab_perf, tab_about = st.tabs(
    ["🩺 Screening", "🖼️ Dataset explorer", "📋 History", "📊 Model performance", "ℹ️ About"])

# ================================================================ SCREENING
AQ = [  # (wording, answer that scores 1)
    ("I often notice small sounds when others do not.", "Agree"),
    ("I usually concentrate more on the whole picture than on the small details.", "Disagree"),
    ("I find it easy to do more than one thing at once.", "Disagree"),
    ("If there is an interruption, I can switch back to what I was doing very quickly.", "Disagree"),
    ("I find it easy to 'read between the lines' when someone is talking to me.", "Disagree"),
    ("I know how to tell if someone listening to me is getting bored.", "Disagree"),
    ("When I read a story, I find it difficult to work out the characters' intentions.", "Agree"),
    ("I like to collect information about categories of things (cars, birds, trains, plants...).", "Agree"),
    ("I find it easy to work out what someone is thinking or feeling just by looking at their face.", "Disagree"),
    ("I find it difficult to work out people's intentions.", "Agree"),
]

with tab_screen:
    left, right = st.columns(2, gap="large")

    # ---- step 1: eye tracking
    with left:
        st.subheader("Step 1 - Eye-tracking input")
        src_opts = ["Upload scanpath image(s)", "Upload raw gaze CSV"]
        if manifest is not None:
            src_opts.append("Use a sample from the dataset")
        source = st.radio("Input type", src_opts, label_visibility="collapsed")
        items, truth, held_out = [], None, None          # items = list of (name, image array)
        if source.startswith("Upload scanpath"):
            ups = st.file_uploader("PNG / JPG scanpath image(s). Add several recordings of the SAME child "
                                   "for a more reliable child-level result.",
                                   type=["png", "jpg", "jpeg"], accept_multiple_files=True)
            items = [(u.name, pil_to_array(Image.open(u))) for u in ups]
        elif source == "Upload raw gaze CSV":
            st.caption("CSV with X and Y gaze columns (one child, one recording).")
            up = st.file_uploader("Gaze CSV", type=["csv"])
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
                        items = [(up.name, pil_to_array(render_scanpath(nx, ny)))]
        else:
            c1, c2 = st.columns(2)
            cls = c1.selectbox("Group", ["ASD", "TD"])
            sub = manifest[manifest.label == cls]
            kids = sorted(sub.participant.unique(), key=str)
            if test_children is not None:
                only_test = st.checkbox("Only held-out test children (never seen in training)", value=True)
                if only_test:
                    kids = [k for k in kids if str(k) in test_children] or kids
            child = c2.selectbox("Child ID", kids)
            rec = sub[sub.participant == child].reset_index(drop=True)
            use_all = st.checkbox(f"Use all {len(rec)} recordings of this child (child-level result)", value=True)
            if use_all or len(rec) == 1:
                chosen = rec
            else:
                k = st.slider("Recording", 1, len(rec), 1)
                chosen = rec.iloc[[k - 1]]
            items = [(Path(p).name, pil_to_array(Image.open(p))) for p in chosen.path]
            truth = cls
            if test_children is not None:
                held_out = str(child) in test_children
                st.caption("✅ Held-out test child: the model never saw this child." if held_out else
                           "⚠️ This child was used for training/validation - demo only, not a fair test.")
            else:
                st.caption("Sample images may have been used for training, so this is a demo, not a test.")
        if items:
            show_img(items[0][1].astype("uint8"),
                     caption=items[0][0] + (f"  (+{len(items) - 1} more recordings)" if len(items) > 1 else ""))
        else:
            st.info("Provide an eye-tracking input, or analyse with the questionnaire only.")

    # ---- step 2: questionnaire
    with right:
        st.subheader("Step 2 - Questionnaire (AQ-10)")
        answers, score, answered = [], 0, 0
        for i, (q, scoring) in enumerate(AQ):
            a = st.radio(f"**A{i + 1}.** {q}", ["Agree", "Disagree"], index=None, horizontal=True, key=f"aq{i}")
            if a is not None:
                answered += 1
                score += int(a == scoring)
            answers.append(int(a == scoring) if a is not None else 0)
        st.progress(answered / 10, text=f"Answered {answered}/10  |  AQ-10 score: {score}/10 "
                                        f"(this dataset flags score >= 7)")
        d1, d2, d3, d4 = st.columns(4)
        age = d1.number_input("Age", 2, 90, 25)
        gender = d2.selectbox("Sex", ["m", "f"])
        jaundice = d3.selectbox("Jaundice at birth", ["no", "yes"])
        family = d4.selectbox("Family ASD", ["no", "yes"])

    st.divider()
    if st.button("🔍 Analyse", type="primary"):
        if answered < 10:
            st.warning(f"{10 - answered} question(s) unanswered - counted as 0, which lowers the score.")
        rowd = {f"A{i + 1}_Score": s for i, s in enumerate(answers)}
        rowd.update(age=age, gender=int(gender == "m"), jundice=int(jaundice == "yes"), austim=int(family == "yes"))
        p_tab = float(tab["model"].predict_proba(pd.DataFrame([rowd])[tab["features"]])[0, 1])
        res = dict(time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   source=(items[0][0] if len(items) == 1 else f"{len(items)} recordings") if items else "-",
                   aq_score=score, answered=answered, p_tab=p_tab, p_img=None, final=p_tab, truth=truth,
                   held_out=held_out, n_rec=len(items), names=[n for n, _ in items],
                   arrs=[a for _, a in items], per_rec=[])
        if items:
            per_rec = [eye_prob(a) for _, a in items]
            p_img = float(np.mean(per_rec))
            res.update(p_img=p_img, final=fuse(p_img, p_tab, w_img), per_rec=per_rec)
        res["w_img"] = w_img if items else 0.0
        st.session_state.result = res
        st.session_state.history.append({k: v for k, v in res.items() if k not in ("arrs", "per_rec", "names")})

    r = st.session_state.result
    if r is not None:
        st.header("Results")
        st.markdown(badge(r["final"]), unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Eye-tracking CNN", f"{r['p_img']:.0%}" if r["p_img"] is not None else "n/a")
        m2.metric("Questionnaire model", f"{r['p_tab']:.0%}")
        m3.metric("Combined score", f"{r['final']:.0%}")
        st.progress(float(min(max(r["final"], 0), 1)), text="Combined ASD likelihood")

        if r["n_rec"] > 1:
            st.info(f"Child-level eye-tracking score = mean over {r['n_rec']} recordings "
                    f"(range {min(r['per_rec']):.0%} - {max(r['per_rec']):.0%}). "
                    "Averaging recordings is more reliable than judging one image.")
        if r["p_img"] is not None:
            if (r["p_img"] >= 0.5) == (r["p_tab"] >= 0.5):
                st.success("Both models point in the same direction.")
            else:
                st.warning("The two models disagree - treat the result with extra caution.")
        if r["truth"]:
            tag = ""
            if r["held_out"] is True:
                tag = " | held-out test child"
            elif r["held_out"] is False:
                tag = " | child was in training/validation"
            st.caption(f"Dataset label for this sample: **{r['truth']}**{tag} (for demo checking only).")

        t1, t2, t3, t4 = st.tabs(["Explanation (Grad-CAM)", "Per-recording", "Contribution", "Download report"])
        with t1:
            if r["arrs"]:
                idx = 0
                if len(r["arrs"]) > 1:
                    idx = st.selectbox("Recording to explain", range(len(r["arrs"])),
                                       format_func=lambda i: f"{i + 1}. {r['names'][i]}  (P={r['per_rec'][i]:.0%})")
                arr = r["arrs"][idx]
                cam = gradcam(cnn, arr)
                a, b = st.columns(2)
                with a:
                    show_img(arr.astype("uint8"), caption="Scanpath")
                with b:
                    show_img(overlay(arr, cam, alpha), caption="Grad-CAM: warm colours = regions driving the CNN")
                st.caption("The heatmap shows where the network looked. It is an aid for discussion, "
                           "not proof of a clinical sign.")
            else:
                st.info("No eye-tracking input was provided.")
        with t2:
            if r["per_rec"]:
                df_rec = pd.DataFrame({"P(ASD)": r["per_rec"]}, index=[f"{i + 1}" for i in range(len(r["per_rec"]))])
                st.bar_chart(df_rec)
                st.caption(f"Decision cut-off for the eye model is 50% after calibration. "
                           f"{sum(p >= 0.5 for p in r['per_rec'])}/{len(r['per_rec'])} recordings are above it.")
            else:
                st.info("No eye-tracking input was provided.")
        with t3:
            wi = r["w_img"]
            parts = {"Eye-tracking": wi * (r["p_img"] or 0), "Questionnaire": (1 - wi) * r["p_tab"]}
            st.bar_chart(pd.DataFrame({"contribution to combined score": parts}))
            st.caption(f"Combined = {wi:.2f} x eye + {1 - wi:.2f} x questionnaire.")
        with t4:
            txt = (f"ASD Eye-Tracking Screener report\nTime: {r['time']}\nInput: {r['source']}\n"
                   f"Recordings analysed: {r['n_rec']}\n"
                   f"AQ-10 score: {r['aq_score']}/10 ({r['answered']} answered)\n"
                   f"Eye-tracking CNN: {'n/a' if r['p_img'] is None else format(r['p_img'], '.1%')}\n"
                   f"Questionnaire model: {r['p_tab']:.1%}\nCombined: {r['final']:.1%}\n"
                   f"Result: {risk_label(r['final'])}\n\n"
                   "Research prototype. Screening support only, not a medical diagnosis.\n")
            st.download_button("⬇️ Report (.txt)", txt, file_name="asd_screening_report.txt")
            if r["arrs"]:
                buf = io.BytesIO()
                Image.fromarray(overlay(r["arrs"][0], gradcam(cnn, r["arrs"][0]), alpha)).save(buf, format="PNG")
                st.download_button("⬇️ Grad-CAM image (.png)", buf.getvalue(), file_name="gradcam.png",
                                   mime="image/png")

# ================================================================ EXPLORER
with tab_explore:
    st.subheader("Dataset explorer")
    if manifest is None:
        st.info("Run `python src/prepare_carette.py --src data/carette` to enable this tab.")
    else:
        c1, c2, c3 = st.columns(3)
        grp = c1.selectbox("Group", ["Both", "ASD", "TD"], key="ex_grp")
        n = c2.slider("Images", 4, 12, 8, 4, key="ex_n")
        seed = c3.number_input("Random seed", 0, 9999, 1, key="ex_seed")
        pool = manifest if grp == "Both" else manifest[manifest.label == grp]
        if test_children is not None and st.checkbox("Only held-out test children", value=True, key="ex_test"):
            pool = pool[pool.participant.astype(str).isin(test_children)]
        sample = pool.sample(min(n, len(pool)), random_state=int(seed)).reset_index(drop=True)
        if st.button("Predict on these images"):
            cols = st.columns(4)
            correct = 0
            for i, rw in sample.iterrows():
                a = pil_to_array(Image.open(rw.path))
                p = eye_prob(a)
                ok = (p >= 0.5) == (rw.label == "ASD")
                correct += int(ok)
                with cols[i % 4]:
                    show_img(a.astype("uint8"))
                    st.caption(f"True: **{rw.label}** | P(ASD) {p:.0%} {'✅' if ok else '❌'}")
            st.info(f"{correct}/{len(sample)} correct on this sample (single images, not child-level). "
                    "Use the Model performance tab for proper test numbers.")
        st.caption(f"Dataset: {len(manifest)} images from {manifest.participant.nunique()} children.")

# ================================================================ HISTORY
with tab_hist:
    st.subheader("Analyses in this session")
    if not st.session_state.history:
        st.info("No analyses yet. Run one in the Screening tab.")
    else:
        h = pd.DataFrame(st.session_state.history)
        view = h[["time", "source", "aq_score", "p_img", "p_tab", "final"]].copy()
        for c in ("p_img", "p_tab", "final"):
            view[c] = view[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:.0%}")
        view.columns = ["Time", "Input", "AQ-10", "Eye CNN", "Questionnaire", "Combined"]
        st.dataframe(view)
        st.download_button("⬇️ History (.csv)", h.to_csv(index=False), file_name="screening_history.csv")

# ================================================================ PERFORMANCE
with tab_perf:
    st.subheader("Model performance")
    mp = config.OUT_DIR / "eye_cnn_metrics.json"
    cvp = config.OUT_DIR / "cv_results.csv"
    if mp.exists():
        m = json.load(open(mp))
        st.markdown("**Hold-out test (children never seen in training)**")
        st.dataframe(pd.DataFrame({"Per image": m["image_level"], "Per child": m["child_level"]}).round(3))
        st.caption(f"{m['n_test_images']} test images from {m['n_test_children']} children. Very small - unstable.")
    if cvp.exists():
        cv = pd.read_csv(cvp)
        st.markdown("**5-fold child-level cross-validation (mean ± std)**")
        cols = [c for c in cv.columns if c.startswith(("img_", "child_"))]
        st.dataframe(pd.DataFrame({"mean": cv[cols].mean(), "std": cv[cols].std()}).round(3))
    for f, cap in (("eye_cnn_evaluation.png", "Confusion matrix, ROC curve, training curve"),
                   ("eye_cnn_gradcam.png", "Grad-CAM on test images")):
        p = config.OUT_DIR / f
        if p.exists():
            show_img(str(p), caption=cap)
    if not mp.exists() and not cvp.exists():
        st.info("Train the model first to see metrics here.")

# ================================================================ ABOUT
with tab_about:
    st.subheader("How it works")
    st.markdown("""
1. **Eye-tracking track:** each scanpath image goes into a CNN (MobileNetV2 transfer learning) -> P(ASD). Several recordings of one child are averaged.
2. **Questionnaire track:** AQ-10 answers + age/sex/history go into an ensemble (LR + RF + SVM) -> P(ASD).
3. **Late fusion:** `combined = w x eye + (1 - w) x questionnaire` (w is the sidebar slider).
4. **Grad-CAM** shows which parts of the scanpath drove the CNN.
""")
    st.subheader("Limitations")
    st.markdown("""
- The two models were trained on **different people** (children for eye-tracking, adults for the questionnaire), so fusion is decision-level only.
- Eye model: only 54 children, so results are unstable. The cut-off was tuned on a handful of validation children.
- Questionnaire label is defined by the AQ-10 score itself, so its accuracy is not clinically meaningful.
- Not validated clinically. **Never use to diagnose.**
""")
    st.caption("Data: Carette et al. 2019 scanpath images; UCI Autism Screening Adult.")