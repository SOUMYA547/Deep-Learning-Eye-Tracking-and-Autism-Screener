# AI Eye-Tracking + Questionnaire Screener for Autism Spectrum Disorder (ASD)

Research prototype. **Screening support only. Not a medical diagnosis.**

## 1. What this project does

The original project predicted ASD from questionnaire data using classical machine learning.
This extension adds **deep learning on eye-tracking images**.

- Input 1: an image of a child's eye-gaze path (a "scanpath") while watching a screen.
- Input 2: the child's AQ-10 screening questionnaire answers.
- Output: ASD likelihood from each model, a combined score, and a heatmap showing what the CNN looked at.

## 2. Full pipeline

```
 EYE-TRACKING TRACK (deep learning)               QUESTIONNAIRE TRACK (classical ML)
 ----------------------------------               ----------------------------------
 Carette scanpath images (547 images,             UCI "Autism Screening Adult"
 54 children: 26 ASD, 28 TD)                      (704 rows, AQ-10 answers + age, sex ...)
        |                                                  |
 prepare_carette.py                                train_questionnaire.py
  - read labels from Metadata CSVs                  - drop leaky "result" column
  - brighten, thicken lines, resize 160x160         - Voting ensemble: LR + RF + SVM
  - write images/ + manifest.csv                    - save models/questionnaire_model.joblib
        |                                                  |
 train_eye_cnn.py                                          |
  - split BY CHILD (no child in train and test)            |
  - CNN: MobileNetV2 (pretrained) or small CNN             |
  - tune decision threshold on validation children         |
  - evaluate per image and per child, Grad-CAM             |
  - save models/asd_eye_cnn.keras                          |
        |                                                  |
        +------------------ LATE FUSION -------------------+
          P_final = w * P_eye + (1 - w) * P_questionnaire   (w = slider in app, default 0.5)
                              |
                         app.py (Streamlit)
          upload scanpath image + tick AQ-10 -> scores + risk level + Grad-CAM heatmap
```

## 3. Folder structure

```
asd_project/
  README.md
  requirements.txt
  app.py                       Streamlit screener app
  convert_arff.py              (only if questionnaire came as .arff from UCI)
  src/
    config.py                  all paths and settings
    eye_utils.py               helpers: scanpath drawing, Grad-CAM, fusion, risk label
    prepare_carette.py         STEP 1: images + labels -> images/ + manifest.csv
    train_eye_cnn.py           STEP 2: train and evaluate the CNN
    train_questionnaire.py     STEP 3: train the questionnaire model
  tests/
    check_pipeline.py          self-test (expect 13/13 PASS)
  data/                        (you put the datasets here)
    carette/
      Images/TCImages/         328 png  (TC = typically developing control)
      Images/TSImages/         219 png  (TS = ASD group)
      Metadata/                Metadata_Participants.csv, Metadata_TC.csv, Metadata_TS.csv
    questionnaire/
      autism_screening.csv
  images/                      (generated) processed images + manifest.csv
  models/                      (generated) asd_eye_cnn.keras, eye_threshold.json, questionnaire_model.joblib
  outputs/                     (generated) metrics, figures, cv_results.csv
```

## 4. Datasets and where to get them

| Data | Source |
|---|---|
| Scanpath images (used) | GitHub: `ItaFel/Visualization-of-Eye-Tracking-Scanpaths-in-Autism-Spectrum-Disorder-Image-Dataset` (Images.rar, Metadata.rar; extract with WinRAR). Original: Carette et al., HEALTHINF 2019, on figshare. |
| Questionnaire (used) | UCI "Autism Screening Adult", https://archive.ics.uci.edu/dataset/426/autism+screening+adult (file is .arff; convert with `convert_arff.py`). |
| Raw gaze CSVs (not used) | Cilia et al. 2022, https://figshare.com/articles/dataset/Eye_tracking_Dataset_to_Support_the_Research_on_Autism_Spectrum_Disorder/20113592 or Kaggle `imtkaggleteam/eye-tracking-autism` |
| Second eye-tracking set (not used) | Saliency4ASD, Zenodo 10.5281/zenodo.2647418 |

In the Carette metadata, TS = ASD group (29 children with CARS scores) and TC = control group.

## 5. Setup and run (Windows 11, VS Code terminal)

Python 3.10 to 3.12 is needed (3.11 recommended).

```powershell
cd <project folder>
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python src/prepare_carette.py --src data/carette
python src/train_eye_cnn.py
python src/train_questionnaire.py
python tests/check_pipeline.py
python -m streamlit run app.py
```

Expected output of each step:

| Step | Expected |
|---|---|
| prepare_carette.py | ASD 219 images / 26 children, TD 328 images / 28 children |
| train_eye_cnn.py | prints per-image and per-child test results; saves model and figures |
| train_questionnaire.py | CV accuracy about 0.99; saves model |
| check_pipeline.py | `13/13 checks passed` |
| app | opens http://localhost:8501 |

Options:
- No internet or slow PC: `python src/train_eye_cnn.py --arch small --weights none --epochs-head 30 --epochs-ft 0`
- Numbers for the report (5-fold cross-validation, slow): `python src/train_eye_cnn.py --mode cv`

## 6. Results so far

Small CNN trained from scratch, 5-fold cross-validation, split by child (every child is in the test set exactly once):

| Metric | Per image | Per child (mean of its images) |
|---|---|---|
| Accuracy | 0.724 +- 0.050 | 0.800 +- 0.135 |
| ROC-AUC | 0.793 +- 0.064 | 0.873 +- 0.157 |

- Each fold has only about 11 test children, so the spread is large.
- The pretrained MobileNetV2 run (the default) has not been tested yet. Record its numbers here once run.
- Questionnaire model: 0.99 CV accuracy, ROC-AUC 1.0. This is expected, because the dataset label is defined by the AQ-10 score itself.

## 7. Key concepts (for explaining to the group)

- **Scanpath**: a drawing of where the eyes moved over time.
- **CNN**: a neural network that learns visual patterns from images.
- **Transfer learning**: start from a network pretrained on millions of images (MobileNetV2) and fine-tune it on our small dataset.
- **Child-level split**: one child produces about 10 images. If some of a child's images are in training and others in testing, the model can recognise the child instead of ASD, giving fake high accuracy. We keep each child entirely in train OR test.
- **Threshold tuning**: the decision cut-off is chosen on validation children only, never on test children.
- **ROC-AUC**: how well the model ranks ASD above TD (0.5 = guessing, 1.0 = perfect).
- **Grad-CAM**: a heatmap showing which image regions drove the CNN's prediction.
- **Late fusion**: each model predicts separately, then the probabilities are averaged with weight w.

## 8. Known limitations (must be stated in the report)

1. Image and questionnaire data belong to different people. There are no paired records, so fusion is decision-level, not jointly trained.
2. Small data: 54 children. Results are unstable; always quote mean +- std.
3. The questionnaire label is defined by the A1-A10 scores, so its high accuracy is expected and not clinically meaningful.
4. The original notebook kept the `result` column (sum of A1-A10), which caused label leakage and about 100% accuracy. It is dropped here.
5. The demo from-scratch CNN gives low-confidence outputs, so the app often shows "moderate".
6. Screening support only. Not validated clinically. Never use it to diagnose a child.

## 9. Suggested work split (7 people)

| # | Task | Files |
|---|---|---|
| 1 | Data: download, check counts, explain labels | data/, prepare_carette.py |
| 2 | CNN training with pretrained weights, record results | train_eye_cnn.py |
| 3 | Cross-validation, figures, result tables | outputs/ |
| 4 | Questionnaire model, leakage fix, compare with old notebook | train_questionnaire.py |
| 5 | App: UI, fusion weight, Grad-CAM display | app.py, eye_utils.py |
| 6 | Testing and documentation | tests/, README.md |
| 7 | Report writing: method, limitations, ethics | report |

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `python` not recognized | use `py`, or reinstall Python with "Add to PATH" |
| `No module named tensorflow` | unsupported Python version; use 3.11 |
| `Cannot find Images/TSImages ...` | wrong folder layout; check section 3 |
| 0-byte PNG files | bad extraction; re-extract the RAR with WinRAR |
| `manifest.csv` not found | run prepare_carette.py first |
| `Models not found` in the app | run both training scripts first |
| Weights download fails | use the `--arch small --weights none` command from section 5 |

## 11. Future work

- Run and report the pretrained MobileNetV2 results.
- Test on a second dataset (Saliency4ASD) for a cross-dataset result.
- Add the raw-gaze route (Cilia CSVs to scanpath images) and the live webcam gaze capture.
- Collect paired data (images plus questionnaire from the same children) for real multimodal training.
- Add probability calibration.
- Compare with other CNNs (EfficientNet, ResNet).

## 12. References

- Carette et al. (2019). Learning to Predict ASD Based on the Visual Patterns of Eye-Tracking Scanpaths. HEALTHINF.
- Cilia et al. (2022). Eye-Tracking Dataset to Support the Research on ASD. IJCAI-ECAI SDAIH workshop.
- Duan et al. (2019). A Dataset of Eye Movements for the Children with ASD. ACM MMSys.
- Thabtah, F. Autism Screening Adult dataset. UCI Machine Learning Repository.