"""All settings in one place. Edit here only if a script tells you to."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR   = ROOT / "data" / "eye_raw"                         # Cilia CSV files + Participants file
QUEST_CSV = ROOT / "data" / "questionnaire" / "autism_screening.csv"
IMG_DIR   = ROOT / "images"                                   # generated scanpath images
MODEL_DIR = ROOT / "models"                                   # generated models
OUT_DIR   = ROOT / "outputs"                                  # generated figures / metrics

SEED = 42
IMG_SIZE = 160            # CNN input size (160 is fast on CPU; 224 is slower)
SCREEN_W, SCREEN_H = 1920, 1080   # used only if gaze values are in pixels
MIN_POINTS = 50           # skip recordings with fewer gaze points
MAX_POINTS = 600          # down-sample long recordings to this many points

# ---- Manual overrides. Leave None to auto-detect. ----
# Run `python src/inspect_data.py` and only fill these if auto-detect is wrong.
COL_PARTICIPANT = None    # e.g. "Participant"
COL_X = None              # e.g. "Gaze point X"
COL_Y = None              # e.g. "Gaze point Y"
COL_TIME = None           # e.g. "Recording timestamp"
META_FILE = None          # e.g. "Participants.csv" (file inside data/eye_raw)
META_ID_COL = None        # e.g. "Participant"
META_LABEL_COL = None     # e.g. "Class"

for _d in (IMG_DIR, MODEL_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
