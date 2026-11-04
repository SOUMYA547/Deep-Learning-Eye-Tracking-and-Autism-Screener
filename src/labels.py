"""Find the metadata file and map participant ID -> ASD / TD."""
import re
import config
from eye_utils import read_table, _nospace


def norm_id(v):
    s = str(v).strip().lower()
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def parse_label(v):
    s = str(v).strip().lower()
    if s in ("1", "1.0"):
        return "ASD"
    if s in ("0", "0.0"):
        return "TD"
    if "asd" in s or "autis" in s:
        return "ASD"
    if s == "td" or "typical" in s or "control" in s or "normal" in s or "healthy" in s:
        return "TD"
    return None


def find_meta_file():
    if config.META_FILE:
        p = config.RAW_DIR / config.META_FILE
        return p if p.exists() else None
    cands = [p for p in config.RAW_DIR.glob("*.csv") if re.search(r"participants?|metadata", p.name, re.I)]
    cands.sort(key=lambda p: ("participants" not in p.name.lower(), p.name))
    return cands[0] if cands else None


def load_label_map():
    """Return (dict id->label, meta_path, id_col, label_col) or (None, ...) if not found."""
    meta = find_meta_file()
    if meta is None:
        return None, None, None, None
    df = read_table(meta)
    ns = {c: _nospace(c) for c in df.columns}
    id_col = config.META_ID_COL if config.META_ID_COL in df.columns else None
    if id_col is None:
        for want in ("participant", "participantid", "participantname", "subjectid", "id"):
            for c, n in ns.items():
                if n == want:
                    id_col = c
                    break
            if id_col:
                break
    if id_col is None:
        for c, n in ns.items():
            if "participant" in n or n.endswith("id"):
                id_col = c
                break
    lab_col = config.META_LABEL_COL if config.META_LABEL_COL in df.columns else None
    if lab_col is None:
        for key in ("class", "group", "diagnos", "label", "asd", "status"):
            for c, n in ns.items():
                if key in n and c != id_col:
                    lab_col = c
                    break
            if lab_col:
                break
    if id_col is None or lab_col is None:
        return None, meta, id_col, lab_col
    mp = {}
    for pid, lab in zip(df[id_col], df[lab_col]):
        parsed = parse_label(lab)
        if parsed:
            mp[norm_id(pid)] = parsed
    return mp, meta, id_col, lab_col


def label_for(pid, label_map):
    """Label from metadata; fall back to the participant name itself (e.g. 'ASD_07')."""
    key = norm_id(pid)
    if label_map and key in label_map:
        return label_map[key]
    return parse_label(pid) if re.search(r"asd|autis|^td|typical|control", key) else None
