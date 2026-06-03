from __future__ import annotations
import re
from typing import Tuple, Optional
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 0. Universal helpers
# -----------------------------------------------------------------------------

NULL_PATTERN = re.compile(r"^\s*(null|nan|na|n/a|#name\?|[-–—]|\?|none)?\s*$", re.I)


def normalise_nulls(series: pd.Series) -> pd.Series:
    """Replace common textual tokens with genuine NaN."""
    return series.replace(NULL_PATTERN, np.nan, regex=True)


# -----------------------------------------------------------------------------
# 1. Column‑specific parsers
# -----------------------------------------------------------------------------

def split_user_name(x: str | float | None) -> Tuple[Optional[int], Optional[str]]:
    """`324_Onco` → (324, "Onco")."""
    if pd.isna(x):
        return np.nan, np.nan
    if isinstance(x, float):  # occasionally stored as number
        x = str(int(x))
    parts = str(x).split("_", 1)
    id_part = re.sub(r"\D", "", parts[0])
    role_part = parts[1] if len(parts) == 2 else np.nan
    return (int(id_part) if id_part else np.nan, role_part)


def parse_stage_basis(x: str | None) -> str | pd.NA:
    mapping = {"c - clinical": "clinical", "p - pathological": "pathological", "r - reccurent": "recurrent"}
    x = str(x).strip().lower() if x is not None else ""
    return mapping.get(x, pd.NA)


def parse_histgrade(x: str | None) -> Optional[int]:
    if pd.isna(x):
        return np.nan
    m = re.search(r"g([1-4])", str(x).lower())
    return int(m.group(1)) if m else np.nan


def parse_ivi(x: str | None) -> Optional[int]:
    if pd.isna(x):
        return np.nan
    x_low = str(x).lower()
    if re.search(r"(pos|\+|yes|present)", x_low):
        return 1
    if re.search(r"(neg|\-|no|absent)", x_low):
        return 0
    return np.nan

# Ki‑67 -----------------------------------------------------------------------
_KI67_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)")
_KI67_SINGLE = re.compile(r"(\d+(?:\.\d+)?)")


def parse_ki67(x: str | None) -> Optional[float]:
    if pd.isna(x):
        return np.nan
    s = str(x).lower().strip()
    # Excel auto‑date such as 06‑Sep interpreted instead of 6.9
    if re.match(r"\d{1,2}[-/](jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", s):
        return np.nan
    if m := _KI67_RANGE.search(s):
        return (float(m.group(1)) + float(m.group(2))) / 2
    if m := _KI67_SINGLE.fullmatch(s.rstrip("%")):
        val = float(m.group(1))
        return val if 0 <= val <= 100 else np.nan
    if any(tok in s for tok in ("<", "less")):
        m = _KI67_SINGLE.search(s)
        return float(m.group(1)) if m else 5.0
    return np.nan

# HER2 ------------------------------------------------------------------------
# Normalise common Hebrew words first
_HEB_MAP = {
    "חיובי": "positive",
    "שלילי": "negative",
    "בינוני": "equivocal",
}

_POS_PAT = re.compile(
    r"(amplified|amplific|fish[^a-zА-я]*\+|cish[^a-z]*\+|/neu[^a-z]*pos|\bpos(?![a-z])|\bpositive\b)",
    re.I,
)
_NEG_PAT = re.compile(
    r"(not\s+amplified|non\s+amplified|fish[^a-zА-я]*-|cish[^a-z]*-|/neu[^a-z]*neg|\bneg(?![a-z])|\bnegative\b)",
    re.I,
)
_INDET_PAT = re.compile(
    r"(equivocal|indeterminate|intermediate|borderline|pending|indeterm|interm)",
    re.I,
)

# “2+”, “3 +”, “( +1 )”, “score 2”, “+++"  … but NOT the 2 in “ratio 2.3”
_IHC_PAT = re.compile(
    r"""
    (?:
        ([0-3])\s*\+                # classic “2+”
      | score\D*?([0-3])            # “score 1”
      | \(\+\s*([0-3])\)            # “(+1)”
      | ^\s*([0-3])\s*$             # bare “1”
    )
    """,
    re.X,
)

_PLUS3_PAT = re.compile(r"\+\s*\+\s*\+")

_SLASH_RANGE_PAT = re.compile(r"([0-3])\s*[-/]\s*([0-3])")  # “0-1”  or “1/2”


# ------------------------------------------------------------------
def _normalise_hebrew(txt: str) -> str:
    for heb, eng in _HEB_MAP.items():
        txt = txt.replace(heb, eng)
    return txt


def parse_her2_cell(raw) -> tuple[float | np.nan, str | pd.NA]:
    """
    Parameters
    ----------
    raw : str | float | None

    Returns
    -------
    ihc_score : float  (0-3 or np.nan)
    fish_call : 'positive' | 'negative' | 'indeterminate' | <NA>
    """
    if pd.isna(raw):
        return np.nan, pd.NA

    s = _normalise_hebrew(str(raw).lower())

    # --------  FISH call first  --------
    fish_call = pd.NA
    if _POS_PAT.search(s):
        fish_call = "positive"
    elif _NEG_PAT.search(s):
        fish_call = "negative"
    elif _INDET_PAT.search(s):
        fish_call = "indeterminate"

    # --------  IHC score 0-3  ----------
    if _PLUS3_PAT.search(s):
        ihc = 3
    else:
        m = _IHC_PAT.search(s)
        if m:
            ihc = max(int(g) for g in m.groups() if g)
        else:
            # Patterns like “0-1” → take the higher of the two
            r = _SLASH_RANGE_PAT.search(s)
            ihc = max(int(r.group(1)), int(r.group(2))) if r else np.nan

    return ihc, fish_call


# ------------------------------------------------------------------
def her2_status_from_components(ihc, fish):
    """
    Apply ASCO-CAP 2023 rules (+ ‘HER2-low’ convention).
    """
    if pd.isna(ihc) and pd.isna(fish):
        return pd.NA

    # 1. definite answers from IHC
    if ihc == 3:
        return "positive"
    if ihc in (0, 1):
        return "negative"

    # 2. IHC 2+ needs fish
    if ihc == 2:
        if fish == "positive":
            return "positive"
        if fish == "negative":
            return "low"
        return "equivocal"

    # 3. no IHC – rely on fish only
    if fish == "positive":
        return "positive"
    if fish == "negative":
        return "negative"

    return pd.NA


# ------------------------------------------------------------------
def add_her2_features(df: pd.DataFrame, source_col="Her2") -> pd.DataFrame:
    """
    Vectorised wrapper that adds all clean HER2 columns to *df*.
    """
    ihc, fish = zip(*df[source_col].apply(parse_her2_cell))
    df["her2_ihc_score"] = pd.to_numeric(ihc, errors="coerce")
    df["her2_fish_call"] = pd.Categorical(fish, categories=["positive", "negative", "indeterminate"])

    df["her2_status"] = [
        her2_status_from_components(i, f) for i, f in zip(df["her2_ihc_score"], df["her2_fish_call"])
    ]
    df["her2_status"] = df["her2_status"].astype("category")

    # Indicator flags
    for lab, col in {"positive": "her2_positive", "low": "her2_low", "equivocal": "her2_equivocal",
                     "negative": "her2_negative"}.items():
        df[col] = (df["her2_status"] == lab).astype("int8")

    # Remove status column (perhaps we'll add it back later)
    df.drop(columns=["her2_status"], inplace=True)

    return df

# Lymphatic penetration -------------------------------------------------------

def parse_l_stage(x: str | None) -> Optional[int]:
    if pd.isna(x):
        return np.nan
    m = re.search(r"l(\d)", str(x).lower())
    return int(m.group(1)) if m else np.nan

# TNM M & N -------------------------------------------------------------------

def parse_tnm_m(x: str | None) -> Tuple[Optional[int], Optional[str]]:
    if pd.isna(x):
        return np.nan, np.nan
    s = str(x).lower()
    if s.startswith("m0"):
        return 0, ""
    if s.startswith("m1"):
        return 1, s[2:]
    return np.nan, np.nan


def parse_tnm_n(x: str | None) -> Tuple[Optional[int], Optional[str]]:
    if pd.isna(x):
        return np.nan, np.nan
    s = str(x).lower()
    if s.startswith("nx"):
        return np.nan, np.nan
    m = re.match(r"n(\d)([a-z]{0,3})", s)
    return (int(m.group(1)), m.group(2) or "") if m else (np.nan, np.nan)

# Margin type (Hebrew) --------------------------------------------------------

def parse_margins(x: str | None) -> Optional[int]:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s in ("נקיים", "נקי", "נקיות"):
        return 0
    if s in ("נגועים", "מעורבים", "לא נקיים"):
        return 1
    return np.nan

# Side (Hebrew) ---------------------------------------------------------------
SIDE_MAP = {"שמאל": "left", "ימין": "right", "דו צדדי": "bilateral"}

def parse_side(x: str | None) -> str | pd.NA:
    if pd.isna(x):
        return pd.NA
    return SIDE_MAP.get(str(x).strip(), pd.NA)

# -----------------------------------------------------------------------------
# 2. Main cleaning orchestrator
# -----------------------------------------------------------------------------

COL_MAP = {
    " Form Name": "form_name",
    " Hospital": "hospital_code",
    "User Name": "user_name",
    "אבחנה-Age": "age",
    "אבחנה-Basic stage": "stage_basis",
    "אבחנה-Diagnosis date": "diagnosis_date",
    "אבחנה-Her2": "her2_raw",
    "אבחנה-Histological diagnosis": "histology",
    "אבחנה-Histopatological degree": "grade_raw",
    "אבחנה-Ivi -Lymphovascular invasion": "ivi_raw",
    "אבחנה-KI67 protein": "ki67_raw",
    "אבחנה-Lymphatic penetration": "l_penetration_raw",
    "אבחנה-M -metastases mark (TNM)": "m_raw",
    "אבחנה-N -lymph nodes mark (TNM)": "n_raw",
    "אבחנה-Margin Type": "margin_raw",
    "אבחנה-Nodes exam": "nodes_exam",
    "אבחנה-Positive nodes": "nodes_pos",
    "אבחנה-Side": "side_raw",
}


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned DataFrame ready for ML pipelines."""
    df = df.copy()
    df.columns = [
        col.strip() if isinstance(col, str) else col
        for col in df.columns
    ]

    # Rename and NULL‑standardise
    col_map = {k.strip(): v for k, v in COL_MAP.items()}
    for k in col_map:
        if k not in df.columns:
            print(f"Warning: Column '{k}' not found in DataFrame")
    df = df.rename(columns=col_map)
    for c in df.columns:
        df[c] = normalise_nulls(df[c])

    # Reporter split: `324_Onco` → (324, "Onco") :: commented out for now
    # user_split = df["user_name"].apply(lambda x: pd.Series(split_user_name(x)))
    # user_split.columns = ["reporter_id", "reporter_role"]
    # df = pd.concat([df, user_split], axis=1)

    # Simple casts
    df["diagnosis_date"] = pd.to_datetime(df["diagnosis_date"], errors="raise", format="%d/%m/%Y %H:%M")
    df["diagnosis_time"] = df["diagnosis_date"].dt.time
    df["diagnosis_date"] = df["diagnosis_date"].dt.date
    df["stage_basis"] = df["stage_basis"].apply(parse_stage_basis).astype("category")
    df["age"] = pd.to_numeric(df["age"], errors="coerce").round(1)

    # HER2, Ki‑67, grade, etc.
    add_her2_features(df, source_col="her2_raw")

    df["grade"] = df["grade_raw"].apply(parse_histgrade)
    df["lvi"] = df["ivi_raw"].apply(parse_ivi)
    df["ki67"] = df["ki67_raw"].apply(parse_ki67)
    df["l_stage"] = df["l_penetration_raw"].apply(parse_l_stage)

    m_stage, m_sub = zip(*df["m_raw"].apply(parse_tnm_m))
    df["m_stage"] = pd.to_numeric(m_stage, errors="coerce")
    df["m_sub"] = m_sub

    n_stage, n_det = zip(*df["n_raw"].apply(parse_tnm_n))
    df["n_stage"] = pd.to_numeric(n_stage, errors="coerce")
    df["n_detail"] = n_det

    df["margin_pos"] = df["margin_raw"].apply(parse_margins)
    df["side"] = df["side_raw"].apply(parse_side).astype("category")

    # Numeric coercions for simple counts
    for col in ("nodes_exam", "nodes_pos"): #, "reporter_id"
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop raw/unparsed helpers
    df.drop(columns=[c for c in df.columns if c.endswith("_raw") or c == "user_name"], inplace=True)

    return df

def parse_hormone_marker_column(series: pd.Series) -> pd.DataFrame:
    import re
    def parse_single(value):
        if pd.isna(value):
            return pd.Series(["Unclear", None, None], index=["status", "percentage", "intensity"])

        val = str(value).strip().lower()

        if any(k in val for k in ["negative", "-", "0%", "no", "none", "not expressed"]):
            status = "Negative"
        elif any(k in val for k in ["positive", "+", "1+", "2+", "3+", "strong", "moderate", "weak"]):
            status = "Positive"
        else:
            status = "Unclear"

        percent_match = re.search(r'(\d+)%', val)
        percentage = int(percent_match.group(1)) if percent_match else None

        if "strong" in val or "3+" in val:
            intensity = "Strong"
        elif "moderate" in val or "2+" in val:
            intensity = "Moderate"
        elif "weak" in val or "1+" in val or "w" in val:
            intensity = "Weak"
        else:
            intensity = None

        return pd.Series([status, percentage, intensity], index=["status", "percentage", "intensity"])

    return series.apply(parse_single)

def _preprocess_columns(Z: pd.DataFrame):
    DF = Z.copy()

    if "אבחנה-Stage" in DF.columns:
        DF["grade_letter"] = np.where(
            DF["אבחנה-Stage"] == "LA",
            "LA",
            DF["אבחנה-Stage"].str.slice(start=6)
        )
        DF["grade_number"] = pd.to_numeric(
            DF["אבחנה-Stage"].str.extract(r'(\d+)', expand=False),
            errors="coerce"
        )
        DF.drop(columns=["אבחנה-Stage"], inplace=True)
        DF["grade_letter"] = DF["grade_letter"].fillna(pd.NA)
        DF["grade_number"] = DF["grade_number"].fillna(pd.NA)
#T,U,V
    for col in ["Surgery date1", "Surgery date2", "Surgery date3"]:
        orig_col = f"אבחנה-{col}"
        if orig_col in DF.columns:
            DF[col] = pd.to_datetime(DF[orig_col], errors="coerce", dayfirst=True)
            DF[col] = DF[col].fillna(pd.NA)
            DF.drop(columns=[orig_col], inplace=True)


# --- One-hot encode surgery name columns (W, X, Y) ---
    surgery_columns = [
    "אבחנה-Surgery name1",  # W
    "אבחנה-Surgery name2",  # X
    "אבחנה-Surgery name3"  # Y
    ]

# Only encode if all surgery columns exist
    if all(col in DF.columns for col in surgery_columns):
        df_encoded = pd.get_dummies( DF[surgery_columns], prefix=["surgery1", "surgery2", "surgery3"],
            dtype=int )
    DF.drop(columns=surgery_columns, inplace=True)
    DF = pd.concat([DF, df_encoded], axis=1)

# Z - מס ניתוחים נשאר מס מי שהערך שלו ריק הופך ל0
    col = "אבחנה-Surgery sum"
    if col in DF.columns:
        DF["surgery_sum"] = pd.to_numeric(DF[col], errors="coerce").fillna(0).astype(int)
        DF.drop(columns=[col], inplace=True)

# --- One-hot encode אבחנה-T (TNM Tumor size category) into columns named T<value> ---
    if "אבחנה-T -Tumor mark (TNM)" in DF.columns:
        DF["אבחנה-T -Tumor mark (TNM)"] = DF["אבחנה-T -Tumor mark (TNM)"].astype(str).str.strip()
        t_dummies = pd.get_dummies(DF["אבחנה-T -Tumor mark (TNM)"], prefix='', prefix_sep='')
        t_dummies.columns = [f"T<{val}>" for val in t_dummies.columns]
        DF.drop(columns=["אבחנה-T -Tumor mark (TNM)"], inplace=True)
        DF = pd.concat([DF, t_dummies], axis=1)

    if "אבחנה-Tumor depth" in DF.columns:
        DF.drop(columns=["אבחנה-Tumor depth"], inplace=True)

    if "אבחנה-Tumor width" in DF.columns:
        DF.drop(columns=["אבחנה-Tumor width"], inplace=True)

# AE AD
    if "אבחנה-er" in DF.columns:
        parsed_er = parse_hormone_marker_column(DF["אבחנה-er"])
        parsed_er.columns = ["er_status", "er_percentage", "er_intensity"]
        DF = pd.concat([DF.drop(columns=["אבחנה-er"]), parsed_er], axis=1)

    if "אבחנה-pr" in DF.columns:
        parsed_pr = parse_hormone_marker_column(DF["אבחנה-pr"])
        parsed_pr.columns = ["pr_status", "pr_percentage", "pr_intensity"]
        DF = pd.concat([DF.drop(columns=["אבחנה-pr"]), parsed_pr], axis=1)

        # --- טיפול בעמודה: "surgery before or after-Activity date" ---
    if "surgery before or after-Activity date" in DF.columns:
        DF["surgery_activity_date"] = pd.to_datetime(
            DF["surgery before or after-Activity date"],
            errors="coerce"
        ).dt.date

        DF["surgery_activity_date"] = DF["surgery_activity_date"].where(
            pd.notna(DF["surgery_activity_date"]), pd.NA
        )

        DF.drop(columns=["surgery before or after-Activity date"], inplace=True)

    if "surgery before or after-Actual activity" in DF.columns:
        # נמיר את הערכים על פי אפשרויות קבועות או ניצור תווך למילוי חסר
        surgery_map = {
            "כיר-לאפ-הוצ טבעת/שנוי מי": "laparoscopic ring removal",
            "כירו-שד-למפקטומי+בלוטות": "lumpectomy + nodes",
            "כירו-שד-מסטקטומי+בלוטות": "mastectomy + nodes",
            "כירורגיה-שד למפקטומי": "lumpectomy",
            "שד-כריתה בגישה זעירה+בלוטות": "mastectomy (minimally invasive) + nodes",
            "כירו-שד-למפקטומי+בלוטות+קרינה תוך ניתוחית (intrabeam)": "lumpectomy + nodes + intrabeam",
            "שד-כריתה בגישה זעירה דרך העטרה": "mastectomy - minimal via areola",
            "כירור-הוצאת בלוטות לימפה": "lymph node removal",
            "כיר-שד-הוצ.בלוטות בית שח": "armpit lymph removal",
            "כירורגיה-שד מסטקטומי": "mastectomy"
        }

        DF["surgery before or after-Actual activity"] = DF[
            "surgery before or after-Actual activity"].map(surgery_map)

        # אם יש ערכים חסרים או ערכים שלא מצאנו במפה, נמלא ב-pd.NA
        DF["surgery before or after-Actual activity"].fillna(pd.NA, inplace=True)

        # נמחק את העמודה המקורית
        DF.drop(columns=["surgery before or after-Actual activity"], inplace=True)
    return DF



# -----------------------------------------------------------------------------
# 3. Quick test utility --------------------------------------------------------
# -----------------------------------------------------------------------------

def _selftest(path: str):
    df = pd.read_csv(path, dtype=str)
    print("Loaded", df.shape)
    df_clean = clean_dataframe(df)
    return df_clean

def drop_columns_with_many_nans(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """
    Remove columns with more than `threshold` fraction of NaN values.
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    threshold : float
        Fraction of allowed NaNs (e.g., 0.5 means drop columns with >50% NaN).
    Returns
    -------
    pd.DataFrame
        DataFrame with columns dropped.
    """
    nan_frac = df.isna().mean()
    cols_to_drop = nan_frac[nan_frac > threshold].index
    return df.drop(columns=cols_to_drop)

def print_column_info(df: pd.DataFrame):
    """
    Print information about each column in the DataFrame.
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    """
    for col in df.columns:
        print(f"Column: {col}\nUnique values: {df[col].unique()}\n")

def impute(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["ki67", "her2_ihc_score", "her2_fish_call"]:
        df[f"{col}_was_missing"] = df[col].isna().astype("int8")
    return df

def preprocess(df: pd.DataFrame, save_to_csv=False, out_name="") -> pd.DataFrame:
    """
    Preprocess the DataFrame by cleaning and transforming it.
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame.
    """
    df = clean_dataframe(df)
    df = _preprocess_columns(df)
    df = drop_columns_with_many_nans(df, threshold=0.85)
    df = impute(df)
    if save_to_csv:
        df.to_csv(out_name + ".csv" if out_name != "" else "preprocessed_data.csv", index=False)
        print("Preprocessed data saved to 'preprocessed_data.csv'")
    else:
        print("Preprocessing complete, DataFrame ready for use.")
    return df

def prepare_features(X):
    """
    Preprocesses the input DataFrame by converting columns based on their names and data types.
    - Columns containing 'date' in their name are parsed as datetime objects with the format '%d/%m/%Y'.
    - Columns containing 'time' in their name are parsed as datetime objects with the format '%H:%M'.
    - Other columns are attempted to be converted to numeric types. If conversion fails, they are cast to categorical type.
    Parameters:
        X (pd.DataFrame): The input DataFrame to preprocess.
    Returns:
        pd.DataFrame: A copy of the input DataFrame with processed feature columns.
    """

    X = X.copy()
    for col in X.columns:
        if col.find('date') >= 0:
            X[col] = pd.to_datetime(X[col], errors='coerce', format='%d/%m/%Y')
        elif col.find('time') >= 0:
            X[col] = pd.to_datetime(X[col], errors='coerce', format='%H:%M')
        else:
            try:
                X[col] = pd.to_numeric(X[col], errors='raise')
            except ValueError:
                # If conversion fails, treat it as categorical
                X[col] = X[col].astype('category')
    return X