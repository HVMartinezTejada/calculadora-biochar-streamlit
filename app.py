# VERSIÓN ENERO 02 2026 4:55PM  (fix StreamlitValueBelowMinError persistente por session_state)
import os
import io
import re
import warnings
import unicodedata
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================

APP_VERSION = "v3.4 (QC + XGBoost robusto + PDF→Google Sheets + Secrets auto + test RW)"
DEFAULT_EXCEL_PATH = "Biochar_Prescriptor_Sistema_Completo_v1.0.xlsx"

st.set_page_config(page_title="Prescriptor Híbrido Biochar", layout="wide", page_icon="🌱")

st.markdown("""
<style>
    .main-header { font-size: 2.4rem; color: #2E7D32; margin-bottom: 0.3rem; }
    .sub-header  { color: #4b5563; margin-top: -0.6rem; margin-bottom: 1rem; }
    .expert-box {
        background: linear-gradient(135deg, #e3f2fd, #f3e5f5);
        padding: 1.0rem; border-radius: 10px; border: 2px solid #5c6bc0; margin: 0.5rem 0 1rem 0;
    }
    .qc-box {
        background: #0b1220;
        padding: 1rem; border-radius: 12px; border: 1px solid #1f2a44; color: #e5e7eb;
        margin: 0.75rem 0;
    }
    .qc-pill { display:inline-block; padding: 0.18rem 0.6rem; border-radius: 999px; margin-right: 0.4rem; font-weight: 600; }
    .pill-ok { background: #065f46; color: #d1fae5; }
    .pill-warn { background: #92400e; color: #ffedd5; }
    .pill-bad { background: #7f1d1d; color: #fee2e2; }
    .floral-header {
        background: linear-gradient(135deg, #f9f0ff, #e6fffa);
        padding: 1.25rem; border-radius: 15px; border: 2px dashed #d63384; text-align: center; margin: 1rem 0;
    }
    .stButton > button { background-color: #4CAF50; color: white; font-weight: 700; border: none; border-radius: 6px; }
    .hint {
        background: #f3f4f6;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        color: #111827;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# UTILIDADES GENERALES
# =============================================================================

def strip_accents(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )

_EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)

def strip_emojis(s: Any) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    s = _EMOJI_RE.sub("", s)
    s = s.replace("•", " ").replace("·", " ").strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_category_value(v: Any) -> Any:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = strip_emojis(v).strip()
    return s if s else np.nan

def safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).strip().replace(",", ".")
        return float(s)
    except Exception:
        return default

# -------------------------------------------------------------------------
# (1) HELPER CLAMP (evita StreamlitValueBelowMinError / AboveMaxError)
# -------------------------------------------------------------------------
def clamp_float(x: Any, min_value: float, max_value: float, default: Optional[float] = None) -> float:
    v = safe_float(x, default=np.nan)
    if not np.isfinite(v):
        v = default if default is not None else min_value
    v = float(v)
    if v < min_value:
        v = float(min_value)
    if v > max_value:
        v = float(max_value)
    return float(v)

def clamp_int(x: Any, min_value: int, max_value: int, default: Optional[int] = None) -> int:
    v = safe_float(x, default=np.nan)
    if not np.isfinite(v):
        v = default if default is not None else min_value
    v = int(round(float(v)))
    if v < min_value:
        v = int(min_value)
    if v > max_value:
        v = int(max_value)
    return int(v)

# -------------------------------------------------------------------------
# FIX CRÍTICO:
# Streamlit usa st.session_state[key] si ya existía el widget.
# Si ese valor queda por debajo del min_value, explota.
# Estas funciones "sanitizan" el session_state ANTES de crear el widget.
# -------------------------------------------------------------------------
def ensure_state_float_in_range(key: str, min_value: float, max_value: float, default: float) -> None:
    if key in st.session_state:
        v = safe_float(st.session_state.get(key), default=np.nan)
        if not np.isfinite(v):
            st.session_state[key] = float(default)
        else:
            st.session_state[key] = float(np.clip(float(v), float(min_value), float(max_value)))
    else:
        st.session_state[key] = float(np.clip(float(default), float(min_value), float(max_value)))

def ensure_state_int_in_range(key: str, min_value: int, max_value: int, default: int) -> None:
    if key in st.session_state:
        v = safe_float(st.session_state.get(key), default=np.nan)
        if not np.isfinite(v):
            st.session_state[key] = int(default)
        else:
            st.session_state[key] = int(np.clip(int(round(float(v))), int(min_value), int(max_value)))
    else:
        st.session_state[key] = int(np.clip(int(default), int(min_value), int(max_value)))

def safe_number_input(
    label: str,
    key: str,
    min_value: float,
    max_value: float,
    value: Any,
    step: float = 0.1,
    help: Optional[str] = None,
    format: Optional[str] = None,
) -> float:
    default = clamp_float(value, min_value, max_value, default=min_value)
    ensure_state_float_in_range(key, min_value, max_value, default)
    return st.number_input(
        label,
        min_value=float(min_value),
        max_value=float(max_value),
        value=float(st.session_state[key]),
        step=float(step),
        key=key,
        help=help,
        format=format
    )

def safe_int_input(
    label: str,
    key: str,
    min_value: int,
    max_value: int,
    value: Any,
    step: int = 1,
    help: Optional[str] = None,
) -> int:
    default = clamp_int(value, min_value, max_value, default=min_value)
    ensure_state_int_in_range(key, min_value, max_value, default)
    return st.number_input(
        label,
        min_value=int(min_value),
        max_value=int(max_value),
        value=int(st.session_state[key]),
        step=int(step),
        key=key,
        help=help,
    )

def norm_tamano(label: str) -> str:
    s = (label or "").strip().lower()
    s = strip_accents(s)

    if s.startswith("fino"):
        return "Fino"
    if s.startswith("medio"):
        return "Medio"
    if s.startswith("grueso"):
        return "Grueso"

    if "<" in s:
        return "Fino"
    if "-" in s:
        return "Medio"
    return "Medio"

def is_flor(cultivo: str) -> bool:
    s = (cultivo or "").lower()
    return any(k in s for k in ["rosa", "clavel", "crisant", "orquí", "orqui", "flor", "ornamental", "bulbo"])

def unique_sorted(values: List[Any]) -> List[str]:
    out = []
    seen = set()
    for v in values:
        v2 = clean_category_value(v)
        if isinstance(v2, float) and np.isnan(v2):
            continue
        if v2 not in seen:
            seen.add(v2)
            out.append(v2)
    return sorted(out, key=lambda x: strip_accents(x).lower())

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def extract_sheet_id(value: str) -> str:
    """
    Acepta ID directo o URL completa de Google Sheets y devuelve solo el sheet_id.
    """
    if not value:
        return ""
    s = str(value).strip()
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", s)
    if m:
        return m.group(1)
    return s

# =============================================================================
# LISTAS UI (fallback) + UTILIDAD DE OPCIONES DINÁMICAS
# =============================================================================

DEFAULT_FEEDSTOCKS = unique_sorted([
    "Madera", "Cáscara cacao", "Cáscara coco", "Cáscara café",
    "Paja trigo", "Paja arroz", "Bambú", "Estiércol", "Cáscara arroz",
    "Lodo papel", "Paja", "Turba", "Otro"
])

DEFAULT_TEXTURAS = unique_sorted([
    "Arenoso", "Arena", "Franco-arenoso", "Franco", "Franco-limoso",
    "Franco-arcilloso", "Franco-arcillo-arenoso", "Arcilloso",
    "Ultisol", "Chernozem", "Turba"
])

DEFAULT_ESTADOS_SUELO = unique_sorted([
    "Ácido", "Muy ácido", "Ligeramente ácido", "Ácido degradado", "Muy degradado",
    "Contaminado", "Moderado", "Seco", "Alcalino", "Salino-sódico", "Ácido extremo"
])

DEFAULT_OBJETIVOS = unique_sorted([
    "Fertilidad", "Remediación", "Resiliencia", "Secuestro", "Estructura", "Supresión patógenos"
])

DEFAULT_TAMANOS = unique_sorted([
    "<0.5 mm", "<1 mm", "<2 mm", "1-3 mm", "2-4 mm", "2-5 mm", "3-6 mm",
    "Fino", "Medio", "Grueso"
])

def get_dataset_categories(colname: str) -> List[str]:
    cats = st.session_state.get("dataset_cats", {})
    return cats.get(colname, [])

def ui_options(colname: str, defaults: List[str]) -> List[str]:
    ds = get_dataset_categories(colname)
    merged = unique_sorted((ds or []) + (defaults or []))
    if "Otro" in merged:
        merged = [x for x in merged if x != "Otro"] + ["Otro"]
    return merged

# =============================================================================
# (a) LECTURA ROBUSTA CSV + (b) NORMALIZACIÓN DE COLUMNAS
# =============================================================================

_CANON_COL_MAP = {
    # target
    "dosis_efectiva": "dosis_efectiva",
    "dosis": "dosis_efectiva",

    # suelo
    "ph": "ph",
    "ph_suelo": "ph",
    "phsuelo": "ph",
    "mo": "mo",
    "materia_organica": "mo",
    "materia_organica_pct": "mo",
    "cic": "CIC",
    "metales": "Metales",

    # biochar
    "t_pirolisis": "T_pirolisis",
    "t_pirólisis": "T_pirolisis",
    "temperatura_pirolisis": "T_pirolisis",
    "temperatura_pirólisis": "T_pirolisis",

    "ph_biochar": "pH_biochar",
    "bet": "Area_BET",
    "area_bet": "Area_BET",
    "área_bet": "Area_BET",
    "area_superficial_bet": "Area_BET",

    "tamaño_biochar": "Tamaño_biochar",
    "tamano_biochar": "Tamaño_biochar",
    "tamaño": "Tamaño_biochar",
    "tamano": "Tamaño_biochar",

    "feedstock": "Feedstock",
    "materia_prima": "Feedstock",

    # categóricas
    "estado_suelo": "Estado_suelo",
    "textura": "Textura",
    "objetivo": "Objetivo",
    "tipo": "Tipo",
    "cultivo": "Tipo",  # si llega como "Cultivo" lo mapeamos a "Tipo"

    # metadata
    "fuente": "Fuente",
}

def canonicalize_column_name(col: str) -> str:
    raw = str(col).strip()
    s = strip_accents(raw)
    s = s.replace("%", "_pct")
    s = s.replace(" ", "_").replace("-", "_").replace("/", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    key = s.lower()
    return _CANON_COL_MAP.get(key, raw)

def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [canonicalize_column_name(c) for c in df.columns]

    # coerciones numéricas esperadas
    for num_col in ["ph", "mo", "CIC", "Metales", "T_pirolisis", "pH_biochar", "Area_BET", "dosis_efectiva", "Sensibilidad_salinidad"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    # limpieza suave en categóricas típicas
    for cat_col in ["Feedstock", "Textura", "Estado_suelo", "Tamaño_biochar", "Objetivo", "Tipo", "Riego", "Clima", "Sistema_cultivo",
                    "Tipo_producto", "Objetivo_calidad", "Metodo_enfriamiento", "Fuente"]:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].apply(clean_category_value)

    return df

def detect_csv_separator(sample_text: str) -> str:
    lines = [ln for ln in sample_text.splitlines() if ln.strip()]
    if not lines:
        return ","
    first = lines[0]
    return ";" if first.count(";") > first.count(",") else ","

def robust_read_csv_from_upload(uploaded_file) -> pd.DataFrame:
    raw_bytes = uploaded_file.getvalue()
    encodings = ["utf-8-sig", "utf-8", "latin1"]
    last_err = None

    sep = ","
    for enc in encodings:
        try:
            preview = raw_bytes[:5000].decode(enc)
            sep = detect_csv_separator(preview)
            break
        except Exception as e:
            last_err = e

    for enc in encodings:
        try:
            text = raw_bytes.decode(enc)
            bio = io.StringIO(text)
            df = pd.read_csv(bio, sep=sep)
            return df
        except Exception as e:
            last_err = e

    raise ValueError(f"No se pudo leer el CSV (encoding/sep). Último error: {last_err}")

def split_features_and_metadata(df: pd.DataFrame, metadata_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metadata_cols = metadata_cols or []
    meta_present = [c for c in metadata_cols if c in df.columns]
    df_meta = df[meta_present].copy() if meta_present else pd.DataFrame(index=df.index)
    df_feat = df.drop(columns=meta_present, errors="ignore").copy()
    return df_feat, df_meta

def capture_dataset_categories(df_norm: pd.DataFrame) -> None:
    cats: Dict[str, List[str]] = st.session_state.get("dataset_cats", {})
    for col in ["Feedstock", "Textura", "Estado_suelo", "Objetivo", "Tamaño_biochar"]:
        if col in df_norm.columns:
            vals = unique_sorted(df_norm[col].dropna().tolist())
            cats[col] = vals
    st.session_state.dataset_cats = cats

# =============================================================================
# GOOGLE SHEETS (PERSISTENCIA)
# =============================================================================

try:
    import gspread
    from google.oauth2.service_account import Credentials
    _GS_AVAILABLE = True
except Exception:
    gspread = None
    Credentials = None
    _GS_AVAILABLE = False

def gs_get_client():
    """
    Requiere st.secrets["gcp_service_account"] con el JSON de la service account.
    """
    if not _GS_AVAILABLE:
        raise RuntimeError("Dependencias de Google Sheets no instaladas (gspread/google-auth).")

    if "gcp_service_account" not in st.secrets:
        raise RuntimeError("Falta st.secrets['gcp_service_account'] (service account JSON).")

    sa_info = st.secrets["gcp_service_account"]
    if isinstance(sa_info, str):
        import json as _json
        sa_info = _json.loads(sa_info)

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
    client = gspread.authorize(creds)
    return client

def gs_open_worksheet(sh, worksheet_name: str, rows: int = 200, cols: int = 40):
    try:
        return sh.worksheet(worksheet_name)
    except Exception:
        return sh.add_worksheet(title=worksheet_name, rows=str(rows), cols=str(cols))

@st.cache_data(show_spinner=False, ttl=120)
def gs_read_df(sheet_id: str, worksheet_name: str) -> pd.DataFrame:
    client = gs_get_client()
    sh = client.open_by_key(sheet_id)
    ws = gs_open_worksheet(sh, worksheet_name)
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()
    headers = values[0]
    rows = values[1:]
    if not headers:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame(columns=headers)
    df = pd.DataFrame(rows, columns=headers)
    return df

def gs_ensure_headers(ws, required_headers: List[str]) -> List[str]:
    headers = ws.row_values(1)
    if not headers:
        headers = required_headers
        ws.append_row(headers)
        return headers

    missing = [h for h in required_headers if h not in headers]
    if missing:
        headers_extended = headers + missing
        ws.add_cols(len(missing))
        ws.update("1:1", [headers_extended])
        headers = headers_extended
    return headers

def gs_append_row(sheet_id: str, worksheet_name: str, row_dict: Dict[str, Any], required_headers: List[str]) -> None:
    client = gs_get_client()
    sh = client.open_by_key(sheet_id)
    ws = gs_open_worksheet(sh, worksheet_name)

    headers = gs_ensure_headers(ws, required_headers)

    row = []
    for h in headers:
        v = row_dict.get(h, "")
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = ""
        row.append(str(v))
    ws.append_row(row, value_input_option="USER_ENTERED")

def gs_test_read_write(sheet_id: str, data_ws: str, test_ws: str = "healthcheck") -> Dict[str, Any]:
    client = gs_get_client()
    sh = client.open_by_key(sheet_id)

    ws_data = gs_open_worksheet(sh, data_ws)
    vals = ws_data.get_all_values()
    n_rows = max(0, len(vals) - 1) if vals else 0
    n_cols = len(vals[0]) if vals and len(vals) > 0 else 0

    ws_test = gs_open_worksheet(sh, test_ws)
    headers = gs_ensure_headers(ws_test, ["ping_timestamp", "note"])
    ws_test.append_row([now_iso(), "streamlit_ping_ok"], value_input_option="USER_ENTERED")

    last = ws_test.get_all_values()[-1] if ws_test.get_all_values() else []

    return {
        "data_rows": n_rows,
        "data_cols": n_cols,
        "test_ws": test_ws,
        "last_test_row": last,
    }

# =============================================================================
# PDF → extracción heurística (para revisión humana)
# =============================================================================

try:
    import pdfplumber
    _PDFPLUMBER_OK = True
except Exception:
    pdfplumber = None
    _PDFPLUMBER_OK = False

try:
    import PyPDF2
    _PYPDF2_OK = True
except Exception:
    PyPDF2 = None
    _PYPDF2_OK = False

_DOI_RE = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)
_DOI_FULL_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Z0-9]+$", re.IGNORECASE)

def extract_text_from_pdf(uploaded_pdf, max_pages: int = 6) -> str:
    raw = uploaded_pdf.getvalue()
    if _PDFPLUMBER_OK:
        text_parts = []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages]):
                t = page.extract_text() or ""
                if t.strip():
                    text_parts.append(t)
        return "\n".join(text_parts)

    if _PYPDF2_OK:
        reader = PyPDF2.PdfReader(io.BytesIO(raw))
        text_parts = []
        for i, page in enumerate(reader.pages[:max_pages]):
            t = page.extract_text() or ""
            if t.strip():
                text_parts.append(t)
        return "\n".join(text_parts)

    raise RuntimeError("No hay librerías para leer PDF (instala pdfplumber o PyPDF2).")

def _pick_first_number_in_range(nums: List[float], lo: float, hi: float) -> Optional[float]:
    for n in nums:
        if lo <= n <= hi:
            return float(n)
    return None

def infer_fields_from_text(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    text = text or ""

    m = _DOI_RE.search(text)
    if m:
        doi = m.group(1).strip().rstrip(").,;")
        out["doi"] = doi

    t_norm = strip_accents(text).lower()
    t_num = text.replace(",", ".")

    ref_type = "experimental"
    if any(k in t_norm for k in ["systematic review", "review article", "literature review", "revision bibliografica", "revisión bibliográfica"]):
        ref_type = "review"
    if any(k in t_norm for k in ["meta-analysis", "metaanalysis", "meta analysis", "metaanalálisis", "meta-analisis", "meta análisis"]):
        ref_type = "meta_analysis"
    if any(k in t_norm for k in ["case study", "caso de estudio"]):
        ref_type = "case_study"
    out["ref_type"] = ref_type

    m_bet = re.search(r"\bBET\b[^0-9]{0,40}(\d{1,5}(?:\.\d+)?)\s*(?:m2/g|m²/g|m\^2/g)", t_num, flags=re.IGNORECASE)
    if m_bet:
        out["Area_BET"] = safe_float(m_bet.group(1), default=np.nan)

    ph_matches = list(re.finditer(r"\bpH\b[^0-9]{0,12}(\d(?:\.\d+)?)", t_num, flags=re.IGNORECASE))
    for mm in ph_matches:
        val = safe_float(mm.group(1), default=np.nan)
        if not np.isfinite(val):
            continue

        win = strip_accents(t_num[max(0, mm.start() - 60): mm.end() + 60]).lower()

        bio_ctx = any(k in win for k in [
            "biochar", "char", "bc", "pyrol", "pirol", "carboniz", "hydrochar", "htc"
        ])
        soil_ctx = any(k in win for k in [
            "soil", "suelo", "substrat", "sustrat", "field", "campo", "pot", "maceta"
        ])

        if bio_ctx and ("pH_biochar" not in out):
            out["pH_biochar"] = float(val)
            continue

        if (soil_ctx or not bio_ctx) and ("ph" not in out):
            out["ph"] = float(val)
            continue

        if "ph" not in out:
            out["ph"] = float(val)
        elif "pH_biochar" not in out:
            out["pH_biochar"] = float(val)

    temps = [safe_float(x, np.nan) for x in re.findall(r"(\d{3,4})\s*°?\s*C", t_num, flags=re.IGNORECASE)]
    temps = [float(x) for x in temps if np.isfinite(x)]

    mT = re.search(r"(?:pyrolys(?:is|ed)?|pirol(?:isis|isis|izado)|carboniz|torref)[^0-9]{0,80}(\d{3,4})\s*°?\s*C",
                   t_num, flags=re.IGNORECASE)
    if mT:
        out["T_pirolisis"] = safe_float(mT.group(1), default=np.nan)
    else:
        mHTT = re.search(r"\bHTT\b[^0-9]{0,30}(\d{2,4}(?:\.\d+)?)\s*°?\s*C", t_num, flags=re.IGNORECASE)
        if mHTT:
            out["T_pirolisis"] = safe_float(mHTT.group(1), default=np.nan)
        else:
            out["T_pirolisis"] = _pick_first_number_in_range(temps, 300, 900)

    mD = re.search(r"(\d+(?:\.\d+)?)\s*(?:t\s*/\s*ha|ton\s*/\s*ha|t\s*ha-1|t\s*ha−1|t\s*ha\^-1|t\s*ha\s*-1)",
                   t_num, flags=re.IGNORECASE)
    if not mD:
        mD = re.search(r"(\d+(?:\.\d+)?)\s*(?:mg\s*/\s*ha|mg\s*ha-1|mg\s*ha−1)",
                       t_num, flags=re.IGNORECASE)
    if mD:
        out["dosis_efectiva"] = safe_float(mD.group(1), default=np.nan)

    mmo = re.search(r"(?:materia\s*organica|materia\s*orgánica|organic\s*matter|\bOM\b)[^0-9]{0,25}(\d+(?:\.\d+)?)\s*%?",
                    t_num, flags=re.IGNORECASE)
    if mmo:
        out["mo"] = safe_float(mmo.group(1), default=np.nan)

    def _grab_pct(patterns: List[str]) -> Optional[float]:
        for pat in patterns:
            mm = re.search(pat, t_num, flags=re.IGNORECASE)
            if mm:
                v = safe_float(mm.group(1), default=np.nan)
                if np.isfinite(v) and 0 <= v <= 100:
                    return float(v)
        return None

    hum = _grab_pct([
        r"(?:moisture|humidity|humedad)[^0-9]{0,25}(\d{1,3}(?:\.\d+)?)\s*%?",
        r"\bMC\b[^0-9]{0,15}(\d{1,3}(?:\.\d+)?)\s*%?",
    ])
    vol = _grab_pct([
        r"(?:volatile\s*matter|volatiles|volátiles)[^0-9]{0,25}(\d{1,3}(?:\.\d+)?)\s*%?",
        r"\bVM\b[^0-9]{0,15}(\d{1,3}(?:\.\d+)?)\s*%?",
    ])
    ash = _grab_pct([
        r"(?:ash|ceniz(?:a|as))[^\d]{0,25}(\d{1,3}(?:\.\d+)?)\s*%?",
    ])
    fc = _grab_pct([
        r"(?:fixed\s*carbon|carbono\s*fijo)[^0-9]{0,25}(\d{1,3}(?:\.\d+)?)\s*%?",
        r"\bFC\b[^0-9]{0,15}(\d{1,3}(?:\.\d+)?)\s*%?",
    ])

    if hum is not None:
        out["Humedad_total"] = hum
    if vol is not None:
        out["Volatiles"] = vol
    if ash is not None:
        out["Cenizas_biomasa"] = ash
    if fc is not None:
        out["Carbono_fijo"] = fc

    m_hc = re.search(r"\bH\s*/\s*C\b[^0-9]{0,15}(\d(?:\.\d+)?)", t_num, flags=re.IGNORECASE)
    if m_hc:
        out["H_C_ratio"] = safe_float(m_hc.group(1), default=np.nan)
    m_oc = re.search(r"\bO\s*/\s*C\b[^0-9]{0,15}(\d(?:\.\d+)?)", t_num, flags=re.IGNORECASE)
    if m_oc:
        out["O_C_ratio"] = safe_float(m_oc.group(1), default=np.nan)

    feed_map = [
        (["coconut", "coco"], "Cáscara coco"),
        (["cacao", "cocoa"], "Cáscara cacao"),
        (["coffee", "cafe", "café"], "Cáscara café"),
        (["wood", "madera"], "Madera"),
        (["manure", "estiércol", "estiercol"], "Estiércol"),
        (["bamboo", "bambu", "bambú"], "Bambú"),
        (["rice", "arroz"], "Cáscara arroz"),
        (["cassava", "yuca"], "Yuca (Cassava)"),
    ]
    for keys, label in feed_map:
        if any(k in t_norm for k in keys):
            out["Feedstock"] = label
            break

    return out

# =============================================================================
# EXCEL (OPCIONAL)
# =============================================================================

@st.cache_data(show_spinner=False)
def cargar_datos_excel(excel_path: str) -> Optional[Dict[str, pd.DataFrame]]:
    try:
        if not excel_path or not os.path.exists(excel_path):
            return None
        parametros_df = pd.read_excel(excel_path, sheet_name="Parametros")
        rangos_df = pd.read_excel(excel_path, sheet_name="Rangos_suelo")
        algoritmos_df = pd.read_excel(excel_path, sheet_name="Algoritmos")
        factores_df = pd.read_excel(excel_path, sheet_name="Factores_ajuste")
        metadatos_df = pd.read_excel(excel_path, sheet_name="Metadatos_completos")
        dosis_exp_df = pd.read_excel(excel_path, sheet_name="Dosis_experimental")
        return {
            "parametros": parametros_df,
            "rangos": rangos_df,
            "algoritmos": algoritmos_df,
            "factores": factores_df,
            "metadatos": metadatos_df,
            "dosis_exp": dosis_exp_df,
        }
    except Exception as e:
        st.error(f"Error cargando Excel: {e}")
        return None

# =============================================================================
# QC REPORT
# =============================================================================

@dataclass
class QCReport:
    score: float
    confidence_factor: float
    can_prescribe: bool
    flags: List[str]
    notes: List[str]

def qc_pill(score: float) -> Tuple[str, str]:
    if score >= 80:
        return ("pill-ok", "QC alto")
    if score >= 55:
        return ("pill-warn", "QC medio")
    return ("pill-bad", "QC bajo")

def compute_qc_report(p: Dict[str, Any]) -> QCReport:
    score = 100.0
    flags, notes = [], []

    o2_ppm = safe_float(p.get("O2_ppm"), default=np.nan)
    o2_temp = safe_float(p.get("O2_temp_exposicion"), default=25.0)
    enf = (p.get("Metodo_enfriamiento") or "Desconocido").lower()

    if np.isfinite(o2_ppm):
        if o2_ppm <= 2000:
            notes.append("O₂ bajo: buena condición para pirólisis controlada.")
        elif o2_ppm <= 10000:
            score -= 10
            flags.append("Riesgo oxidación (O₂ medio)")
            notes.append("O₂ medio: posible oxidación parcial / variabilidad.")
        else:
            score -= 30
            flags.append("Riesgo oxidación alto (O₂ alto)")
            notes.append("O₂ alto: probable desviación a combustión/gasificación parcial.")
    else:
        score -= 6
        flags.append("O₂ no reportado")
        notes.append("Sin O₂ medido: aumenta incertidumbre de pirólisis real.")

    if o2_temp >= 100 and np.isfinite(o2_ppm) and o2_ppm > 2000:
        score -= 18
        flags.append("Exposición a O₂ a alta T")
        notes.append("Exposición a oxígeno a T>100°C aumenta oxidación y degrada estabilidad.")

    if "cámara inerte" in enf or "inerte" in enf or "contacto indirecto" in enf:
        notes.append("Enfriamiento controlado: preserva mejor estructura.")
    elif "aire" in enf:
        score -= 10
        flags.append("Enfriamiento en aire")
        notes.append("Aire durante enfriamiento puede oxidar superficialmente.")
    elif "agua" in enf:
        score -= 22
        flags.append("Apagado con agua")
        notes.append("Agua directa eleva variabilidad y oxidación; reduce confiabilidad.")
    else:
        score -= 6
        flags.append("Enfriamiento no especificado")
        notes.append("Método de enfriamiento desconocido: aumenta incertidumbre.")

    hum = safe_float(p.get("Humedad_total"), default=np.nan)
    vol = safe_float(p.get("Volatiles"), default=np.nan)
    ash = safe_float(p.get("Cenizas_biomasa"), default=np.nan)
    fc  = safe_float(p.get("Carbono_fijo"), default=np.nan)

    if all(np.isfinite(v) for v in [hum, vol, ash, fc]):
        cierre = hum + vol + ash + fc
        if abs(cierre - 100) > 8:
            score -= 10
            flags.append("Cierre proximal inconsistente")
            notes.append(f"Cierre proximal={cierre:.1f}%: revisar base seca/húmeda o datos.")
        else:
            notes.append(f"Cierre proximal OK ({cierre:.1f}%).")

    if np.isfinite(ash):
        if ash > 20:
            score -= 12
            flags.append("Cenizas altas")
            notes.append("Cenizas muy altas: riesgo de sales/pH alto; cuidado en cultivos sensibles.")
        elif ash > 12:
            score -= 6
            flags.append("Cenizas moderadas-altas")
            notes.append("Cenizas moderadas: monitorear EC/pH, especialmente en floricultura.")

    hc = safe_float(p.get("H_C_ratio"), default=np.nan)
    oc = safe_float(p.get("O_C_ratio"), default=np.nan)
    if np.isfinite(hc):
        if hc < 0.7:
            notes.append("H/C bajo: mayor aromaticidad (mejor estabilidad).")
        else:
            score -= 8
            flags.append("H/C alto")
            notes.append("H/C alto: menor aromaticidad; posible menor estabilidad.")
    if np.isfinite(oc):
        if oc < 0.2:
            notes.append("O/C bajo: indicador tendencial de alta estabilidad.")
        else:
            score -= 8
            flags.append("O/C alto")
            notes.append("O/C alto: posible menor estabilidad / mayor reactividad.")

    T = safe_float(p.get("T_pirolisis"), default=np.nan)
    if np.isfinite(T):
        if T < 350:
            score -= 15
            flags.append("T baja")
            notes.append("T<350°C: mayor fracción volátil/variabilidad; ojo con estabilidad.")
        elif T > 750:
            score -= 6
            flags.append("T muy alta")
            notes.append("T muy alta puede penalizar algunas funciones agronómicas según feedstock.")
    else:
        score -= 4
        flags.append("T no reportada")
        notes.append("Sin T reportada: aumenta incertidumbre.")

    score = float(np.clip(score, 0, 100))

    can_prescribe = True
    if ("Riesgo oxidación alto (O₂ alto)" in flags) and ("Apagado con agua" in flags):
        can_prescribe = False

    confidence_factor = float(np.clip(score / 85.0, 0.65, 1.15))

    return QCReport(
        score=score,
        confidence_factor=confidence_factor,
        can_prescribe=can_prescribe,
        flags=flags,
        notes=notes
    )

# =============================================================================
# REGLAS DETERMINÍSTICAS
# =============================================================================

def parse_coeficientes(coef_str: Any) -> Dict[str, float]:
    coefs = {}
    if coef_str is None or (isinstance(coef_str, float) and np.isnan(coef_str)):
        return coefs
    for item in str(coef_str).split(","):
        if ":" in item:
            k, v = item.strip().split(":")
            coefs[k.strip()] = safe_float(v, default=0.0)
    return coefs

def aplicar_factores_excel(factores_df: pd.DataFrame, objetivo: str, suelo: dict, biochar: dict, cultivo: dict) -> float:
    if factores_df is None or factores_df.empty:
        return 1.0

    df = factores_df.copy()
    cols = {c.lower(): c for c in df.columns}
    def col(name): return cols.get(name.lower())

    c_obj = col("objetivo")
    c_var = col("variable") or col("parametro") or col("nombre")
    c_min = col("min") or col("desde")
    c_max = col("max") or col("hasta")
    c_fac = col("factor") or col("multiplicador")
    c_app = col("aplica_a") or col("grupo") or col("categoria")

    if not (c_var and c_fac):
        return 1.0

    if c_obj:
        df = df[df[c_obj].astype(str).str.strip().str.lower() == str(objetivo).strip().lower()]

    f_total = 1.0
    for _, r in df.iterrows():
        var = str(r[c_var]).strip()
        factor = safe_float(r[c_fac], default=1.0)

        val = None
        if c_app:
            app = str(r[c_app]).strip().lower()
            if "suelo" in app:
                val = suelo.get(var)
            elif "biochar" in app:
                val = biochar.get(var)
            elif "cultivo" in app:
                val = cultivo.get(var)
            else:
                val = suelo.get(var, biochar.get(var, cultivo.get(var)))
        else:
            val = suelo.get(var, biochar.get(var, cultivo.get(var)))

        if val is None:
            continue

        v = safe_float(val, default=np.nan)
        if not np.isfinite(v):
            continue

        ok = True
        if c_min:
            vmin = safe_float(r[c_min], default=-np.inf)
            if np.isfinite(vmin) and v < vmin:
                ok = False
        if c_max:
            vmax = safe_float(r[c_max], default=np.inf)
            if np.isfinite(vmax) and v >= vmax:
                ok = False

        if ok and np.isfinite(factor) and factor > 0:
            f_total *= factor

    return float(np.clip(f_total, 0.2, 5.0))

def calcular_dosis_deterministica(objetivo: str, suelo: dict, biochar: dict, cultivo: dict, datos_excel: Optional[Dict[str, pd.DataFrame]]) -> float:
    if datos_excel is None:
        base = 20.0
        ph = safe_float(suelo.get("pH"), 6.5)
        mo = safe_float(suelo.get("MO"), 2.0)

        if str(objetivo).lower().startswith("fert"):
            base += 1.5 * (6.5 - ph) + 2.0 * (3.0 - mo)
        if str(objetivo).lower().startswith("rem"):
            met = safe_float(suelo.get("Metales"), 0.0)
            base += 0.03 * met

        t = norm_tamano(biochar.get("Tamaño", "Medio"))
        if t == "Fino":
            base *= 1.15
        elif t == "Grueso":
            base *= 0.9

        return float(np.clip(base, 5, 50))

    algoritmos = datos_excel.get("algoritmos")
    factores = datos_excel.get("factores")

    if algoritmos is None or algoritmos.empty:
        return 20.0

    algo_row = algoritmos[algoritmos["Objetivo"].astype(str).str.strip().str.lower() == str(objetivo).strip().lower()]
    if algo_row.empty:
        return 20.0

    constante = safe_float(algo_row["Constante"].iloc[0], default=20.0)
    coef_str = algo_row["Coeficientes"].iloc[0] if "Coeficientes" in algo_row.columns else None
    coefs = parse_coeficientes(coef_str)

    dosis_base = constante
    for k, a in coefs.items():
        val = suelo.get(k, biochar.get(k, cultivo.get(k)))
        v = safe_float(val, default=np.nan)
        if np.isfinite(v):
            dosis_base += a * v

    f_total = aplicar_factores_excel(factores, objetivo, suelo, biochar, cultivo)

    t = norm_tamano(biochar.get("Tamaño", "Medio"))
    if t == "Fino":
        f_total *= 1.15
    elif t == "Grueso":
        f_total *= 0.90

    return float(np.clip(dosis_base * f_total, 5, 50))

def calcular_dosis_flores(suelo: dict, biochar: dict, flor: dict) -> float:
    dosis_base = 12.0

    sistema = flor.get("Sistema_cultivo", "Campo abierto")
    factores_sistema = {"Campo abierto": 1.0, "Invernadero": 0.8, "Maceta/Contenedor": 0.6, "Hidroponía": 0.4}
    factor_sistema = factores_sistema.get(sistema, 1.0)

    ph = safe_float(suelo.get("pH"), 6.5)
    if ph < 5.5 or ph > 7.0:
        dosis_base += abs(6.0 - ph) * 1.2

    tipo_producto = flor.get("Tipo_producto", "Flores cortadas")
    if tipo_producto == "Plantas en maceta":
        factor_sistema *= 0.9
    elif tipo_producto == "Bulbos":
        dosis_base += 3.0

    sensibilidad = safe_float(flor.get("Sensibilidad_salinidad"), 1.5)
    cenizas = safe_float(biochar.get("Cenizas_biomasa"), np.nan)
    if sensibilidad > 2.0 and np.isfinite(cenizas) and cenizas > 12:
        dosis_base *= 0.8

    objetivo_calidad = flor.get("Objetivo_calidad", "Larga vida en florero")
    if objetivo_calidad == "Larga vida en florero":
        dosis_base += 2.0
    elif objetivo_calidad == "Color intenso":
        dosis_base += 1.5

    return float(np.clip(dosis_base * factor_sistema, 5, 30))

# =============================================================================
# CALCULADORAS (Ingeniería) — versión educativa
# =============================================================================

def calcular_balance_masa_energia(feedstock: str, humedad: float, temperatura: float, tiempo_residencia_min: float) -> Dict[str, float]:
    factores_biomasa = {
        "Madera":        {"pci": 18.5, "yield_char": 0.25, "yield_oil": 0.40},
        "Cáscara cacao": {"pci": 17.8, "yield_char": 0.28, "yield_oil": 0.38},
        "Paja trigo":    {"pci": 15.2, "yield_char": 0.22, "yield_oil": 0.42},
        "Bambú":         {"pci": 19.1, "yield_char": 0.30, "yield_oil": 0.35},
        "Estiércol":     {"pci": 14.5, "yield_char": 0.35, "yield_oil": 0.30},
    }
    f = factores_biomasa.get(feedstock, {"pci": 17.0, "yield_char": 0.25, "yield_oil": 0.40})

    biomasa_seca_pct = float(np.clip(100.0 - humedad, 0, 100))
    y_char = 100.0 * f["yield_char"]
    y_oil = 100.0 * f["yield_oil"]
    y_gas = float(np.clip(100.0 - y_char - y_oil, 0, 100))

    energia = 1.5 * max(0, (temperatura - 20)) * (tiempo_residencia_min / 60.0) / 100.0

    return {
        "biomasa_seca_pct": biomasa_seca_pct,
        "rend_biochar_pct": y_char,
        "rend_biooil_pct": y_oil,
        "rend_gas_pct": y_gas,
        "energia_MJ_kg": energia,
        "pci_MJ_kg_seco": f["pci"],
    }

def calcular_secuestro_carbono(feedstock: str, temperatura: float, rendimiento_biochar_pct: float, carbono_inicial_pct: float = 48.0) -> Dict[str, float]:
    if temperatura < 400:
        f_temp = 0.6
        vida = 50
    elif temperatura < 550:
        f_temp = 0.8
        vida = 100
    else:
        f_temp = 0.9
        vida = 500

    factores_feedstock = {"Madera": 0.85, "Cáscara cacao": 0.78, "Paja trigo": 0.72, "Bambú": 0.88, "Estiércol": 0.65}
    f_feed = factores_feedstock.get(feedstock, 0.80)

    carbono_retenido_pct = carbono_inicial_pct * (rendimiento_biochar_pct / 100.0) * f_feed * f_temp
    sec_C_ton_por_ton_biomasa = carbono_retenido_pct / 100.0
    return {
        "carbono_retenido_pct": float(carbono_retenido_pct),
        "secuestro_tC_t": float(sec_C_ton_por_ton_biomasa),
        "vida_media_anios": float(vida),
    }

# =============================================================================
# ML — ENTRENAMIENTO (robusto por ESQUEMA)
# =============================================================================

TARGET_COL = "dosis_efectiva"

SCHEMA_NUM_COLS = [
    "ph", "mo", "CIC", "Metales",
    "T_pirolisis", "pH_biochar", "Area_BET",
    "Sensibilidad_salinidad",
    "Humedad_total", "Volatiles", "Cenizas_biomasa", "Carbono_fijo",
    "O2_ppm", "O2_temp_exposicion", "H_C_ratio", "O_C_ratio",
]

META_COLS = [
    "Fuente","Fuente_raw","doi","ref_type","doi_format_ok","doi_url","ref_id","ref_quality",
    "verification_status","verified_title","verified_journal","verified_year","verified_authors",
    "verification_notes","Fuente_display","Fuente_status","Fuente_public",
    "ingest_timestamp","pdf_filename"
]

def entrenar_modelo_xgb_pipeline(df_raw: pd.DataFrame, target: str = TARGET_COL) -> Tuple[Pipeline, float, List[str], pd.DataFrame]:
    if target not in df_raw.columns:
        raise ValueError(f"El dataset debe incluir la columna '{target}'")

    df = df_raw.copy()

    if "verification_status" in df.columns:
        vs = df["verification_status"].astype(str).str.strip().str.lower()
        df = df[vs == "user_confirmed"].copy()

    if "ref_quality" in df.columns:
        rq = df["ref_quality"].astype(str).str.strip().str.lower()
        df = df[rq != "inferred"].copy()
    if "Fuente_status" in df.columns:
        fs = df["Fuente_status"].astype(str).str.strip().str.lower()
        df = df[fs != "inferred"].copy()

    df_feat, df_meta = split_features_and_metadata(df, metadata_cols=META_COLS)

    y = pd.to_numeric(df_feat[target], errors="coerce")
    keep = y.notna()
    df_feat = df_feat.loc[keep].copy()
    df_meta = df_meta.loc[keep].copy()
    y = y.loc[keep].copy()

    if len(df_feat) < 10:
        raise ValueError("Dataset insuficiente (post-filtros): se requieren al menos 10 filas VALIDADAS con dosis_efectiva válida.")

    X = df_feat.drop(columns=[target]).copy()
    expected_cols = list(X.columns)

    num_cols = [c for c in expected_cols if c in SCHEMA_NUM_COLS]
    cat_cols = [c for c in expected_cols if c not in num_cols]

    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in cat_cols:
        X[c] = X[c].apply(clean_category_value).astype(object)

    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=450,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    if len(Xtr) == 0 or len(Xte) == 0:
        raise ValueError("Split inválido: revisa tamaño del dataset para holdout.")

    pipe.fit(Xtr, ytr)
    r2 = r2_score(yte, pipe.predict(Xte))

    return pipe, float(r2), expected_cols, df_meta

def build_flat_features_for_model(suelo: dict, biochar: dict, cultivo_d: dict, objetivo: str) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    flat["ph"] = safe_float(suelo.get("pH"), np.nan)
    flat["mo"] = safe_float(suelo.get("MO"), np.nan)
    flat["CIC"] = safe_float(suelo.get("CIC"), np.nan)
    flat["Metales"] = safe_float(suelo.get("Metales"), np.nan)
    flat["Textura"] = clean_category_value(suelo.get("Textura"))
    flat["Estado_suelo"] = clean_category_value(suelo.get("Estado_suelo"))

    flat["Feedstock"] = clean_category_value(biochar.get("Feedstock"))
    flat["T_pirolisis"] = safe_float(biochar.get("T_pirolisis"), np.nan)
    flat["pH_biochar"] = safe_float(biochar.get("pH_biochar"), np.nan)
    flat["Area_BET"] = safe_float(biochar.get("BET"), np.nan)
    flat["Tamaño_biochar"] = clean_category_value(biochar.get("Tamaño_biochar"))
    flat["Objetivo"] = clean_category_value(objetivo)

    flat["Tipo"] = clean_category_value(cultivo_d.get("Tipo"))
    flat["Riego"] = clean_category_value(cultivo_d.get("Riego"))
    flat["Clima"] = clean_category_value(cultivo_d.get("Clima"))
    flat["Sistema_cultivo"] = clean_category_value(cultivo_d.get("Sistema_cultivo"))
    flat["Tipo_producto"] = clean_category_value(cultivo_d.get("Tipo_producto"))
    flat["Objetivo_calidad"] = clean_category_value(cultivo_d.get("Objetivo_calidad"))
    flat["Sensibilidad_salinidad"] = safe_float(cultivo_d.get("Sensibilidad_salinidad"), np.nan)

    flat["Humedad_total"] = safe_float(biochar.get("Humedad_total"), np.nan)
    flat["Volatiles"] = safe_float(biochar.get("Volatiles"), np.nan)
    flat["Cenizas_biomasa"] = safe_float(biochar.get("Cenizas_biomasa"), np.nan)
    flat["Carbono_fijo"] = safe_float(biochar.get("Carbono_fijo"), np.nan)
    flat["O2_ppm"] = safe_float(biochar.get("O2_ppm"), np.nan)
    flat["O2_temp_exposicion"] = safe_float(biochar.get("O2_temp_exposicion"), np.nan)
    flat["Metodo_enfriamiento"] = clean_category_value(biochar.get("Metodo_enfriamiento"))
    flat["H_C_ratio"] = safe_float(biochar.get("H_C_ratio"), np.nan)
    flat["O_C_ratio"] = safe_float(biochar.get("O_C_ratio"), np.nan)

    return flat

def preparar_input_modelo(expected_cols: List[str], params_flat: Dict[str, Any]) -> pd.DataFrame:
    row = {}
    for c in expected_cols:
        if c in SCHEMA_NUM_COLS:
            row[c] = safe_float(params_flat.get(c), np.nan)
        else:
            row[c] = clean_category_value(params_flat.get(c))
    return pd.DataFrame([row])

def compute_peso_xgb(r2v: float, qc_score: float) -> Tuple[float, str]:
    r2v = float(r2v)
    qc_score = float(qc_score)

    xp = np.array([-0.2, 0.0, 0.3, 0.5, 0.7, 0.85, 0.95])
    fp = np.array([0.10, 0.15, 0.25, 0.40, 0.60, 0.75, 0.80])
    w = float(np.interp(np.clip(r2v, xp.min(), xp.max()), xp, fp))

    if qc_score < 55:
        w *= 0.65
        qc_msg = "QC bajo → damos más peso a la regla (más conservadora)."
    elif qc_score < 70:
        w *= 0.85
        qc_msg = "QC medio → equilibramos regla y ML."
    else:
        qc_msg = "QC alto → el ML puede pesar más si su desempeño es bueno."

    w = float(np.clip(w, 0.15, 0.80))

    expl = (
        f"Peso ML = {w:.2f}. "
        f"Se calcula con (1) el desempeño del modelo (R²={r2v:.2f}) y (2) la confiabilidad del proceso (QC={qc_score:.0f}/100). "
        f"{qc_msg}"
    )
    return w, expl

# =============================================================================
# UI — Sidebar: Excel opcional + Google Sheets opcional + autoentreno
# =============================================================================

st.sidebar.markdown(f"###⚙️ Configuración ({APP_VERSION})")

excel_choice = st.sidebar.radio(
    "Base Excel (opcional)",
    ["Usar ruta por defecto", "Subir Excel", "No usar Excel"],
    index=0
)

excel_path = None
uploaded_excel = None

if excel_choice == "Usar ruta por defecto":
    excel_path = DEFAULT_EXCEL_PATH
elif excel_choice == "Subir Excel":
    uploaded_excel = st.sidebar.file_uploader("Sube el Excel del sistema", type=["xlsx"])
    if uploaded_excel is not None:
        tmp_path = os.path.join(".", "__tmp_biochar_system.xlsx")
        with open(tmp_path, "wb") as f:
            f.write(uploaded_excel.getbuffer())
        excel_path = tmp_path
else:
    excel_path = None

datos_excel = cargar_datos_excel(excel_path) if excel_path else None

st.sidebar.markdown("---")
st.sidebar.markdown("### 🗂️ Persistencia (Google Sheets)")

_secrets_gs = st.secrets.get("google_sheets", {})
DEFAULT_GS_SHEET_ID = extract_sheet_id(_secrets_gs.get("sheet_id", "")) if isinstance(_secrets_gs, dict) else ""
DEFAULT_GS_WORKSHEET = (_secrets_gs.get("worksheet", "data") if isinstance(_secrets_gs, dict) else "data")
DEFAULT_GS_TEST_WS = (_secrets_gs.get("test_worksheet", "healthcheck") if isinstance(_secrets_gs, dict) else "healthcheck")

gs_enabled = st.sidebar.checkbox(
    "Activar Google Sheets",
    value=bool(DEFAULT_GS_SHEET_ID),
    help="Guarda/lee la base desde una hoja para trabajo colaborativo."
)

gs_sheet_id_in = st.sidebar.text_input(
    "Sheet ID o URL",
    value=DEFAULT_GS_SHEET_ID,
    help="Pega el ID o el link del Google Sheet; la app extrae el ID automáticamente.",
    disabled=not gs_enabled
)
gs_sheet_id = extract_sheet_id(gs_sheet_id_in)

gs_worksheet = st.sidebar.text_input(
    "Worksheet (dataset)",
    value=DEFAULT_GS_WORKSHEET,
    help="Nombre de la pestaña dentro del Sheet.",
    disabled=not gs_enabled
)
gs_test_worksheet = st.sidebar.text_input(
    "Worksheet (pruebas)",
    value=DEFAULT_GS_TEST_WS,
    help="Pestaña usada para test de escritura (se crea si no existe).",
    disabled=not gs_enabled
)

auto_train = st.sidebar.checkbox(
    "Autoentrenar si hay ≥ N nuevas filas validadas",
    value=False,
    disabled=not gs_enabled,
)
N_AUTO = st.sidebar.number_input("N (umbral autoentreno)", min_value=1, max_value=100, value=10, step=1, disabled=(not gs_enabled))

with st.sidebar.expander("🧪 Probar conexión (lee + escribe)", expanded=False):
    if not gs_enabled:
        st.info("Activa Google Sheets para habilitar la prueba.")
    elif not _GS_AVAILABLE:
        st.error("Faltan dependencias: `gspread` y `google-auth` en requirements.txt.")
    elif not gs_sheet_id:
        st.warning("Ingresa un Sheet ID o URL.")
    else:
        if st.button("✅ Probar ahora", use_container_width=True):
            try:
                with st.spinner("Probando lectura/escritura..."):
                    out = gs_test_read_write(gs_sheet_id, gs_worksheet, gs_test_worksheet)
                st.success("Conexión OK ✅")
                st.write(f"Dataset ({gs_worksheet}): **{out['data_rows']}** filas, **{out['data_cols']}** columnas.")
                st.write(f"Escritura en ({out['test_ws']}): última fila:", out["last_test_row"])
                st.caption("Tip: el test escribe SOLO en el worksheet de pruebas para no ensuciar tu dataset.")
            except Exception as e:
                st.error(f"Falló la prueba: {e}")
                st.caption("Revisa: (1) share del Sheet con la Service Account, (2) APIs habilitadas, (3) Secrets correctos.")

with st.sidebar.expander("Cómo configurar Google Sheets", expanded=False):
    st.markdown("""
- En **Secrets** debes tener:
  - `gcp_service_account` (JSON de la Service Account)
  - `google_sheets.sheet_id` (ID del Sheet)
  - `google_sheets.worksheet` (p. ej. `data`)
- Recuerda: el Sheet debe estar **compartido** con el email de la Service Account como **Editor**.
    """)

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="main-header">🧬 Prescriptor Híbrido Biochar</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">QC + reglas determinísticas + XGBoost + PDF→Base</div>', unsafe_allow_html=True)
with col2:
    try:
        st.image("logonanomof.png", width=240)
    except Exception:
        st.markdown("### **NanoMof**")

# =============================================================================
# SESSION STATE
# =============================================================================

if "modelo_pipe" not in st.session_state:
    st.session_state.modelo_pipe = None
if "modelo_activo" not in st.session_state:
    st.session_state.modelo_activo = False
if "r2_score" not in st.session_state:
    st.session_state.r2_score = 0.0
if "expected_cols" not in st.session_state:
    st.session_state.expected_cols = []
if "train_metadata" not in st.session_state:
    st.session_state.train_metadata = pd.DataFrame()
if "parametros_usuario" not in st.session_state:
    st.session_state.parametros_usuario = {}
if "dataset_cats" not in st.session_state:
    st.session_state.dataset_cats = {}
if "last_train_confirmed" not in st.session_state:
    st.session_state.last_train_confirmed = 0
if "pdf_extract" not in st.session_state:
    st.session_state.pdf_extract = {}

# =============================================================================
# Helper: cargar dataset desde Sheets (si está habilitado)
# =============================================================================

def load_dataset_from_gs() -> pd.DataFrame:
    if not gs_enabled:
        return pd.DataFrame()
    if not gs_sheet_id or not gs_worksheet:
        return pd.DataFrame()
    df = gs_read_df(gs_sheet_id, gs_worksheet)
    if df is None or df.empty:
        return pd.DataFrame()
    df = normalize_dataframe_columns(df)
    capture_dataset_categories(df)
    return df

def count_confirmed_rows(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    if "verification_status" in df.columns:
        return int((df["verification_status"].astype(str).str.lower() == "user_confirmed").sum())
    return int(len(df))

if gs_enabled and auto_train and gs_sheet_id and gs_worksheet:
    try:
        df_gs = load_dataset_from_gs()
        confirmed = count_confirmed_rows(df_gs)
        delta = confirmed - int(st.session_state.last_train_confirmed)
        if delta >= int(N_AUTO) and confirmed >= 10 and (TARGET_COL in df_gs.columns):
            pipe, r2v, expected_cols, meta = entrenar_modelo_xgb_pipeline(df_gs, target=TARGET_COL)
            st.session_state.modelo_pipe = pipe
            st.session_state.r2_score = r2v
            st.session_state.expected_cols = expected_cols
            st.session_state.train_metadata = meta
            st.session_state.modelo_activo = True
            st.session_state.last_train_confirmed = confirmed
    except Exception:
        pass

auto_train_excel = st.sidebar.checkbox(
    "Autoentrenar al iniciar usando Dosis_experimental (Excel)",
    value=False,
    help="Usa la hoja Dosis_experimental si está presente y contiene dosis_efectiva."
)

if auto_train_excel and (not st.session_state.modelo_activo) and datos_excel is not None:
    try:
        df0 = datos_excel.get("dosis_exp")
        if df0 is not None and TARGET_COL in df0.columns and len(df0) >= 10:
            df0n = normalize_dataframe_columns(df0)
            capture_dataset_categories(df0n)
            pipe, r2v, expected_cols, meta = entrenar_modelo_xgb_pipeline(df0n, target=TARGET_COL)
            st.session_state.modelo_pipe = pipe
            st.session_state.r2_score = r2v
            st.session_state.expected_cols = expected_cols
            st.session_state.train_metadata = meta
            st.session_state.modelo_activo = True
    except Exception:
        pass

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Prescripción Híbrida",
    "📊 Entrenamiento XGBoost",
    "📚 Base de Conocimiento",
    "⚗️ Ingeniería / QC",
    "📄 PDF → Base (Google Sheets)"
])

# =============================================================================
# TAB 1 — Prescripción
# =============================================================================

with tab1:
    st.header("Prescripción Personalizada")

    col_modo1, col_modo2, col_modo3 = st.columns([2, 3, 2])
    with col_modo2:
        modo_experto = st.checkbox(
            "🔬 ACTIVAR MODO EXPERTO (QC + proceso)",
            value=False,
            help="Activa inputs de O₂, enfriamiento, cierre proximal y ratios H/C y O/C."
        )

    if modo_experto:
        st.markdown("""
        <div class="expert-box">
        <b>Modo experto activo:</b> la prescripción se ajusta por QC (riesgo oxidación + estabilidad).
        </div>
        """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("🏜️ Suelo")
        ph_suelo = st.slider("pH del suelo", 3.5, 9.5, 6.5, 0.1)
        mo_suelo = st.slider("Materia Orgánica (%)", 0.1, 10.0, 2.0, 0.1)
        cic_suelo = st.slider("CIC (cmolc/kg)", 2.0, 50.0, 15.0, 0.5)

        textura = st.selectbox("Textura", ui_options("Textura", DEFAULT_TEXTURAS))
        estado_suelo = st.selectbox("Estado del suelo", ui_options("Estado_suelo", DEFAULT_ESTADOS_SUELO))

        metales = st.number_input("Metales pesados (mg/kg)", 0.0, 500.0, 0.0, 1.0)

    with c2:
        st.subheader("🌿 Biochar (básico)")
        feedstock = st.selectbox("Materia prima", ui_options("Feedstock", DEFAULT_FEEDSTOCKS))

        temp_pirolisis = st.slider("Temperatura pirólisis (°C)", 300, 900, 550, 10)
        ph_biochar = st.slider("pH del biochar", 5.0, 12.0, 9.0, 0.1)

        tamano_ds = st.selectbox("Tamaño partícula", ui_options("Tamaño_biochar", DEFAULT_TAMANOS))
        area_bet = st.slider("Área superficial BET (m²/g)", 10, 600, 300, 10)

        if modo_experto:
            st.markdown("---")
            st.subheader("⚗️ QC + Proceso (experto)")

            hum_total = st.slider("Humedad total biomasa (%)", 0.0, 60.0, 10.0, 0.5)
            volatiles = st.slider("Volátiles biomasa (%)", 0.0, 90.0, 70.0, 0.5)
            cenizas_biomasa = st.slider("Cenizas biomasa (%)", 0.0, 40.0, 3.0, 0.5)
            carbono_fijo = st.slider("Carbono fijo biomasa (%)", 0.0, 80.0, 17.0, 0.5)
            cierre = hum_total + volatiles + cenizas_biomasa + carbono_fijo
            st.caption(f"Cierre proximal = {cierre:.1f}% (ideal ~100%, revisar base seca vs húmeda)")

            o2_ppm = st.number_input("O₂ residual (ppm) en reactor/enfriamiento", 0, 50000, 5000, 250)
            o2_temp = st.slider("Temperatura durante posible exposición a O₂ (°C)", 20, 600, 120, 10)

            metodo_enfriamiento = st.selectbox(
                "Método de enfriamiento",
                ["Cámara inerte", "Contacto indirecto", "En aire", "Agua directa", "Desconocido"],
                index=0
            )

            st.markdown("**Estabilidad (si tienes elemental):**")
            hc_ratio = st.number_input("H/C (molar)", 0.0, 2.0, 0.65, 0.01)
            oc_ratio = st.number_input("O/C (molar)", 0.0, 1.5, 0.18, 0.01)

    with c3:
        st.subheader("🌱 Agronomía")
        objetivo = st.selectbox("Objetivo", ui_options("Objetivo", DEFAULT_OBJETIVOS))

        cultivo = st.selectbox("Cultivo", [
            "Teff", "Hortalizas", "Trigo", "Girasol", "Maíz", "Pasto",
            "Cacao", "Frijol", "Palma", "Sorgo", "Tomate", "Soja",
            "🌺 Rosas", "🌼 Claveles", "🌻 Crisantemos", "🌸 Orquídeas",
            "💐 Flores cortadas", "🌷 Flores ornamentales", "🌹 Flores de bulbo",
            "Otro"
        ])
        sistema_riego = st.selectbox("Riego", ["Gravedad", "Aspersión", "Goteo", "No irrigado"])
        clima = st.selectbox("Clima", ["Árido", "Semiárido", "Mediterráneo", "Tropical", "Templado"])

        es_flor = is_flor(cultivo)

        if es_flor:
            st.markdown("---")
            st.subheader("🌺 Flores")
            sistema_cultivo = st.radio("Sistema", ["Campo abierto", "Invernadero", "Maceta/Contenedor", "Hidroponía"], horizontal=True)
            tipo_producto_floral = st.selectbox("Tipo producto", ["Flores cortadas", "Plantas en maceta", "Bulbos", "Follaje ornamental"])
            colf1, colf2 = st.columns(2)
            with colf1:
                objetivo_calidad = st.selectbox("Objetivo calidad", [
                    "Larga vida en florero", "Color intenso", "Tamaño de flor", "Longitud de tallo", "Producción todo el año"
                ])
            with colf2:
                sensibilidad_salinidad = st.slider("Sensibilidad salinidad", 1.0, 3.0, 1.5, 0.1)
        else:
            sistema_cultivo = "Campo abierto"
            tipo_producto_floral = "No aplica"
            objetivo_calidad = "No aplica"
            sensibilidad_salinidad = 1.5

    suelo = {
        "pH": ph_suelo,
        "MO": mo_suelo,
        "CIC": cic_suelo,
        "Textura": textura,
        "Estado_suelo": estado_suelo,
        "Metales": metales
    }

    biochar = {
        "Feedstock": feedstock,
        "T_pirolisis": temp_pirolisis,
        "pH_biochar": ph_biochar,
        "BET": area_bet,
        "Tamaño_biochar": tamano_ds,
        "Tamaño": norm_tamano(tamano_ds),
    }

    cultivo_d = {
        "Tipo": strip_emojis(cultivo),
        "Riego": sistema_riego,
        "Clima": clima,
        "Sistema_cultivo": sistema_cultivo,
        "Tipo_producto": tipo_producto_floral,
        "Objetivo_calidad": objetivo_calidad,
        "Sensibilidad_salinidad": sensibilidad_salinidad,
    }

    if modo_experto:
        biochar.update({
            "Humedad_total": hum_total,
            "Volatiles": volatiles,
            "Cenizas_biomasa": cenizas_biomasa,
            "Carbono_fijo": carbono_fijo,
            "O2_ppm": o2_ppm,
            "O2_temp_exposicion": o2_temp,
            "Metodo_enfriamiento": metodo_enfriamiento,
            "H_C_ratio": hc_ratio,
            "O_C_ratio": oc_ratio,
        })

    st.session_state.parametros_usuario = {"suelo": suelo, "biochar": biochar, "cultivo": cultivo_d, "objetivo": objetivo, "modo_experto": modo_experto}

    qc = compute_qc_report(biochar)
    pill_class, pill_text = qc_pill(qc.score)
    st.markdown(f"""
    <div class="qc-box">
      <span class="qc-pill {pill_class}">{pill_text}</span>
      <span class="qc-pill pill-ok">Score: {qc.score:.0f}/100</span>
      <span class="qc-pill pill-warn">Factor confianza: {qc.confidence_factor:.2f}</span>
      {"<span class='qc-pill pill-bad'>GATE: NO PRESCRIBIR</span>" if not qc.can_prescribe else ""}
      <div style="margin-top:0.5rem; color:#cbd5e1;">
        {"<br>".join([f"• {n}" for n in qc.notes[:6]])}
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not qc.can_prescribe:
        st.error("GATE activado: con O₂ alto + apagado con agua, la app no recomienda dosis.")

    if st.button("🎯 Calcular Dosis Híbrida", type="primary", use_container_width=True, disabled=(not qc.can_prescribe)):
        with st.spinner("Calculando..."):
            if es_flor:
                dosis_det = calcular_dosis_flores(suelo, biochar, cultivo_d)
                tipo_det = "Especializado floricultura"
            else:
                dosis_det = calcular_dosis_deterministica(objetivo, suelo, biochar, cultivo_d, datos_excel)
                tipo_det = "Determinístico"

            dosis_det_qc = float(np.clip(dosis_det * qc.confidence_factor, 3, 60))

            dosis_xgb = None
            metodo = tipo_det
            peso_xgb = 0.0
            peso_expl = "Modelo inactivo."

            if st.session_state.modelo_activo and st.session_state.modelo_pipe is not None:
                try:
                    flat = build_flat_features_for_model(suelo, biochar, cultivo_d, objetivo)
                    df_in = preparar_input_modelo(st.session_state.expected_cols, flat)
                    dosis_xgb = float(st.session_state.modelo_pipe.predict(df_in)[0])
                    peso_xgb, peso_expl = compute_peso_xgb(st.session_state.r2_score, qc.score)
                except Exception as e:
                    st.warning(f"Predicción XGBoost no disponible: {e}")
                    dosis_xgb = None

            if dosis_xgb is not None:
                dosis_final = (1 - peso_xgb) * dosis_det_qc + peso_xgb * dosis_xgb
                metodo = f"Híbrido (Regla+QC + XGBoost | peso_ml={peso_xgb:.2f})"
                mostrar_xgb = True
            else:
                dosis_final = dosis_det_qc
                mostrar_xgb = False

            dosis_final = float(np.clip(dosis_final, 3, 60))

            st.markdown("---")

            if es_flor:
                st.markdown("### 🌸 **PRESCRIPCIÓN PARA FLORICULTURA**")
                cc1, cc2, cc3 = st.columns([1, 2, 1])
                with cc2:
                    st.markdown(f"""
                    <div class="floral-header">
                        <h1 style='color: #d63384;'>🌸 {dosis_final:.1f} t/Ha 🌸</h1>
                        <p style='color: #666;'><strong>Dosis recomendada para {cultivo}</strong></p>
                        <p style='color: #666;'>{metodo}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success(f"### 📊 Dosis Recomendada: **{dosis_final:.1f} t/Ha**")
                st.caption(metodo)

            r1, r2c, r3 = st.columns(3)
            with r1:
                st.metric("Regla (con QC)", f"{dosis_det_qc:.1f} t/Ha", delta=f"Base {dosis_det:.1f} × QC {qc.confidence_factor:.2f}")
            with r2c:
                if mostrar_xgb:
                    st.metric("XGBoost", f"{dosis_xgb:.1f} t/Ha", delta=f"R² (holdout): {st.session_state.r2_score:.3f}")
                else:
                    st.metric("XGBoost", "—", delta="Modelo inactivo")
            with r3:
                st.metric("QC Score", f"{qc.score:.0f}/100", delta=pill_text)

            st.markdown(f"<div class='hint'><b>¿Por qué el peso del ML fue {peso_xgb:.2f}?</b><br>{peso_expl}</div>", unsafe_allow_html=True)

            with st.expander("📋 Detalles y trazabilidad", expanded=False):
                st.markdown(f"""
**Suelo:** pH={ph_suelo}, MO={mo_suelo}%, CIC={cic_suelo}, Textura={textura}, Estado={estado_suelo}, Metales={metales}  
**Biochar:** Feedstock={feedstock}, T={temp_pirolisis}°C, pH={ph_biochar}, Tamaño={tamano_ds}, BET={area_bet}  
**Cultivo:** {cultivo} | Riego={sistema_riego} | Clima={clima} | Objetivo={objetivo}  
**QC:** score={qc.score:.0f}, flags={", ".join(qc.flags) if qc.flags else "—"}
                """)
                if modo_experto:
                    st.markdown(f"""
**Experto:** O₂={o2_ppm} ppm (Texp={o2_temp}°C) | Enfriamiento={metodo_enfriamiento} | Proximal cierre={cierre:.1f}% | H/C={hc_ratio} | O/C={oc_ratio}
                    """)

            out = pd.DataFrame({
                "Parámetro": [
                    "Dosis_final_t_ha", "Metodo", "peso_ml", "R2_holdout", "QC_score", "QC_factor", "QC_flags",
                    "Objetivo", "Cultivo", "Riego", "Clima",
                    "pH_suelo", "MO", "CIC", "Textura", "Estado_suelo", "Metales",
                    "Feedstock", "T_pirolisis", "pH_biochar", "Tamaño_biochar", "BET",
                ],
                "Valor": [
                    f"{dosis_final:.2f}", metodo, f"{peso_xgb:.2f}", f"{st.session_state.r2_score:.4f}", f"{qc.score:.0f}", f"{qc.confidence_factor:.2f}", "; ".join(qc.flags),
                    objetivo, cultivo, sistema_riego, clima,
                    ph_suelo, mo_suelo, cic_suelo, textura, estado_suelo, metales,
                    feedstock, temp_pirolisis, ph_biochar, tamano_ds, area_bet,
                ]
            })

            if modo_experto:
                extra = pd.DataFrame({
                    "Parámetro": ["O2_ppm", "O2_temp_exposicion", "Metodo_enfriamiento", "H_C_ratio", "O_C_ratio",
                                 "Humedad_total", "Volatiles", "Cenizas_biomasa", "Carbono_fijo"],
                    "Valor": [o2_ppm, o2_temp, metodo_enfriamiento, hc_ratio, oc_ratio,
                              hum_total, volatiles, cenizas_biomasa, carbono_fijo]
                })
                out = pd.concat([out, extra], ignore_index=True)

            csv = out.to_csv(index=False)
            st.download_button("📥 Descargar resultados (CSV)", data=csv, file_name="prescripcion_biochar.csv", mime="text/csv")

# =============================================================================
# TAB 2 — Entrenamiento XGBoost
# =============================================================================

with tab2:
    st.header("Entrenamiento del Modelo (XGBoost)")

    src = st.radio("Fuente de datos para entrenar", ["Subir CSV", "Google Sheets", "Excel (Dosis_experimental)"], horizontal=True)

    if src == "Subir CSV":
        uploaded_csv = st.file_uploader("📤 Subir dataset (CSV)", type=["csv"], key="train_csv")
        if uploaded_csv is not None:
            try:
                df_raw = robust_read_csv_from_upload(uploaded_csv)
                df_raw = normalize_dataframe_columns(df_raw)
                capture_dataset_categories(df_raw)

                st.subheader("Vista previa")
                st.dataframe(df_raw.head(20), use_container_width=True)

                if TARGET_COL not in df_raw.columns:
                    st.error(f"Falta la columna '{TARGET_COL}'.")
                else:
                    st.caption(f"Filas: {len(df_raw)} | Columnas: {len(df_raw.columns)}")
                    if st.button("🚀 Entrenar XGBoost", type="primary", key="btn_train_csv"):
                        with st.spinner("Entrenando..."):
                            pipe, r2v, expected_cols, meta = entrenar_modelo_xgb_pipeline(df_raw, target=TARGET_COL)

                            st.session_state.modelo_pipe = pipe
                            st.session_state.r2_score = r2v
                            st.session_state.expected_cols = expected_cols
                            st.session_state.train_metadata = meta
                            st.session_state.modelo_activo = True

                        st.success("Modelo entrenado y activado ✅")
                        st.metric("R² (holdout)", f"{r2v:.4f}")

            except Exception as e:
                st.error(f"Error entrenando: {e}")

    elif src == "Google Sheets":
        if not gs_enabled:
            st.info("Activa Google Sheets en el panel izquierdo para usar esta opción.")
        elif not _GS_AVAILABLE:
            st.error("Faltan dependencias: agrega `gspread` y `google-auth` a requirements.txt.")
        elif not gs_sheet_id or not gs_worksheet:
            st.warning("Completa Sheet ID y Worksheet en el panel izquierdo (o en Secrets).")
        else:
            if st.button("🔄 Cargar/Refrescar dataset de Google Sheets", key="btn_load_gs"):
                st.cache_data.clear()
            try:
                df_gs = load_dataset_from_gs()
                if df_gs.empty:
                    st.warning("La hoja está vacía o no se pudo leer.")
                else:
                    st.subheader("Vista previa (Google Sheets)")
                    st.dataframe(df_gs.head(20), use_container_width=True)
                    st.caption(f"Filas: {len(df_gs)} | Columnas: {len(df_gs.columns)}")
                    if TARGET_COL not in df_gs.columns:
                        st.error(f"Falta la columna '{TARGET_COL}' en el Sheet.")
                    else:
                        confirmed = count_confirmed_rows(df_gs)
                        st.info(f"Filas validadas (verification_status='user_confirmed'): {confirmed}")
                        if st.button("🚀 Entrenar ahora (Google Sheets)", type="primary", key="btn_train_gs"):
                            with st.spinner("Entrenando..."):
                                pipe, r2v, expected_cols, meta = entrenar_modelo_xgb_pipeline(df_gs, target=TARGET_COL)
                                st.session_state.modelo_pipe = pipe
                                st.session_state.r2_score = r2v
                                st.session_state.expected_cols = expected_cols
                                st.session_state.train_metadata = meta
                                st.session_state.modelo_activo = True
                                st.session_state.last_train_confirmed = confirmed

                            st.success("Modelo entrenado con Google Sheets ✅")
                            st.metric("R² (holdout)", f"{r2v:.4f}")

                    with st.expander("📌 Categorías capturadas del dataset (para alinear UI)", expanded=False):
                        cats = st.session_state.get("dataset_cats", {})
                        for k, v in cats.items():
                            st.write(f"**{k}** ({len(v)}):", v[:50] + (["…"] if len(v) > 50 else []))

            except Exception as e:
                st.error(f"No se pudo leer/entrenar con Google Sheets: {e}")

    else:
        if datos_excel is None:
            st.info("No hay Excel cargado (puedes cargarlo en el panel izquierdo).")
        else:
            df_excel = datos_excel.get("dosis_exp")
            if df_excel is None or df_excel.empty:
                st.warning("El Excel no trae 'Dosis_experimental' o está vacío.")
            else:
                df_excel_n = normalize_dataframe_columns(df_excel)
                capture_dataset_categories(df_excel_n)
                st.caption(f"Filas en Dosis_experimental: {len(df_excel_n)}")
                st.dataframe(df_excel_n.head(15), use_container_width=True)
                if st.button("🚀 Entrenar ahora (Excel)", type="primary", key="btn_train_excel"):
                    try:
                        with st.spinner("Entrenando..."):
                            pipe, r2v, expected_cols, meta = entrenar_modelo_xgb_pipeline(df_excel_n, target=TARGET_COL)

                            st.session_state.modelo_pipe = pipe
                            st.session_state.r2_score = r2v
                            st.session_state.expected_cols = expected_cols
                            st.session_state.train_metadata = meta
                            st.session_state.modelo_activo = True

                        st.success("Modelo entrenado con Excel ✅")
                        st.metric("R² (holdout)", f"{r2v:.4f}")
                    except Exception as e:
                        st.error(f"No se pudo entrenar con el Excel: {e}")

# =============================================================================
# TAB 3 — Base de Conocimiento
# =============================================================================

with tab3:
    st.header("Base de Conocimiento")
    if datos_excel is None:
        st.warning("No hay Excel cargado. Esta pestaña muestra más contenido cuando el Excel está disponible.")
    else:
        t1, t2, t3b = st.tabs(["📊 Parámetros", "⚙️ Algoritmos/Factores", "📈 Casos históricos"])
        with t1:
            st.dataframe(datos_excel.get("parametros"), use_container_width=True)
        with t2:
            st.subheader("Algoritmos")
            st.dataframe(datos_excel.get("algoritmos"), use_container_width=True)
            st.subheader("Factores de ajuste")
            st.dataframe(datos_excel.get("factores"), use_container_width=True)
        with t3b:
            st.dataframe(datos_excel.get("metadatos"), use_container_width=True)

# =============================================================================
# TAB 4 — Ingeniería / QC
# =============================================================================

with tab4:
    st.header("⚗️ Calculadoras de Ingeniería de Biochar")
    st.markdown("Herramientas comparativas para producción, rendimientos y secuestro (modo educativo).")

    col_eng1, col_eng2 = st.columns(2)

    with col_eng1:
        eng_feedstock = st.selectbox(
            "Materia prima para cálculo",
            ["Madera", "Cáscara cacao", "Paja trigo", "Bambú", "Estiércol"],
            key="eng_feedstock"
        )
        eng_humedad = st.slider("Humedad (%)", 5.0, 30.0, 10.0, 1.0, key="eng_humedad")
        eng_carbono = st.slider("Carbono en biomasa seca (%)", 30.0, 60.0, 48.0, 1.0, key="eng_carbono")

    with col_eng2:
        eng_temperatura = st.slider("Temperatura (°C)", 300, 900, 550, 10, key="eng_temperatura")
        eng_tiempo = st.slider("Tiempo residencia (min)", 15, 180, 60, 5, key="eng_tiempo")
        eng_capacidad = st.number_input("Capacidad reactor (kg/h)", 10, 1000, 100, 10, key="eng_capacidad")

    if st.button("⚖️ Calcular Balance Completo", type="primary", key="btn_balance_completo"):
        with st.spinner("Calculando..."):
            bal = calcular_balance_masa_energia(eng_feedstock, eng_humedad, eng_temperatura, eng_tiempo)
            sec = calcular_secuestro_carbono(eng_feedstock, eng_temperatura, bal["rend_biochar_pct"], eng_carbono)

            prod_char_kg_h = eng_capacidad * (bal["rend_biochar_pct"] / 100.0)
            prod_char_t_d = (prod_char_kg_h * 24.0) / 1000.0
            prod_char_t_y = prod_char_t_d * 300.0

            co2_t_y = prod_char_t_y * sec["secuestro_tC_t"] * 3.67

            st.markdown("---")
            st.subheader("📊 Resultados")

            cA, cB, cC = st.columns(3)
            with cA:
                st.metric("Rendimiento Biochar", f"{bal['rend_biochar_pct']:.1f}%")
                st.metric("Producción Biochar", f"{prod_char_kg_h:.1f} kg/h")
                st.metric("Producción anual", f"{prod_char_t_y:.1f} t/año")
            with cB:
                st.metric("Rendimiento Bio-oil", f"{bal['rend_biooil_pct']:.1f}%")
                st.metric("Rendimiento Gas", f"{bal['rend_gas_pct']:.1f}%")
                st.metric("Energía (proxy)", f"{bal['energia_MJ_kg']:.3f} MJ/kg")
            with cC:
                st.metric("Carbono retenido", f"{sec['carbono_retenido_pct']:.2f}% (proxy)")
                st.metric("CO₂ secuestrado/año", f"{co2_t_y:.1f} t CO₂/año (proxy)")
                st.metric("Vida media (proxy)", f"{sec['vida_media_anios']:.0f} años")

            st.markdown("### 📈 Distribución de productos")
            df_plot = pd.DataFrame({
                "Producto": ["Biochar", "Bio-oil", "Gas"],
                "Porcentaje": [bal["rend_biochar_pct"], bal["rend_biooil_pct"], bal["rend_gas_pct"]],
            })
            st.bar_chart(df_plot.set_index("Producto"))

# =============================================================================
# TAB 5 — PDF → Base (Google Sheets)
# =============================================================================

with tab5:
    st.header("📄 Ingesta de Artículos (PDF) → Base persistente")

    st.markdown("""
Este módulo está diseñado para **ingestar artículos científicos de manera controlada**:
1) Subes un PDF.  
2) El sistema **intenta** extraer campos (heurístico).  
3) Tú **confirmas/editar** antes de guardar.  
4) Guardas como **BORRADOR** o como **VALIDADO** (`user_confirmed`).  
5) El entrenamiento usa **solo filas VALIDADAS** y excluye `inferred`.
    """)

    if not gs_enabled:
        st.info("Activa Google Sheets en el panel izquierdo para habilitar guardado persistente.")
    elif not _GS_AVAILABLE:
        st.error("Faltan dependencias: agrega `gspread` y `google-auth` a requirements.txt.")
    elif not gs_sheet_id or not gs_worksheet:
        st.warning("Completa Sheet ID y Worksheet en el panel izquierdo (o en Secrets).")
    else:
        st.subheader("🧾 Modo artículo (PDF)")
        st.caption("Consejo: extrae, revisa, ajusta, y guarda. Si falta dosis o no estás 100% seguro, guarda BORRADOR.")

        pdf_file = st.file_uploader("Sube PDF del artículo", type=["pdf"], key="t5_pdf_upl_v2")

        colA, colB, colC = st.columns([1, 1, 1])
        with colA:
            extract_btn = st.button("🔎 Extraer campos", type="primary", disabled=(pdf_file is None), key="t5_extract_btn_v2")
        with colB:
            clear_btn = st.button("🧹 Limpiar", key="t5_clear_btn_v2")
        with colC:
            max_pages = st.number_input("Páginas a leer (rápido)", min_value=1, max_value=20, value=6, step=1, key="t5_max_pages_v2")

        if clear_btn:
            st.session_state.pdf_extract = {}

        if extract_btn and pdf_file is not None:
            try:
                with st.spinner("Leyendo PDF y extrayendo..."):
                    txt = extract_text_from_pdf(pdf_file, max_pages=int(max_pages))
                    fields = infer_fields_from_text(txt)

                    fields["pdf_filename"] = pdf_file.name
                    fields["ingest_timestamp"] = now_iso()
                    fields["ref_quality"] = "inferred"

                    st.session_state.pdf_extract = fields
                st.success("Extracción realizada. Revisa/ajusta antes de guardar.")
            except Exception as e:
                st.error(f"No se pudo extraer del PDF: {e}")

        fields = st.session_state.get("pdf_extract", {}) or {}

        if fields:
            st.markdown("---")
            st.subheader("✅ Revisión y confirmación (antes de guardar)")

            def _inject_and_index(options: List[str], suggested: Any) -> Tuple[List[str], int]:
                opts = list(options) if options else []
                sug = clean_category_value(suggested)
                if isinstance(sug, float) and np.isnan(sug):
                    return opts, 0
                if sug and (sug not in opts):
                    opts = unique_sorted(opts + [sug])
                idx = opts.index(sug) if (sug in opts) else 0
                return opts, idx

            def _safe_str(x: Any) -> str:
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return ""
                return str(x)

            m1, m2, m3 = st.columns(3)
            with m1:
                doi_in = st.text_input("DOI (si aplica)", value=_safe_str(fields.get("doi", "")), key="t5_doi_v2")
                fuente_in = st.text_area("Fuente (texto corto / cita)", value=_safe_str(fields.get("Fuente", "")), height=90, key="t5_fuente_v2")
            with m2:
                ref_type_opts = ["experimental", "review", "meta_analysis", "case_study", "other"]
                ref_type_guess = _safe_str(fields.get("ref_type", "experimental")).strip().lower()
                if ref_type_guess not in ref_type_opts:
                    ref_type_guess = "experimental"
                ref_type_in = st.selectbox("Tipo de artículo (ref_type)", ref_type_opts, index=ref_type_opts.index(ref_type_guess), key="t5_ref_type_v2")

                tipo_cultivo_in = st.text_input("Cultivo / sistema reportado (Tipo)", value=_safe_str(fields.get("Tipo", "")), key="t5_tipo_v2")
                objetivo_opts, objetivo_idx = _inject_and_index(ui_options("Objetivo", DEFAULT_OBJETIVOS), fields.get("Objetivo"))
                objetivo_in = st.selectbox("Objetivo (si se reporta)", objetivo_opts, index=objetivo_idx if len(objetivo_opts) else 0, key="t5_objetivo_v2")
            with m3:
                notes_in = st.text_area("Notas de verificación", value=_safe_str(fields.get("verification_notes", "")), height=90, key="t5_notes_v2")
                st.caption("Tip: pega aquí cualquier duda: unidades, si es maceta/campo, si el pH es del suelo o del biochar, etc.")

            st.markdown("### 🧪 Variables (suelo + biochar + caracterización)")

            ph_def  = clamp_float(fields.get("ph"), 3.0, 9.5, default=6.5)
            mo_def  = clamp_float(fields.get("mo"), 0.0, 20.0, default=2.0)
            cic_def = clamp_float(fields.get("CIC"), 0.0, 100.0, default=15.0)
            met_def = clamp_float(fields.get("Metales"), 0.0, 5000.0, default=0.0)

            T_def   = clamp_float(fields.get("T_pirolisis"), 250.0, 950.0, default=550.0)
            phb_def = clamp_float(fields.get("pH_biochar"), 3.0, 14.0, default=9.0)
            bet_def = clamp_float(fields.get("Area_BET"), 0.0, 2000.0, default=300.0)

            hum_def = clamp_float(fields.get("Humedad_total"), 0.0, 60.0, default=10.0)
            vol_def = clamp_float(fields.get("Volatiles"), 0.0, 95.0, default=70.0)
            ash_def = clamp_float(fields.get("Cenizas_biomasa"), 0.0, 60.0, default=3.0)
            fc_def  = clamp_float(fields.get("Carbono_fijo"), 0.0, 95.0, default=17.0)

            hc_def  = clamp_float(fields.get("H_C_ratio"), 0.0, 2.0, default=0.65)
            oc_def  = clamp_float(fields.get("O_C_ratio"), 0.0, 1.5, default=0.18)

            o2_def  = clamp_int(fields.get("O2_ppm"), 0, 50000, default=5000)
            o2t_def = clamp_float(fields.get("O2_temp_exposicion"), 20.0, 600.0, default=120.0)

            feed_opts, feed_idx = _inject_and_index(ui_options("Feedstock", DEFAULT_FEEDSTOCKS), fields.get("Feedstock"))
            tex_opts,  tex_idx  = _inject_and_index(ui_options("Textura", DEFAULT_TEXTURAS), fields.get("Textura"))
            est_opts,  est_idx  = _inject_and_index(ui_options("Estado_suelo", DEFAULT_ESTADOS_SUELO), fields.get("Estado_suelo"))
            tam_opts,  tam_idx  = _inject_and_index(ui_options("Tamaño_biochar", DEFAULT_TAMANOS), fields.get("Tamaño_biochar"))

            v1, v2, v3 = st.columns(3)
            with v1:
                st.markdown("**Suelo**")
                ph_in  = safe_number_input("ph (suelo)", key="t5_ph_v2", min_value=3.0, max_value=9.5, value=ph_def, step=0.1)
                mo_in  = safe_number_input("mo (Materia orgánica %)", key="t5_mo_v2", min_value=0.0, max_value=20.0, value=mo_def, step=0.1)
                cic_in = safe_number_input("CIC (cmolc/kg)", key="t5_cic_v2", min_value=0.0, max_value=100.0, value=cic_def, step=0.5)
                met_in = safe_number_input("Metales (mg/kg)", key="t5_met_v2", min_value=0.0, max_value=5000.0, value=met_def, step=1.0)
                textura_in = st.selectbox("Textura", tex_opts, index=tex_idx if len(tex_opts) else 0, key="t5_textura_v2")
                estado_in  = st.selectbox("Estado_suelo", est_opts, index=est_idx if len(est_opts) else 0, key="t5_estado_v2")

            with v2:
                st.markdown("**Biochar**")
                feedstock_in = st.selectbox("Feedstock", feed_opts, index=feed_idx if len(feed_opts) else 0, key="t5_feedstock_v2")
                T_in   = safe_number_input("T_pirolisis (°C)", key="t5_T_v2", min_value=250.0, max_value=950.0, value=T_def, step=10.0)
                phb_in = safe_number_input("pH_biochar", key="t5_phb_v2", min_value=3.0, max_value=14.0, value=phb_def, step=0.1)
                bet_in = safe_number_input("Area_BET (m²/g)", key="t5_bet_v2", min_value=0.0, max_value=2000.0, value=bet_def, step=10.0)
                tam_in = st.selectbox("Tamaño_biochar", tam_opts, index=tam_idx if len(tam_opts) else 0, key="t5_tamano_v2")

            with v3:
                st.markdown("**Caracterización / Proceso (opcional)**")
                hum_in = safe_number_input("Humedad_total (%)", key="t5_hum_v2", min_value=0.0, max_value=60.0, value=hum_def, step=0.5)
                vol_in = safe_number_input("Volatiles (%)", key="t5_vol_v2", min_value=0.0, max_value=95.0, value=vol_def, step=0.5)
                ash_in = safe_number_input("Cenizas_biomasa (%)", key="t5_ash_v2", min_value=0.0, max_value=60.0, value=ash_def, step=0.5)
                fc_in  = safe_number_input("Carbono_fijo (%)", key="t5_fc_v2", min_value=0.0, max_value=95.0, value=fc_def, step=0.5)

                metodo_enf_in = st.selectbox(
                    "Metodo_enfriamiento",
                    ["Cámara inerte", "Contacto indirecto", "En aire", "Agua directa", "Desconocido"],
                    index=4,
                    key="t5_enf_v2"
                )
                o2_in  = safe_int_input("O2_ppm (si se reporta)", key="t5_o2_v2", min_value=0, max_value=50000, value=o2_def, step=250)
                o2t_in = safe_number_input("O2_temp_exposicion (°C)", key="t5_o2t_v2", min_value=20.0, max_value=600.0, value=o2t_def, step=10.0)
                hc_in  = safe_number_input("H_C_ratio (molar)", key="t5_hc_v2", min_value=0.0, max_value=2.0, value=hc_def, step=0.01)
                oc_in  = safe_number_input("O_C_ratio (molar)", key="t5_oc_v2", min_value=0.0, max_value=1.5, value=oc_def, step=0.01)

            st.markdown("### 🎯 Dosis (gating de validación)")
            dosis_def = clamp_float(fields.get("dosis_efectiva"), 0.0, 200.0, default=0.0)
            dosis_in = safe_number_input("dosis_efectiva (t/ha)", key="t5_dosis_v2", min_value=0.0, max_value=200.0, value=dosis_def, step=0.1)

            st.markdown("---")
            st.subheader("🗃️ Estado del registro (y entrenamiento)")

            estado_registro = st.radio(
                "¿Cómo deseas guardar este registro?",
                ["BORRADOR (no entrena)", "VALIDADO (user_confirmed → entra a entrenamiento)"],
                index=0,
                horizontal=True,
                key="t5_estado_registro_v2",
            )

            confirm_dose = False
            if estado_registro.startswith("VALIDADO"):
                st.warning("VALIDADO exige confirmar la dosis.")
                confirm_dose = st.checkbox(
                    "Confirmo que la dosis está reportada en el artículo y corresponde a t/ha (o Mg/ha).",
                    value=False,
                    key="t5_confirm_dose_v2",
                )

            can_validate = (estado_registro.startswith("VALIDADO") and (dosis_in > 0) and confirm_dose)

            if estado_registro.startswith("VALIDADO") and (dosis_in <= 0):
                st.error("No puedes guardar VALIDADO sin una dosis > 0.")
            if estado_registro.startswith("VALIDADO") and (not confirm_dose):
                st.info("Marca la casilla de confirmación para habilitar guardado VALIDADO.")

            verification_status = "user_confirmed" if can_validate else "draft"
            ref_quality = "confirmed" if can_validate else "inferred"

            doi_format_ok = bool(_DOI_FULL_RE.match(doi_in.strip())) if doi_in.strip() else False
            doi_url = f"https://doi.org/{doi_in.strip()}" if doi_in.strip() and doi_format_ok else ""

            fn = _safe_str(fields.get("pdf_filename", "pdf")).strip()
            fn_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", fn)[:80]
            ref_id = f"pdf_{now_iso().replace(':','').replace('-','')}_{fn_slug}"

            row = {
                "ph": ph_in,
                "mo": mo_in,
                "CIC": cic_in,
                "Metales": met_in,
                "Textura": textura_in,
                "Estado_suelo": estado_in,

                "Feedstock": feedstock_in,
                "T_pirolisis": T_in,
                "pH_biochar": phb_in,
                "Area_BET": bet_in,
                "Tamaño_biochar": tam_in,

                "Objetivo": objetivo_in,
                "Tipo": tipo_cultivo_in,

                "Humedad_total": hum_in,
                "Volatiles": vol_in,
                "Cenizas_biomasa": ash_in,
                "Carbono_fijo": fc_in,
                "Metodo_enfriamiento": metodo_enf_in,
                "O2_ppm": int(o2_in),
                "O2_temp_exposicion": o2t_in,
                "H_C_ratio": hc_in,
                "O_C_ratio": oc_in,

                "dosis_efectiva": float(dosis_in) if dosis_in > 0 else "",

                "Fuente": fuente_in,
                "Fuente_raw": fn,
                "doi": doi_in.strip(),
                "doi_format_ok": str(doi_format_ok),
                "doi_url": doi_url,
                "ref_type": ref_type_in,
                "ref_id": ref_id,
                "ref_quality": ref_quality,
                "verification_status": verification_status,
                "verification_notes": notes_in,
                "ingest_timestamp": fields.get("ingest_timestamp", now_iso()),
                "pdf_filename": fn,
            }

            required_headers = unique_sorted(
                list(row.keys()) +
                META_COLS +
                [TARGET_COL] +
                [
                    "ph", "mo", "CIC", "Metales", "Textura", "Estado_suelo",
                    "Feedstock", "T_pirolisis", "pH_biochar", "Area_BET", "Tamaño_biochar",
                    "Objetivo", "Tipo",
                    "Humedad_total", "Volatiles", "Cenizas_biomasa", "Carbono_fijo",
                    "Metodo_enfriamiento", "O2_ppm", "O2_temp_exposicion", "H_C_ratio", "O_C_ratio",
                ]
            )

            st.markdown("---")
            save_disabled = (estado_registro.startswith("VALIDADO") and (not can_validate))
            if st.button("💾 Guardar fila en Google Sheets", type="primary", key="t5_save_v2", disabled=save_disabled):
                try:
                    gs_append_row(gs_sheet_id, gs_worksheet, row, required_headers=required_headers)
                    st.success(f"Fila guardada ✅  (verification_status = {verification_status}, ref_quality = {ref_quality})")
                    st.cache_data.clear()
                    st.caption("Tip: ve a 'Entrenamiento XGBoost' para entrenar, o activa autoentreno en el panel izquierdo.")
                except Exception as e:
                    st.error(f"No se pudo guardar en Sheets: {e}")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(
    f"""
    <div style='text-align:center;color:#666;padding:0.9rem;'>
    <b>Prescriptor Híbrido Biochar {APP_VERSION} 🌱⚗️</b><br>
    QC + reglas + XGBoost • NanoMof 2025 ©️
    </div>
    """,
    unsafe_allow_html=True
)
