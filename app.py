import os
import io
import re
import warnings
import unicodedata
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

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

APP_VERSION = "v3.3 (QC + XGBoost robusto + UI alineada + explicación peso_xgb)"
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
    s = strip_emojis(v)
    s = s.strip()
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
# EXPLICACIÓN "DUMMY" PARA peso_xgb (visible en la app)
# =============================================================================

def explain_peso_xgb_for_users(
    peso_xgb: float,
    r2_holdout: Optional[float],
    qc_score: float,
    train_n: int = 0,
    train_sources_n: Optional[int] = None
) -> str:
    """
    Mensaje corto y accionable para no técnicos.
    """
    p = float(np.clip(peso_xgb, 0.0, 1.0))
    ml_pct = int(round(p * 100))
    rules_pct = 100 - ml_pct

    # lectura rápida
    if p >= 0.65:
        lectura = "El modelo (XGBoost) está pesando más: se confía bastante en los datos."
    elif p >= 0.45:
        lectura = "La recomendación está balanceada: mezcla datos + reglas de forma prudente."
    else:
        lectura = "El sistema está siendo conservador con el modelo: pesan más las reglas + QC."

    # por qué
    motivos = []
    if r2_holdout is not None:
        if r2_holdout >= 0.8:
            motivos.append("el modelo mostró buen desempeño en validación")
        elif r2_holdout >= 0.6:
            motivos.append("el modelo es aceptable, pero no sobresaliente")
        else:
            motivos.append("el desempeño del modelo es bajo")

    if qc_score < 55:
        motivos.append("el QC es bajo (más incertidumbre del material)")
    elif qc_score < 80:
        motivos.append("el QC es medio (hay señales de variabilidad)")
    else:
        motivos.append("el QC es alto (material más confiable)")

    if train_n > 0:
        motivos.append(f"entrenado con ~{train_n} casos")
    if train_sources_n is not None and train_sources_n > 0:
        motivos.append(f"{train_sources_n} fuentes")

    motivos_txt = "; ".join(motivos)

    # recomendación práctica (1 línea)
    if p >= 0.65:
        consejo = "Si el caso es similar a los datos, puedes tomar la dosis como punto principal y afinar en campo."
    elif p >= 0.45:
        consejo = "Úsala como dosis base y ajusta con tu criterio agronómico (especialmente si hay restricciones)."
    else:
        consejo = "Prioriza la dosis conservadora (reglas+QC) y valida con prueba piloto antes de escalar."

    return (
        f"**¿Qué significa `peso_xgb`?** {ml_pct}% de la dosis viene del modelo XGBoost y {rules_pct}% de reglas+QC. "
        f"{lectura} (Se ajusta automáticamente por: {motivos_txt}). "
        f"{consejo}"
    )

# =============================================================================
# (a) LECTURA ROBUSTA CSV + (b) NORMALIZACIÓN DE COLUMNAS
# =============================================================================

_CANON_COL_MAP = {
    "dosis_efectiva": "dosis_efectiva",
    "dosis": "dosis_efectiva",

    "ph": "ph",
    "ph_suelo": "ph",
    "mo": "mo",

    "t_pirolisis": "T_pirolisis",
    "t_pirólisis": "T_pirolisis",
    "temperatura_pirolisis": "T_pirolisis",
    "temperatura_pirólisis": "T_pirolisis",

    "ph_biochar": "pH_biochar",
    "area_bet": "Area_BET",
    "área_bet": "Area_BET",
    "area_superficial_bet": "Area_BET",
    "tamaño_biochar": "Tamaño_biochar",
    "tamano_biochar": "Tamaño_biochar",
    "feedstock": "Feedstock",

    "estado_suelo": "Estado_suelo",
    "textura": "Textura",
    "objetivo": "Objetivo",
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

    for num_col in ["ph", "mo", "T_pirolisis", "pH_biochar", "Area_BET", "dosis_efectiva"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    for cat_col in ["Feedstock", "Textura", "Estado_suelo", "Tamaño_biochar", "Objetivo", "Fuente"]:
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

SCHEMA_NUM_COLS = [
    "ph", "mo", "CIC", "Metales",
    "T_pirolisis", "pH_biochar", "Area_BET",
    "Sensibilidad_salinidad",
    "Humedad_total", "Volatiles", "Cenizas_biomasa", "Carbono_fijo",
    "O2_ppm", "O2_temp_exposicion", "H_C_ratio", "O_C_ratio",
]

FORCE_CATEGORICAL_COLS = {
    "Tipo", "Cultivo",
    "Textura", "Feedstock", "Estado_suelo", "Tamaño_biochar", "Objetivo",
    "Riego", "Clima", "Sistema_cultivo", "Tipo_producto", "Objetivo_calidad",
    "Metodo_enfriamiento",
}

def entrenar_modelo_xgb_pipeline(
    df_raw: pd.DataFrame,
    target: str = "dosis_efectiva"
) -> Tuple[Pipeline, float, List[str], pd.DataFrame, int, Optional[int]]:
    """
    Retorna:
    - pipe
    - r2 holdout
    - expected_cols
    - metadata (no features)
    - train_n (filas usadas con target válido)
    - train_sources_n (si existe alguna columna de fuente)
    """
    if target not in df_raw.columns:
        raise ValueError(f"El dataset debe incluir la columna '{target}'")

    META_COLS = [
        "Fuente","Fuente_raw","doi","ref_type","doi_format_ok","doi_url","ref_id","ref_quality",
        "verification_status","verified_title","verified_journal","verified_year","verified_authors",
        "verification_notes","Fuente_display","Fuente_status","Fuente_public"
    ]

    df_feat, df_meta = split_features_and_metadata(df_raw, metadata_cols=META_COLS)

    y = pd.to_numeric(df_feat[target], errors="coerce")
    keep = y.notna()
    df_feat = df_feat.loc[keep].copy()
    df_meta = df_meta.loc[keep].copy()
    y = y.loc[keep].copy()

    train_n = int(len(df_feat))
    if train_n < 10:
        raise ValueError("Dataset insuficiente: se requieren al menos 10 filas con dosis_efectiva válida.")

    X = df_feat.drop(columns=[target]).copy()
    expected_cols = list(X.columns)

    num_cols = [c for c in expected_cols if c in SCHEMA_NUM_COLS]
    cat_cols = [c for c in expected_cols if c not in num_cols]

    for c in FORCE_CATEGORICAL_COLS:
        if c in num_cols:
            num_cols.remove(c)
            if c not in cat_cols:
                cat_cols.append(c)

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
        n_estimators=400,
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

    # intenta contar "fuentes" si existe alguna columna típica en metadata
    train_sources_n = None
    for cand in ["Fuente_public", "Fuente_display", "Fuente_raw", "Fuente"]:
        if cand in df_meta.columns:
            try:
                train_sources_n = int(pd.Series(df_meta[cand]).dropna().nunique())
            except Exception:
                train_sources_n = None
            break

    return pipe, float(r2), expected_cols, df_meta, train_n, train_sources_n

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

    # QC features si existen en entrenamiento (no hace daño si el modelo no las usa: quedan como NaN)
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

# =============================================================================
# UI — Sidebar: Excel opcional + autoentreno opcional
# =============================================================================

st.sidebar.markdown(f"### ⚙️ Configuración ({APP_VERSION})")

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

auto_train = st.sidebar.checkbox(
    "Autoentrenar XGBoost al iniciar (si hay dataset en Excel)",
    value=False,
    help="Se activa si el Excel trae hoja 'Dosis_experimental' con 'dosis_efectiva'."
)

# header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="main-header">🧬 Prescriptor Híbrido Biochar</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">QC + reglas determinísticas + XGBoost</div>', unsafe_allow_html=True)
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
if "train_n" not in st.session_state:
    st.session_state.train_n = 0
if "train_sources_n" not in st.session_state:
    st.session_state.train_sources_n = None
if "parametros_usuario" not in st.session_state:
    st.session_state.parametros_usuario = {}
if "dataset_cats" not in st.session_state:
    st.session_state.dataset_cats = {}

# Autoentreno opcional
if auto_train and (not st.session_state.modelo_activo) and datos_excel is not None:
    try:
        df0 = datos_excel.get("dosis_exp")
        if df0 is not None and "dosis_efectiva" in df0.columns and len(df0) >= 10:
            df0n = normalize_dataframe_columns(df0)
            capture_dataset_categories(df0n)
            pipe, r2v, expected_cols, meta, train_n, train_sources_n = entrenar_modelo_xgb_pipeline(df0n, target="dosis_efectiva")
            st.session_state.modelo_pipe = pipe
            st.session_state.r2_score = r2v
            st.session_state.expected_cols = expected_cols
            st.session_state.train_metadata = meta
            st.session_state.train_n = train_n
            st.session_state.train_sources_n = train_sources_n
            st.session_state.modelo_activo = True
    except Exception:
        pass

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Prescripción Híbrida",
    "📊 Entrenamiento XGBoost",
    "📚 Base de Conocimiento",
    "⚗️ Ingeniería / QC"
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

            if st.session_state.modelo_activo and st.session_state.modelo_pipe is not None:
                try:
                    flat = build_flat_features_for_model(suelo, biochar, cultivo_d, objetivo)
                    df_in = preparar_input_modelo(st.session_state.expected_cols, flat)
                    dosis_xgb = float(st.session_state.modelo_pipe.predict(df_in)[0])
                except Exception as e:
                    st.warning(f"Predicción XGBoost no disponible: {e}")
                    dosis_xgb = None

            if dosis_xgb is not None:
                # peso dinámico (prudente): depende de desempeño + QC
                r2v = float(st.session_state.r2_score or 0.0)

                # base: parte de 0.60 pero se ajusta suave
                peso_xgb = 0.60

                # si el desempeño baja, baja el peso
                if r2v < 0.5:
                    peso_xgb = 0.30
                elif r2v < 0.7:
                    peso_xgb = 0.45
                elif r2v > 0.9:
                    peso_xgb = 0.62  # pequeño premio, no “manda todo”

                # QC penaliza si hay incertidumbre
                if qc.score < 60:
                    peso_xgb = min(peso_xgb, 0.35)
                elif qc.score < 80:
                    peso_xgb = min(peso_xgb, 0.55)

                # clamp final
                peso_xgb = float(np.clip(peso_xgb, 0.20, 0.70))

                dosis_final = (1 - peso_xgb) * dosis_det_qc + peso_xgb * dosis_xgb
                metodo = f"Híbrido (Det+QC + XGBoost | peso_xgb={peso_xgb:.2f})"
                mostrar_xgb = True
            else:
                dosis_final = dosis_det_qc
                mostrar_xgb = False

            dosis_final = float(np.clip(dosis_final, 3, 60))

            st.markdown("---")

            # explicación dummy (visible)
            explain_txt = None
            if mostrar_xgb:
                explain_txt = explain_peso_xgb_for_users(
                    peso_xgb=peso_xgb,
                    r2_holdout=st.session_state.r2_score,
                    qc_score=qc.score,
                    train_n=int(st.session_state.train_n or 0),
                    train_sources_n=st.session_state.train_sources_n
                )

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
                if explain_txt:
                    st.info(explain_txt)
            else:
                st.success(f"### 📊 Dosis Recomendada: **{dosis_final:.1f} t/Ha**")
                st.caption(metodo)
                if explain_txt:
                    st.info(explain_txt)

            r1, r2c, r3 = st.columns(3)
            with r1:
                st.metric("Determinístico", f"{dosis_det:.1f} t/Ha", delta=f"QC×{qc.confidence_factor:.2f} → {dosis_det_qc:.1f}")
            with r2c:
                if mostrar_xgb:
                    st.metric("XGBoost", f"{dosis_xgb:.1f} t/Ha", delta=f"R² (holdout): {st.session_state.r2_score:.3f}")
                else:
                    st.metric("XGBoost", "—", delta="Modelo inactivo")
            with r3:
                st.metric("QC Score", f"{qc.score:.0f}/100", delta=pill_text)

            with st.expander("📋 Detalles y trazabilidad", expanded=True):
                st.markdown(f"""
**Suelo:** pH={ph_suelo}, MO={mo_suelo}%, CIC={cic_suelo}, Textura={textura}, Estado={estado_suelo}, Metales={metales}  
**Biochar:** Feedstock={feedstock}, T={temp_pirolisis}°C, pH={ph_biochar}, Tamaño={tamano_ds}, BET={area_bet}  
**Cultivo (Tipo):** {cultivo} | Riego={sistema_riego} | Clima={clima} | Objetivo={objetivo}  
**QC:** score={qc.score:.0f}, flags={", ".join(qc.flags) if qc.flags else "—"}
                """)
                if modo_experto:
                    st.markdown(f"""
**Experto:** O₂={o2_ppm} ppm (Texp={o2_temp}°C) | Enfriamiento={metodo_enfriamiento} | Proximal cierre={cierre:.1f}% | H/C={hc_ratio} | O/C={oc_ratio}
                    """)

            out = pd.DataFrame({
                "Parámetro": [
                    "Dosis_final_t_ha", "Metodo", "QC_score", "QC_factor", "QC_flags",
                    "Objetivo", "Cultivo", "Riego", "Clima",
                    "pH_suelo", "MO", "CIC", "Textura", "Estado_suelo", "Metales",
                    "Feedstock", "T_pirolisis", "pH_biochar", "Tamaño_biochar", "BET",
                ],
                "Valor": [
                    f"{dosis_final:.2f}", metodo, f"{qc.score:.0f}", f"{qc.confidence_factor:.2f}", "; ".join(qc.flags),
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

    uploaded_csv = st.file_uploader("📤 Subir dataset (CSV)", type=["csv"])

    if uploaded_csv is not None:
        try:
            df_raw = robust_read_csv_from_upload(uploaded_csv)
            df_raw = normalize_dataframe_columns(df_raw)

            capture_dataset_categories(df_raw)

            st.subheader("Vista previa (incluye metadata si existe)")
            st.dataframe(df_raw.head(20), use_container_width=True)

            if "dosis_efectiva" not in df_raw.columns:
                st.error("Falta la columna 'dosis_efectiva'.")
            else:
                st.caption(f"Filas: {len(df_raw)} | Columnas: {len(df_raw.columns)}")

                if st.button("🚀 Entrenar XGBoost", type="primary"):
                    with st.spinner("Entrenando..."):
                        pipe, r2v, expected_cols, meta, train_n, train_sources_n = entrenar_modelo_xgb_pipeline(df_raw, target="dosis_efectiva")

                        st.session_state.modelo_pipe = pipe
                        st.session_state.r2_score = r2v
                        st.session_state.expected_cols = expected_cols
                        st.session_state.train_metadata = meta
                        st.session_state.train_n = train_n
                        st.session_state.train_sources_n = train_sources_n
                        st.session_state.modelo_activo = True

                    st.success("Modelo entrenado y activado ✅")
                    st.metric("R² (holdout)", f"{r2v:.4f}")
                    st.caption(f"Filas usadas (con dosis_efectiva válida): {train_n}")
                    if train_sources_n is not None:
                        st.caption(f"Fuentes distintas (según metadata): {train_sources_n}")
                    st.caption(f"Columnas usadas por el modelo: {len(expected_cols)}")

                    if len(meta.columns) > 0:
                        st.info("Metadata preservada (no usada como feature): " + ", ".join(list(meta.columns)))

                    with st.expander("📌 Categorías capturadas del dataset (para alinear UI)", expanded=False):
                        cats = st.session_state.get("dataset_cats", {})
                        for k, v in cats.items():
                            st.write(f"**{k}** ({len(v)}):", v[:50] + (["…"] if len(v) > 50 else []))

        except Exception as e:
            st.error(f"Error entrenando: {e}")

    st.markdown("---")
    st.subheader("Entrenar con dataset del Excel (opcional)")
    if datos_excel is None:
        st.info("No hay Excel cargado (puedes cargarlo en el panel izquierdo).")
    else:
        df_excel = datos_excel.get("dosis_exp")
        if df_excel is None or df_excel.empty:
            st.warning("El Excel no trae 'Dosis_experimental' o está vacío.")
        else:
            st.caption(f"Filas en Dosis_experimental: {len(df_excel)}")
            if st.button("🔄 Entrenar con Dosis_experimental del Excel"):
                try:
                    with st.spinner("Entrenando..."):
                        df_excel_n = normalize_dataframe_columns(df_excel)
                        capture_dataset_categories(df_excel_n)
                        pipe, r2v, expected_cols, meta, train_n, train_sources_n = entrenar_modelo_xgb_pipeline(df_excel_n, target="dosis_efectiva")

                        st.session_state.modelo_pipe = pipe
                        st.session_state.r2_score = r2v
                        st.session_state.expected_cols = expected_cols
                        st.session_state.train_metadata = meta
                        st.session_state.train_n = train_n
                        st.session_state.train_sources_n = train_sources_n
                        st.session_state.modelo_activo = True

                    st.success("Modelo entrenado con Excel ✅")
                    st.metric("R² (holdout)", f"{r2v:.4f}")
                    st.caption(f"Filas usadas (con dosis_efectiva válida): {train_n}")
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
