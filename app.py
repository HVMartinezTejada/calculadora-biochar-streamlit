import os
import warnings
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

APP_VERSION = "v3.1 (QC + Pipeline XGBoost)"
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
# UTILIDADES
# =============================================================================

def norm_tamano(label: str) -> str:
    s = (label or "").strip().lower()
    if s.startswith("fino"):
        return "Fino"
    if s.startswith("medio"):
        return "Medio"
    if s.startswith("grueso"):
        return "Grueso"
    return "Medio"

def is_flor(cultivo: str) -> bool:
    s = (cultivo or "").lower()
    return any(k in s for k in ["rosa", "clavel", "crisant", "orquí", "orqui", "flor", "ornamental", "bulbo"])

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


# =============================================================================
# EXCEL (OPCIONAL) — reglas determinísticas + KB + dataset
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
# QC REPORT — “biochar serio” como gate + score + incertidumbre
# =============================================================================

@dataclass
class QCReport:
    score: float
    confidence_factor: float  # multiplica dosis (o ajusta peso ML)
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
    """
    QC enfocado en:
    - Riesgo de oxidación: O2 + T_exposición + enfriamiento
    - Coherencia de línea base (proximal) si existe
    - Ratios H/C y O/C si existe (estabilidad)
    - Cenizas como proxy de sales/alkalinidad
    """
    score = 100.0
    flags, notes = [], []

    # ----- Riesgo oxidación (no absolutista)
    o2_ppm = safe_float(p.get("O2_ppm"), default=np.nan)
    o2_temp = safe_float(p.get("O2_temp_exposicion"), default=25.0)  # °C
    enf = (p.get("Metodo_enfriamiento") or "Desconocido").lower()

    # “exposición peligrosa” = aire/O2 significativo a T elevada
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

    # ----- Enfriamiento
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

    # ----- Proximal (si existe)
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
    else:
        # no castigar duro: modo simple no lo tendrá
        pass

    # ----- Cenizas como proxy sales/alkalinidad
    if np.isfinite(ash):
        if ash > 20:
            score -= 12
            flags.append("Cenizas altas")
            notes.append("Cenizas muy altas: riesgo de sales/pH alto; cuidado en cultivos sensibles.")
        elif ash > 12:
            score -= 6
            flags.append("Cenizas moderadas-altas")
            notes.append("Cenizas moderadas: monitorear EC/pH, especialmente en floricultura.")
    # ----- Ratios H/C y O/C (si elemental disponible)
    hc = safe_float(p.get("H_C_ratio"), default=np.nan)
    oc = safe_float(p.get("O_C_ratio"), default=np.nan)
    if np.isfinite(hc):
        if hc < 0.7:
            notes.append("H/C bajo: indica mayor aromaticidad (mejor estabilidad).")
        else:
            score -= 8
            flags.append("H/C alto")
            notes.append("H/C alto: menor aromaticidad; posible biochar menos estable.")
    if np.isfinite(oc):
        if oc < 0.2:
            notes.append("O/C bajo: indicador de alta estabilidad (tendencial).")
        else:
            score -= 8
            flags.append("O/C alto")
            notes.append("O/C alto: posible menor estabilidad / mayor reactividad.")

    # ----- Temperatura: no dogma, pero coherencia
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

    # “gate” suave: solo bloquea si riesgo oxidación extremo + enfriamiento malo
    can_prescribe = True
    if ("Riesgo oxidación alto (O₂ alto)" in flags) and ("Apagado con agua" in flags):
        can_prescribe = False

    # confidence_factor: afecta dosis y peso del ML
    # (más QC → más confianza. QC bajo → conservador)
    confidence_factor = float(np.clip(score / 85.0, 0.65, 1.15))

    return QCReport(
        score=score,
        confidence_factor=confidence_factor,
        can_prescribe=can_prescribe,
        flags=flags,
        notes=notes
    )


# =============================================================================
# REGLAS DETERMINÍSTICAS — ahora usa Factores_ajuste si existe
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
    """
    Espera (ideal):
    columnas: Objetivo, Variable, Min, Max, Factor, Aplica_a (suelo/biochar/cultivo/any)
    Si tu Excel no tiene exactamente estas columnas, hace best-effort.
    """
    if factores_df is None or factores_df.empty:
        return 1.0

    df = factores_df.copy()
    # normalizaciones suaves de nombres
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

        # dónde buscar el valor
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

        # rangos opcionales
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
    """
    - Si hay Excel: usa 'Algoritmos' + 'Factores_ajuste'
    - Si no: fallback simple (tu lógica actual)
    """
    # Fallback simple (si no hay Excel)
    if datos_excel is None:
        base = 20.0
        ph = safe_float(suelo.get("pH"), 6.5)
        mo = safe_float(suelo.get("MO"), 2.0)

        if objetivo.lower().startswith("fert"):
            base += 1.5 * (6.5 - ph) + 2.0 * (3.0 - mo)
        if objetivo.lower().startswith("rem"):
            met = safe_float(suelo.get("Metales"), 0.0)
            base += 0.03 * met

        # tamaño
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

    # Ejemplo: si hay coefs, aplica sobre variables disponibles (suelo + biochar + cultivo)
    # Nota: esto es general; puedes especializar por objetivo con reglas adicionales.
    for k, a in coefs.items():
        # buscar k en suelo/biochar/cultivo
        val = suelo.get(k, biochar.get(k, cultivo.get(k)))
        v = safe_float(val, default=np.nan)
        if np.isfinite(v):
            # aquí tu Excel debería definir el sentido; usamos una forma "lineal" genérica:
            dosis_base += a * v

    # Factores desde Excel (si están bien estructurados)
    f_total = aplicar_factores_excel(factores, objetivo, suelo, biochar, cultivo)

    # Ajuste por tamaño (ahora normalizado)
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
    # Si sensibilidad alta y cenizas altas: reducir
    if sensibilidad > 2.0 and np.isfinite(cenizas) and cenizas > 12:
        dosis_base *= 0.8

    objetivo_calidad = flor.get("Objetivo_calidad", "Larga vida en florero")
    if objetivo_calidad == "Larga vida en florero":
        dosis_base += 2.0
    elif objetivo_calidad == "Color intenso":
        dosis_base += 1.5

    return float(np.clip(dosis_base * factor_sistema, 5, 30))


# =============================================================================
# ML — XGBoost (Pipeline + OneHot + Imputer) + R² holdout
# =============================================================================

def entrenar_modelo_xgb_pipeline(df: pd.DataFrame, target: str = "dosis_efectiva") -> Tuple[Pipeline, float, List[str]]:
    if target not in df.columns:
        raise ValueError(f"El dataset debe incluir la columna '{target}'")

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # columnas esperadas (para alinear predicción)
    expected_cols = list(X.columns)

    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

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

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("model", model),
    ])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(Xtr, ytr)
    r2 = r2_score(yte, pipe.predict(Xte))

    return pipe, float(r2), expected_cols


def preparar_input_modelo(expected_cols: List[str], params_flat: Dict[str, Any]) -> pd.DataFrame:
    row = {}
    for c in expected_cols:
        row[c] = params_flat.get(c, np.nan)
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
        # guardar en disco temporalmente (Streamlit necesita path)
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
    help="Solo se activa si el Excel tiene hoja 'Dosis_experimental' con 'dosis_efectiva'."
)

# header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="main-header">🧬 Prescriptor Híbrido Biochar</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">QC + Reglas determinísticas + XGBoost (pipeline)</div>', unsafe_allow_html=True)
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
if "parametros_usuario" not in st.session_state:
    st.session_state.parametros_usuario = {}

# Autoentreno opcional (solo si el usuario lo pidió)
if auto_train and (not st.session_state.modelo_activo) and datos_excel is not None:
    try:
        df0 = datos_excel.get("dosis_exp")
        if df0 is not None and "dosis_efectiva" in df0.columns and len(df0) >= 30:
            pipe, r2, expected_cols = entrenar_modelo_xgb_pipeline(df0.dropna())
            st.session_state.modelo_pipe = pipe
            st.session_state.r2_score = r2
            st.session_state.expected_cols = expected_cols
            st.session_state.modelo_activo = True
    except Exception:
        # no interrumpir la app si falla autoentreno
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
            help="Activa inputs de línea base, O₂, ratios H/C y O/C, y enfriamiento."
        )

    if modo_experto:
        st.markdown("""
        <div class="expert-box">
        <b>Modo experto activo:</b> la prescripción se ajusta por un QC Report (riesgo de oxidación + estabilidad).
        </div>
        """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("🏜️ Suelo")
        ph_suelo = st.slider("pH del suelo", 3.5, 9.5, 6.5, 0.1)
        mo_suelo = st.slider("Materia Orgánica (%)", 0.1, 10.0, 2.0, 0.1)
        cic_suelo = st.slider("CIC (cmolc/kg)", 2.0, 50.0, 15.0, 0.5)
        textura = st.selectbox("Textura", ["Arena", "Franco-arenoso", "Franco", "Franco-arcilloso", "Arcilloso"])
        metales = st.number_input("Metales pesados (mg/kg)", 0.0, 500.0, 0.0, 1.0)

    with c2:
        st.subheader("🌿 Biochar (básico)")
        feedstock = st.selectbox("Materia prima", [
            "Madera", "Cáscara cacao", "Paja trigo", "Bambú",
            "Estiércol", "Paja arroz", "Cáscara arroz", "Lodo papel", "Otro"
        ])
        temp_pirolisis = st.slider("Temperatura pirólisis (°C)", 300, 900, 550, 10)
        ph_biochar = st.slider("pH del biochar", 5.0, 12.0, 9.0, 0.1)
        tamaño_ui = st.selectbox("Tamaño partícula", ["Fino (<1 mm)", "Medio (1-5 mm)", "Grueso (>5 mm)"])
        area_bet = st.slider("Área superficial BET (m²/g)", 10, 600, 300, 10)

        # EXPERTO: QC + proceso
        if modo_experto:
            st.markdown("---")
            st.subheader("⚗️ QC + Proceso (experto)")

            # Línea base (proximal)
            hum_total = st.slider("Humedad total biomasa (%)", 0.0, 60.0, 10.0, 0.5)
            volatiles = st.slider("Volátiles biomasa (%)", 0.0, 90.0, 70.0, 0.5)
            cenizas_biomasa = st.slider("Cenizas biomasa (%)", 0.0, 40.0, 3.0, 0.5)
            carbono_fijo = st.slider("Carbono fijo biomasa (%)", 0.0, 80.0, 17.0, 0.5)
            cierre = hum_total + volatiles + cenizas_biomasa + carbono_fijo
            st.caption(f"Cierre proximal = {cierre:.1f}% (ideal ~100%, ojo base seca vs húmeda)")

            # Oxígeno + exposición
            o2_ppm = st.number_input("O₂ residual (ppm) en reactor/enfriamiento", 0, 50000, 5000, 250)
            o2_temp = st.slider("Temperatura durante posible exposición a O₂ (°C)", 20, 600, 120, 10)

            # Enfriamiento
            metodo_enfriamiento = st.selectbox(
                "Método de enfriamiento",
                ["Cámara inerte", "Contacto indirecto", "En aire", "Agua directa", "Desconocido"],
                index=0
            )

            # Ratios (si hay elemental)
            st.markdown("**Estabilidad (si tienes elemental):**")
            hc_ratio = st.number_input("H/C (molar)", 0.0, 2.0, 0.65, 0.01)
            oc_ratio = st.number_input("O/C (molar)", 0.0, 1.5, 0.18, 0.01)

    with c3:
        st.subheader("🌱 Agronomía")
        objetivo = st.selectbox("Objetivo", ["Fertilidad", "Remediación", "Resiliencia hídrica", "Secuestro carbono", "Supresión patógenos"])
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

    # Guardar parámetros (flat + por grupos)
    suelo = {"pH": ph_suelo, "MO": mo_suelo, "CIC": cic_suelo, "Textura": textura, "Metales": metales}
    biochar = {
        "Feedstock": feedstock,
        "T_pirolisis": temp_pirolisis,
        "pH_biochar": ph_biochar,
        "Tamaño": norm_tamano(tamaño_ui),
        "BET": area_bet,
    }
    cultivo_d = {
        "Tipo": cultivo,
        "Riego": sistema_riego,
        "Clima": clima,
        "Sistema_cultivo": sistema_cultivo,
        "Tipo_producto": tipo_producto_floral,
        "Objetivo_calidad": objetivo_calidad,
        "Sensibilidad_salinidad": sensibilidad_salinidad,
    }

    # Agregar experto a biochar
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

    # QC Report (si modo experto; si no, hace QC “ligero” con lo disponible)
    qc = compute_qc_report(biochar)

    # Render QC
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
        st.error("GATE activado: con O₂ alto + apagado con agua, la app no recomienda dosis (alta probabilidad de material no controlado).")

    # Botón principal
    if st.button("🎯 Calcular Dosis Híbrida", type="primary", use_container_width=True, disabled=(not qc.can_prescribe)):
        with st.spinner("Calculando..."):
            # Determinístico (o flores)
            if es_flor:
                dosis_det = calcular_dosis_flores(suelo, biochar, cultivo_d)
                tipo_det = "Especializado floricultura"
            else:
                dosis_det = calcular_dosis_deterministica(objetivo, suelo, biochar, cultivo_d, datos_excel)
                tipo_det = "Determinístico"

            # Ajuste por QC (conservador cuando QC baja)
            dosis_det_qc = float(np.clip(dosis_det * qc.confidence_factor, 3, 60))

            dosis_xgb = None
            metodo = tipo_det

            # ML: predicción si modelo activo
            if st.session_state.modelo_activo and st.session_state.modelo_pipe is not None:
                try:
                    # Flatten: usar nombres de columnas esperadas
                    # Nota: si tu dataset usa nombres distintos, el usuario debe entrenar con esas columnas.
                    flat = {}
                    flat.update(suelo)
                    flat.update(biochar)
                    flat.update(cultivo_d)
                    flat["Objetivo"] = objetivo

                    df_in = preparar_input_modelo(st.session_state.expected_cols, flat)
                    dosis_xgb = float(st.session_state.modelo_pipe.predict(df_in)[0])
                except Exception as e:
                    st.warning(f"Predicción XGBoost no disponible: {e}")
                    dosis_xgb = None

            # Peso dinámico del ML
            if dosis_xgb is not None:
                r2 = st.session_state.r2_score
                peso_xgb = 0.6
                if r2 < 0.5:
                    peso_xgb = 0.3
                if qc.score < 60:
                    peso_xgb = min(peso_xgb, 0.35)

                dosis_final = (1 - peso_xgb) * dosis_det_qc + peso_xgb * dosis_xgb
                metodo = f"Híbrido (Det+QC + XGBoost | peso_xgb={peso_xgb:.2f})"
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
                st.metric("Determinístico", f"{dosis_det:.1f} t/Ha", delta=f"QC×{qc.confidence_factor:.2f} → {dosis_det_qc:.1f}")
            with r2c:
                if mostrar_xgb:
                    st.metric("XGBoost", f"{dosis_xgb:.1f} t/Ha", delta=f"R² holdout: {st.session_state.r2_score:.3f}")
                else:
                    st.metric("XGBoost", "—", delta="Modelo inactivo")
            with r3:
                st.metric("QC Score", f"{qc.score:.0f}/100", delta=pill_text)

            with st.expander("📋 Detalles y trazabilidad", expanded=True):
                st.markdown(f"""
**Suelo:** pH={ph_suelo}, MO={mo_suelo}%, CIC={cic_suelo}, Textura={textura}, Metales={metales}  
**Biochar:** Feedstock={feedstock}, T={temp_pirolisis}°C, pH={ph_biochar}, Tamaño={norm_tamano(tamaño_ui)}, BET={area_bet}  
**Cultivo:** {cultivo} | Riego={sistema_riego} | Clima={clima} | Objetivo={objetivo}  
**QC:** score={qc.score:.0f}, flags={", ".join(qc.flags) if qc.flags else "—"}
                """)
                if modo_experto:
                    st.markdown(f"""
**Experto:** O₂={o2_ppm} ppm (Texp={o2_temp}°C) | Enfriamiento={metodo_enfriamiento} | Proximal cierre={cierre:.1f}% | H/C={hc_ratio} | O/C={oc_ratio}
                    """)

            # Export
            out = pd.DataFrame({
                "Parámetro": [
                    "Dosis_final_t_ha", "Metodo", "QC_score", "QC_factor", "QC_flags",
                    "Objetivo", "Cultivo", "Riego", "Clima",
                    "pH_suelo", "MO", "CIC", "Textura", "Metales",
                    "Feedstock", "T_pirolisis", "pH_biochar", "Tamaño", "BET",
                ],
                "Valor": [
                    f"{dosis_final:.2f}", metodo, f"{qc.score:.0f}", f"{qc.confidence_factor:.2f}", "; ".join(qc.flags),
                    objetivo, cultivo, sistema_riego, clima,
                    ph_suelo, mo_suelo, cic_suelo, textura, metales,
                    feedstock, temp_pirolisis, ph_biochar, norm_tamano(tamaño_ui), area_bet,
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
    st.markdown("""
- El modelo **solo se activa** si entrenas con un dataset que tenga `dosis_efectiva`.
- El R² mostrado es **holdout** (20% test), no sobre los datos de entrenamiento.
""")

    uploaded_csv = st.file_uploader("📤 Subir dataset (CSV) con 'dosis_efectiva'", type=["csv"])

    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv)
            st.subheader("Vista previa")
            st.dataframe(df.head(), use_container_width=True)

            if "dosis_efectiva" not in df.columns:
                st.error("Falta la columna 'dosis_efectiva'.")
            else:
                if st.button("🚀 Entrenar XGBoost (Pipeline)", type="primary"):
                    with st.spinner("Entrenando..."):
                        pipe, r2, expected_cols = entrenar_modelo_xgb_pipeline(df.dropna())
                        st.session_state.modelo_pipe = pipe
                        st.session_state.r2_score = r2
                        st.session_state.expected_cols = expected_cols
                        st.session_state.modelo_activo = True

                    st.success("Modelo entrenado y activado ✅")
                    st.metric("R² (holdout)", f"{r2:.4f}")
                    st.caption(f"Columnas esperadas por el modelo: {len(expected_cols)}")

        except Exception as e:
            st.error(f"Error entrenando: {e}")

    st.markdown("---")
    st.subheader("Entrenar con dataset del Excel (opcional)")
    if datos_excel is None:
        st.info("No hay Excel cargado. Puedes entrenar subiendo un CSV.")
    else:
        df_excel = datos_excel.get("dosis_exp")
        if df_excel is None or df_excel.empty:
            st.warning("El Excel no trae 'Dosis_experimental' o está vacío.")
        else:
            st.caption(f"Filas en Dosis_experimental: {len(df_excel)}")
            if st.button("🔄 Entrenar con Dosis_experimental del Excel"):
                try:
                    with st.spinner("Entrenando..."):
                        pipe, r2, expected_cols = entrenar_modelo_xgb_pipeline(df_excel.dropna())
                        st.session_state.modelo_pipe = pipe
                        st.session_state.r2_score = r2
                        st.session_state.expected_cols = expected_cols
                        st.session_state.modelo_activo = True
                    st.success("Modelo entrenado con Excel ✅")
                    st.metric("R² (holdout)", f"{r2:.4f}")
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

    with st.expander("⚗️ Ciencia del Biochar (QC)", expanded=False):
        st.markdown("""
**Qué “cambia el juego” en biochar de ingeniería (en la app):**
- **Riesgo de oxidación**: O₂ + temperatura de exposición + método de enfriamiento.
- **Línea base (proximal)**: cierre de humedad/volátiles/cenizas/C fijo.
- **Estabilidad**: ratios **H/C** y **O/C** cuando hay elemental.
- **Cenizas** como proxy de sales/pH (especialmente importante en floricultura).
        """)


# =============================================================================
# TAB 4 — Ingeniería / QC
# =============================================================================

with tab4:
    st.header("⚗️ Futura calculadora de Ingeniería de Biochar")

    col_eng1, col_eng2 = st.columns(2)

    with col_eng1:
        eng_feedstock = st.selectbox(
            "Materia prima para cálculo",
            ["Madera", "Cáscara cacao", "Paja trigo", "Bambú", "Estiércol"],
            key="eng_feedstock"
        )
        eng_humedad = st.slider("Humedad (%)", 5.0, 30.0, 10.0, 1.0, key="eng_humedad")

    with col_eng2:
        eng_temperatura = st.slider("Temperatura (°C)", 300, 900, 550, 10, key="eng_temperatura")
        eng_tiempo = st.slider("Tiempo residencia (min)", 15, 180, 60, 5, key="eng_tiempo")
        eng_capacidad = st.number_input("Capacidad reactor (kg/h)", 10, 1000, 100, 10, key="eng_capacidad")

    if st.button("⚖️ Calcular Balance Completo", type="primary", key="btn_balance_completo"):
        st.write("✅ Botón funciona (Las funciones de balance/secuestro deben responder a un balance termoquímico riguroso con composición, O/C, H/C, rendimientos medidos, etc).")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align:center;color:#666;padding:0.9rem;'>
    <b>Prescriptor Híbrido Biochar {APP_VERSION} 🌱⚗️</b><br>
    QC (biochar serio) + reglas + XGBoost (pipeline) • NanoMof 2025 ©️
    </div>
    """,
    unsafe_allow_html=True
)




