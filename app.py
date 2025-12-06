import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CARGA Y PROCESAMIENTO DE DATOS DEL EXCEL
# ============================================================================

@st.cache_resource
def cargar_datos_excel():
    """Carga todas las hojas del Excel para reglas determinísticas"""
    try:
        # Si el Excel está en el mismo directorio
        excel_path = "Biochar_Prescriptor_Sistema_Completo_v1.0.xlsx"
        
        # Cargar todas las hojas necesarias
        parametros_df = pd.read_excel(excel_path, sheet_name='Parametros')
        rangos_df = pd.read_excel(excel_path, sheet_name='Rangos_suelo')
        algoritmos_df = pd.read_excel(excel_path, sheet_name='Algoritmos')
        factores_df = pd.read_excel(excel_path, sheet_name='Factores_ajuste')
        metadatos_df = pd.read_excel(excel_path, sheet_name='Metadatos_completos')
        dosis_exp_df = pd.read_excel(excel_path, sheet_name='Dosis_experimental')
        
        return {
            'parametros': parametros_df,
            'rangos': rangos_df,
            'algoritmos': algoritmos_df,
            'factores': factores_df,
            'metadatos': metadatos_df,
            'dosis_exp': dosis_exp_df
        }
    except Exception as e:
        st.error(f"Error cargando Excel: {e}")
        return None

# Cargar datos del Excel
datos_excel = cargar_datos_excel()

# ============================================================================
# 2. FUNCIONES DE MODELO HÍBRIDO
# ============================================================================

def calcular_dosis_deterministica(objetivo, parametros_suelo, parametros_biochar):
    """
    Calcula dosis usando ecuaciones determinísticas del Excel
    """
    if datos_excel is None:
        return 20.0  # Valor por defecto
    
    algoritmos = datos_excel['algoritmos']
    factores = datos_excel['factores']
    
    # Filtrar algoritmo para el objetivo
    algo_row = algoritmos[algoritmos['Objetivo'] == objetivo]
    
    if algo_row.empty:
        return 20.0
    
    # Extraer ecuación y coeficientes
    ecuacion = algo_row['Ecuación_dosis_base'].iloc[0]
    constante = float(algo_row['Constante'].iloc[0])
    
    # Parsear coeficientes (ejemplo: "pH:1.5, CIC:0.3, ...")
    coef_str = algo_row['Coeficientes'].iloc[0]
    coeficientes = {}
    for item in str(coef_str).split(','):
        if ':' in item:
            key, val = item.strip().split(':')
            coeficientes[key.strip()] = float(val.strip())
    
    # Calcular dosis base según parámetros críticos
    dosis_base = constante
    
    # Para simplificar, calculamos con los parámetros disponibles
    # En una implementación completa, aquí se aplicaría la lógica específica
    if objetivo == 'Fertilidad':
        # Ejemplo simplificado
        if 'pH' in parametros_suelo and 'pH' in coeficientes:
            dosis_base += coeficientes['pH'] * (6.5 - parametros_suelo['pH'])
        if 'MO' in parametros_suelo and 'MO' in coeficientes:
            dosis_base += coeficientes['MO'] * (3.0 - parametros_suelo['MO'])
    
    elif objetivo == 'Remediación':
        if 'Metales' in parametros_suelo and 'Metales' in coeficientes:
            dosis_base += coeficientes['Metales'] * parametros_suelo['Metales']
    
    # Aplicar factores de ajuste
    factor_total = 1.0
    
    # Ajuste por pH del suelo
    ph_suelo = parametros_suelo.get('pH', 6.5)
    if ph_suelo < 4.5:
        factor_total *= 2.5
    elif ph_suelo < 5.5:
        factor_total *= 1.8
    elif ph_suelo < 6.0 or ph_suelo > 7.5:
        factor_total *= 1.3
    
    # Ajuste por MO
    mo_suelo = parametros_suelo.get('MO', 2.0)
    if mo_suelo < 0.5:
        factor_total *= 3.0
    elif mo_suelo < 1.5:
        factor_total *= 2.0
    elif mo_suelo < 3.0:
        factor_total *= 1.4
    
    # Ajuste por tamaño de biochar
    tamaño = parametros_biochar.get('Tamaño', 'Medio')
    if tamaño == 'Fino':
        factor_total *= 1.3
    elif tamaño == 'Grueso':
        factor_total *= 0.8
    
    return max(5.0, min(50.0, dosis_base * factor_total))

def entrenar_modelo_hibrido(df_entrenamiento):
    """
    Entrena un modelo XGBoost con múltiples características
    """
    # Codificar variables categóricas
    categorical_cols = df_entrenamiento.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_entrenamiento[col] = le.fit_transform(df_entrenamiento[col].astype(str))
    
    # Separar características y objetivo
    X = df_entrenamiento.drop('dosis_efectiva', axis=1)
    y = df_entrenamiento['dosis_efectiva']
    
    # Entrenar modelo
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    
    model.fit(X, y)
    
    # Calcular R2
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    return model, r2, list(X.columns)

# ============================================================================
# 3. INICIALIZACIÓN DE LA APLICACIÓN
# ============================================================================

# Configuración de página
st.set_page_config(
    page_title="Prescriptor Híbrido Biochar",
    layout="wide",
    page_icon="🌱"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<h1 class="main-header">🧬 Prescriptor Híbrido Biochar</h1>', unsafe_allow_html=True)
    st.markdown("**Modelo combinado: XGBoost + Reglas Determinísticas**")
with col2:
    st.image("logonanomof.png", width=300)

# ============================================================================
# 4. ESTADO DE LA SESIÓN
# ============================================================================

# Inicializar estado de sesión
if 'modelo_hibrido' not in st.session_state:
    # Modelo inicial con datos de ejemplo del Excel
    if datos_excel is not None:
        df_ejemplo = datos_excel['dosis_exp'].copy()
        # Seleccionar columnas numéricas para entrenamiento inicial
        numeric_cols = df_ejemplo.select_dtypes(include=[np.number]).columns
        if 'dosis_efectiva' in df_ejemplo.columns:
            cols_entrenamiento = list(numeric_cols)
            if 'dosis_efectiva' in cols_entrenamiento:
                cols_entrenamiento.remove('dosis_efectiva')
            df_train = df_ejemplo[cols_entrenamiento + ['dosis_efectiva']].dropna()
            
            if len(df_train) > 10:
                modelo, r2, features = entrenar_modelo_hibrido(df_train)
                st.session_state.modelo_hibrido = modelo
                st.session_state.r2_score = r2
                st.session_state.features_modelo = features
                st.session_state.modelo_activo = True
            else:
                st.session_state.modelo_activo = False
        else:
            st.session_state.modelo_activo = False
    else:
        st.session_state.modelo_activo = False

if 'parametros_usuario' not in st.session_state:
    st.session_state.parametros_usuario = {
        'suelo': {},
        'biochar': {},
        'cultivo': {},
        'objetivo': 'Fertilidad'
    }

# ============================================================================
# 5. INTERFAZ PRINCIPAL - PESTAÑAS
# ============================================================================

tab1, tab2, tab3 = st.tabs([
    "🎯 Prescripción Híbrida", 
    "📊 Entrenamiento Avanzado", 
    "📚 Base de Conocimiento"
])

# ============================================================================
# PESTAÑA 1: PRESCRIPCIÓN HÍBRIDA
# ============================================================================

with tab1:
    st.header("Prescripción Personalizada")
    
    col_input1, col_input2, col_input3 = st.columns(3)
    
    with col_input1:
        st.subheader("🏜️ Parámetros del Suelo")
        
        # Parámetros principales del suelo
        ph_suelo = st.slider("pH del suelo", 3.5, 9.5, 6.5, 0.1)
        mo_suelo = st.slider("Materia Orgánica (%)", 0.1, 10.0, 2.0, 0.1)
        cic_suelo = st.slider("CIC (cmolc/kg)", 2.0, 50.0, 15.0, 0.5)
        textura = st.selectbox("Textura del suelo", [
            "Arena", "Franco-arenoso", "Franco", "Franco-arcilloso", "Arcilloso"
        ])
        
        # Metales pesados (si aplica)
        metales = st.number_input("Metales pesados (mg/kg)", 0.0, 500.0, 0.0, 1.0)
    
    with col_input2:
        st.subheader("🌿 Propiedades del Biochar")
        
        feedstock = st.selectbox("Materia prima", [
            "Madera", "Cáscara cacao", "Paja trigo", "Bambú", 
            "Estiércol", "Paja arroz", "Cáscara arroz", "Lodo papel"
        ])
        
        temp_pirolisis = st.slider("Temperatura pirólisis (°C)", 300, 900, 550, 10)
        ph_biochar = st.slider("pH del biochar", 5.0, 12.0, 9.0, 0.1)
        tamaño = st.selectbox("Tamaño partícula", ["Fino (<1 mm)", "Medio (1-5 mm)", "Grueso (>5 mm)"])
        area_bet = st.slider("Área superficial (m²/g)", 10, 600, 300, 10)
    
    with col_input3:
        st.subheader("🌱 Condiciones Agronómicas")
        
        objetivo = st.selectbox("Objetivo principal", [
            "Fertilidad", "Remediación", "Resiliencia hídrica", 
            "Secuestro carbono", "Supresión patógenos"
        ])
        
        cultivo = st.selectbox("Tipo de cultivo", [
            "Teff", "Hortalizas", "Trigo", "Girasol", "Maíz", "Pasto", 
            "Cacao", "Frijol", "Palma", "Sorgo", "Tomate", "Soja"
        ])
        
        sistema_riego = st.selectbox("Sistema de riego", [
            "Gravedad", "Aspersión", "Goteo", "No irrigado"
        ])
        
        clima = st.selectbox("Clima", [
            "Árido", "Semiárido", "Mediterráneo", "Tropical", "Templado"
        ])
    
    # Almacenar parámetros
    st.session_state.parametros_usuario = {
        'suelo': {
            'pH': ph_suelo,
            'MO': mo_suelo,
            'CIC': cic_suelo,
            'Textura': textura,
            'Metales': metales
        },
        'biochar': {
            'Feedstock': feedstock,
            'T_pirolisis': temp_pirolisis,
            'pH_biochar': ph_biochar,
            'Tamaño': tamaño,
            'BET': area_bet
        },
        'cultivo': {
            'Tipo': cultivo,
            'Riego': sistema_riego,
            'Clima': clima
        },
        'objetivo': objetivo
    }
    
    # Botón para calcular
    if st.button("🎯 Calcular Dosis Híbrida", type="primary", use_container_width=True):
        
        with st.spinner("Calculando dosis óptima..."):
            # 1. Calcular dosis determinística
            dosis_det = calcular_dosis_deterministica(
                objetivo,
                st.session_state.parametros_usuario['suelo'],
                st.session_state.parametros_usuario['biochar']
            )
            
            # 2. Si hay modelo XGBoost activo, calcular predicción
            dosis_xgb = None
            if st.session_state.get('modelo_activo', False):
                try:
                    # Preparar datos para XGBoost
                    features_modelo = st.session_state.get('features_modelo', [])
                    
                    # Crear dataframe con valores por defecto para todas las features
                    input_data = {}
                    for feat in features_modelo:
                        # Buscar el valor en los parámetros del usuario
                        found = False
                        for categoria in ['suelo', 'biochar', 'cultivo']:
                            for key, value in st.session_state.parametros_usuario[categoria].items():
                                if key.lower() == feat.lower():
                                    input_data[feat] = float(value) if isinstance(value, (int, float)) else 0.0
                                    found = True
                                    break
                            if found:
                                break
                        if not found:
                            input_data[feat] = 0.0  # Valor por defecto
                    
                    df_input = pd.DataFrame([input_data])
                    dosis_xgb = st.session_state.modelo_hibrido.predict(df_input)[0]
                except Exception as e:
                    st.warning(f"Predicción XGBoost no disponible: {e}")
            
            # 3. Combinar resultados (promedio ponderado)
            if dosis_xgb is not None:
                dosis_final = (dosis_det * 0.4 + dosis_xgb * 0.6)
                metodo = "Híbrido (Determinístico + XGBoost)"
            else:
                dosis_final = dosis_det
                metodo = "Determinístico"
            
            # Mostrar resultados
            st.markdown("---")
            
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.metric(
                    label="Dosis Recomendada",
                    value=f"{dosis_final:.1f} t/Ha",
                    delta=f"{metodo}"
                )
            
            with col_res2:
                st.metric(
                    label="Dosis Determinística",
                    value=f"{dosis_det:.1f} t/Ha"
                )
            
            if dosis_xgb:
                with col_res3:
                    st.metric(
                        label="Dosis XGBoost",
                        value=f"{dosis_xgb:.1f} t/Ha"
                    )
            
            # Explicación detallada
            with st.expander("📋 Detalles de la prescripción"):
                st.markdown(f"""
                **Parámetros considerados:**
                - **Suelo:** pH {ph_suelo}, MO {mo_suelo}%, CIC {cic_suelo} cmolc/kg, Textura {textura}
                - **Biochar:** {feedstock} a {temp_pirolisis}°C, pH {ph_biochar}, {tamaño}
                - **Cultivo:** {cultivo} en clima {clima}
                - **Objetivo:** {objetivo}
                
                **Metodología:** {metodo}
                
                **Rango típico para {objetivo.lower()}:** {{
                    'Fertilidad': '10-30 t/Ha',
                    'Remediación': '15-50 t/Ha',
                    'Resiliencia hídrica': '10-25 t/Ha',
                    'Secuestro carbono': '20-50 t/Ha',
                    'Supresión patógenos': '10-20 t/Ha'
                }}.get(objetivo, 'Consultar referencia')
                """)
            
            # Recomendaciones adicionales
            st.info(f"""
            **💡 Recomendaciones adicionales:**
            1. Aplicar en dosis divididas para mejor incorporación
            2. Considerar época de aplicación según ciclo del cultivo
            3. Monitorear pH del suelo después de 3 meses
            4. Evaluar respuesta del cultivo a los 60 días
            """)

# ============================================================================
# PESTAÑA 2: ENTRENAMIENTO AVANZADO
# ============================================================================

with tab2:
    st.header("Entrenamiento del Modelo XGBoost")
    
    st.markdown("""
    Sube tu dataset con múltiples parámetros para entrenar un modelo XGBoost personalizado.
    El dataset debe incluir **dosis_efectiva** como variable objetivo.
    """)
    
    uploaded_file = st.file_uploader("📤 Subir dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_cargado = pd.read_csv(uploaded_file)
            
            st.subheader("Vista previa del dataset")
            st.dataframe(df_cargado.head(), use_container_width=True)
            
            st.subheader("Estadísticas descriptivas")
            st.dataframe(df_cargado.describe(), use_container_width=True)
            
            # Verificar si tiene columna objetivo
            if 'dosis_efectiva' not in df_cargado.columns:
                st.error("❌ El dataset debe contener la columna 'dosis_efectiva'")
            else:
                if st.button("🚀 Entrenar Modelo XGBoost", type="primary"):
                    with st.spinner("Entrenando modelo..."):
                        modelo_nuevo, r2_nuevo, features_nuevas = entrenar_modelo_hibrido(df_cargado)
                        
                        # Actualizar estado de sesión
                        st.session_state.modelo_hibrido = modelo_nuevo
                        st.session_state.r2_score = r2_nuevo
                        st.session_state.features_modelo = features_nuevas
                        st.session_state.modelo_activo = True
                        
                        st.success(f"✅ Modelo entrenado exitosamente!")
                        
                        col_metric1, col_metric2 = st.columns(2)
                        with col_metric1:
                            st.metric("Coeficiente R²", f"{r2_nuevo:.4f}")
                        with col_metric2:
                            st.metric("Número de características", len(features_nuevas))
                        
                        st.info(f"""
                        **Características utilizadas:** {', '.join(features_nuevas[:5])}...
                        """)
                        
        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")
    
    # Opción de usar datos del Excel
    elif datos_excel is not None:
        st.markdown("---")
        st.subheader("Usar datos de referencia del Excel")
        
        if st.button("🔄 Entrenar con datos del Excel"):
            df_excel = datos_excel['dosis_exp'].copy()
            if 'dosis_efectiva' in df_excel.columns:
                with st.spinner("Entrenando con datos de referencia..."):
                    modelo_ref, r2_ref, features_ref = entrenar_modelo_hibrido(df_excel)
                    
                    st.session_state.modelo_hibrido = modelo_ref
                    st.session_state.r2_score = r2_ref
                    st.session_state.features_modelo = features_ref
                    st.session_state.modelo_activo = True
                    
                    st.success("Modelo de referencia entrenado exitosamente!")
                    st.metric("R² del modelo de referencia", f"{r2_ref:.4f}")

# ============================================================================
# PESTAÑA 3: BASE DE CONOCIMIENTO
# ============================================================================

with tab3:
    st.header("Base de Conocimiento y Referencias")
    
    if datos_excel is None:
        st.warning("No se pudo cargar la base de conocimientos del Excel")
    else:
        tab_ref1, tab_ref2, tab_ref3 = st.tabs(["📊 Parámetros", "⚙️ Algoritmos", "📈 Casos Históricos"])
        
        with tab_ref1:
            st.subheader("Parámetros del sistema")
            st.dataframe(datos_excel['parametros'], use_container_width=True)
        
        with tab_ref2:
            st.subheader("Algoritmos determinísticos")
            st.dataframe(datos_excel['algoritmos'], use_container_width=True)
            
            st.subheader("Factores de ajuste")
            st.dataframe(datos_excel['factores'], use_container_width=True)
        
        with tab_ref3:
            st.subheader("Casos históricos documentados")
            st.dataframe(datos_excel['metadatos'], use_container_width=True)

# ============================================================================
# 6. PANEL DE INFORMACIÓN
# ============================================================================

st.markdown("---")

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.markdown("### 🔬 Método Híbrido")
    st.markdown("""
    - **XGBoost:** Aprendizaje automático no lineal
    - **Determinístico:** Ecuaciones basadas en investigación
    - **Combinación:** Mayor robustez y precisión
    """)

with col_info2:
    st.markdown("### 📋 Estado del Sistema")
    if st.session_state.get('modelo_activo', False):
        st.success(f"✅ Modelo XGBoost activo (R²: {st.session_state.get('r2_score', 'N/A'):.4f})")
    else:
        st.warning("⚠️ Usando solo modelo determinístico")
    
    if datos_excel:
        st.success("✅ Base de conocimientos cargada")

with col_info3:
    st.markdown("### 📚 Referencias")
    st.markdown("""
    - Woolf et al. (2010) Nature Communications
    - Jeffery et al. (2017) Agronomy for Sustainable Development
    - Lehmann & Joseph (2015) Biochar for Environmental Management
    """)

# ============================================================================
# 7. FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
    <b>Prescriptor Híbrido Biochar v2.0</b> • 
    Combinando inteligencia artificial con conocimiento agronómico • 
    NanoMof 2025 ©️
    </div>
    """,
    unsafe_allow_html=True
)
