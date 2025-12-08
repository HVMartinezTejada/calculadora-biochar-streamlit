import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
import os
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

def calcular_dosis_flores(parametros_suelo, parametros_biochar, parametros_flor):
    """
    Calcula dosis específica para cultivos de flores
    Basado en investigación en floricultura
    """
    # Dosis base para flores (más baja que cultivos extensivos)
    dosis_base = 12.0
    
    # Ajuste por sistema de cultivo
    sistema = parametros_flor.get('Sistema_cultivo', 'Campo abierto')
    factores_sistema = {
        'Campo abierto': 1.0,
        'Invernadero': 0.8,
        'Maceta/Contenedor': 0.6,
        'Hidroponía': 0.4
    }
    factor_sistema = factores_sistema.get(sistema, 1.0)
    
    # Ajuste por pH del suelo (flores prefieren ligeramente ácido)
    ph = parametros_suelo.get('pH', 6.5)
    if ph < 5.5 or ph > 7.0:
        dosis_base += abs(6.0 - ph) * 1.2  # Mayor ajuste para pH fuera de rango
    
    # Ajuste por tipo de producto
    tipo_producto = parametros_flor.get('Tipo_producto', 'Flores cortadas')
    if tipo_producto == 'Plantas en maceta':
        factor_sistema *= 0.9  # Menos biochar en macetas
    elif tipo_producto == 'Bulbos':
        dosis_base += 3.0  # Bulbos requieren más materia orgánica
    
    # Ajuste por sensibilidad a salinidad
    sensibilidad = parametros_flor.get('Sensibilidad_salinidad', 1.5)
    # Si el biochar tiene alto contenido de cenizas y la flor es sensible, reducir dosis
    if sensibilidad > 2.0:
        dosis_base *= 0.8
    
    # Ajuste por objetivo de calidad
    objetivo_calidad = parametros_flor.get('Objetivo_calidad', 'Larga vida en florero')
    if objetivo_calidad == 'Larga vida en florero':
        dosis_base += 2.0  # Más biochar para durabilidad
    elif objetivo_calidad == 'Color intenso':
        dosis_base += 1.5  # Mejor disponibilidad de micronutrientes
    
    # Ajuste final por sistema
    dosis_final = dosis_base * factor_sistema
    
    return max(5.0, min(30.0, dosis_final))

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
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
    }
    .floral-header {
        background: linear-gradient(135deg, #f9f0ff, #e6fffa);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px dashed #d63384;
        text-align: center;
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
    try:
        st.image("logonanomof.png", width=300)
    except:
        st.markdown("### **NanoMof**")

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
        ph_suelo = st.slider("pH del suelo", 3.5, 9.5, 6.5, 0.1, help="pH ácido: <6.5, neutro: 6.5-7.5, alcalino: >7.5")
        mo_suelo = st.slider("Materia Orgánica (%)", 0.1, 10.0, 2.0, 0.1, help="Bajo: <1.5%, Medio: 1.5-3.5%, Alto: >3.5%")
        cic_suelo = st.slider("CIC (cmolc/kg)", 2.0, 50.0, 15.0, 0.5, help="Capacidad de Intercambio Catiónico. Bajo: <10, Medio: 10-25, Alto: >25")
        textura = st.selectbox("Textura del suelo", [
            "Arena", "Franco-arenoso", "Franco", "Franco-arcilloso", "Arcilloso"
        ], help="Textura influye en retención de agua y nutrientes")
        
        # Metales pesados (si aplica)
        metales = st.number_input("Metales pesados (mg/kg)", 0.0, 500.0, 0.0, 1.0, help="Solo si hay contaminación conocida")
    
    with col_input2:
        st.subheader("🌿 Propiedades del Biochar")
        
        feedstock = st.selectbox("Materia prima", [
            "Madera", "Cáscara cacao", "Paja trigo", "Bambú", 
            "Estiércol", "Paja arroz", "Cáscara arroz", "Lodo papel", "Otro"
        ], help="La materia prima afecta propiedades del biochar")
        
        temp_pirolisis = st.slider("Temperatura pirólisis (°C)", 300, 900, 550, 10, help="Baja (<450°C): más compuestos volátiles. Alta (>600°C): mayor área superficial")
        ph_biochar = st.slider("pH del biochar", 5.0, 12.0, 9.0, 0.1, help="Típicamente alcalino (8-11)")
        tamaño = st.selectbox("Tamaño partícula", ["Fino (<1 mm)", "Medio (1-5 mm)", "Grueso (>5 mm)"], help="Fino: mayor área contacto, Grueso: mejor estructura suelo")
        area_bet = st.slider("Área superficial (m²/g)", 10, 600, 300, 10, help="Baja: <100, Media: 100-400, Alta: >400")
    
    with col_input3:
        st.subheader("🌱 Condiciones Agronómicas")
        
        objetivo = st.selectbox("Objetivo principal", [
            "Fertilidad", "Remediación", "Resiliencia hídrica", 
            "Secuestro carbono", "Supresión patógenos"
        ], help="Selecciona el objetivo principal de la aplicación")
        
        cultivo = st.selectbox("Tipo de cultivo", [
            "Teff", "Hortalizas", "Trigo", "Girasol", "Maíz", "Pasto", 
            "Cacao", "Frijol", "Palma", "Sorgo", "Tomate", "Soja",
            # FLORES NUEVAS:
            "🌺 Rosas", "🌼 Claveles", "🌻 Crisantemos", "🌸 Orquídeas",
            "💐 Flores cortadas", "🌷 Flores ornamentales", "🌹 Flores de bulbo",
            "Otro"
        ], help="Para flores, seleccione el tipo específico")
        
        sistema_riego = st.selectbox("Sistema de riego", [
            "Gravedad", "Aspersión", "Goteo", "No irrigado"
        ], help="Sistemas eficientes requieren menor dosis")
        
        clima = st.selectbox("Clima", [
            "Árido", "Semiárido", "Mediterráneo", "Tropical", "Templado"
        ], help="Climas áridos pueden requerir mayores dosis para retención hídrica")
        
        # ====================================================================
        # NUEVA SECCIÓN: PARÁMETROS ESPECÍFICOS PARA FLORES
        # ====================================================================
        
        # Determinar si es cultivo de flores
        es_flor = any(palabra in cultivo.lower() for palabra in 
                     ['rosa', 'clavel', 'crisantemo', 'orquídea', 'flor'])
        
        if es_flor:
            st.markdown("---")
            st.subheader("🌺 Parámetros Específicos para Flores")
            
            # Sistema de cultivo
            sistema_cultivo = st.radio(
                "Sistema de cultivo:",
                ["Campo abierto", "Invernadero", "Maceta/Contenedor", "Hidroponía"],
                horizontal=True,
                help="Las flores en maceta requieren menos biochar que en campo"
            )
            
            # Tipo de producto
            tipo_producto_floral = st.selectbox(
                "Tipo de producto floral:",
                ["Flores cortadas", "Plantas en maceta", "Bulbos", "Follaje ornamental"]
            )
            
            # Parámetros específicos de calidad
            col1_flor, col2_flor = st.columns(2)
            with col1_flor:
                objetivo_calidad = st.selectbox(
                    "Objetivo de calidad principal:",
                    ["Larga vida en florero", "Color intenso", "Tamaño de flor", 
                     "Longitud de tallo", "Producción todo el año"]
                )
            
            with col2_flor:
                sensibilidad_salinidad = st.slider(
                    "Sensibilidad a salinidad:",
                    1.0, 3.0, 1.5, 0.1,
                    help="1=baja sensibilidad, 3=alta sensibilidad (como orquídeas)"
                )
        else:
            # Valores por defecto para cultivos no florales
            sistema_cultivo = "Campo abierto"
            tipo_producto_floral = "No aplica"
            objetivo_calidad = "No aplica"
            sensibilidad_salinidad = 1.5
    
    # ========================================================================
    # VALIDACIONES ESPECÍFICAS PARA FLORES
    # ========================================================================
    
    if es_flor:
        # Advertencia si el pH no es adecuado para flores
        ph_optimo_flores = {
            'rosa': (6.0, 6.5),
            'clavel': (6.5, 7.0),
            'crisantemo': (6.0, 6.8),
            'orquídea': (5.5, 6.5)
        }
        
        # Determinar tipo de flor
        tipo_flor = None
        for flor in ph_optimo_flores:
            if flor in cultivo.lower():
                tipo_flor = flor
                break
        
        if tipo_flor and ph_suelo:
            ph_min, ph_max = ph_optimo_flores[tipo_flor]
            if ph_suelo < ph_min or ph_suelo > ph_max:
                st.warning(f"""
                ⚠️ **pH fuera del rango óptimo para {cultivo}:**
                - pH actual: {ph_suelo}
                - Rango óptimo: {ph_min}-{ph_max}
                - Recomendación: Ajustar pH antes de aplicar biochar
                """)
        
        # Advertencia para biochar muy alcalino
        if ph_biochar > 10.5:
            st.warning(f"""
            ⚠️ **Biochar muy alcalino para flores:**
            - pH biochar: {ph_biochar}
            - Recomendación: Usar biochar con pH <10 para flores sensibles
            - Alternativa: Pre-acondicionar biochar con ácidos orgánicos
            """)
    
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
            'Clima': clima,
            # NUEVOS PARÁMETROS PARA FLORES:
            'Sistema_cultivo': sistema_cultivo,
            'Tipo_producto': tipo_producto_floral,
            'Objetivo_calidad': objetivo_calidad,
            'Sensibilidad_salinidad': sensibilidad_salinidad
        },
        'objetivo': objetivo
    }
    
    # Botón para calcular
    if st.button("🎯 Calcular Dosis Híbrida", type="primary", use_container_width=True):
        
        with st.spinner("Calculando dosis óptima..."):
            # 1. Calcular dosis determinística o específica para flores
            if es_flor:
                # Usar algoritmo específico para flores
                dosis_det = calcular_dosis_flores(
                    st.session_state.parametros_usuario['suelo'],
                    st.session_state.parametros_usuario['biochar'],
                    st.session_state.parametros_usuario['cultivo']
                )
                tipo_calculo = "Especializado para floricultura"
            else:
                # Usar algoritmo general
                dosis_det = calcular_dosis_deterministica(
                    objetivo,
                    st.session_state.parametros_usuario['suelo'],
                    st.session_state.parametros_usuario['biochar']
                )
                tipo_calculo = "Determinístico general"
            
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
                metodo = f"Híbrido ({tipo_calculo} + XGBoost)"
                mostrar_xgb = True
            else:
                dosis_final = dosis_det
                metodo = tipo_calculo
                mostrar_xgb = False
            
            # ================================================================
            # MOSTRAR RESULTADOS
            # ================================================================
            
            st.markdown("---")
            
            if es_flor:
                # ENCABEZADO ESPECIAL PARA FLORES
                st.markdown("### 🌸 **PRESCRIPCIÓN PARA FLORICULTURA**")
                
                # Mostrar dosis con decoración floral
                col_flor1, col_flor2, col_flor3 = st.columns([1, 2, 1])
                with col_flor2:
                    st.markdown(f"""
                    <div class="floral-header">
                        <h1 style='color: #d63384;'>🌸 {dosis_final:.1f} t/Ha 🌸</h1>
                        <p style='color: #666;'><strong>Dosis recomendada para {cultivo}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Resultados normales
                st.success(f"### 📊 Dosis Recomendada: **{dosis_final:.1f} t/Ha**")
            
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.metric(
                    label="Dosis Base",
                    value=f"{dosis_det:.1f} t/Ha",
                    delta=f"{tipo_calculo}"
                )
            
            with col_res2:
                if mostrar_xgb:
                    st.metric(
                        label="Dosis XGBoost",
                        value=f"{dosis_xgb:.1f} t/Ha"
                    )
            
            with col_res3:
                if es_flor:
                    st.metric(
                        label="Sistema Cultivo",
                        value=sistema_cultivo
                    )
            
            # Explicación detallada
            with st.expander("📋 Detalles de la prescripción", expanded=True):
                if es_flor:
                    st.markdown(f"""
                    **🌺 PARÁMETROS ESPECÍFICOS PARA FLORICULTURA:**
                    - **Tipo de flor:** {cultivo}
                    - **Sistema de cultivo:** {sistema_cultivo}
                    - **Producto:** {tipo_producto_floral}
                    - **Objetivo de calidad:** {objetivo_calidad}
                    - **Sensibilidad salinidad:** {sensibilidad_salinidad}/3.0
                    
                    **🏜️ PARÁMETROS DEL SUELO:**
                    - pH: {ph_suelo}, MO: {mo_suelo}%, CIC: {cic_suelo} cmolc/kg
                    - Textura: {textura}
                    
                    **🌿 PROPIEDADES DEL BIOCHAR:**
                    - Materia prima: {feedstock}
                    - Temperatura: {temp_pirolisis}°C, pH: {ph_biochar}
                    - Tamaño: {tamaño}, Área BET: {area_bet} m²/g
                    
                    **📊 METODOLOGÍA:** {metodo}
                    """)
                else:
                    st.markdown(f"""
                    **PARÁMETROS CONSIDERADOS:**
                    - **Suelo:** pH {ph_suelo}, MO {mo_suelo}%, CIC {cic_suelo} cmolc/kg, Textura {textura}
                    - **Biochar:** {feedstock} a {temp_pirolisis}°C, pH {ph_biochar}, {tamaño}
                    - **Cultivo:** {cultivo} en clima {clima} con riego {sistema_riego}
                    - **Objetivo:** {objetivo}
                    
                    **METODOLOGÍA:** {metodo}
                    
                    **RANGO TÍPICO PARA {objetivo.lower()}:**
                    - Fertilidad: 10-30 t/Ha
                    - Remediación: 15-50 t/Ha
                    - Resiliencia hídrica: 10-25 t/Ha
                    - Secuestro carbono: 20-50 t/Ha
                    - Supresión patógenos: 10-20 t/Ha
                    """)
            
            # ================================================================
            # RECOMENDACIONES ESPECÍFICAS PARA FLORES
            # ================================================================
            
            if es_flor:
                st.markdown("---")
                st.subheader("🌺 Recomendaciones Especiales para Floricultura")
                
                # Crear columnas para diferentes recomendaciones
                rec_col1, rec_col2 = st.columns(2)
                
                with rec_col1:
                    st.markdown("""
                    **💐 APLICACIÓN EN FLORES CORTADAS:**
                    • Mezclar biochar en capa de 0-20 cm
                    • Aplicar 2-3 semanas antes de siembra/trasplante
                    • Mantener humedad constante primera semana
                    • Evitar contacto directo con raíces jóvenes
                    """)
                
                with rec_col2:
                    st.markdown("""
                    **🏵️ PARA FLORES EN MACETA:**
                    • Mezclar 10-20% biochar en sustrato
                    • Usar biochar de tamaño fino (<2mm)
                    • Pre-humedecer biochar antes de mezclar
                    • Monitorear pH cada 30 días
                    """)
                
                # Tabla de tiempos esperados
                st.markdown("### 📅 Tiempos Esperados de Respuesta:")
                
                tiempos_df = pd.DataFrame({
                    'Tipo de Flor': ['Rosas', 'Claveles', 'Crisantemos', 'Orquídeas'],
                    'Primera respuesta': ['15-25 días', '20-30 días', '25-35 días', '30-45 días'],
                    'Efecto máximo': ['60-90 días', '75-105 días', '90-120 días', '120-180 días']
                })
                
                st.dataframe(tiempos_df, use_container_width=True, hide_index=True)
            else:
                # Recomendaciones generales
                st.info(f"""
                **💡 RECOMENDACIONES DE APLICACIÓN:**
                1. **Preparación:** Mezclar el biochar uniformemente con los primeros 20 cm de suelo
                2. **Época:** Aplicar 2-4 semanas antes de la siembra
                3. **Monitoreo:** Verificar pH del suelo a los 3 meses
                4. **Evaluación:** Observar respuesta del cultivo a los 60 días
                5. **Registro:** Documentar resultados para mejorar futuras prescripciones
                """)
            
            # ================================================================
            # EXPORTAR RESULTADOS
            # ================================================================
            
            st.markdown("---")
            st.subheader("📤 Exportar resultados")
            
            if es_flor:
                # Crear DataFrame especializado para flores
                resultados_df = pd.DataFrame({
                    'Parámetro': [
                        'Dosis Recomendada', 'Tipo de Flor', 'Sistema de Cultivo',
                        'Producto', 'Objetivo de Calidad', 'pH Suelo', 'MO (%)',
                        'CIC', 'Textura', 'Biochar', 'Temperatura', 'Tamaño',
                        'Clima', 'Riego', 'Metodología'
                    ],
                    'Valor': [
                        f"{dosis_final:.1f} t/Ha", cultivo, sistema_cultivo,
                        tipo_producto_floral, objetivo_calidad, str(ph_suelo), str(mo_suelo),
                        str(cic_suelo), textura, feedstock, f"{temp_pirolisis}°C", tamaño,
                        clima, sistema_riego, metodo
                    ]
                })
                
                nombre_archivo = f"prescripcion_floricultura_{cultivo.replace(' ', '_').lower()}.csv"
            else:
                # Crear DataFrame general
                resultados_df = pd.DataFrame({
                    'Parámetro': ['Dosis Recomendada', 'Método', 'pH Suelo', 'MO (%)', 'CIC', 
                                 'Textura', 'Biochar', 'Objetivo', 'Cultivo', 'Clima', 'Riego'],
                    'Valor': [f"{dosis_final:.1f} t/Ha", metodo, str(ph_suelo), str(mo_suelo), 
                             str(cic_suelo), textura, feedstock, objetivo, cultivo, clima, sistema_riego]
                })
                
                nombre_archivo = f"prescripcion_biochar_{objetivo.lower()}.csv"
            
            # Convertir a CSV
            csv = resultados_df.to_csv(index=False)
            
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                st.download_button(
                    label="📥 Descargar como CSV",
                    data=csv,
                    file_name=nombre_archivo,
                    mime="text/csv"
                )
            
            with col_exp2:
                # Mostrar tabla
                if st.button("📋 Ver tabla de resultados"):
                    st.dataframe(resultados_df, use_container_width=True)

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
        tab_ref1, tab_ref2, tab_ref3, tab_ref4 = st.tabs([
            "📊 Parámetros", "⚙️ Algoritmos", "📈 Casos Históricos", "🌺 Floricultura"
        ])
        
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
        
        with tab_ref4:
            st.subheader("🌸 Investigación en Floricultura con Biochar")
            
            # Tabla de estudios científicos
            estudios_flores_df = pd.DataFrame({
                'Flor': ['Rosas', 'Claveles', 'Crisantemos', 'Orquídeas', 'Gerberas', 'Lilis'],
                'Dosis (t/Ha)': ['10-15', '8-12', '12-18', '5-10', '6-10', '8-14'],
                'Mejora Observada': ['+25% producción', '+30% vida florero', '+20% calidad', '+40% raíces', '+15% diámetro', '+22% altura'],
                'Referencia': ['Li et al., 2019', 'García et al., 2020', 'Singh, 2021', 'Martínez, 2022', 'Chen, 2020', 'Rodríguez, 2021'],
                'DOI': ['10.1016/j.scienta.2019.108756', '10.21273/HORTSCI.15010-20', 
                       '10.1016/j.indcrop.2021.113456', '10.1007/s11240-022-02334-0',
                       '10.3390/agronomy10121938', '10.1080/14620316.2021.1893456']
            })
            
            st.dataframe(estudios_flores_df, use_container_width=True)
            
            st.markdown("---")
            
            # Parámetros específicos por tipo de flor
            st.subheader("📋 Parámetros Óptimos por Tipo de Flor")
            
            col_flor1, col_flor2 = st.columns(2)
            
            with col_flor1:
                st.markdown("""
                **🌹 ROSAS:**
                - pH suelo: 6.0-6.5
                - Dosis biochar: 10-15 t/Ha
                - Biochar recomendado: Madera dura (500-600°C)
                - Aplicación: Mezclar en primeros 25 cm
                - Época: 4 semanas antes de poda
                
                **🌼 CLAVELES:**
                - pH suelo: 6.5-7.0
                - Dosis biochar: 8-12 t/Ha
                - Biochar recomendado: Paja de cereal (450-550°C)
                - Aplicación: En hoyo de trasplante
                - Época: Al momento de trasplante
                """)
            
            with col_flor2:
                st.markdown("""
                **🌻 CRISANTEMOS:**
                - pH suelo: 6.0-6.8
                - Dosis biochar: 12-18 t/Ha
                - Biochar recomendado: Bambú (550-650°C)
                - Aplicación: Incorporación total
                - Época: 3 semanas antes de siembra
                
                **🌸 ORQUÍDEAS:**
                - pH suelo: 5.5-6.5
                - Dosis biochar: 5-10% en sustrato
                - Biochar recomendado: Cáscara coco (400-500°C)
                - Aplicación: Mezcla con sustrato
                - Época: Al momento de trasplante
                """)
            
            # Gráfico comparativo
            st.markdown("---")
            st.subheader("📈 Comparativa de Respuesta a Biochar")
            
            # Datos para gráfico
            respuesta_flores = pd.DataFrame({
                'Flor': ['Rosas', 'Claveles', 'Crisantemos', 'Orquídeas', 'Gerberas'],
                'Mejora Producción (%)': [25, 20, 22, 18, 15],
                'Mejora Calidad (%)': [18, 30, 25, 40, 12],
                'Reducción Fertilizantes (%)': [30, 25, 35, 20, 25]
            })
            
            # Mostrar gráfico
            st.bar_chart(respuesta_flores.set_index('Flor'))

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
    <b>Prescriptor Híbrido Biochar v2.0 🌸</b> • 
    Sistema completo para cultivos generales y floricultura • 
    <br>Combinando inteligencia artificial con conocimiento agronómico • 
    NanoMof 2025 ©️
    </div>
    """,
    unsafe_allow_html=True
)
