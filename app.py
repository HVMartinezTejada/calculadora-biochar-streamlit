import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
import os
from io import BytesIO
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURACIÓN PARA STREAMLIT CLOUD
# ============================================================================

# Configuración de página PRIMERO (requerido por Streamlit)
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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. FUNCIONES PARA CARGAR ARCHIVOS (COMPATIBLE CON STREAMLIT CLOUD)
# ============================================================================

@st.cache_resource
def cargar_datos_excel():
    """Carga datos del Excel o usa datos de ejemplo embebidos"""
    try:
        # PRIMERO: Intentar cargar desde archivo en el mismo directorio
        excel_path = "Biochar_Prescriptor_Sistema_Completo_v1.0.xlsx"
        
        # Verificar si el archivo existe localmente
        if os.path.exists(excel_path):
            # Cargar todas las hojas necesarias
            parametros_df = pd.read_excel(excel_path, sheet_name='Parametros')
            rangos_df = pd.read_excel(excel_path, sheet_name='Rangos_suelo')
            algoritmos_df = pd.read_excel(excel_path, sheet_name='Algoritmos')
            factores_df = pd.read_excel(excel_path, sheet_name='Factores_ajuste')
            metadatos_df = pd.read_excel(excel_path, sheet_name='Metadatos_completos')
            dosis_exp_df = pd.read_excel(excel_path, sheet_name='Dosis_experimental')
            
            st.success("✅ Base de datos cargada desde archivo Excel")
            
            return {
                'parametros': parametros_df,
                'rangos': rangos_df,
                'algoritmos': algoritmos_df,
                'factores': factores_df,
                'metadatos': metadatos_df,
                'dosis_exp': dosis_exp_df,
                'fuente': 'archivo_local'
            }
        
        # SEGUNDO: Intentar cargar desde URL (alternativa)
        try:
            # Esta es una URL de ejemplo - puedes reemplazarla con tu Google Sheet o Dropbox
            excel_url = "https://raw.githubusercontent.com/tu_usuario/tu_repositorio/main/Biochar_Prescriptor_Sistema_Completo_v1.0.xlsx"
            parametros_df = pd.read_excel(excel_url, sheet_name='Parametros')
            # ... cargar otras hojas
            
            st.success("✅ Base de datos cargada desde URL")
            return {
                'parametros': parametros_df,
                # ... otras hojas
                'fuente': 'url'
            }
            
        except:
            # TERCERO: Crear datos de ejemplo embebidos en el código
            st.warning("⚠️ No se encontró el archivo Excel. Usando datos de ejemplo embebidos.")
            return crear_datos_ejemplo()
            
    except Exception as e:
        st.error(f"❌ Error cargando datos: {str(e)[:100]}...")
        return crear_datos_ejemplo()

def crear_datos_ejemplo():
    """Crea datos de ejemplo embebidos para cuando no hay Excel"""
    # Datos mínimos para que la aplicación funcione
    parametros_df = pd.DataFrame({
        'Parámetro': ['pH del suelo', 'Materia orgánica', 'CIC', 'Textura arena%', 'Metales pesados'],
        'Nombre corto': ['pH', 'MO', 'CIC', 'Textura_arena', 'Metales'],
        'Categoría': ['Suelo químico', 'Suelo químico', 'Suelo químico', 'Suelo físico', 'Suelo químico'],
        'Unidad': ['pH', '%', 'cmolc/kg', '%', 'mg/kg'],
        'Mínimo': [3.5, 0.1, 2, 0, 0],
        'Máximo': [9.5, 10, 50, 100, 500],
        'Dosis_recomendada_t_ha': ['10-30', '20-40', '10-20', '15-25', '20-50']
    })
    
    algoritmos_df = pd.DataFrame({
        'Objetivo': ['Fertilidad', 'Remediación', 'Resiliencia hídrica'],
        'Ecuación_dosis_base': ['D_base = 12 + (6.5-pH)×1.5 + (3-MO)×0.8', 
                               'D_base = 25 + Metales/20',
                               'D_base = 18 + (25-CRA)×0.4'],
        'Constante': [12, 25, 18],
        'Coeficientes': ['pH:1.5, MO:0.8', 'Metales:0.05', 'CRA:0.4'],
        'Rango_típico_t/ha': ['10-30', '15-50', '10-25']
    })
    
    # Datos de entrenamiento de ejemplo
    np.random.seed(42)
    n = 30
    dosis_exp_df = pd.DataFrame({
        'ph': np.random.uniform(4, 8, n),
        'mo': np.random.uniform(1, 5, n),
        'dosis_efectiva': np.random.uniform(10, 30, n),
        'Textura': np.random.choice(['Arena', 'Franco', 'Arcilloso'], n),
        'Feedstock': np.random.choice(['Madera', 'Cáscara cacao', 'Paja trigo'], n)
    })
    
    return {
        'parametros': parametros_df,
        'algoritmos': algoritmos_df,
        'dosis_exp': dosis_exp_df,
        'fuente': 'ejemplo'
    }

# ============================================================================
# 3. CARGAR LOGO CON MANEJO DE ERRORES
# ============================================================================

def cargar_logo():
    """Intenta cargar el logo, si no existe usa texto alternativo"""
    try:
        # Intentar cargar desde archivo local
        if os.path.exists("logonanomof.png"):
            return "logonanomof.png"
        
        # Si estás en Streamlit Cloud y el logo está en GitHub
        # Puedes usar una URL:
        logo_url = "https://raw.githubusercontent.com/tu_usuario/tu_repositorio/main/logonanomof.png"
        
        # Verificar si la URL es accesible
        import requests
        response = requests.head(logo_url)
        if response.status_code == 200:
            return logo_url
        
        # Si nada funciona, retornar None
        return None
        
    except:
        return None

# ============================================================================
# 4. FUNCIONES DEL MODELO HÍBRIDO
# ============================================================================

def calcular_dosis_deterministica(objetivo, parametros_suelo, parametros_biochar, datos_excel):
    """
    Calcula dosis usando ecuaciones determinísticas
    """
    if datos_excel is None or 'algoritmos' not in datos_excel:
        # Valores por defecto basados en objetivo
        dosis_default = {
            'Fertilidad': 15.0,
            'Remediación': 25.0,
            'Resiliencia hídrica': 18.0,
            'Secuestro carbono': 30.0,
            'Supresión patógenos': 12.0
        }
        return dosis_default.get(objetivo, 20.0)
    
    try:
        algoritmos = datos_excel['algoritmos']
        
        # Filtrar algoritmo para el objetivo
        if objetivo not in algoritmos['Objetivo'].values:
            return 20.0
        
        algo_row = algoritmos[algoritmos['Objetivo'] == objetivo].iloc[0]
        
        # Calcular dosis base (simplificado para demo)
        dosis_base = 0.0
        
        # Para fertilidad
        if objetivo == 'Fertilidad':
            dosis_base = 12.0
            ph = parametros_suelo.get('pH', 6.5)
            mo = parametros_suelo.get('MO', 2.0)
            dosis_base += (6.5 - ph) * 1.5 + (3.0 - mo) * 0.8
        
        # Para remediación
        elif objetivo == 'Remediación':
            dosis_base = 25.0
            metales = parametros_suelo.get('Metales', 0.0)
            dosis_base += metales / 20.0
        
        # Para otros objetivos
        else:
            dosis_base = 20.0
        
        # Factores de ajuste
        factor = 1.0
        
        # Ajuste por pH
        ph = parametros_suelo.get('pH', 6.5)
        if ph < 5.0:
            factor *= 1.5
        elif ph > 8.0:
            factor *= 1.2
        
        # Ajuste por MO
        mo = parametros_suelo.get('MO', 2.0)
        if mo < 1.0:
            factor *= 1.4
        
        return max(5.0, min(50.0, dosis_base * factor))
        
    except Exception as e:
        st.warning(f"Advertencia en cálculo determinístico: {e}")
        return 20.0

def entrenar_modelo_hibrido(df_entrenamiento):
    """
    Entrena un modelo XGBoost con múltiples características
    """
    try:
        # Hacer copia para no modificar el original
        df = df_entrenamiento.copy()
        
        # Codificar variables categóricas
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != 'dosis_efectiva':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        
        # Separar características y objetivo
        X = df.drop('dosis_efectiva', axis=1)
        y = df['dosis_efectiva']
        
        # Entrenar modelo
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        model.fit(X, y)
        
        # Calcular R2
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        return model, r2, list(X.columns)
        
    except Exception as e:
        st.error(f"Error entrenando modelo: {e}")
        return None, 0.0, []

# ============================================================================
# 5. INICIALIZACIÓN DE LA APLICACIÓN
# ============================================================================

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<h1 class="main-header">🧬 Prescriptor Híbrido Biochar</h1>', unsafe_allow_html=True)
    st.markdown("**Modelo combinado: XGBoost + Reglas Determinísticas**")

with col2:
    logo_path = cargar_logo()
    if logo_path:
        st.image(logo_path, width=250)
    else:
        st.markdown("### **NanoMof**")

# Cargar datos del Excel
with st.spinner("Cargando base de datos..."):
    datos_excel = cargar_datos_excel()

# Mostrar fuente de datos
if datos_excel and 'fuente' in datos_excel:
    fuente = datos_excel['fuente']
    if fuente == 'archivo_local':
        st.success("✅ Usando archivo Excel local")
    elif fuente == 'url':
        st.info("🌐 Usando datos desde URL")
    elif fuente == 'ejemplo':
        st.warning("📋 Usando datos de ejemplo. Para máxima precisión, sube tu Excel a GitHub.")

# ============================================================================
# 6. ESTADO DE LA SESIÓN
# ============================================================================

# Inicializar estado de sesión
if 'modelo_hibrido' not in st.session_state:
    # Intentar entrenar modelo inicial con datos disponibles
    if datos_excel is not None and 'dosis_exp' in datos_excel:
        df_ejemplo = datos_excel['dosis_exp'].copy()
        
        # Verificar columnas necesarias
        if 'dosis_efectiva' in df_ejemplo.columns and len(df_ejemplo) > 5:
            try:
                modelo, r2, features = entrenar_modelo_hibrido(df_ejemplo)
                if modelo:
                    st.session_state.modelo_hibrido = modelo
                    st.session_state.r2_score = r2
                    st.session_state.features_modelo = features
                    st.session_state.modelo_activo = True
                    st.success(f"✅ Modelo XGBoost inicial entrenado (R²: {r2:.3f})")
                else:
                    st.session_state.modelo_activo = False
            except:
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
# 7. INTERFAZ PRINCIPAL - PESTAÑAS
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
        
        ph_suelo = st.slider("pH del suelo", 3.5, 9.5, 6.5, 0.1,
                            help="pH ácido: <6.5, neutro: 6.5-7.5, alcalino: >7.5")
        mo_suelo = st.slider("Materia Orgánica (%)", 0.1, 10.0, 2.0, 0.1,
                            help="Bajo: <1.5%, Medio: 1.5-3.5%, Alto: >3.5%")
        cic_suelo = st.slider("CIC (cmolc/kg)", 2.0, 50.0, 15.0, 0.5,
                             help="Capacidad de Intercambio Catiónico. Bajo: <10, Medio: 10-25, Alto: >25")
        textura = st.selectbox("Textura del suelo", [
            "Arena", "Franco-arenoso", "Franco", "Franco-arcilloso", "Arcilloso"
        ], help="Textura influye en retención de agua y nutrientes")
        
        metales = st.number_input("Metales pesados (mg/kg)", 0.0, 500.0, 0.0, 1.0,
                                 help="Solo si hay contaminación conocida")
    
    with col_input2:
        st.subheader("🌿 Propiedades del Biochar")
        
        feedstock = st.selectbox("Materia prima", [
            "Madera", "Cáscara cacao", "Paja trigo", "Bambú", 
            "Estiércol", "Paja arroz", "Cáscara arroz", "Lodo papel", "Otro"
        ], help="La materia prima afecta propiedades del biochar")
        
        temp_pirolisis = st.slider("Temperatura pirólisis (°C)", 300, 900, 550, 10,
                                  help="Baja (<450°C): más compuestos volátiles. Alta (>600°C): mayor área superficial")
        ph_biochar = st.slider("pH del biochar", 5.0, 12.0, 9.0, 0.1,
                              help="Típicamente alcalino (8-11)")
        
        tamaño_opciones = ["Fino (<1 mm)", "Medio (1-5 mm)", "Grueso (>5 mm)"]
        tamaño = st.selectbox("Tamaño partícula", tamaño_opciones,
                             help="Fino: mayor área contacto, Grueso: mejor estructura suelo")
        
        area_bet = st.slider("Área superficial (m²/g)", 10, 600, 300, 10,
                            help="Baja: <100, Media: 100-400, Alta: >400")
    
    with col_input3:
        st.subheader("🌱 Condiciones Agronómicas")
        
        objetivo = st.selectbox("Objetivo principal", [
            "Fertilidad", "Remediación", "Resiliencia hídrica", 
            "Secuestro carbono", "Supresión patógenos"
        ], help="Selecciona el objetivo principal de la aplicación")
        
        cultivo = st.selectbox("Tipo de cultivo", [
            "Teff", "Hortalizas", "Trigo", "Girasol", "Maíz", "Pasto", 
            "Cacao", "Frijol", "Palma", "Sorgo", "Tomate", "Soja", "Otro"
        ], help="El tipo de cultivo influye en requerimientos nutricionales")
        
        sistema_riego = st.selectbox("Sistema de riego", [
            "Gravedad", "Aspersión", "Goteo", "No irrigado"
        ], help="Sistemas eficientes requieren menor dosis")
        
        clima = st.selectbox("Clima", [
            "Árido", "Semiárido", "Mediterráneo", "Tropical", "Templado"
        ], help="Climas áridos pueden requerir mayores dosis para retención hídrica")
    
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
                st.session_state.parametros_usuario['biochar'],
                datos_excel
            )
            
            # 2. Si hay modelo XGBoost activo, calcular predicción
            dosis_xgb = None
            if st.session_state.get('modelo_activo', False):
                try:
                    modelo = st.session_state.modelo_hibrido
                    features = st.session_state.get('features_modelo', [])
                    
                    # Preparar datos para el modelo
                    input_dict = {}
                    for feature in features:
                        # Buscar el valor en los parámetros del usuario
                        valor = None
                        
                        # Buscar en todas las categorías
                        for categoria in ['suelo', 'biochar', 'cultivo']:
                            for key, value in st.session_state.parametros_usuario[categoria].items():
                                if key.lower() == feature.lower():
                                    if isinstance(value, (int, float)):
                                        valor = float(value)
                                    else:
                                        # Para variables categóricas, usar 0 como placeholder
                                        valor = 0.0
                                    break
                            if valor is not None:
                                break
                        
                        if valor is None:
                            valor = 0.0  # Valor por defecto
                        
                        input_dict[feature] = valor
                    
                    # Crear DataFrame y predecir
                    if input_dict:
                        df_input = pd.DataFrame([input_dict])
                        dosis_xgb = float(modelo.predict(df_input)[0])
                        
                except Exception as e:
                    st.warning(f"Predicción XGBoost no disponible: {str(e)[:50]}")
            
            # 3. Combinar resultados
            if dosis_xgb is not None and dosis_xgb > 0:
                dosis_final = (dosis_det * 0.4 + dosis_xgb * 0.6)
                metodo = "Híbrido (Determinístico + XGBoost)"
                mostrar_xgb = True
            else:
                dosis_final = dosis_det
                metodo = "Determinístico (basado en investigación)"
                mostrar_xgb = False
            
            # 4. Mostrar resultados
            st.markdown("---")
            st.success(f"### 📊 Dosis Recomendada: **{dosis_final:.1f} t/Ha**")
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.metric(
                    label="Dosis Determinística",
                    value=f"{dosis_det:.1f} t/Ha",
                    help="Basada en ecuaciones científicas del Excel"
                )
            
            with col_res2:
                if mostrar_xgb:
                    st.metric(
                        label="Dosis XGBoost",
                        value=f"{dosis_xgb:.1f} t/Ha",
                        help="Basada en aprendizaje automático"
                    )
            
            # 5. Explicación detallada
            with st.expander("📋 Detalles de la prescripción", expanded=True):
                st.markdown(f"""
                **Parámetros considerados:**
                - **Suelo:** pH {ph_suelo}, MO {mo_suelo}%, CIC {cic_suelo} cmolc/kg, Textura {textura}
                - **Biochar:** {feedstock} a {temp_pirolisis}°C, pH {ph_biochar}, {tamaño}
                - **Cultivo:** {cultivo} en clima {clima} con riego {sistema_riego}
                - **Objetivo:** {objetivo}
                
                **Metodología:** {metodo}
                
                **Rango típico para {objetivo.lower()}:**
                - Fertilidad: 10-30 t/Ha
                - Remediación: 15-50 t/Ha
                - Resiliencia hídrica: 10-25 t/Ha
                - Secuestro carbono: 20-50 t/Ha
                - Supresión patógenos: 10-20 t/Ha
                """)
                
                # Barra de progreso visual
                rangos = {
                    'Fertilidad': (10, 30),
                    'Remediación': (15, 50),
                    'Resiliencia hídrica': (10, 25),
                    'Secuestro carbono': (20, 50),
                    'Supresión patógenos': (10, 20)
                }
                
                if objetivo in rangos:
                    min_rango, max_rango = rangos[objetivo]
                    porcentaje = min(100, max(0, (dosis_final - min_rango) / (max_rango - min_rango) * 100))
                    
                    st.markdown(f"**Posición en rango típico:**")
                    st.progress(int(porcentaje)/100)
                    st.caption(f"{dosis_final:.1f} t/Ha está en el {porcentaje:.0f}% del rango típico ({min_rango}-{max_rango} t/Ha)")
            
            # 6. Recomendaciones
            st.info(f"""
            **💡 Recomendaciones de aplicación:**
            1. **Preparación:** Mezclar el biochar uniformemente con los primeros 20 cm de suelo
            2. **Época:** Aplicar 2-4 semanas antes de la siembra
            3. **Monitoreo:** Verificar pH del suelo a los 3 meses
            4. **Evaluación:** Observar respuesta del cultivo a los 60 días
            5. **Registro:** Documentar resultados para mejorar futuras prescripciones
            """)
            
            # 7. Exportar resultados
            st.markdown("---")
            st.subheader("📤 Exportar resultados")
            
            # Crear DataFrame con los resultados
            resultados_df = pd.DataFrame({
                'Parámetro': ['Dosis Recomendada', 'Método', 'pH Suelo', 'MO (%)', 'CIC', 
                             'Textura', 'Biochar', 'Objetivo', 'Cultivo'],
                'Valor': [f"{dosis_final:.1f} t/Ha", metodo, str(ph_suelo), str(mo_suelo), 
                         str(cic_suelo), textura, feedstock, objetivo, cultivo]
            })
            
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                # Convertir a CSV
                csv = resultados_df.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar como CSV",
                    data=csv,
                    file_name=f"prescripcion_biochar_{objetivo.lower()}.csv",
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
    ### 📖 Instrucciones para entrenamiento personalizado
    
    Para mejorar la precisión del modelo, puedes:
    1. **Usar los datos de referencia** del Excel
    2. **Subir tu propio dataset** con casos reales
    3. **Combinar múltiples fuentes** de datos
    
    **Formato requerido del CSV:**
    - Debe incluir columna `dosis_efectiva`
    - Puede incluir columnas como: `ph`, `mo`, `textura`, `feedstock`, etc.
    - Separador: coma (,) o punto y coma (;)
    """)
    
    # Opción 1: Usar datos del Excel
    st.subheader("Opción 1: Usar datos de referencia")
    
    if datos_excel and 'dosis_exp' in datos_excel:
        df_referencia = datos_excel['dosis_exp'].copy()
        st.write(f"Datos disponibles: {len(df_referencia)} registros")
        
        if st.button("🔄 Entrenar modelo con datos de referencia"):
            with st.spinner("Entrenando modelo con datos de referencia..."):
                modelo_ref, r2_ref, features_ref = entrenar_modelo_hibrido(df_referencia)
                
                if modelo_ref:
                    st.session_state.modelo_hibrido = modelo_ref
                    st.session_state.r2_score = r2_ref
                    st.session_state.features_modelo = features_ref
                    st.session_state.modelo_activo = True
                    
                    st.success(f"✅ Modelo entrenado exitosamente!")
                    st.metric("Coeficiente R²", f"{r2_ref:.4f}")
                    st.metric("Características usadas", len(features_ref))
                    
                    # Mostrar primeras características
                    with st.expander("Ver características del modelo"):
                        st.write("**Variables utilizadas:**", ", ".join(features_ref[:10]))
                else:
                    st.error("No se pudo entrenar el modelo con los datos de referencia")
    
    # Opción 2: Subir datos propios
    st.subheader("Opción 2: Subir tu propio dataset")
    
    uploaded_file = st.file_uploader("📤 Subir archivo CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Detectar separador
            content = uploaded_file.getvalue().decode('utf-8')
            if ';' in content.split('\n')[0]:
                separator = ';'
            else:
                separator = ','
            
            uploaded_file.seek(0)  # Volver al inicio del archivo
            df_cargado = pd.read_csv(uploaded_file, sep=separator)
            
            st.success(f"✅ Archivo cargado: {len(df_cargado)} registros")
            
            # Mostrar vista previa
            with st.expander("📊 Vista previa de datos", expanded=True):
                st.dataframe(df_cargado.head(), use_container_width=True)
                st.write(f"**Columnas:** {list(df_cargado.columns)}")
            
            # Verificar columna objetivo
            if 'dosis_efectiva' not in df_cargado.columns:
                st.error("❌ El dataset debe contener la columna 'dosis_efectiva'")
                st.info("Sugerencia: Renombra tu columna de dosis a 'dosis_efectiva'")
            else:
                if st.button("🚀 Entrenar modelo personalizado", type="primary"):
                    with st.spinner("Entrenando modelo personalizado..."):
                        modelo_personal, r2_personal, features_personal = entrenar_modelo_hibrido(df_cargado)
                        
                        if modelo_personal:
                            st.session_state.modelo_hibrido = modelo_personal
                            st.session_state.r2_score = r2_personal
                            st.session_state.features_modelo = features_personal
                            st.session_state.modelo_activo = True
                            
                            st.balloons()
                            st.success(f"✅ Modelo personalizado entrenado!")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("R²", f"{r2_personal:.4f}")
                            with col2:
                                st.metric("Registros", len(df_cargado))
                            with col3:
                                st.metric("Variables", len(features_personal))
                            
                            # Guardar dataset en sesión para uso futuro
                            st.session_state.dataset_personal = df_cargado
                            
                            st.info("💡 Ahora puedes usar el modelo personalizado en la pestaña de Prescripción")
                        else:
                            st.error("No se pudo entrenar el modelo con los datos subidos")
                            
        except Exception as e:
            st.error(f"Error procesando el archivo: {str(e)[:100]}")

# ============================================================================
# PESTAÑA 3: BASE DE CONOCIMIENTO
# ============================================================================

with tab3:
    st.header("Base de Conocimiento y Referencias")
    
    if datos_excel is None:
        st.warning("No se pudieron cargar los datos de referencia")
    else:
        tab_ref1, tab_ref2, tab_ref3 = st.tabs(["📊 Parámetros", "⚙️ Algoritmos", "📈 Casos de Estudio"])
        
        with tab_ref1:
            st.subheader("Parámetros del sistema")
            if 'parametros' in datos_excel:
                st.dataframe(datos_excel['parametros'], use_container_width=True)
            else:
                st.info("No hay datos de parámetros disponibles")
        
        with tab_ref2:
            st.subheader("Algoritmos determinísticos")
            if 'algoritmos' in datos_excel:
                st.dataframe(datos_excel['algoritmos'], use_container_width=True)
            else:
                st.info("No hay datos de algoritmos disponibles")
            
            st.subheader("Factores de ajuste")
            if 'factores' in datos_excel:
                st.dataframe(datos_excel['factores'].head(10), use_container_width=True)
            else:
                st.info("No hay datos de factores disponibles")
        
        with tab_ref3:
            st.subheader("Casos históricos documentados")
            if 'metadatos' in datos_excel:
                st.dataframe(datos_excel['metadatos'].head(15), use_container_width=True)
            else:
                st.info("No hay casos históricos disponibles")
            
            # Mostrar estadísticas si hay datos de dosis experimental
            if 'dosis_exp' in datos_excel and len(datos_excel['dosis_exp']) > 0:
                st.subheader("📊 Estadísticas de dosis experimentales")
                df_dosis = datos_excel['dosis_exp']
                
                if 'dosis_efectiva' in df_dosis.columns:
                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                    with col_stats1:
                        st.metric("Dosis mínima", f"{df_dosis['dosis_efectiva'].min():.1f} t/Ha")
                    with col_stats2:
                        st.metric("Dosis promedio", f"{df_dosis['dosis_efectiva'].mean():.1f} t/Ha")
                    with col_stats3:
                        st.metric("Dosis máxima", f"{df_dosis['dosis_efectiva'].max():.1f} t/Ha")
                    
                    # Histograma de dosis
                    st.bar_chart(df_dosis['dosis_efectiva'].value_counts().sort_index())

# ============================================================================
# 8. PANEL DE INFORMACIÓN Y FOOTER
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
        r2 = st.session_state.get('r2_score', 0)
        color = "green" if r2 > 0.7 else "orange" if r2 > 0.5 else "red"
        st.markdown(f"<span style='color:{color}'>✅ Modelo XGBoost activo (R²: {r2:.3f})</span>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Usando solo modelo determinístico")
    
    if datos_excel:
        fuente = datos_excel.get('fuente', 'desconocida')
        if fuente == 'archivo_local':
            st.success("✅ Base de datos local cargada")
        elif fuente == 'ejemplo':
            st.warning("📋 Usando datos de ejemplo")

with col_info3:
    st.markdown("### 📚 Referencias Científicas")
    st.markdown("""
    - Woolf et al. (2010) Nature Communications
    - Jeffery et al. (2017) Agronomy for Sustainable Development
    - Lehmann & Joseph (2015) Biochar for Environmental Management
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
    <b>Prescriptor Híbrido Biochar v2.0</b> • 
    Combinando inteligencia artificial con conocimiento agronómico • 
    <br>NanoMof 2025 ©️ • 
    <a href='https://github.com/tu_usuario/tu_repositorio' target='_blank'>Código en GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
