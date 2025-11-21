import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import r2_score

# --- FUNCIONES DE INICIALIZACIÓN Y MODELO DUMMY ---
# Función para entrenar el modelo de simulación inicial (dummy)
# Este modelo es el que se usa por defecto hasta que el usuario entrena uno real.
# @st.cache_resource asegura que el entrenamiento solo ocurre una vez por sesión de Streamlit.
@st.cache_resource
def entrenar_modelo_simulado():
    # Datos sintéticos basados en la hipótesis (pH bajo = Dosis alta)
    np.random.seed(42)
    n = 500
    ph = np.random.uniform(4, 9, n)
    mo = np.random.uniform(1, 5, n)
    dosis = (9 - ph) * 2 + (6 - mo) + np.random.normal(0, 0.2, n)
    df_sim = pd.DataFrame({'ph': ph, 'mo': mo, 'dosis_efectiva': dosis})
    
    model_sim = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model_sim.fit(df_sim[['ph', 'mo']], df_sim['dosis_efectiva'])
    return model_sim

# 1. INICIALIZACIÓN DEL ESTADO DE SESIÓN
# El MASTER_MODEL es la única fuente de verdad para todas las predicciones.
if 'master_model' not in st.session_state:
    st.session_state['master_model'] = entrenar_modelo_simulado()
    # Bandera para saber si el modelo es real (True) o solo el dummy (False)
    st.session_state['is_real_model'] = False
    # Almacena el R2 del último entrenamiento real
    st.session_state['r2_score'] = "N/A"

# Configuración de la página
st.set_page_config(page_title="Calculadora Biochar", layout="wide")

# --- AJUSTES ESTÉTICOS (Preservados) ---
# 1. Definir dos columnas: [Título, Logo]. Proporción 4:1.5
col_titulo, col_logo = st.columns([4, 1.5]) 

# 2. Colocar el Título en la primera columna (izquierda)
with col_titulo:
    st.title("🧪 Prescriptor Edafológico")

# 3. Colocar el Logo en la segunda columna (derecha)
with col_logo:
    # Manteniendo el width=500 solicitado por el usuario
    st.image("logonanomof.png", width=500) 

# Footer
st.markdown("---") 
st.markdown(
    """
    *NanomofXGBoost*©️ Created by: HV Martínez-Tejada. **NanoMof 2025**.
    """
)
# --- FIN DE AJUSTES ESTÉTICOS ---


# Creamos pestañas
tab1, tab2 = st.tabs(["🤖 Simulación Servicios B2B", "📂 Entrenamiento"])


# --- PESTAÑA 1: SIMULACIÓN (PREDICCIÓN) ---
with tab1:
    st.header("Prescripción de Dosis")

    # 1. ADVERTENCIA DE CONFIANZA (Muestra el estado actual del modelo maestro)
    if st.session_state.get('is_real_model', False):
        st.success(f"✅ **MODELO MAESTRO ACTIVO:** Usando el algoritmo XGBoost entrenado con sus datos reales. Coeficiente de Determinación (R²): **{st.session_state.get('r2_score')}**")
    else:
        st.warning("🚨 **ATENCIÓN:** Usando el **Modelo Preliminar de Simulación**. Para resultados de confianza, entrene su modelo en la pestaña 'Proyecto de Servicios B2B'.")

    st.markdown("---")
    
    # Inputs para el usuario
    col1, col2 = st.columns(2)
    with col1:
        ph_input = st.slider("pH del Suelo", min_value=3.0, max_value=9.0, value=6.5, step=0.1, help="Rango de pH del suelo a ser enmendado (3.0 a 9.0).")
    with col2:
        mo_input = st.slider("Materia Orgánica (%)", min_value=0.5, max_value=50.0, value=2.0, step=0.1, help="Contenido de Materia Orgánica en porcentaje (0.5% a 50.0%).")

    if st.button("Calcular Dosis Óptima"):
        # Lógica de predicción que SIEMPRE usa el MASTER_MODEL
        model_to_use = st.session_state.master_model
        
        input_data = pd.DataFrame({'ph': [ph_input], 'mo': [mo_input]})
        
        # Ejecuta la predicción
        dosis_predicha = model_to_use.predict(input_data)[0]
        
        st.markdown("---")
        st.subheader(f"Resultado de la Prescripción:")
        
        st.metric(label="Dosis de Biochar Recomendada", 
                  value=f"{dosis_predicha:.2f} t/Ha", 
                  delta_color="off")
        
        st.markdown(f"""
        Esta dosis (**{dosis_predicha:.2f} t/Ha**) es la prescripción del **Modelo Maestro** para un suelo con **pH {ph_input}** y **{mo_input}% de Materia Orgánica**.
        """)


# --- PESTAÑA 2: CARGA DE DATOS REALES (ENTRENAMIENTO) ---
with tab2:
    st.header("Datos de Entrenamiento")
    # Instrucciones
    st.info("Sube tu archivo CSV con columnas: 'ph', 'mo', 'dosis_efectiva'. **Separador: punto y coma ';'**.")
    
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Lectura del archivo con los parámetros correctos
            df_real = pd.read_csv(uploaded_file, encoding='latin1', sep=';')
            st.write("Vista previa de los datos cargados:")
            st.dataframe(df_real.head())
            
            # Verificar las columnas necesarias
            required_cols = ['ph', 'mo', 'dosis_efectiva']
            if not all(col in df_real.columns for col in required_cols):
                st.error("Error: Asegúrate de que el CSV contenga las columnas 'ph', 'mo', y 'dosis_efectiva'.")
            
            elif st.button("🚀 Entrenar y Actualizar Modelo Maestro"):
                # 1. Definir X e y
                X = df_real[['ph', 'mo']]
                y = df_real['dosis_efectiva']
                
                # 2. Instanciar y entrenar
                model_real = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
                model_real.fit(X, y)
                
                # 3. Evaluar el rendimiento (R2)
                score = r2_score(y, model_real.predict(X))
                
                # 4. ACTUALIZAR EL MODELO MAESTRO Y EL ESTADO DE CONFIANZA
                st.session_state['master_model'] = model_real
                st.session_state['is_real_model'] = True
                st.session_state['r2_score'] = f"{score:.4f}"
                
                # 5. Mostrar SOLO la métrica de confianza (R2)
                st.success("🎉 **¡MODELO MAESTRO ACTUALIZADO!**")
                st.info(f"El rendimiento del modelo XGBoost en sus datos es: **Coeficiente de Determinación (R²): {st.session_state['r2_score']}**")
                
                st.markdown("---")
                # Instrucción para el siguiente paso
                st.warning("⚠️ **Siguiente Paso:** Use la pestaña **'Simulación (Prescripción para Servicios B2B)'** para consultar la Dosis Óptima, ya que ahora está utilizando este nuevo modelo de alta precisión.")

        except KeyError:
            st.error("Error: Asegúrate de que tu archivo CSV contenga las columnas 'ph', 'mo', y 'dosis_efectiva'.")
        except Exception as e:
            st.error(f"Error desconocido durante la carga/entrenamiento: {e}. Revisa el formato y el delimitador (';').")




