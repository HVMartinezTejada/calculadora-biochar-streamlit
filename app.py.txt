import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np

# Configuración
st.set_page_config(page_title="Calculadora Biochar", layout="wide")

st.title("🧪 Prototipo: IA para Dosis de Biochar")
st.markdown("""
Esta aplicación demuestra el uso del algoritmo **XGBoost** (usado en *Scientific Reports, 2025*) 
para predecir la dosis de Biochar.
""")

# Creamos pestañas
tab1, tab2 = st.tabs(["🤖 Modo Simulación", "📂 Carga tus Datos Reales"])

# --- PESTAÑA 1: SIMULACIÓN ---
with tab1:
    st.warning("⚠️ **AVISO:** Este modo utiliza datos sintéticos para demostrar la interfaz.")
    
    # Modelo para Simulación
    @st.cache_resource
    def entrenar_modelo_simulado():
        np.random.seed(42)
        n = 500
        ph = np.random.uniform(4, 9, n)
        mo = np.random.uniform(1, 5, n)
        dosis = (9 - ph) * 2 + (6 - mo) + np.random.normal(0, 0.2, n)
        df_sim = pd.DataFrame({'pH': ph, 'MO': mo, 'Dosis': dosis})
        model_sim = xgb.XGBRegressor(objective='reg:squarederror')
        model_sim.fit(df_sim[['pH', 'MO']], df_sim['Dosis'])
        return model_sim

    if st.button("Generar Modelo de Simulación"):
        model_sim = entrenar_modelo_simulado()
        st.session_state['model_sim'] = model_sim
        st.success("Modelo simulado entrenado exitosamente. Ahora puedes ajustar los parámetros.")

    if 'model_sim' in st.session_state:
        model_sim = st.session_state['model_sim']
        st.divider()
        st.subheader("Simulación en Tiempo Real")
        col1, col2 = st.columns(2)
        with col1:
            val_ph = st.slider("pH del Suelo", 4.0, 9.0, 6.5)
        with col2:
            val_mo = st.slider("Materia Orgánica (%)", 1.0, 5.0, 2.5)
            
        pred = model_sim.predict(pd.DataFrame([[val_ph, val_mo]], columns=['pH', 'MO']))[0]
        st.metric(label="Dosis Sugerida (Simulada)", value=f"{pred:.2f} Ton/Ha")

# --- PESTAÑA 2: CARGA DE DATOS REALES ---
with tab2:
    st.header("Entrena con tus datos de laboratorio")
    st.info("Sube un archivo CSV con columnas: 'ph', 'mo', 'dosis_efectiva'")
    
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    
    if uploaded_file is not None:
        df_real = pd.read_csv(uploaded_file)
        
        if st.button("Entrenar XGBoost con mis datos"):
            try:
                # Asumimos que las columnas son 'ph', 'mo', 'dosis_efectiva'
                X = df_real[['ph', 'mo']]
                y = df_real['dosis_efectiva']
                
                model_real = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
                model_real.fit(X, y)
                score = model_real.score(X, y)
                
                st.session_state['model_real'] = model_real
                st.success(f"¡Modelo entrenado con TUS datos! Precisión (R2): {score:.4f}")
                
            except KeyError:
                st.error("Error: Asegúrate de que tu archivo CSV contenga las columnas 'ph', 'mo', y 'dosis_efectiva'.")
            except Exception as e:
                st.error(f"Error desconocido durante el entrenamiento: {e}")
                
    if 'model_real' in st.session_state:
        st.divider()
        st.subheader("Calculadora basada en TUS datos")
        r_ph = st.number_input("Ingresa pH real", value=6.0)
        r_mo = st.number_input("Ingresa MO real", value=2.0)
        
        if st.button("Calcular (Modelo Real)"):
            pred_real = st.session_state['model_real'].predict(pd.DataFrame([[r_ph, r_mo]], columns=['ph', 'mo']))[0]
            st.success(f"Dosis calculada: {pred_real:.2f} Ton/Ha")