import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --- CONFIGURACIÓN DE NIVEL PROFESIONAL ---
st.set_page_config(page_title="AI PRO TRADING DASHBOARD", layout="wide", initial_sidebar_state="collapsed")

# Inyectamos CSS para imitar la interfaz de trading oscura y compacta
st.markdown("""
    <style>
    /* Tema Oscuro Global */
    .stApp {
        background-color: #0e1621;
        color: #e4e4e4;
    }
    /* Estilo de los contenedores de mercado */
    .market-box {
        background-color: #1c2a38;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #2c3e50;
    }
    /* Encabezados de sección */
    .market-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #00f2ff;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    /* Filas de apuestas compactas */
    .bet-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #2c3e50;
    }
    .bet-label { font-weight: 600; font-size: 0.95rem; }
    
    /* Cajas de Cuota y Probabilidad IA */
    .odds-box {
        background-color: #263238;
        color: #aab8c2;
        padding: 5px 12px;
        border-radius: 4px;
        font-weight: bold;
        margin-left: 10px;
        min-width: 60px;
        text-align: center;
    }
    .ai-prob-box {
        background: linear-gradient(90deg, #00b894, #00cec9);
        color: black;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: 900;
        margin-left: 10px;
        min-width: 70px;
        text-align: center;
        box-shadow: 0 0 10px rgba(0, 206, 201, 0.3);
    }
    /* Ocultar elementos nativos de Streamlit que molestan */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stSelectbox label { color: #00f2ff !important; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTOR DE DATOS E IA ---
@st.cache_data(ttl=3600)
def load_data(url):
    try:
        df = pd.read_csv(url).dropna(subset=['FTR', 'B365H', 'FTHG', 'FTAG', 'HC', 'AC'])
        # Feature Engineering básico para modelos específicos
        df['TotalGoals'] = df['FTHG'] + df['FTAG']
        df['Over25'] = (df['TotalGoals'] > 2.5).astype(int)
        df['TotalCorners'] = df['HC'] + df['AC']
        return df
    except: return None

ligas = {
    "🇪🇸 ESPAÑA - La Liga": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "🇬🇧 INGLATERRA - Premier": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "🇮🇹 ITALIA - Serie A": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "🇩🇪 ALEMANIA - Bundesliga": "https://www.football-data.co.uk/mmz4281/2526/D1.csv"
}

# Función auxiliar para crear filas de la interfaz
def crear_fila_apuesta(label, cuota_ref, prob_ia):
    st.markdown(f"""
        <div class="bet-row">
            <span class="bet-label">{label}</span>
            <div style="display:flex; align-items:center;">
                <span style="color:#aaa; font-size:0.8rem; margin-right:5px;">Ref. Casa:</span>
                <span class="odds-box">{cuota_ref:.2f}</span>
                <span style="color:#00f2ff; font-size:0.8rem; margin:0 5px;">IA PROB:</span>
                <span class="ai-prob-box">{prob_ia:.1f}%</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- INTERFAZ PRINCIPAL ---
st.markdown("<h1 style='text-align:center; color:#00f2ff; margin-bottom:5px;'>AI PRO TRADING DASHBOARD</h1>", unsafe_allow_html=True)

# Barra superior de selección compacta
col_liga, col_l, col_v = st.columns([2, 3, 3])
sel_liga = col_liga.selectbox("Competición", list(ligas.keys()), label_visibility="collapsed")
df = load_data(ligas[sel_liga])

if df is not None:
    le = LabelEncoder()
    teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    le.fit(teams)
    t1 = col_l.selectbox("Local", teams, label_visibility="collapsed")
    t2 = col_v.selectbox("Visitante", teams, index=min(1, len(teams)-1), label_visibility="collapsed")

    if t1 == t2:
        st.error("Selecciona equipos diferentes.")
        st.stop()

    st.markdown("---")

    # --- NÚCLEO DE IA: ENTRENAMIENTO MULTI-MODELO ---
    with st.spinner("⚡ La IA está escaneando miles de partidos históricos..."):
        # Preparación de datos
        df['H_c'] = le.transform(df['HomeTeam'])
        df['A_c'] = le.transform(df['AwayTeam'])
        X = df[['H_c', 'A_c', 'B365H', 'B365D', 'B365A']].values
        
        # 1. Modelo Ganador (1X2) - Clasificador
        y_1x2 = df['FTR'].apply(lambda x: 0 if x == 'H' else (2 if x == 'A' else 1)) # 0:Local, 1:Empate, 2:Visita
        m_1x2 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42).fit(X, y_1x2)
        
        # 2. Modelo Goles Binario (Más/Menos 2.5) - Clasificador específico
        y_o25 = df['Over25']
        m_o25 = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42).fit(X, y_o25)

        # 3. Modelo Córners y Goles Totales - Regresores para promedios
        m_corn_reg = RandomForestRegressor(n_estimators=150, random_state=42).fit(X, df['TotalCorners'])
        m_goals_reg = RandomForestRegressor(n_estimators=150, random_state=42).fit(X, df['TotalGoals'])

        # --- PREDICCIÓN DEL PARTIDO SELECCIONADO ---
        v_in = [[le.transform([t1])[0], le.transform([t2])[0], 2.5, 3.2, 2.8]] # Usamos cuotas medias de ref.

        # Probabilidades 1X2
        probs_1x2 = m_1x2.predict_proba(v_in)[0] * 100 # [Prob Local, Prob Empate, Prob Visita]
        
        # Probabilidades Goles
        prob_over25 = m_o25.predict_proba(v_in)[0][1] * 100
        prob_under25 = 100 - prob_over25
        avg_goals = m_goals_reg.predict(v_in)[0]

        # Probabilidades Córners (Estimación basada en regresión)
        avg_corners = m_corn_reg.predict(v_in)[0]
        # Estimación simple de prob para corners (se puede mejorar con modelos binarios)
        prob_over95_corn = min(95, max(5, (avg_corners / 9.5) * 50))
        prob_under95_corn = 100 - prob_over95_corn

    # --- VISUALIZACIÓN TIPO SPORTSBOOK ---
    
    # PANEL 1: PRINCIPAL (1X2)
    st.markdown('<div class="market-header">🔥 Mercado Principal (1X2)</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="market-box">', unsafe_allow_html=True)
        # Usamos cuotas de referencia genéricas, en una app real se conectarían a una API
        crear_fila_apuesta(f"Victoria {t1}", 2.30, probs_1x2[0])
        crear_fila_apuesta("Empate (X)", 3.40, probs_1x2[1])
        crear_fila_apuesta(f"Victoria {t2}", 3.10, probs_1x2[2])
        st.markdown('</div>', unsafe_allow_html=True)

    # PANEL 2: GOLES
    st.markdown(f'<div class="market-header">⚽ Mercado de Goles (Promedio IA: {avg_goals:.1f})</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="market-box">', unsafe_allow_html=True)
        crear_fila_apuesta("Más de 2.5 Goles", 1.90, prob_over25)
        crear_fila_apuesta("Menos de 2.5 Goles", 1.90, prob_under25)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # PANEL 3: CÓRNERS
    st.markdown(f'<div class="market-header">⛳ Mercado de Córners (Promedio IA: {avg_corners:.0f})</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="market-box">', unsafe_allow_html=True)
        # Nota: Estas probs de córners son estimaciones basadas en el promedio, no modelos binarios puros aún.
        crear_fila_apuesta("Más de 9.5 Córners", 1.85, prob_over95_corn)
        crear_fila_apuesta("Menos de 9.5 Córners", 1.85, prob_under95_corn)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("Error cargando los datos. Intenta refrescar.")

# Aviso legal importante al estilo profesional
st.markdown("---")
st.caption("⚠️ **ADVERTENCIA DE RIESGO PROFESIONAL:** Esta herramienta proporciona probabilidades estadísticas basadas en datos históricos. No garantiza resultados futuros. El trading deportivo implica riesgo de pérdida de capital. Utilice esta información como soporte para su análisis, no como verdades absolutas.")
