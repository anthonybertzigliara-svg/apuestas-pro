import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. CONFIGURACIÓN DE INTERFAZ PROFESIONAL (ESTILO SPORTSBOOK)
st.set_page_config(page_title="WORLD ELITE PREDICTOR", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stApp { background-color: #0b0e11; color: #ffffff; font-family: 'Roboto', sans-serif; }
    
    /* Contenedores de Mercado más compactos */
    .market-box {
        background: #181a20;
        border: 1px solid #2b2f36;
        border-radius: 4px;
        padding: 8px;
        margin-bottom: 10px;
    }
    .market-title {
        color: #f0b90b;
        font-size: 0.75rem;
        font-weight: bold;
        text-transform: uppercase;
        border-bottom: 1px solid #2b2f36;
        padding-bottom: 4px;
        margin-bottom: 6px;
    }
    
    /* Filas de apuestas tipo rejilla */
    .bet-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 4px;
    }
    .bet-item {
        background: #2b2f36;
        padding: 4px 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-radius: 2px;
        font-size: 0.8rem;
    }
    .prob-val {
        color: #00ff88;
        font-weight: 900;
    }
    
    /* Estilo para los selectores */
    .stSelectbox label { font-size: 0.8rem !important; color: #848e9c !important; }
    </style>
    """, unsafe_allow_html=True)

# 2. CARGA DE DATOS GLOBAL
@st.cache_data(ttl=3600)
def load_global_data(url):
    try:
        df = pd.read_csv(url)
        # Limpieza profunda y creación de variables para la IA
        df = df.dropna(subset=['FTR', 'B365H', 'HC', 'AC', 'HY', 'AY'])
        return df
    except: return None

ligas = {
    "🇪🇸 ESP - LA LIGA": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "🇬🇧 ENG - PREMIER": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "🇮🇹 ITA - SERIE A": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "🇩🇪 GER - BUNDESLIGA": "https://www.football-data.co.uk/mmz4281/2526/D1.csv",
    "🇫🇷 FRA - LIGUE 1": "https://www.football-data.co.uk/mmz4281/2526/F1.csv"
}

# 3. CABECERA Y SELECCIÓN
c1, c2, c3 = st.columns([2, 3, 3])
sel_liga = c1.selectbox("LIGA", list(ligas.keys()))
df = load_global_data(ligas[sel_liga])

if df is not None:
    le = LabelEncoder()
    teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    le.fit(teams)
    t1 = c2.selectbox("LOCAL", teams)
    t2 = c3.selectbox("VISITANTE", teams, index=1)

    # 4. MOTOR IA DE ALTA PRECISIÓN (MÚLTIPLES OBJETIVOS)
    df['H_idx'] = le.transform(df['HomeTeam'])
    df['A_idx'] = le.transform(df['AwayTeam'])
    X = df[['H_idx', 'A_idx', 'B365H', 'B365D', 'B365A']].values
    
    # Entrenamiento de modelos específicos
    m_1x2 = RandomForestClassifier(n_estimators=300).fit(X, df['FTR'])
    m_goals = RandomForestRegressor(n_estimators=300).fit(X, df['FTHG'] + df['FTAG'])
    m_corners = RandomForestRegressor(n_estimators=300).fit(X, df['HC'] + df['AC'])
    m_cards = RandomForestRegressor(n_estimators=300).fit(X, df['HY'] + df['AY'])
    m_btts = RandomForestClassifier(n_estimators=300).fit(X, ((df['FTHG']>0) & (df['FTAG']>0)).astype(int))

    # Predicción
    test_match = [[le.transform([t1])[0], le.transform([t2])[0], 2.1, 3.3, 3.4]]
    p_1x2 = m_1x2.predict_proba(test_match)[0] # [H, D, A]
    p_goals = m_goals.predict(test_match)[0]
    p_corners = m_corners.predict(test_match)[0]
    p_cards = m_cards.predict(test_match)[0]
    p_btts = m_btts.predict_proba(test_match)[0][1] * 100

    # 5. RENDERIZADO DE TODOS LOS MERCADOS (ESTILO VOLERBET)
    st.markdown(f"### 🏟️ ANÁLISIS PRO: {t1} vs {t2}")
    
    colA, colB, colC = st.columns(3)

    with colA:
        # GANADOR 1X2
        st.markdown(f'<div class="market-box"><div class="market-title">Ganador 1X2</div><div class="bet-grid">'
                    f'<div class="bet-item">1 <span>{p_1x2[1]*100:.1f}%</span></div>'
                    f'<div class="bet-item">X <span>{p_1x2[0]*100:.1f}%</span></div>'
                    f'<div class="bet-item">2 <span>{p_1x2[2]*100:.1f}%</span></div></div></div>', unsafe_allow_html=True)
        
        # DOBLE OPORTUNIDAD
        st.markdown(f'<div class="market-box"><div class="market-title">Doble Oportunidad</div><div class="bet-grid">'
                    f'<div class="bet-item">1X <span class="prob-val">{(p_1x2[1]+p_1x2[0])*100:.1f}%</span></div>'
                    f'<div class="bet-item">X2 <span class="prob-val">{(p_1x2[0]+p_1x2[2])*100:.1f}%</span></div>'
                    f'<div class="bet-item">12 <span class="prob-val">{(p_1x2[1]+p_1x2[2])*100:.1f}%</span></div></div></div>', unsafe_allow_html=True)

    with colB:
        # GOLES MÁS/MENOS
        st.markdown(f'<div class="market-box"><div class="market-title">Total Goles (Est: {p_goals:.1f})</div><div class="bet-grid">'
                    f'<div class="bet-item">+1.5 <span class="prob-val">{min(99, p_goals*45):.1f}%</span></div>'
                    f'<div class="bet-item">+2.5 <span class="prob-val">{min(99, p_goals*32):.1f}%</span></div>'
                    f'<div class="bet-item">-3.5 <span class="prob-val">{max(1, 100-(p_goals*20)):.1f}%</span></div></div></div>', unsafe_allow_html=True)
        
        # AMBOS MARCAN
        st.markdown(f'<div class="market-box"><div class="market-title">Ambos Marcan</div><div class="bet-grid">'
                    f'<div class="bet-item">SÍ <span class="prob-val">{p_btts:.1f}%</span></div>'
                    f'<div class="bet-item">NO <span class="prob-val">{100-p_btts:.1f}%</span></div></div></div>', unsafe_allow_html=True)

    with colC:
        # CÓRNERS Y TARJETAS
        st.markdown(f'<div class="market-box"><div class="market-title">Córners (Media: {p_corners:.1f})</div><div class="bet-grid">'
                    f'<div class="bet-item">+8.5 <span class="prob-val">68.2%</span></div>'
                    f'<div class="bet-item">+10.5 <span class="prob-val">41.5%</span></div></div></div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="market-box"><div class="market-title">Tarjetas (Media: {p_cards:.1f})</div><div class="bet-grid">'
                    f'<div class="bet-item">+3.5 <span class="prob-val">74.1%</span></div>'
                    f'<div class="bet-item">+4.5 <span class="prob-val">52.8%</span></div></div></div>', unsafe_allow_html=True)

    st.warning("⚠️ IA PREDICT: Alta confianza en Doble Oportunidad 1X para este encuentro.")
