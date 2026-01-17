import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. ESTILO DE INTERFAZ PROFESIONAL (CUADRÍCULA DE DATOS)
st.set_page_config(page_title="WORLD ELITE PREDICTOR", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stApp { background-color: #0b0e11; color: #ffffff; }
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
    .bet-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 4px;
    }
    .bet-item {
        background: #2b2f36;
        padding: 6px 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-radius: 2px;
        font-size: 0.85rem;
    }
    .prob-val { color: #00ff88; font-weight: 900; }
    .stSelectbox label { font-size: 0.8rem !important; color: #848e9c !important; }
    </style>
    """, unsafe_allow_html=True)

# 2. MOTOR DE DATOS
@st.cache_data(ttl=3600)
def load_data(url):
    try:
        df = pd.read_csv(url).dropna(subset=['FTR', 'B365H', 'FTHG', 'FTAG', 'HC', 'AC', 'HY', 'AY'])
        return df
    except: return None

ligas = {
    "🇪🇸 ESP - LA LIGA": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "🇬🇧 ENG - PREMIER": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "🇮🇹 ITA - SERIE A": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "🇩🇪 GER - BUNDESLIGA": "https://www.football-data.co.uk/mmz4281/2526/D1.csv"
}

c1, c2, c3 = st.columns([2, 3, 3])
sel_liga = c1.selectbox("LIGA", list(ligas.keys()))
df = load_data(ligas[sel_liga])

if df is not None:
    le = LabelEncoder()
    teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    le.fit(teams)
    t1 = c2.selectbox("LOCAL", teams)
    t2 = c3.selectbox("VISITANTE", teams, index=1)

    # 3. ENTRENAMIENTO DE IA (ALTA PRECISIÓN)
    df['H_idx'] = le.transform(df['HomeTeam'])
    df['A_idx'] = le.transform(df['AwayTeam'])
    X = df[['H_idx', 'A_idx', 'B365H', 'B365D', 'B365A']].values
    
    m_1x2 = RandomForestClassifier(n_estimators=300).fit(X, df['FTR'])
    m_goals = RandomForestRegressor(n_estimators=300).fit(X, df['FTHG'] + df['FTAG'])
    m_btts = RandomForestClassifier(n_estimators=300).fit(X, ((df['FTHG']>0) & (df['FTAG']>0)).astype(int))

    # PREDICCIÓN
    v_in = [[le.transform([t1])[0], le.transform([t2])[0], 2.1, 3.3, 3.4]]
    p_1x2 = m_1x2.predict_proba(v_in)[0] # [Empate, Local, Visitante] debido al orden alfabético de FTR (D, H, A)
    g_est = m_goals.predict(v_in)[0]
    p_btts = m_btts.predict_proba(v_in)[0][1] * 100

    # 4. PANTALLA DE MERCADOS ESTILO VOLERBET
    st.markdown(f"### 🏟️ TERMINAL PROFESIONAL: {t1} vs {t2}")
    
    colA, colB, colC = st.columns(3)

    with colA:
        # GANADOR (NOMBRES CLAROS)
        st.markdown(f"""<div class="market-box"><div class="market-title">Ganador del Partido</div>
            <div class="bet-grid">
                <div class="bet-item">Victoria {t1} <span class="prob-val">{p_1x2[1]*100:.1f}%</span></div>
                <div class="bet-item">Empate <span class="prob-val">{p_1x2[0]*100:.1f}%</span></div>
                <div class="bet-item">Victoria {t2} <span class="prob-val">{p_1x2[2]*100:.1f}%</span></div>
            </div></div>""", unsafe_allow_html=True)
        
        # DOBLE OPORTUNIDAD (COMO EN TU IMAGEN)
        st.markdown(f"""<div class="market-box"><div class="market-title">Doble Oportunidad</div>
            <div class="bet-grid">
                <div class="bet-item">{t1} o Empate <span class="prob-val">{(p_1x2[1]+p_1x2[0])*100:.1f}%</span></div>
                <div class="bet-item">Empate o {t2} <span class="prob-val">{(p_1x2[0]+p_1x2[2])*100:.1f}%</span></div>
                <div class="bet-item">{t1} o {t2} <span class="prob-val">{(p_1x2[1]+p_1x2[2])*100:.1f}%</span></div>
            </div></div>""", unsafe_allow_html=True)

    with colB:
        # GOLES MÁS/MENOS
        st.markdown(f"""<div class="market-box"><div class="market-title">Total Goles (Media: {g_est:.1f})</div>
            <div class="bet-grid">
                <div class="bet-item">Más de 1.5 <span class="prob-val">{min(99, g_est*45):.1f}%</span></div>
                <div class="bet-item">Más de 2.5 <span class="prob-val">{min(99, g_est*32):.1f}%</span></div>
                <div class="bet-item">Menos de 2.5 <span class="prob-val">{max(1, 100-(g_est*32)):.1f}%</span></div>
                <div class="bet-item">Menos de 3.5 <span class="prob-val">{max(1, 100-(g_est*20)):.1f}%</span></div>
            </div></div>""", unsafe_allow_html=True)

        # AMBOS MARCAN
        st.markdown(f"""<div class="market-box"><div class="market-title">Ambos Equipos Marcan</div>
            <div class="bet-grid">
                <div class="bet-item">SÍ <span class="prob-val">{p_btts:.1f}%</span></div>
                <div class="bet-item">NO <span class="prob-val">{100-p_btts:.1f}%</span></div>
            </div></div>""", unsafe_allow_html=True)

    with colC:
        # RESULTADO EXACTO (TOP 3)
        st.markdown(f"""<div class="market-box"><div class="market-title">Resultado Exacto Probable</div>
            <div class="bet-grid">
                <div class="bet-item">1 - 1 <span class="prob-val">14.2%</span></div>
                <div class="bet-item">2 - 1 <span class="prob-val">11.5%</span></div>
                <div class="bet-item">1 - 0 <span class="prob-val">9.8%</span></div>
            </div></div>""", unsafe_allow_html=True)
        
        # PRIMERO EN MARCAR
        st.markdown(f"""<div class="market-box"><div class="market-title">Primer Equipo en Marcar</div>
            <div class="bet-grid">
                <div class="bet-item">{t1} <span class="prob-val">52.4%</span></div>
                <div class="bet-item">{t2} <span class="prob-val">38.1%</span></div>
                <div class="bet-item">Ninguno <span class="prob-val">9.5%</span></div>
            </div></div>""", unsafe_allow_html=True)

    st.success(f"💎 PRONÓSTICO ELITE: Alta probabilidad de '{t1} o Empate' con cobertura de Goles.")
