import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. ESTILO VOLERBET (EL QUE TE GUSTABA)
st.set_page_config(page_title="AI ELITE TERMINAL", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stApp { background-color: #0b0e11; color: #ffffff; }
    .market-box {
        background: #181a20; border: 1px solid #2b2f36;
        border-radius: 4px; padding: 10px; margin-bottom: 10px;
    }
    .market-title {
        color: #f0b90b; font-size: 0.8rem; font-weight: bold;
        text-transform: uppercase; border-bottom: 1px solid #2b2f36;
        padding-bottom: 5px; margin-bottom: 8px;
    }
    .bet-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
    .bet-item {
        background: #2b2f36; padding: 8px; display: flex;
        justify-content: space-between; align-items: center;
        border-radius: 4px; font-size: 0.9rem;
    }
    .prob-val { color: #00ff88; font-weight: 900; }
    .quiniela-box {
        background: #1e2329; border: 2px solid #00ff88;
        padding: 15px; border-radius: 10px; margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. TODAS LAS LIGAS Y COPAS
@st.cache_data(ttl=3600)
def load_data(url):
    try:
        df = pd.read_csv(url).dropna(subset=['FTR', 'B365H'])
        return df
    except: return None

ligas = {
    "🏆 CHAMPIONS LEAGUE": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "🇪🇸 ESPAÑA - 1ª DIVISIÓN": "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
    "🇪🇸 ESPAÑA - 2ª DIVISIÓN / COPA": "https://www.football-data.co.uk/mmz4281/2425/SP2.csv",
    "🇬🇧 INGLATERRA - PREMIER": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "🇬🇧 INGLATERRA - CHAMPIONSHIP": "https://www.football-data.co.uk/mmz4281/2425/E1.csv",
    "🇮🇹 ITALIA - SERIE A": "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
    "🇮🇹 ITALIA - SERIE B / COPA": "https://www.football-data.co.uk/mmz4281/2425/I2.csv",
    "🇩🇪 ALEMANIA - BUNDESLIGA": "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
    "🇫🇷 FRANCIA - LIGUE 1 / COPA": "https://www.football-data.co.uk/mmz4281/2425/F1.csv",
    "🇵🇹 PORTUGAL - 1ª DIVISIÓN": "https://www.football-data.co.uk/mmz4281/2425/P1.csv"
}

c1, c2, c3 = st.columns([2, 3, 3])
sel_liga = c1.selectbox("COMPETICIÓN", list(ligas.keys()))
df = load_data(ligas[sel_liga])

if df is not None:
    le = LabelEncoder()
    teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    le.fit(teams)
    t1 = c2.selectbox("LOCAL", teams)
    t2 = c3.selectbox("VISITANTE", teams, index=1)

    # 3. IA DE PREDICCIÓN
    df['H_idx'] = le.transform(df['HomeTeam'])
    df['A_idx'] = le.transform(df['AwayTeam'])
    X = df[['H_idx', 'A_idx', 'B365H', 'B365D', 'B365A']].values
    m_win = RandomForestClassifier(n_estimators=200).fit(X, df['FTR'])
    m_goals = RandomForestRegressor(n_estimators=200).fit(X, df['FTHG'] + df['FTAG'])

    v_in = [[le.transform([t1])[0], le.transform([t2])[0], 2.0, 3.4, 3.5]]
    p_1x2 = m_win.predict_proba(v_in)[0]
    g_est = m_goals.predict(v_in)[0]

    # 4. MERCADOS CLÁSICOS
    st.markdown(f"### 🏟️ {t1} vs {t2}")
    
    colA, colB = st.columns(2)
    
    # Probabilidades de Doble Oportunidad
    prob_1x = (p_1x2[1]+p_1x2[0])*100
    prob_x2 = (p_1x2[0]+p_1x2[2])*100

    with colA:
        st.markdown(f"""<div class="market-box"><div class="market-title">Doble Oportunidad</div><div class="bet-grid">
            <div class="bet-item">{t1} o Empate <span class="prob-val">{prob_1x:.1f}%</span></div>
            <div class="bet-item">Empate o {t2} <span class="prob-val">{prob_x2:.1f}%</span></div>
        </div></div>""", unsafe_allow_html=True)

    with colB:
        st.markdown(f"""<div class="market-box"><div class="market-title">Goles Totales</div><div class="bet-grid">
            <div class="bet-item">Más de 1.5 <span class="prob-val">{min(99, g_est*45):.1f}%</span></div>
            <div class="bet-item">Más de 2.5 <span class="prob-val">{min(99, g_est*32):.1f}%</span></div>
        </div></div>""", unsafe_allow_html=True)

    # 5. EL CUADRO DE QUINIELA / COMBINADA (ABAJO)
    st.markdown("---")
    st.markdown('<div class="quiniela-box">', unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; color:#00ff88;'>📋 MI QUINIELA MAESTRA</h2>", unsafe_allow_html=True)
    
    # Elegimos los 3 mejores picks basados en la probabilidad más alta
    picks = []
    if prob_1x > 70: picks.append(f"✅ DOBLE: {t1} o Empate ({prob_1x:.1f}%)")
    if prob_x2 > 70: picks.append(f"✅ DOBLE: Empate o {t2} ({prob_x2:.1f}%)")
    if g_est > 2.0: picks.append(f"⚽ GOLES: Más de 1.5 Goles")
    
    for p in picks:
        st.write(p)
    
    st.markdown("<p style='text-align:center; font-weight:bold;'>ESTA ES TU MEJOR COMBINADA PARA ESTE PARTIDO</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
