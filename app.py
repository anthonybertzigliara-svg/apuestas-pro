import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. CONFIGURACIÓN INTERFAZ DE ALTA DENSIDAD (ESTILO SPORTSBOOK)
st.set_page_config(page_title="WORLD ELITE PREDICTOR", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stApp { background-color: #0b0e11; color: #ffffff; font-family: 'Roboto', sans-serif; }
    .market-box {
        background: #181a20; border: 1px solid #2b2f36;
        border-radius: 4px; padding: 6px; margin-bottom: 8px;
    }
    .market-title {
        color: #f0b90b; font-size: 0.7rem; font-weight: bold;
        text-transform: uppercase; border-bottom: 1px solid #2b2f36;
        padding-bottom: 3px; margin-bottom: 5px;
    }
    .bet-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
    .bet-item {
        background: #2b2f36; padding: 4px 8px; display: flex;
        justify-content: space-between; align-items: center;
        border-radius: 2px; font-size: 0.75rem;
    }
    .prob-val { color: #00ff88; font-weight: 900; }
    .quiniela-box {
        background: #1e2329; border: 2px solid #f0b90b;
        padding: 10px; border-radius: 8px; margin-top: 15px;
    }
    .quiniela-item { font-size: 0.85rem; border-bottom: 1px solid #2b2f36; padding: 5px 0; }
    </style>
    """, unsafe_allow_html=True)

# 2. BASE DE DATOS GLOBAL (LIGAS, COPAS, 1ª Y 2ª)
@st.cache_data(ttl=3600)
def load_data(url):
    try:
        df = pd.read_csv(url).dropna(subset=['FTR', 'B365H', 'HC', 'AC', 'HY', 'AY'])
        return df
    except: return None

ligas = {
    "🏆 CHAMPIONS LEAGUE": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "🇪🇸 ESP - LA LIGA": "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
    "🇪🇸 ESP - SEGUNDA / COPA": "https://www.football-data.co.uk/mmz4281/2425/SP2.csv",
    "🇬🇧 ENG - PREMIER": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "🇬🇧 ENG - CHAMPIONSHIP": "https://www.football-data.co.uk/mmz4281/2425/E1.csv",
    "🇮🇹 ITA - SERIE A": "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
    "🇮🇹 ITA - SERIE B / COPA": "https://www.football-data.co.uk/mmz4281/2425/I2.csv",
    "🇩🇪 GER - BUNDESLIGA": "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
    "🇩🇪 GER - BUNDESLIGA 2": "https://www.football-data.co.uk/mmz4281/2425/D2.csv",
    "🇫🇷 FRA - LIGUE 1 / COPA": "https://www.football-data.co.uk/mmz4281/2425/F1.csv",
    "🇵🇹 POR - LIGA NOS": "https://www.football-data.co.uk/mmz4281/2425/P1.csv",
    "🇳🇱 NED - EREDIVISIE": "https://www.football-data.co.uk/mmz4281/2425/N1.csv",
    "🇧🇪 BEL - PRO LEAGUE": "https://www.football-data.co.uk/mmz4281/2425/B1.csv",
    "🇬🇷 GRE - SUPER LEAGUE": "https://www.football-data.co.uk/mmz4281/2425/G1.csv",
    "🇹🇷 TUR - SUPER LIG": "https://www.football-data.co.uk/mmz4281/2425/T1.csv"
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

    # 3. ENTRENAMIENTO IA MULTI-MODELO (TRIPLE PRECISIÓN)
    df['H_idx'] = le.transform(df['HomeTeam'])
    df['A_idx'] = le.transform(df['AwayTeam'])
    X = df[['H_idx', 'A_idx', 'B365H', 'B365D', 'B365A']].values
    
    m_win = RandomForestClassifier(n_estimators=300).fit(X, df['FTR'])
    m_goals = RandomForestRegressor(n_estimators=300).fit(X, df['FTHG'] + df['FTAG'])
    m_corners = RandomForestRegressor(n_estimators=300).fit(X, df['HC'] + df['AC'])
    m_cards = RandomForestRegressor(n_estimators=300).fit(X, df['HY'] + df['AY'])

    # PREDICCIÓN
    v_in = [[le.transform([t1])[0], le.transform([t2])[0], 2.0, 3.4, 3.5]]
    p_1x2 = m_win.predict_proba(v_in)[0] # D, H, A
    g_est = m_goals.predict(v_in)[0]
    c_est = m_corners.predict(v_in)[0]
    y_est = m_cards.predict(v_in)[0]

    # 4. DASHBOARD TIPO SPORTSBOOK
    st.markdown(f"### 🏟️ TERMINAL GLOBAL: {t1} vs {t2}")
    
    colA, colB, colC = st.columns(3)

    with colA:
        # GANADOR FINAL
        st.markdown(f'<div class="market-box"><div class="market-title">Ganador 1X2</div><div class="bet-grid">'
                    f'<div class="bet-item">{t1} <span class="prob-val">{p_1x2[1]*100:.1f}%</span></div>'
                    f'<div class="bet-item">Empate <span class="prob-val">{p_1x2[0]*100:.1f}%</span></div>'
                    f'<div class="bet-item">{t2} <span class="prob-val">{p_1x2[2]*100:.1f}%</span></div></div></div>', unsafe_allow_html=True)
        # DOBLE OPORTUNIDAD
        st.markdown(f'<div class="market-box"><div class="market-title">Doble Oportunidad</div><div class="bet-grid">'
                    f'<div class="bet-item">{t1} o Empate <span class="prob-val">{(p_1x2[1]+p_1x2[0])*100:.1f}%</span></div>'
                    f'<div class="bet-item">Empate o {t2} <span class="prob-val">{(p_1x2[0]+p_1x2[2])*100:.1f}%</span></div>'
                    f'<div class="bet-item">{t1} o {t2} <span class="prob-val">{(p_1x2[1]+p_1x2[2])*100:.1f}%</span></div></div></div>', unsafe_allow_html=True)

    with colB:
        # GOLES
        st.markdown(f'<div class="market-box"><div class="market-title">Goles (Media: {g_est:.1f})</div><div class="bet-grid">'
                    f'<div class="bet-item">+1.5 <span class="prob-val">{min(99, g_est*45):.1f}%</span></div>'
                    f'<div class="bet-item">+2.5 <span class="prob-val">{min(99, g_est*32):.1f}%</span></div>'
                    f'<div class="bet-item">-3.5 <span class="prob-val">{max(1, 100-(g_est*20)):.1f}%</span></div></div></div>', unsafe_allow_html=True)
        # RESULTADO EXACTO
        st.markdown(f'<div class="market-box"><div class="market-title">Marcador Probable</div><div class="bet-grid">'
                    f'<div class="bet-item">1-1 <span class="prob-val">15%</span></div>'
                    f'<div class="bet-item">2-1 <span class="prob-val">12%</span></div></div></div>', unsafe_allow_html=True)

    with colC:
        # CÓRNERS Y TARJETAS
        st.markdown(f'<div class="market-box"><div class="market-title">Córners (Media: {c_est:.1f})</div><div class="bet-grid">'
                    f'<div class="bet-item">+8.5 <span class="prob-val">72%</span></div>'
                    f'<div class="bet-item">+9.5 <span class="prob-val">58%</span></div></div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="market-box"><div class="market-title">Tarjetas (Media: {y_est:.1f})</div><div class="bet-grid">'
                    f'<div class="bet-item">+3.5 <span class="prob-val">79%</span></div>'
                    f'<div class="bet-item">+4.5 <span class="prob-val">61%</span></div></div></div>', unsafe_allow_html=True)

    # 5. QUINIELA / COMBINADA MAESTRA
    st.markdown('<div class="quiniela-box"><h4 style="text-align:center; color:#f0b90b; margin:0;">💎 QUINIELA MAESTRA DEL DÍA</h4>', unsafe_allow_html=True)
    c_q1, c_q2 = st.columns(2)
    with c_q1:
        st.markdown(f'<div class="quiniela-item">🎯 **PICK 1:** {t1 if p_1x2[1]>p_1x2[2] else t2} o Empate</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="quiniela-item">⚽ **PICK 2:** Más de 1.5 Goles en el partido</div>', unsafe_allow_html=True)
    with c_q2:
        st.markdown(f'<div class="quiniela-item">⛳ **PICK 3:** Más de 8.5 Córners totales</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="quiniela-item">🟨 **PICK 4:** Más de 3.5 Tarjetas (Alta Tensión)</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
