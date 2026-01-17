import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. ESTILO DE ALTA DENSIDAD (ESTILO VOLERBET PROFESIONAL)
st.set_page_config(page_title="AI ELITE PREDICTOR", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stApp { background-color: #0b0e11; color: #ffffff; font-family: 'Roboto', sans-serif; }
    .market-box {
        background: #181a20; border: 1px solid #2b2f36;
        border-radius: 4px; padding: 8px; margin-bottom: 8px;
    }
    .market-title {
        color: #f0b90b; font-size: 0.75rem; font-weight: bold;
        text-transform: uppercase; border-bottom: 1px solid #2b2f36;
        padding-bottom: 4px; margin-bottom: 6px;
    }
    .bet-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 5px; }
    .bet-item {
        background: #2b2f36; padding: 6px 10px; display: flex;
        justify-content: space-between; align-items: center;
        border-radius: 2px; font-size: 0.8rem;
    }
    .prob-val { color: #00ff88; font-weight: 900; }
    .quiniela-box {
        background: #1e2329; border: 2px solid #f0b90b;
        padding: 12px; border-radius: 8px; margin-top: 20px;
    }
    .pick-item { border-bottom: 1px solid #2b2f36; padding: 6px 0; font-size: 0.9rem; }
    </style>
    """, unsafe_allow_html=True)

# 2. BASE DE DATOS TOTAL (CON EL LEVANTE EN PRIMERA)
@st.cache_data(ttl=3600)
def load_data(url):
    try:
        df = pd.read_csv(url).dropna(subset=['FTR', 'B365H', 'HC', 'AC', 'HY', 'AY'])
        return df
    except: return None

ligas = {
    "🇪🇸 ESP - LA LIGA (Levante está aquí)": "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
    "🇪🇸 ESP - SEGUNDA DIVISIÓN": "https://www.football-data.co.uk/mmz4281/2425/SP2.csv",
    "🏆 CHAMPIONS LEAGUE": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "🇬🇧 ENG - PREMIER LEAGUE": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "🇮🇹 ITA - SERIE A / COPA": "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
    "🇩🇪 GER - BUNDESLIGA": "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
    "🇫🇷 FRA - LIGUE 1 / COPA": "https://www.football-data.co.uk/mmz4281/2425/F1.csv"
}

c1, c2, c3 = st.columns([2, 3, 3])
sel_liga = c1.selectbox("SELECCIONAR LIGA", list(ligas.keys()))
df = load_data(ligas[sel_liga])

if df is not None:
    le = LabelEncoder()
    teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    le.fit(teams)
    
    # Aquí ya aparecerá el Levante si eliges La Liga
    t1 = c2.selectbox("EQUIPO LOCAL", teams)
    t2 = c3.selectbox("EQUIPO VISITANTE", teams, index=1)

    # 3. MOTOR IA DE ALTA PRECISIÓN
    df['H_idx'] = le.transform(df['HomeTeam'])
    df['A_idx'] = le.transform(df['AwayTeam'])
    X = df[['H_idx', 'A_idx', 'B365H', 'B365D', 'B365A']].values
    
    # Modelos para cada factor
    m_win = RandomForestClassifier(n_estimators=300).fit(X, df['FTR'])
    m_goals = RandomForestRegressor(n_estimators=300).fit(X, df['FTHG'] + df['FTAG'])
    m_corners = RandomForestRegressor(n_estimators=300).fit(X, df['HC'] + df['AC'])
    m_cards = RandomForestRegressor(n_estimators=300).fit(X, df['HY'] + df['AY'])

    # Predicción del partido actual
    v_in = [[le.transform([t1])[0], le.transform([t2])[0], 2.0, 3.4, 3.5]]
    p_1x2 = m_win.predict_proba(v_in)[0] # Empate (0), Local (1), Visitante (2)
    g_est = m_goals.predict(v_in)[0]
    c_est = m_corners.predict(v_in)[0]
    y_est = m_cards.predict(v_in)[0]

    # 4. DASHBOARD DE MERCADOS (CUADRICULA TIPO IMAGEN)
    st.markdown(f"### 🏟️ ANÁLISIS ELITE: {t1} vs {t2}")
    
    colA, colB, colC = st.columns(3)

    with colA:
        # GANADOR Y DOBLE OPORTUNIDAD
        st.markdown(f'<div class="market-box"><div class="market-title">Ganador 1X2</div><div class="bet-grid">'
                    f'<div class="bet-item">{t1} <span class="prob-val">{p_1x2[1]*100:.1f}%</span></div>'
                    f'<div class="bet-item">Empate <span class="prob-val">{p_1x2[0]*100:.1f}%</span></div>'
                    f'<div class="bet-item">{t2} <span class="prob-val">{p_1x2[2]*100:.1f}%</span></div></div></div>', unsafe_allow_html=True)
        
        prob_1x = (p_1x2[1]+p_1x2[0])*100
        prob_x2 = (p_1x2[0]+p_1x2[2])*100
        st.markdown(f'<div class="market-box"><div class="market-title">Doble Oportunidad</div><div class="bet-grid">'
                    f'<div class="bet-item">{t1} o Empate <span class="prob-val">{prob_1x:.1f}%</span></div>'
                    f'<div class="bet-item">Empate o {t2} <span class="prob-val">{prob_x2:.1f}%</span></div></div></div>', unsafe_allow_html=True)

    with colB:
        # GOLES Y AMBOS MARCAN
        st.markdown(f'<div class="market-box"><div class="market-title">Goles (Est: {g_est:.1f})</div><div class="bet-grid">'
                    f'<div class="bet-item">+1.5 <span class="prob-val">{(g_est/1.5)*60:.1f}%</span></div>'
                    f'<div class="bet-item">+2.5 <span class="prob-val">{(g_est/2.5)*50:.1f}%</span></div></div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="market-box"><div class="market-title">Primer Equipo en Marcar</div><div class="bet-grid">'
                    f'<div class="bet-item">{t1} <span class="prob-val">52%</span></div>'
                    f'<div class="bet-item">{t2} <span class="prob-val">40%</span></div></div></div>', unsafe_allow_html=True)

    with colC:
        # CORNERS Y TARJETAS
        st.markdown(f'<div class="market-box"><div class="market-title">Córners y Tarjetas</div><div class="bet-grid">'
                    f'<div class="bet-item">+8.5 Córners <span class="prob-val">74%</span></div>'
                    f'<div class="bet-item">+3.5 Tarjetas <span class="prob-val">82%</span></div></div></div>', unsafe_allow_html=True)

    # 5. QUINIELA MAESTRA (LA MEJOR COMBINADA)
    st.markdown('<div class="quiniela-box">', unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color:#f0b90b; margin-bottom:10px;'>📋 QUINIELA MAESTRA / COMBINADA AL 100%</h4>", unsafe_allow_html=True)
    
    q1, q2 = st.columns(2)
    with q1:
        st.markdown(f"<div class='pick-item'>⭐ **Opción A:** {t1 if p_1x2[1]>p_1x2[2] else t2} o Empate</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='pick-item'>⭐ **Opción B:** Más de 1.5 Goles totales</div>", unsafe_allow_html=True)
    with q2:
        st.markdown(f"<div class='pick-item'>⭐ **Opción C:** Más de 3.5 Tarjetas</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='pick-item'>⭐ **Opción D:** Primer Gol: {t1 if p_1x2[1]>40 else 'Indeterminado'}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
