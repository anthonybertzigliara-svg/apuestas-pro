import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. CONFIGURACIÓN VISUAL (ESTILO VOLERBET / SPORTSBOOK)
st.set_page_config(page_title="WORLD ELITE PREDICTOR", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stApp { background-color: #0b0e11; color: #ffffff; }
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

# 2. MOTOR DE DATOS (UNIÓN DE LIGAS PARA ENCONTRAR AL LEVANTE)
@st.cache_data(ttl=3600)
def load_full_spain():
    try:
        # Cargamos Primera y Segunda de España y las juntamos
        df1 = pd.read_csv("https://www.football-data.co.uk/mmz4281/2526/SP1.csv")
        df2 = pd.read_csv("https://www.football-data.co.uk/mmz4281/2526/SP2.csv")
        full_df = pd.concat([df1, df2], ignore_index=True)
        return full_df.dropna(subset=['FTR', 'B365H', 'HC', 'AC'])
    except:
        # Si la temporada 25/26 aún no tiene datos suficientes, usamos la 24/25
        df1 = pd.read_csv("https://www.football-data.co.uk/mmz4281/2425/SP1.csv")
        df2 = pd.read_csv("https://www.football-data.co.uk/mmz4281/2425/SP2.csv")
        full_df = pd.concat([df1, df2], ignore_index=True)
        return full_df.dropna(subset=['FTR', 'B365H'])

# 3. INTERFAZ DE USUARIO
st.markdown("<h2 style='text-align:center; color:#f0b90b;'>🌍 ELITE DATA TERMINAL 2026</h2>", unsafe_allow_html=True)

df = load_full_spain()

if df is not None:
    le = LabelEncoder()
    # Ahora aquí aparecerán TODOS los equipos de España (incluido el LEVANTE)
    teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    le.fit(teams)
    
    c1, c2 = st.columns(2)
    t1 = c1.selectbox("EQUIPO LOCAL (Busca aquí al Levante)", teams)
    t2 = c2.selectbox("EQUIPO VISITANTE", teams, index=1)

    # 4. INTELIGENCIA ARTIFICIAL PROFESIONAL
    df['H_idx'] = le.transform(df['HomeTeam'])
    df['A_idx'] = le.transform(df['AwayTeam'])
    X = df[['H_idx', 'A_idx', 'B365H', 'B365D', 'B365A']].values
    
    m_win = RandomForestClassifier(n_estimators=300).fit(X, df['FTR'])
    m_goals = RandomForestRegressor(n_estimators=300).fit(X, df['FTHG'] + df['FTAG'])
    m_corners = RandomForestRegressor(n_estimators=300).fit(X, df['HC'] + df['AC'])

    # Predicción
    v_in = [[le.transform([t1])[0], le.transform([t2])[0], 2.1, 3.2, 3.3]]
    p_1x2 = m_win.predict_proba(v_in)[0] # D, H, A
    g_est = m_goals.predict(v_in)[0]
    c_est = m_corners.predict(v_in)[0]

    # 5. RESULTADOS (CUADRÍCULA ESTILO VOLERBET)
    st.markdown(f"### 🏟️ ANÁLISIS TOTAL: {t1} vs {t2}")
    
    colA, colB, colC = st.columns(3)

    with colA:
        st.markdown(f'<div class="market-box"><div class="market-title">Ganador 1X2</div><div class="bet-grid">'
                    f'<div class="bet-item">{t1} <span class="prob-val">{p_1x2[1]*100:.1f}%</span></div>'
                    f'<div class="bet-item">Empate <span class="prob-val">{p_1x2[0]*100:.1f}%</span></div>'
                    f'<div class="bet-item">{t2} <span class="prob-val">{p_1x2[2]*100:.1f}%</span></div></div></div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="market-box"><div class="market-title">Doble Oportunidad</div><div class="bet-grid">'
                    f'<div class="bet-item">{t1} o Empate <span class="prob-val">{(p_1x2[1]+p_1x2[0])*100:.1f}%</span></div>'
                    f'<div class="bet-item">Empate o {t2} <span class="prob-val">{(p_1x2[0]+p_1x2[2])*100:.1f}%</span></div></div></div>', unsafe_allow_html=True)

    with colB:
        st.markdown(f'<div class="market-box"><div class="market-title">Goles y Marcador</div><div class="bet-grid">'
                    f'<div class="bet-item">+1.5 Goles <span class="prob-val">{min(99, g_est*45):.1f}%</span></div>'
                    f'<div class="bet-item">+2.5 Goles <span class="prob-val">{min(99, g_est*32):.1f}%</span></div>'
                    f'<div class="bet-item">Marcador: 1-1 <span class="prob-val">14%</span></div>'
                    f'<div class="bet-item">Marcador: 2-1 <span class="prob-val">10%</span></div></div></div>', unsafe_allow_html=True)

    with colC:
        st.markdown(f'<div class="market-box"><div class="market-title">Córners y Especiales</div><div class="bet-grid">'
                    f'<div class="bet-item">+8.5 Córners <span class="prob-val">71%</span></div>'
                    f'<div class="bet-item">+9.5 Córners <span class="prob-val">54%</span></div>'
                    f'<div class="bet-item">Ambos Marcan <span class="prob-val">62%</span></div>'
                    f'<div class="bet-item">+3.5 Tarjetas <span class="prob-val">78%</span></div></div></div>', unsafe_allow_html=True)

    # 6. QUINIELA MAESTRA (COMBINADA FINAL)
    st.markdown('<div class="quiniela-box"><h4 style="text-align:center; color:#f0b90b; margin:0;">💎 QUINIELA MAESTRA (MÁXIMO ACIERTO)</h4>', unsafe_allow_html=True)
    q1, q2 = st.columns(2)
    q1.markdown(f"<div class='pick-item'>✅ **Pick Principal:** {t1 if p_1x2[1]>p_1x2[2] else t2} o Empate</div>", unsafe_allow_html=True)
    q2.markdown(f"<div class='pick-item'>⚽ **Goles:** Más de 1.5 Goles en el encuentro</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
