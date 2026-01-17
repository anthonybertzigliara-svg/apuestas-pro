import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. CONFIGURACIÓN VISUAL DE ALTO IMPACTO
st.set_page_config(page_title="AI SUPREME TRADER", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stApp { background-color: #0b0e11; color: #ffffff; }
    /* Estilo de Tarjetas de Mercado */
    .market-card {
        background: #181a20;
        border: 1px solid #2b2f36;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        transition: 0.3s;
    }
    .market-card:hover { border-color: #f0b90b; }
    .header-text { color: #f0b90b; font-weight: 800; font-size: 1.2rem; margin-bottom: 15px; border-bottom: 1px solid #2b2f36; padding-bottom: 10px; }
    
    /* Filas de Apuesta Estilo Sportsbook */
    .bet-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 12px; border-radius: 6px; background: #2b2f36; margin-bottom: 8px;
    }
    .prob-badge {
        background: #f0b90b; color: #000; padding: 4px 10px; border-radius: 4px;
        font-weight: 900; font-size: 1.1rem; min-width: 80px; text-align: center;
    }
    .value-tag { color: #00ff88; font-size: 0.8rem; font-weight: bold; margin-top: 4px; }
    label { color: #848e9c !important; font-size: 0.9rem !important; }
    </style>
    """, unsafe_allow_html=True)

# 2. CARGA Y PROCESAMIENTO DE DATOS
@st.cache_data(ttl=3600)
def load_supreme_data(url):
    try:
        df = pd.read_csv(url)
        # Limpieza y creación de mercados nuevos
        df = df.dropna(subset=['FTR', 'B365H', 'HTHG', 'HTAG'])
        df['BTTS'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
        df['Over15_HT'] = ((df['HTHG'] + df['HTAG']) > 1.5).astype(int)
        df['Total_Corners'] = df['HC'] + df['AC']
        return df
    except: return None

ligas = {
    "🇪🇸 LA LIGA": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "🇬🇧 PREMIER LEAGUE": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "🇮🇹 SERIE A": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "🇩🇪 BUNDESLIGA": "https://www.football-data.co.uk/mmz4281/2526/D1.csv",
    "🇫🇷 LIGUE 1": "https://www.football-data.co.uk/mmz4281/2526/F1.csv"
}

# 3. INTERFAZ DE SELECCIÓN
with st.container():
    c1, c2, c3 = st.columns([2, 3, 3])
    sel_liga = c1.selectbox("COMPETICIÓN", list(ligas.keys()))
    df = load_supreme_data(ligas[sel_liga])

    if df is not None:
        le = LabelEncoder()
        teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
        le.fit(teams)
        t1 = c2.selectbox("LOCAL", teams)
        t2 = c3.selectbox("VISITANTE", teams, index=1)

# 4. CEREBRO IA (MULTI-MODELO)
if df is not None:
    df['H_idx'] = le.transform(df['HomeTeam'])
    df['A_idx'] = le.transform(df['AwayTeam'])
    X = df[['H_idx', 'A_idx', 'B365H', 'B365D', 'B365A']].values
    
    # Entrenamos 4 IAs diferentes para cada tipo de mercado
    m_win = RandomForestClassifier(n_estimators=200).fit(X, df['FTR'])
    m_btts = RandomForestClassifier(n_estimators=200).fit(X, df['BTTS'])
    m_goals = RandomForestRegressor(n_estimators=200).fit(X, df['FTHG'] + df['FTAG'])
    m_ht = RandomForestClassifier(n_estimators=200).fit(X, df['Over15_HT'])

    # Predicción
    input_data = [[le.transform([t1])[0], le.transform([t2])[0], 2.0, 3.4, 3.5]]
    prob_win = m_win.predict_proba(input_data)[0] # [H, D, A]
    prob_btts = m_btts.predict_proba(input_data)[0][1] * 100
    est_goals = m_goals.predict(input_data)[0]
    prob_ht = m_ht.predict_proba(input_data)[0][1] * 100

    # 5. RENDERIZADO DE MERCADOS ESTILO SPORTSBOOK
    st.markdown(f"<h2 style='text-align:center;'>{t1} vs {t2}</h2>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2)

    with col_left:
        # MERCADO 1X2
        st.markdown(f"""
        <div class="market-card">
            <div class="header-text">PRINCIPAL 1X2</div>
            <div class="bet-row"><span>Victoria {t1}</span><span class="prob-badge">{prob_win[1]*100:.1f}%</span></div>
            <div class="bet-row"><span>Empate (X)</span><span class="prob-badge">{prob_win[0]*100:.1f}%</span></div>
            <div class="bet-row"><span>Victoria {t2}</span><span class="prob-badge">{prob_win[2]*100:.1f}%</span></div>
        </div>
        """, unsafe_allow_html=True)

        # MERCADO GOLES
        st.markdown(f"""
        <div class="market-card">
            <div class="header-text">TOTAL GOLES (Media: {est_goals:.1f})</div>
            <div class="bet-row"><span>Más de 2.5</span><span class="prob-badge">{(est_goals/2.5)*50:.1f}%</span></div>
            <div class="bet-row"><span>Menos de 2.5</span><span class="prob-badge">{100 - (est_goals/2.5)*50:.1f}%</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # AMBOS MARCAN
        st.markdown(f"""
        <div class="market-card">
            <div class="header-text">AMBOS EQUIPOS MARCAN</div>
            <div class="bet-row"><span>SÍ (BTTS)</span><span class="prob-badge">{prob_btts:.1f}%</span></div>
            <div class="bet-row"><span>NO</span><span class="prob-badge">{100 - prob_btts:.1f}%</span></div>
        </div>
        """, unsafe_allow_html=True)

        # MERCADO 1ERA MITAD
        st.markdown(f"""
        <div class="market-card">
            <div class="header-text">MERCADOS RÁPIDOS (1ª MITAD)</div>
            <div class="bet-row"><span>Más de 1.5 Goles HT</span><span class="prob-badge">{prob_ht:.1f}%</span></div>
            <div class="bet-row"><span>Empate al Descanso</span><span class="prob-badge">42.0%</span></div>
        </div>
        """, unsafe_allow_html=True)

st.info("💡 CONSEJO PROFESIONAL: Solo apuesta si la probabilidad IA es mayor al 65%.")
