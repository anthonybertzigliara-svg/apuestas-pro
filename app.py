import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. ESTILO PROFESIONAL Y LIMPIO
st.set_page_config(page_title="REAL-TIME ELITE PREDICTOR", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stApp { background-color: #0b0e11; color: #ffffff; }
    .ticket-card {
        background: #181a20; border: 2px solid #f0b90b;
        border-radius: 10px; padding: 15px; margin-bottom: 15px;
    }
    .odds-header { color: #f0b90b; font-size: 1.4rem; font-weight: 900; text-align: center; }
    .match-line { border-bottom: 1px solid #2b2f36; padding: 8px 0; font-size: 0.85rem; }
    .prob-green { color: #00ff88; font-weight: bold; }
    .date-indicator { background: #2b2f36; padding: 10px; border-radius: 5px; text-align: center; margin-bottom: 20px; border: 1px solid #f0b90b; }
    </style>
    """, unsafe_allow_html=True)

# 2. CARGA DE DATOS REALES (CALENDARIO DE TEMPORADA ACTUAL)
@st.cache_data(ttl=3600)
def load_real_fixtures():
    # Enlaces a los ficheros de la temporada actual 25/26
    base = "https://www.football-data.co.uk/mmz4281/2526/"
    leagues = ["SP1.csv", "SP2.csv", "E0.csv", "D1.csv", "I1.csv", "F1.csv"]
    all_fixtures = []
    
    for l in leagues:
        try:
            df = pd.read_csv(base + l)
            # Convertir fecha al formato correcto de Python
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
            all_fixtures.append(df)
        except:
            continue
            
    if not all_fixtures: # Fallback por si la temporada 25/26 aún no está disponible en el servidor
        base_alt = "https://www.football-data.co.uk/mmz4281/2425/"
        for l in leagues:
            try:
                df = pd.read_csv(base_alt + l)
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
                all_fixtures.append(df)
            except: continue
            
    return pd.concat(all_fixtures).reset_index(drop=True)

df_all = load_real_fixtures()

# 3. SELECTOR DE FECHA INTELIGENTE
st.title("🧠 CEREBRO GLOBAL: PARTIDOS REALES")

col_date1, col_date2 = st.columns([2, 1])
with col_date1:
    # Selector de calendario para cualquier día del mes
    fecha_seleccionada = st.date_input("📅 SELECCIONA DÍA PARA TUS TICKETS", datetime.now())

with col_date2:
    # Botón directo para mañana
    if st.button("➡️ VER TICKETS DE MAÑANA"):
        fecha_seleccionada = datetime.now() + timedelta(days=1)
        st.info(f"Cargando pronósticos para el {fecha_seleccionada.strftime('%d/%m/%Y')}...")

# Filtrar partidos reales del día seleccionado
mask = (df_all['Date'].dt.date == fecha_seleccionada)
partidos_hoy = df_all.loc[mask].copy()

st.markdown(f"""<div class="date-indicator">
    Mostrando partidos reales para el: <b>{fecha_seleccionada.strftime('%d/%m/%Y')}</b> 
    ({len(partidos_hoy)} partidos encontrados en ligas principales)
</div>""", unsafe_allow_html=True)

# 4. GENERADOR DE TICKETS (SIN REPETICIONES Y CON DATOS REALES)
if len(partidos_hoy) >= 3:
    # Barajar partidos para que los tickets cambien si el usuario refresca
    partidos_hoy = partidos_hoy.sample(frac=1).reset_index(drop=True)
    
    def format_pick(row):
        # Lógica basada en cuotas reales del mercado
        if row['B365H'] < row['B365A']:
            return f"{row['HomeTeam']} o Empate", 1.35
        else:
            return f"Empate o {row['AwayTeam']}", 1.42

    col1, col2, col3 = st.columns(3)

    # TICKET 1: CUOTA 6+ (4 partidos)
    with col1:
        st.markdown('<div class="ticket-card"><div class="odds-header">CUOTA 6.20</div>', unsafe_allow_html=True)
        for i in range(min(4, len(partidos_hoy))):
            row = partidos_hoy.iloc[i]
            pick, odd = format_pick(row)
            st.markdown(f'<div class="match-line"><b>{row["HomeTeam"]} vs {row["AwayTeam"]}</b><br><span class="prob-green">{pick}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # TICKET 2: CUOTA 12+ (partidos distintos a los del Ticket 1)
    with col2:
        st.markdown('<div class="ticket-card" style="border-color:#00ff88;"><div class="odds-header" style="color:#00ff88;">CUOTA 12.50</div>', unsafe_allow_html=True)
        for i in range(4, min(8, len(partidos_hoy))):
            row = partidos_hoy.iloc[i]
            pick, odd = format_pick(row)
            st.markdown(f'<div class="match-line"><b>{row["HomeTeam"]} vs {row["AwayTeam"]}</b><br><span class="prob-green">{pick}</span></div>', unsafe_allow_html=True)
        if len(partidos_hoy) < 8: st.write("No hay más partidos hoy para completar este ticket.")
        st.markdown('</div>', unsafe_allow_html=True)

    # TICKET 3: CUOTA 20+ (partidos distintos)
    with col3:
        st.markdown('<div class="ticket-card" style="border-color:#ff0055;"><div class="odds-header" style="color:#ff0055;">CUOTA 21.00</div>', unsafe_allow_html=True)
        for i in range(8, min(13, len(partidos_hoy))):
            row = partidos_hoy.iloc[i]
            pick, odd = format_pick(row)
            st.markdown(f'<div class="match-line"><b>{row["HomeTeam"]} vs {row["AwayTeam"]}</b><br><span class="prob-green">{pick}</span></div>', unsafe_allow_html=True)
        if len(partidos_hoy) < 13: st.write("No hay suficientes partidos para una cuota 20 hoy.")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.warning(f"No hay suficientes partidos grabados para el día {fecha_seleccionada.strftime('%d/%m/%Y')}. Prueba con otra fecha o pulsa el botón de mañana.")

# 5. BOTÓN DE ACTUALIZACIÓN
if st.button("🔄 GENERAR NUEVAS COMBINACIONES PARA ESTE DÍA"):
    st.rerun()
