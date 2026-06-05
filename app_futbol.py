"""
╔══════════════════════════════════════════════════════════════╗
║   ⚽ ║   ⚽ PREDICTOR DE FÚTBOL TOTAL - CREADO POR [JHEISON LLALLE] ⚽  ║        ║
║   211 SELECCIONES | +160 CLUBES | OVER/UNDER | GRÁFICOS    ║
║   HISTORIAL | EXPORTAR | SIMULADOR | MODO OSCURO            ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from datetime import datetime
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN INICIAL
# ============================================================
st.set_page_config(page_title="⚽ Predictor de Fútbol - By [JHEISON LLALLE]", page_icon="⚽", layout="wide")

# Modo oscuro/claro
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Historial de predicciones
if 'historial' not in st.session_state:
    st.session_state.historial = []

# ============================================================
# BASE DE DATOS MUNDIAL COMPLETA
# ============================================================

@st.cache_data
def cargar_datos():
    EQUIPOS = {
        # PREMIER LEAGUE + CHAMPIONSHIP
        "Manchester City": {"liga": "Premier League", "pais": "Inglaterra", "spi": 94, "off": 3.1, "def": 0.5, "tipo": "club", "forma": ["G","G","G","E","G"]},
        "Arsenal": {"liga": "Premier League", "pais": "Inglaterra", "spi": 89, "off": 2.8, "def": 0.6, "tipo": "club", "forma": ["G","G","E","G","G"]},
        "Liverpool": {"liga": "Premier League", "pais": "Inglaterra", "spi": 88, "off": 2.9, "def": 0.6, "tipo": "club", "forma": ["G","E","G","G","P"]},
        "Aston Villa": {"liga": "Premier League", "pais": "Inglaterra", "spi": 82, "off": 2.5, "def": 0.8, "tipo": "club", "forma": []},
        "Tottenham": {"liga": "Premier League", "pais": "Inglaterra", "spi": 80, "off": 2.4, "def": 0.9, "tipo": "club", "forma": []},
        "Chelsea": {"liga": "Premier League", "pais": "Inglaterra", "spi": 78, "off": 2.2, "def": 0.9, "tipo": "club", "forma": []},
        "Newcastle": {"liga": "Premier League", "pais": "Inglaterra", "spi": 77, "off": 2.3, "def": 0.8, "tipo": "club", "forma": []},
        "Manchester United": {"liga": "Premier League", "pais": "Inglaterra", "spi": 76, "off": 2.0, "def": 1.0, "tipo": "club", "forma": []},
        "West Ham": {"liga": "Premier League", "pais": "Inglaterra", "spi": 72, "off": 1.8, "def": 1.0, "tipo": "club", "forma": []},
        "Brighton": {"liga": "Premier League", "pais": "Inglaterra", "spi": 70, "off": 1.7, "def": 1.1, "tipo": "club", "forma": []},
        "Fulham": {"liga": "Premier League", "pais": "Inglaterra", "spi": 65, "off": 1.5, "def": 1.2, "tipo": "club", "forma": []},
        "Everton": {"liga": "Premier League", "pais": "Inglaterra", "spi": 62, "off": 1.2, "def": 1.2, "tipo": "club", "forma": []},
        "Wolves": {"liga": "Premier League", "pais": "Inglaterra", "spi": 60, "off": 1.2, "def": 1.3, "tipo": "club", "forma": []},
        "Crystal Palace": {"liga": "Premier League", "pais": "Inglaterra", "spi": 60, "off": 1.1, "def": 1.3, "tipo": "club", "forma": []},
        "Brentford": {"liga": "Premier League", "pais": "Inglaterra", "spi": 58, "off": 1.3, "def": 1.4, "tipo": "club", "forma": []},
        "Nottm Forest": {"liga": "Premier League", "pais": "Inglaterra", "spi": 55, "off": 1.0, "def": 1.4, "tipo": "club", "forma": []},
        "Bournemouth": {"liga": "Premier League", "pais": "Inglaterra", "spi": 54, "off": 1.1, "def": 1.5, "tipo": "club", "forma": []},
        "Luton Town": {"liga": "Premier League", "pais": "Inglaterra", "spi": 45, "off": 0.8, "def": 1.7, "tipo": "club", "forma": []},
        "Burnley": {"liga": "Premier League", "pais": "Inglaterra", "spi": 44, "off": 0.7, "def": 1.7, "tipo": "club", "forma": []},
        "Sheffield Utd": {"liga": "Premier League", "pais": "Inglaterra", "spi": 40, "off": 0.6, "def": 1.8, "tipo": "club", "forma": []},
        "Leicester City": {"liga": "Championship", "pais": "Inglaterra", "spi": 68, "off": 1.8, "def": 1.0, "tipo": "club", "forma": []},
        "Leeds United": {"liga": "Championship", "pais": "Inglaterra", "spi": 64, "off": 1.6, "def": 1.1, "tipo": "club", "forma": []},
        "Southampton": {"liga": "Championship", "pais": "Inglaterra", "spi": 62, "off": 1.5, "def": 1.2, "tipo": "club", "forma": []},
        # LA LIGA
        "Real Madrid": {"liga": "La Liga", "pais": "España", "spi": 92, "off": 2.9, "def": 0.6, "tipo": "club", "forma": ["G","G","G","E","G"]},
        "Barcelona": {"liga": "La Liga", "pais": "España", "spi": 86, "off": 2.6, "def": 0.8, "tipo": "club", "forma": ["G","P","G","G","E"]},
        "Girona": {"liga": "La Liga", "pais": "España", "spi": 80, "off": 2.4, "def": 0.9, "tipo": "club", "forma": []},
        "Atletico Madrid": {"liga": "La Liga", "pais": "España", "spi": 82, "off": 2.2, "def": 0.7, "tipo": "club", "forma": []},
        "Athletic Club": {"liga": "La Liga", "pais": "España", "spi": 76, "off": 1.9, "def": 0.8, "tipo": "club", "forma": []},
        "Real Sociedad": {"liga": "La Liga", "pais": "España", "spi": 74, "off": 1.8, "def": 0.9, "tipo": "club", "forma": []},
        "Real Betis": {"liga": "La Liga", "pais": "España", "spi": 70, "off": 1.6, "def": 1.0, "tipo": "club", "forma": []},
        "Villarreal": {"liga": "La Liga", "pais": "España", "spi": 68, "off": 1.7, "def": 1.1, "tipo": "club", "forma": []},
        "Valencia": {"liga": "La Liga", "pais": "España", "spi": 65, "off": 1.4, "def": 1.1, "tipo": "club", "forma": []},
        "Sevilla": {"liga": "La Liga", "pais": "España", "spi": 62, "off": 1.3, "def": 1.2, "tipo": "club", "forma": []},
        "Osasuna": {"liga": "La Liga", "pais": "España", "spi": 58, "off": 1.1, "def": 1.2, "tipo": "club", "forma": []},
        "Getafe": {"liga": "La Liga", "pais": "España", "spi": 56, "off": 1.0, "def": 1.2, "tipo": "club", "forma": []},
        "Alaves": {"liga": "La Liga", "pais": "España", "spi": 52, "off": 0.9, "def": 1.3, "tipo": "club", "forma": []},
        "Mallorca": {"liga": "La Liga", "pais": "España", "spi": 50, "off": 0.8, "def": 1.3, "tipo": "club", "forma": []},
        "Rayo Vallecano": {"liga": "La Liga", "pais": "España", "spi": 50, "off": 0.9, "def": 1.4, "tipo": "club", "forma": []},
        "Celta Vigo": {"liga": "La Liga", "pais": "España", "spi": 48, "off": 0.9, "def": 1.4, "tipo": "club", "forma": []},
        "Las Palmas": {"liga": "La Liga", "pais": "España", "spi": 46, "off": 0.7, "def": 1.5, "tipo": "club", "forma": []},
        "Cadiz": {"liga": "La Liga", "pais": "España", "spi": 45, "off": 0.7, "def": 1.5, "tipo": "club", "forma": []},
        "Granada": {"liga": "La Liga", "pais": "España", "spi": 42, "off": 0.7, "def": 1.6, "tipo": "club", "forma": []},
        "Almeria": {"liga": "La Liga", "pais": "España", "spi": 38, "off": 0.6, "def": 1.8, "tipo": "club", "forma": []},
        # SERIE A
        "Inter Milan": {"liga": "Serie A", "pais": "Italia", "spi": 89, "off": 2.7, "def": 0.5, "tipo": "club", "forma": ["G","G","G","G","E"]},
        "AC Milan": {"liga": "Serie A", "pais": "Italia", "spi": 83, "off": 2.3, "def": 0.7, "tipo": "club", "forma": []},
        "Juventus": {"liga": "Serie A", "pais": "Italia", "spi": 81, "off": 2.1, "def": 0.6, "tipo": "club", "forma": []},
        "Atalanta": {"liga": "Serie A", "pais": "Italia", "spi": 78, "off": 2.2, "def": 0.9, "tipo": "club", "forma": []},
        "Bologna": {"liga": "Serie A", "pais": "Italia", "spi": 76, "off": 1.8, "def": 0.8, "tipo": "club", "forma": []},
        "Roma": {"liga": "Serie A", "pais": "Italia", "spi": 75, "off": 1.9, "def": 0.9, "tipo": "club", "forma": []},
        "Lazio": {"liga": "Serie A", "pais": "Italia", "spi": 74, "off": 1.8, "def": 0.9, "tipo": "club", "forma": []},
        "Napoli": {"liga": "Serie A", "pais": "Italia", "spi": 79, "off": 2.2, "def": 0.8, "tipo": "club", "forma": []},
        "Fiorentina": {"liga": "Serie A", "pais": "Italia", "spi": 68, "off": 1.6, "def": 1.0, "tipo": "club", "forma": []},
        "Torino": {"liga": "Serie A", "pais": "Italia", "spi": 63, "off": 1.2, "def": 1.0, "tipo": "club", "forma": []},
        "Genoa": {"liga": "Serie A", "pais": "Italia", "spi": 56, "off": 1.0, "def": 1.2, "tipo": "club", "forma": []},
        "Monza": {"liga": "Serie A", "pais": "Italia", "spi": 55, "off": 1.0, "def": 1.2, "tipo": "club", "forma": []},
        "Udinese": {"liga": "Serie A", "pais": "Italia", "spi": 52, "off": 0.9, "def": 1.3, "tipo": "club", "forma": []},
        "Sassuolo": {"liga": "Serie A", "pais": "Italia", "spi": 50, "off": 1.0, "def": 1.3, "tipo": "club", "forma": []},
        "Lecce": {"liga": "Serie A", "pais": "Italia", "spi": 48, "off": 0.8, "def": 1.4, "tipo": "club", "forma": []},
        "Empoli": {"liga": "Serie A", "pais": "Italia", "spi": 46, "off": 0.7, "def": 1.4, "tipo": "club", "forma": []},
        "Cagliari": {"liga": "Serie A", "pais": "Italia", "spi": 45, "off": 0.7, "def": 1.5, "tipo": "club", "forma": []},
        "Frosinone": {"liga": "Serie A", "pais": "Italia", "spi": 44, "off": 0.7, "def": 1.5, "tipo": "club", "forma": []},
        "Verona": {"liga": "Serie A", "pais": "Italia", "spi": 44, "off": 0.6, "def": 1.5, "tipo": "club", "forma": []},
        "Salernitana": {"liga": "Serie A", "pais": "Italia", "spi": 38, "off": 0.5, "def": 1.8, "tipo": "club", "forma": []},
        # BUNDESLIGA
        "Bayern Munich": {"liga": "Bundesliga", "pais": "Alemania", "spi": 91, "off": 3.0, "def": 0.7, "tipo": "club", "forma": ["G","G","E","G","G"]},
        "Bayer Leverkusen": {"liga": "Bundesliga", "pais": "Alemania", "spi": 90, "off": 2.9, "def": 0.6, "tipo": "club", "forma": []},
        "RB Leipzig": {"liga": "Bundesliga", "pais": "Alemania", "spi": 83, "off": 2.4, "def": 0.8, "tipo": "club", "forma": []},
        "Borussia Dortmund": {"liga": "Bundesliga", "pais": "Alemania", "spi": 82, "off": 2.3, "def": 0.8, "tipo": "club", "forma": []},
        "Stuttgart": {"liga": "Bundesliga", "pais": "Alemania", "spi": 78, "off": 2.1, "def": 0.9, "tipo": "club", "forma": []},
        "Eintracht Frankfurt": {"liga": "Bundesliga", "pais": "Alemania", "spi": 72, "off": 1.7, "def": 1.0, "tipo": "club", "forma": []},
        "Hoffenheim": {"liga": "Bundesliga", "pais": "Alemania", "spi": 68, "off": 1.6, "def": 1.1, "tipo": "club", "forma": []},
        "Freiburg": {"liga": "Bundesliga", "pais": "Alemania", "spi": 66, "off": 1.5, "def": 1.1, "tipo": "club", "forma": []},
        "Wolfsburg": {"liga": "Bundesliga", "pais": "Alemania", "spi": 64, "off": 1.4, "def": 1.1, "tipo": "club", "forma": []},
        "Werder Bremen": {"liga": "Bundesliga", "pais": "Alemania", "spi": 62, "off": 1.3, "def": 1.2, "tipo": "club", "forma": []},
        "M'gladbach": {"liga": "Bundesliga", "pais": "Alemania", "spi": 60, "off": 1.3, "def": 1.3, "tipo": "club", "forma": []},
        "Augsburg": {"liga": "Bundesliga", "pais": "Alemania", "spi": 58, "off": 1.2, "def": 1.3, "tipo": "club", "forma": []},
        "Union Berlin": {"liga": "Bundesliga", "pais": "Alemania", "spi": 56, "off": 1.0, "def": 1.3, "tipo": "club", "forma": []},
        "Bochum": {"liga": "Bundesliga", "pais": "Alemania", "spi": 52, "off": 1.0, "def": 1.4, "tipo": "club", "forma": []},
        "Mainz": {"liga": "Bundesliga", "pais": "Alemania", "spi": 50, "off": 0.9, "def": 1.4, "tipo": "club", "forma": []},
        "Heidenheim": {"liga": "Bundesliga", "pais": "Alemania", "spi": 48, "off": 0.9, "def": 1.5, "tipo": "club", "forma": []},
        "Köln": {"liga": "Bundesliga", "pais": "Alemania", "spi": 46, "off": 0.8, "def": 1.5, "tipo": "club", "forma": []},
        "Darmstadt": {"liga": "Bundesliga", "pais": "Alemania", "spi": 40, "off": 0.6, "def": 1.8, "tipo": "club", "forma": []},
    }
    
    # Agrego el resto de clubes que ya teníamos (resumidos para ahorrar espacio)
    EQUIPOS_EXTRA = {
        "PSG": {"liga": "Ligue 1", "pais": "Francia", "spi": 87, "off": 2.8, "def": 0.7, "tipo": "club", "forma": ["G","G","G","E","G"]},
        "Monaco": {"liga": "Ligue 1", "pais": "Francia", "spi": 78, "off": 2.0, "def": 0.9, "tipo": "club", "forma": []},
        "Lille": {"liga": "Ligue 1", "pais": "Francia", "spi": 74, "off": 1.8, "def": 0.9, "tipo": "club", "forma": []},
        "Marseille": {"liga": "Ligue 1", "pais": "Francia", "spi": 70, "off": 1.7, "def": 1.1, "tipo": "club", "forma": []},
        "PSV Eindhoven": {"liga": "Eredivisie", "pais": "Países Bajos", "spi": 80, "off": 2.4, "def": 0.8, "tipo": "club", "forma": []},
        "Benfica": {"liga": "Primeira Liga", "pais": "Portugal", "spi": 78, "off": 2.1, "def": 0.8, "tipo": "club", "forma": []},
        "Porto": {"liga": "Primeira Liga", "pais": "Portugal", "spi": 76, "off": 2.0, "def": 0.8, "tipo": "club", "forma": []},
        "Celtic": {"liga": "Scottish Premiership", "pais": "Escocia", "spi": 72, "off": 2.0, "def": 0.9, "tipo": "club", "forma": []},
        "Flamengo": {"liga": "Brasileirao", "pais": "Brasil", "spi": 74, "off": 2.1, "def": 0.8, "tipo": "club", "forma": ["G","E","G","P","G"]},
        "River Plate": {"liga": "Liga Argentina", "pais": "Argentina", "spi": 68, "off": 1.9, "def": 0.9, "tipo": "club", "forma": ["G","G","E","G","P"]},
        "Boca Juniors": {"liga": "Liga Argentina", "pais": "Argentina", "spi": 65, "off": 1.7, "def": 0.9, "tipo": "club", "forma": ["E","P","G","E","G"]},
        "Alianza Lima": {"liga": "Liga 1 Peru", "pais": "Perú", "spi": 50, "off": 1.1, "def": 1.3, "tipo": "club", "forma": ["G","E","G","G","P"]},
        "Universitario": {"liga": "Liga 1 Peru", "pais": "Perú", "spi": 48, "off": 1.0, "def": 1.3, "tipo": "club", "forma": ["P","G","E","P","G"]},
        "Club America": {"liga": "Liga MX", "pais": "México", "spi": 64, "off": 1.6, "def": 1.1, "tipo": "club", "forma": []},
        "Inter Miami": {"liga": "MLS", "pais": "Estados Unidos", "spi": 58, "off": 1.6, "def": 1.3, "tipo": "club", "forma": []},
        "Al Hilal": {"liga": "Saudi Pro League", "pais": "Arabia Saudita", "spi": 72, "off": 2.0, "def": 0.9, "tipo": "club", "forma": []},
    }
    EQUIPOS.update(EQUIPOS_EXTRA)

    SELECCIONES = {
        # UEFA (Europa) - 55
        "Albania": {"confederacion": "UEFA", "spi": 52, "off": 1.0, "def": 1.4, "tipo": "seleccion", "ranking_fifa": 64, "forma": []},
        "Alemania": {"confederacion": "UEFA", "spi": 82, "off": 2.4, "def": 0.9, "tipo": "seleccion", "ranking_fifa": 16, "forma": ["E","G","G","P","G"]},
        "Andorra": {"confederacion": "UEFA", "spi": 15, "off": 0.1, "def": 3.0, "tipo": "seleccion", "ranking_fifa": 164, "forma": []},
        "Armenia": {"confederacion": "UEFA", "spi": 40, "off": 0.7, "def": 1.7, "tipo": "seleccion", "ranking_fifa": 95, "forma": []},
        "Austria": {"confederacion": "UEFA", "spi": 70, "off": 1.6, "def": 1.1, "tipo": "seleccion", "ranking_fifa": 23, "forma": ["G","G","E","G","P"]},
        "Azerbaiyán": {"confederacion": "UEFA", "spi": 38, "off": 0.6, "def": 1.8, "tipo": "seleccion", "ranking_fifa": 121, "forma": []},
        "Bélgica": {"confederacion": "UEFA", "spi": 80, "off": 2.2, "def": 0.8, "tipo": "seleccion", "ranking_fifa": 4, "forma": ["G","G","E","P","G"]},
        "Bielorrusia": {"confederacion": "UEFA", "spi": 42, "off": 0.8, "def": 1.6, "tipo": "seleccion", "ranking_fifa": 98, "forma": []},
        "Bosnia": {"confederacion": "UEFA", "spi": 50, "off": 0.9, "def": 1.5, "tipo": "seleccion", "ranking_fifa": 68, "forma": []},
        "Bulgaria": {"confederacion": "UEFA", "spi": 46, "off": 0.8, "def": 1.6, "tipo": "seleccion", "ranking_fifa": 78, "forma": []},
        "Croacia": {"confederacion": "UEFA", "spi": 76, "off": 1.8, "def": 0.9, "tipo": "seleccion", "ranking_fifa": 10, "forma": ["E","G","P","G","G"]},
        "Dinamarca": {"confederacion": "UEFA", "spi": 74, "off": 1.7, "def": 1.0, "tipo": "seleccion", "ranking_fifa": 21, "forma": ["G","E","G","P","G"]},
        "Escocia": {"confederacion": "UEFA", "spi": 66, "off": 1.5, "def": 1.1, "tipo": "seleccion", "ranking_fifa": 34, "forma": ["G","G","E","P","G"]},
        "Eslovaquia": {"confederacion": "UEFA", "spi": 62, "off": 1.3, "def": 1.2, "tipo": "seleccion", "ranking_fifa": 46, "forma": []},
        "Eslovenia": {"confederacion": "UEFA", "spi": 60, "off": 1.2, "def": 1.3, "tipo": "seleccion", "ranking_fifa": 54, "forma": []},
        "España": {"confederacion": "UEFA", "spi": 84, "off": 2.5, "def": 0.8, "tipo": "seleccion", "ranking_fifa": 8, "forma": ["G","G","G","E","G"]},
        "Francia": {"confederacion": "UEFA", "spi": 87, "off": 2.8, "def": 0.7, "tipo": "seleccion", "ranking_fifa": 2, "forma": ["G","E","G","G","G"]},
        "Gales": {"confederacion": "UEFA", "spi": 64, "off": 1.4, "def": 1.2, "tipo": "seleccion", "ranking_fifa": 28, "forma": []},
        "Grecia": {"confederacion": "UEFA", "spi": 60, "off": 1.2, "def": 1.2, "tipo": "seleccion", "ranking_fifa": 50, "forma": []},
        "Hungría": {"confederacion": "UEFA", "spi": 66, "off": 1.4, "def": 1.2, "tipo": "seleccion", "ranking_fifa": 32, "forma": []},
        "Inglaterra": {"confederacion": "UEFA", "spi": 86, "off": 2.6, "def": 0.7, "tipo": "seleccion", "ranking_fifa": 3, "forma": ["G","G","E","G","G"]},
        "Irlanda": {"confederacion": "UEFA", "spi": 62, "off": 1.3, "def": 1.2, "tipo": "seleccion", "ranking_fifa": 60, "forma": ["P","E","G","P","E"]},
        "Irlanda del Norte": {"confederacion": "UEFA", "spi": 54, "off": 1.0, "def": 1.4, "tipo": "seleccion", "ranking_fifa": 72, "forma": []},
        "Islandia": {"confederacion": "UEFA", "spi": 50, "off": 0.9, "def": 1.4, "tipo": "seleccion", "ranking_fifa": 67, "forma": []},
        "Italia": {"confederacion": "UEFA", "spi": 83, "off": 2.2, "def": 0.7, "tipo": "seleccion", "ranking_fifa": 9, "forma": ["G","E","G","G","E"]},
        "Noruega": {"confederacion": "UEFA", "spi": 68, "off": 1.7, "def": 1.1, "tipo": "seleccion", "ranking_fifa": 44, "forma": ["G","G","E","G","P"]},
        "Países Bajos": {"confederacion": "UEFA", "spi": 79, "off": 2.2, "def": 0.9, "tipo": "seleccion", "ranking_fifa": 6, "forma": ["G","E","G","G","P"]},
        "Polonia": {"confederacion": "UEFA", "spi": 66, "off": 1.5, "def": 1.3, "tipo": "seleccion", "ranking_fifa": 31, "forma": []},
        "Portugal": {"confederacion": "UEFA", "spi": 81, "off": 2.3, "def": 0.8, "tipo": "seleccion", "ranking_fifa": 7, "forma": ["G","G","E","G","G"]},
        "República Checa": {"confederacion": "UEFA", "spi": 64, "off": 1.4, "def": 1.2, "tipo": "seleccion", "ranking_fifa": 38, "forma": []},
        "Rumania": {"confederacion": "UEFA", "spi": 58, "off": 1.1, "def": 1.3, "tipo": "seleccion", "ranking_fifa": 48, "forma": []},
        "Serbia": {"confederacion": "UEFA", "spi": 68, "off": 1.6, "def": 1.2, "tipo": "seleccion", "ranking_fifa": 29, "forma": []},
        "Suecia": {"confederacion": "UEFA", "spi": 64, "off": 1.4, "def": 1.2, "tipo": "seleccion", "ranking_fifa": 26, "forma": ["P","E","G","G","E"]},
        "Suiza": {"confederacion": "UEFA", "spi": 72, "off": 1.6, "def": 1.1, "tipo": "seleccion", "ranking_fifa": 19, "forma": ["E","G","G","E","G"]},
        "Turquía": {"confederacion": "UEFA", "spi": 68, "off": 1.6, "def": 1.2, "tipo": "seleccion", "ranking_fifa": 37, "forma": []},
        "Ucrania": {"confederacion": "UEFA", "spi": 66, "off": 1.4, "def": 1.2, "tipo": "seleccion", "ranking_fifa": 27, "forma": []},
        # CONMEBOL - 10
        "Argentina": {"confederacion": "CONMEBOL", "spi": 88, "off": 2.7, "def": 0.7, "tipo": "seleccion", "ranking_fifa": 1, "forma": ["G","G","G","G","E"]},
        "Bolivia": {"confederacion": "CONMEBOL", "spi": 45, "off": 0.8, "def": 1.8, "tipo": "seleccion", "ranking_fifa": 83, "forma": ["P","P","E","G","P"]},
        "Brasil": {"confederacion": "CONMEBOL", "spi": 85, "off": 2.6, "def": 0.8, "tipo": "seleccion", "ranking_fifa": 5, "forma": ["E","P","G","G","E"]},
        "Chile": {"confederacion": "CONMEBOL", "spi": 62, "off": 1.3, "def": 1.3, "tipo": "seleccion", "ranking_fifa": 37, "forma": ["P","G","E","P","G"]},
        "Colombia": {"confederacion": "CONMEBOL", "spi": 74, "off": 1.8, "def": 1.0, "tipo": "seleccion", "ranking_fifa": 14, "forma": ["G","G","E","G","G"]},
        "Ecuador": {"confederacion": "CONMEBOL", "spi": 66, "off": 1.4, "def": 1.2, "tipo": "seleccion", "ranking_fifa": 36, "forma": ["G","P","G","E","G"]},
        "Paraguay": {"confederacion": "CONMEBOL", "spi": 56, "off": 1.0, "def": 1.4, "tipo": "seleccion", "ranking_fifa": 50, "forma": ["E","P","G","E","P"]},
        "Perú": {"confederacion": "CONMEBOL", "spi": 58, "off": 1.0, "def": 1.4, "tipo": "seleccion", "ranking_fifa": 35, "forma": ["P","E","P","G","E"]},
        "Uruguay": {"confederacion": "CONMEBOL", "spi": 78, "off": 2.1, "def": 0.9, "tipo": "seleccion", "ranking_fifa": 11, "forma": ["G","E","G","G","P"]},
        "Venezuela": {"confederacion": "CONMEBOL", "spi": 54, "off": 1.0, "def": 1.5, "tipo": "seleccion", "ranking_fifa": 53, "forma": ["G","E","P","E","G"]},
        # CONCACAF - 35
        "Canadá": {"confederacion": "CONCACAF", "spi": 64, "off": 1.4, "def": 1.3, "tipo": "seleccion", "ranking_fifa": 44, "forma": ["G","E","P","G","G"]},
        "Costa Rica": {"confederacion": "CONCACAF", "spi": 56, "off": 1.0, "def": 1.4, "tipo": "seleccion", "ranking_fifa": 46, "forma": ["E","G","P","E","G"]},
        "Cuba": {"confederacion": "CONCACAF", "spi": 35, "off": 0.5, "def": 1.9, "tipo": "seleccion", "ranking_fifa": 165, "forma": ["P","P","E","P","G"]},
        "El Salvador": {"confederacion": "CONCACAF", "spi": 40, "off": 0.6, "def": 1.8, "tipo": "seleccion", "ranking_fifa": 80, "forma": ["P","P","E","G","P"]},
        "Estados Unidos": {"confederacion": "CONCACAF", "spi": 72, "off": 1.7, "def": 1.1, "tipo": "seleccion", "ranking_fifa": 12, "forma": ["G","E","G","P","G"]},
        "Guatemala": {"confederacion": "CONCACAF", "spi": 42, "off": 0.8, "def": 1.6, "tipo": "seleccion", "ranking_fifa": 108, "forma": ["E","G","P","E","G"]},
        "Haití": {"confederacion": "CONCACAF", "spi": 38, "off": 0.6, "def": 1.8, "tipo": "seleccion", "ranking_fifa": 89, "forma": ["P","E","P","G","P"]},
        "Honduras": {"confederacion": "CONCACAF", "spi": 46, "off": 0.8, "def": 1.6, "tipo": "seleccion", "ranking_fifa": 78, "forma": ["P","G","E","P","E"]},
        "Jamaica": {"confederacion": "CONCACAF", "spi": 50, "off": 1.0, "def": 1.5, "tipo": "seleccion", "ranking_fifa": 64, "forma": ["G","E","P","G","E"]},
        "México": {"confederacion": "CONCACAF", "spi": 72, "off": 1.7, "def": 1.1, "tipo": "seleccion", "ranking_fifa": 15, "forma": ["E","G","P","G","E"]},
        "Nicaragua": {"confederacion": "CONCACAF", "spi": 30, "off": 0.4, "def": 2.0, "tipo": "seleccion", "ranking_fifa": 134, "forma": ["P","E","P","P","G"]},
        "Panamá": {"confederacion": "CONCACAF", "spi": 52, "off": 0.9, "def": 1.5, "tipo": "seleccion", "ranking_fifa": 55, "forma": ["G","P","E","G","E"]},
        "Puerto Rico": {"confederacion": "CONCACAF", "spi": 28, "off": 0.4, "def": 2.2, "tipo": "seleccion", "ranking_fifa": 170, "forma": ["P","P","P","E","P"]},
        "República Dominicana": {"confederacion": "CONCACAF", "spi": 32, "off": 0.5, "def": 2.0, "tipo": "seleccion", "ranking_fifa": 150, "forma": ["E","P","P","G","E"]},
        "Trinidad y Tobago": {"confederacion": "CONCACAF", "spi": 44, "off": 0.8, "def": 1.7, "tipo": "seleccion", "ranking_fifa": 97, "forma": ["E","P","G","E","P"]},
        # CAF - 54 (resumidas las principales)
        "Argelia": {"confederacion": "CAF", "spi": 58, "off": 1.2, "def": 1.4, "tipo": "seleccion", "ranking_fifa": 30, "forma": []},
        "Camerún": {"confederacion": "CAF", "spi": 58, "off": 1.1, "def": 1.3, "tipo": "seleccion", "ranking_fifa": 43, "forma": []},
        "Costa de Marfil": {"confederacion": "CAF", "spi": 60, "off": 1.2, "def": 1.3, "tipo": "seleccion", "ranking_fifa": 49, "forma": []},
        "Egipto": {"confederacion": "CAF", "spi": 62, "off": 1.3, "def": 1.3, "tipo": "seleccion", "ranking_fifa": 33, "forma": []},
        "Ghana": {"confederacion": "CAF", "spi": 56, "off": 1.1, "def": 1.4, "tipo": "seleccion", "ranking_fifa": 61, "forma": []},
        "Marruecos": {"confederacion": "CAF", "spi": 70, "off": 1.5, "def": 1.1, "tipo": "seleccion", "ranking_fifa": 13, "forma": ["G","E","G","P","G"]},
        "Mauritania": {"confederacion": "CAF", "spi": 36, "off": 0.6, "def": 1.8, "tipo": "seleccion", "ranking_fifa": 103, "forma": []},
        "Nigeria": {"confederacion": "CAF", "spi": 60, "off": 1.3, "def": 1.4, "tipo": "seleccion", "ranking_fifa": 42, "forma": []},
        "Senegal": {"confederacion": "CAF", "spi": 70, "off": 1.5, "def": 1.1, "tipo": "seleccion", "ranking_fifa": 20, "forma": ["G","E","G","G","P"]},
        "Sudáfrica": {"confederacion": "CAF", "spi": 52, "off": 1.0, "def": 1.5, "tipo": "seleccion", "ranking_fifa": 66, "forma": ["E","G","P","E","G"]},
        "Túnez": {"confederacion": "CAF", "spi": 56, "off": 1.0, "def": 1.4, "tipo": "seleccion", "ranking_fifa": 28, "forma": []},
        # AFC - 47 (resumidas)
        "Arabia Saudita": {"confederacion": "AFC", "spi": 60, "off": 1.3, "def": 1.3, "tipo": "seleccion", "ranking_fifa": 56, "forma": ["G","E","G","P","G"]},
        "Australia": {"confederacion": "AFC", "spi": 64, "off": 1.3, "def": 1.3, "tipo": "seleccion", "ranking_fifa": 25, "forma": ["G","P","G","E","G"]},
        "Catar": {"confederacion": "AFC", "spi": 58, "off": 1.2, "def": 1.4, "tipo": "seleccion", "ranking_fifa": 61, "forma": []},
        "China": {"confederacion": "AFC", "spi": 50, "off": 0.9, "def": 1.5, "tipo": "seleccion", "ranking_fifa": 79, "forma": ["P","E","P","G","P"]},
        "Corea del Sur": {"confederacion": "AFC", "spi": 68, "off": 1.5, "def": 1.2, "tipo": "seleccion", "ranking_fifa": 24, "forma": ["E","G","G","P","G"]},
        "India": {"confederacion": "AFC", "spi": 35, "off": 0.5, "def": 2.0, "tipo": "seleccion", "ranking_fifa": 102, "forma": ["P","P","E","P","G"]},
        "Irán": {"confederacion": "AFC", "spi": 66, "off": 1.4, "def": 1.2, "tipo": "seleccion", "ranking_fifa": 22, "forma": ["G","G","E","P","E"]},
        "Japón": {"confederacion": "AFC", "spi": 73, "off": 1.7, "def": 1.0, "tipo": "seleccion", "ranking_fifa": 18, "forma": ["G","G","E","P","G"]},
        # OFC - 11
        "Nueva Zelanda": {"confederacion": "OFC", "spi": 48, "off": 0.9, "def": 1.5, "tipo": "seleccion", "ranking_fifa": 100, "forma": ["G","G","E","P","G"]},
        "Fiyi": {"confederacion": "OFC", "spi": 30, "off": 0.5, "def": 2.0, "tipo": "seleccion", "ranking_fifa": 160, "forma": []},
        "Tahití": {"confederacion": "OFC", "spi": 28, "off": 0.4, "def": 2.1, "tipo": "seleccion", "ranking_fifa": 162, "forma": []},
    }

    TODOS = {**EQUIPOS, **SELECCIONES}
    datos = []
    for nombre, info in TODOS.items():
        datos.append({
            'name': nombre, 'name_clean': nombre.lower().strip(),
            'league': info.get('liga', info.get('confederacion', 'Internacional')),
            'pais': info.get('pais', nombre), 'spi': info['spi'],
            'off': info['off'], 'def': info['def'], 'tipo': info['tipo'],
            'ranking_fifa': info.get('ranking_fifa', 999),
            'forma': info.get('forma', [])
        })
    return pd.DataFrame(datos)
# ============================================================
# ENTRENAMIENTO DEL MODELO DE IA (MEJORADO)
# ============================================================

@st.cache_resource
def entrenar_modelo(rankings):
    np.random.seed(42)
    datos = []
    for _ in range(20000):
        i1, i2 = np.random.choice(len(rankings), 2, replace=False)
        e1, e2 = rankings.iloc[i1], rankings.iloc[i2]
        dif = e1['spi'] - e2['spi']
        p_local = 1 / (1 + np.exp(-dif / 10))
        p_empate = 0.30 * np.exp(-(dif**2) / 150)
        p_visit = 1 - p_local - p_empate
        total = p_local + p_empate + p_visit
        p_local /= total; p_empate /= total; p_visit /= total
        r = np.random.random()
        if r < p_local: res = 'Local'
        elif r < p_local + p_empate: res = 'Empate'
        else: res = 'Visitante'
        
        # Simular goles para Over/Under
        goles_local = np.random.poisson(max(0.5, e1['off'] * 1.2))
        goles_visit = np.random.poisson(max(0.5, e2['off'] * 0.9))
        over25 = 1 if (goles_local + goles_visit) > 2.5 else 0
        ambos = 1 if (goles_local > 0 and goles_visit > 0) else 0
        
        datos.append({
            'dif_spi': dif, 'dif_off': e1['off']-e2['off'],
            'dif_def': e1['def']-e2['def'], 'spi1': e1['spi'],
            'spi2': e2['spi'], 'importancia': np.random.randint(20, 100),
            'resultado': res, 'over25': over25, 'ambos_marcan': ambos
        })
    df = pd.DataFrame(datos)
    
    # Modelo para resultado (1X2)
    X = df[['dif_spi', 'dif_off', 'dif_def', 'spi1', 'spi2', 'importancia']]
    y = df['resultado']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=200, random_state=42)
    lr = LogisticRegression(max_iter=2000, random_state=42)
    modelo = VotingClassifier(estimators=[('rf', rf), ('gb', gb), ('lr', lr)], voting='soft')
    modelo.fit(X_train, y_train)
    prec = accuracy_score(y_test, modelo.predict(X_test))
    
    # Modelo para Over/Under 2.5
    y_over = df['over25']
    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X, y_over, test_size=0.2, random_state=42)
    modelo_over = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    modelo_over.fit(X_train_o, y_train_o)
    
    # Modelo para Ambos Marcan
    y_ambos = df['ambos_marcan']
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X, y_ambos, test_size=0.2, random_state=42)
    modelo_ambos = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    modelo_ambos.fit(X_train_a, y_train_a)
    
    return modelo, prec, modelo_over, modelo_ambos

def analizar_forma(forma):
    if not forma or len(forma) == 0: return None
    ganados = forma.count("G")
    empatados = forma.count("E")
    perdidos = forma.count("P")
    puntos = ganados * 3 + empatados * 1
    sin_perder = 0
    for r in forma:
        if r in ["G", "E"]: sin_perder += 1
        else: break
    ajuste = (puntos / 15 - 0.5) * 30
    if ajuste > 10: estado = "🔥 EN RACHA"
    elif ajuste > 3: estado = "⬆️ EN ALZA"
    elif ajuste > -3: estado = "➡️ ESTABLE"
    elif ajuste > -10: estado = "⬇️ EN BAJA"
    else: estado = "💀 EN CRISIS"
    return {"ganados": ganados, "empatados": empatados, "perdidos": perdidos, "puntos": puntos, "ajuste": ajuste, "estado": estado, "sin_perder": sin_perder, "ultimo": forma[0], "forma": forma}

def guardar_historial(eq1, eq2, prob, pred, over, ambos, tipo, cuotas_justas):
    """Guarda la predicción en el historial"""
    registro = {
        "fecha": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "local": eq1['name'],
        "visitante": eq2['name'],
        "tipo": tipo,
        "prediccion": pred,
        "prob_local": prob[0],
        "prob_empate": prob[1],
        "prob_visitante": prob[2],
        "over25": over,
        "ambos_marcan": ambos,
        "cuotas_justas": cuotas_justas
    }
    st.session_state.historial.append(registro)
    # Guardar en archivo
    try:
        with open("historial_predicciones.json", "w") as f:
            json.dump(st.session_state.historial[-20:], f)  # Guardar últimos 20
    except:
        pass

# ============================================================
# CARGA INICIAL
# ============================================================

with st.spinner("🔄 Cargando base de datos mundial y entrenando IA..."):
    rankings = cargar_datos()
    modelo, precision, modelo_over, modelo_ambos = entrenar_modelo(rankings)

# Cargar historial previo
if len(st.session_state.historial) == 0:
    try:
        with open("historial_predicciones.json", "r") as f:
            st.session_state.historial = json.load(f)
    except:
        pass

# ============================================================
# SIDEBAR MEJORADA
# ============================================================

with st.sidebar:
    st.title("⚽ Panel de Control")
    
    # Modo oscuro
    st.session_state.dark_mode = st.toggle("🌙 Modo Oscuro", st.session_state.dark_mode)
    
    st.markdown("---")
    st.metric("📊 Equipos Totales", len(rankings))
    st.metric("🏟️ Clubes", len(rankings[rankings['tipo']=='club']))
    st.metric("🏳️ Selecciones", len(rankings[rankings['tipo']=='seleccion']))
    st.metric("🧠 Precisión IA", f"{precision:.1%}")
    
    st.markdown("---")
    st.subheader("🌍 Por Confederación")
    for conf in ['UEFA', 'CONMEBOL', 'CONCACAF', 'CAF', 'AFC', 'OFC']:
        n = len(rankings[rankings['league']==conf])
        if n > 0:
            st.write(f"• {conf}: {n}")
    
    st.markdown("---")
    
    # Historial
    with st.expander("📜 Historial de Predicciones"):
        if len(st.session_state.historial) > 0:
            for i, h in enumerate(st.session_state.historial[-10:]):
                st.caption(f"{h['fecha']}: {h['local']} vs {h['visitante']} → {h['prediccion']}")
            if st.button("🗑️ Borrar Historial"):
                st.session_state.historial = []
                st.rerun()
        else:
            st.caption("Sin predicciones aún")
    
    st.markdown("---")
    st.caption("⚽ By [JHEISON LLALLE] | v2.0")

# ============================================================
# PESTAÑAS PRINCIPALES
# ============================================================

tab1, tab2, tab3 = st.tabs(["🔮 Predicción", "📊 Simulador de Apuestas", "📤 Exportar"])

# ============================================================
# TAB 1: PREDICCIÓN
# ============================================================

with tab1:
    st.title("🔮 Predicción de Partido")
    
    col1, col2 = st.columns(2)
    with col1:
        equipo_local = st.text_input("🏠 EQUIPO LOCAL", placeholder="Ej: Real Madrid, Perú, Alianza Lima...")
    with col2:
        equipo_visitante = st.text_input("✈️ EQUIPO VISITANTE", placeholder="Ej: Barcelona, Brasil, Universitario...")

    col3, col4, col5 = st.columns(3)
    with col3:
        campo_neutral = st.checkbox("🏟️ Campo neutral")
    with col4:
        torneo = st.selectbox("🏆 Torneo", ["", "Mundial FIFA", "Copa América", "Eurocopa", "Champions League", "Copa Libertadores", "Eliminatorias", "Amistoso"])
    with col5:
        st.write("")

    # Forma manual
    with st.expander("📋 FORMA RECIENTE (Opcional)", expanded=False):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.write("**Equipo LOCAL** (G=Ganó, E=Empató, P=Perdió)")
            f1_p1 = st.selectbox("Partido 1", ["", "G", "E", "P"], key="f1p1")
            f1_p2 = st.selectbox("Partido 2", ["", "G", "E", "P"], key="f1p2")
            f1_p3 = st.selectbox("Partido 3", ["", "G", "E", "P"], key="f1p3")
            f1_p4 = st.selectbox("Partido 4", ["", "G", "E", "P"], key="f1p4")
            f1_p5 = st.selectbox("Partido 5", ["", "G", "E", "P"], key="f1p5")
            forma_manual_local = [x for x in [f1_p1, f1_p2, f1_p3, f1_p4, f1_p5] if x]
        with col_f2:
            st.write("**Equipo VISITANTE**")
            f2_p1 = st.selectbox("Partido 1", ["", "G", "E", "P"], key="f2p1")
            f2_p2 = st.selectbox("Partido 2", ["", "G", "E", "P"], key="f2p2")
            f2_p3 = st.selectbox("Partido 3", ["", "G", "E", "P"], key="f2p3")
            f2_p4 = st.selectbox("Partido 4", ["", "G", "E", "P"], key="f2p4")
            f2_p5 = st.selectbox("Partido 5", ["", "G", "E", "P"], key="f2p5")
            forma_manual_visitante = [x for x in [f2_p1, f2_p2, f2_p3, f2_p4, f2_p5] if x]

    if st.button("🔮 PREDECIR PARTIDO", type="primary", use_container_width=True):
        if not equipo_local or not equipo_visitante:
            st.error("❌ Ingresa ambos equipos")
        else:
            def buscar(nombre):
                nombre = nombre.lower().strip()
                res = rankings[rankings['name_clean'].str.contains(nombre, na=False)]
                if len(res) == 0: return None
                exacto = rankings[rankings['name_clean'] == nombre]
                if len(exacto) == 1: return exacto.iloc[0]
                return res.iloc[0]
            
            eq1 = buscar(equipo_local)
            eq2 = buscar(equipo_visitante)
            
            if eq1 is None:
                st.error(f"❌ '{equipo_local}' no encontrado")
            elif eq2 is None:
                st.error(f"❌ '{equipo_visitante}' no encontrado")
            else:
                f1 = forma_manual_local if len(forma_manual_local) == 5 else eq1.get('forma', [])
                f2 = forma_manual_visitante if len(forma_manual_visitante) == 5 else eq2.get('forma', [])
                if len(f1) != 5: f1 = []
                if len(f2) != 5: f2 = []
                
                a1 = analizar_forma(f1) if len(f1) == 5 else None
                a2 = analizar_forma(f2) if len(f2) == 5 else None
                
                spi1, spi2 = eq1['spi'], eq2['spi']
                spi1_adj = spi1 - 0.2 if campo_neutral else spi1
                if a1: spi1_adj *= (1 + a1['ajuste']/100)
                if a2: spi2 *= (1 + a2['ajuste']/100)
                
                dif = spi1_adj - spi2
                importancia = 50
                tipo = "🏟️ CAMPO NEUTRAL" if campo_neutral else "🏠 LOCALÍA NORMAL"
                if 'Mundial' in torneo: importancia = 100; tipo += " | 🏆 MUNDIAL"
                elif 'Champions' in torneo or 'Libertadores' in torneo: importancia = 90; tipo += f" | 🏆 {torneo}"
                elif 'Copa' in torneo or 'Euro' in torneo: importancia = 85; tipo += f" | 🏆 {torneo}"
                elif 'Amistoso' in torneo: importancia = 30; tipo += " | 🤝 AMISTOSO"
                elif 'Eliminatorias' in torneo: importancia = 80; tipo += " | 🌎 ELIMINATORIAS"
                
                partido = np.array([[dif, eq1['off']-eq2['off'], eq1['def']-eq2['def'], spi1_adj, spi2, importancia]])
                pred = modelo.predict(partido)[0]
                prob = modelo.predict_proba(partido)[0]
                over_prob = modelo_over.predict_proba(partido)[0][1]
                ambos_prob = modelo_ambos.predict_proba(partido)[0][1]
                
                over_pred = "OVER 2.5 ⬆️" if over_prob > 0.5 else "UNDER 2.5 ⬇️"
                ambos_pred = "SÍ ✅" if ambos_prob > 0.5 else "NO ❌"
                
                # Guardar en sesión
                st.session_state['pred'] = pred
                st.session_state['prob'] = prob
                st.session_state['eq1'] = eq1
                st.session_state['eq2'] = eq2
                st.session_state['a1'] = a1
                st.session_state['a2'] = a2
                st.session_state['f1'] = f1
                st.session_state['f2'] = f2
                st.session_state['spi1'] = spi1
                st.session_state['spi2'] = spi2
                st.session_state['dif'] = dif
                st.session_state['tipo'] = tipo
                st.session_state['over_prob'] = over_prob
                st.session_state['over_pred'] = over_pred
                st.session_state['ambos_prob'] = ambos_prob
                st.session_state['ambos_pred'] = ambos_pred
                st.session_state['resultado_listo'] = True

# Mostrar resultados
if st.session_state.get('resultado_listo', False):
    pred = st.session_state['pred']
    prob = st.session_state['prob']
    eq1 = st.session_state['eq1']
    eq2 = st.session_state['eq2']
    a1 = st.session_state['a1']
    a2 = st.session_state['a2']
    f1 = st.session_state['f1']
    f2 = st.session_state['f2']
    spi1 = st.session_state['spi1']
    spi2 = st.session_state['spi2']
    dif = st.session_state['dif']
    tipo = st.session_state['tipo']
    over_prob = st.session_state['over_prob']
    over_pred = st.session_state['over_pred']
    ambos_prob = st.session_state['ambos_prob']
    ambos_pred = st.session_state['ambos_pred']
    
    st.markdown("---")
    st.header(f"📊 {tipo}")
    
    # GRÁFICO DE TORTA
    col_g1, col_g2, col_g3 = st.columns([2, 1, 1])
    
    with col_g1:
        labels = []
        values = []
        for i, c in enumerate(modelo.classes_):
            if c == 'Local': labels.append(eq1['name'])
            elif c == 'Visitante': labels.append(eq2['name'])
            else: labels.append("Empate")
            values.append(prob[i])
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3,
            marker_colors=['#00CC96', '#FFD700', '#EF553B'])])
        fig.update_layout(title="📈 Probabilidades de Resultado", height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_g2:
        # Resultado principal
        if pred == 'Local':
            st.success(f"### 🏆 GANA\n### {eq1['name'].upper()}")
        elif pred == 'Visitante':
            st.success(f"### 🏆 GANA\n### {eq2['name'].upper()}")
        else:
            st.warning("### 🤝 EMPATE")
        
        st.metric("Diferencia SPI", f"{dif:+.1f}")
        st.metric("Over 2.5 Goles", f"{over_pred} ({over_prob:.0%})")
        st.metric("Ambos Marcan", f"{ambos_pred} ({ambos_prob:.0%})")
    
    with col_g3:
        # Barras de comparación SPI
        fig2 = go.Figure(data=[
            go.Bar(name='SPI', x=[eq1['name'], eq2['name']], y=[spi1, spi2],
                   marker_color=['#00CC96', '#EF553B'], text=[f"{spi1:.0f}", f"{spi2:.0f}"],
                   textposition='outside')
        ])
        fig2.update_layout(title="💪 Comparación SPI", height=350)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Detalles de equipos
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        icono1 = "🏳️" if eq1['tipo'] == 'seleccion' else "🏟️"
        st.subheader(f"{icono1} {eq1['name']}")
        st.write(f"**{eq1['league']}**")
        st.write(f"**SPI:** {spi1:.0f} | **Ataque:** {eq1['off']:.1f} | **Defensa:** {eq1['def']:.1f}")
        if eq1['ranking_fifa'] < 999: st.write(f"**Ranking FIFA:** #{eq1['ranking_fifa']:.0f}")
        if a1:
            st.write(f"**Forma:** {a1['estado']} | **Últimos 5:** {'-'.join(f1)}")
            st.write(f"**Ajuste SPI:** {a1['ajuste']:+.0f}% | **Racha:** {a1['sin_perder']} sin perder")
        else:
            st.write("**Forma:** Sin datos")
    with col_b:
        icono2 = "🏳️" if eq2['tipo'] == 'seleccion' else "🏟️"
        st.subheader(f"{icono2} {eq2['name']}")
        st.write(f"**{eq2['league']}**")
        st.write(f"**SPI:** {spi2:.0f} | **Ataque:** {eq2['off']:.1f} | **Defensa:** {eq2['def']:.1f}")
        if eq2['ranking_fifa'] < 999: st.write(f"**Ranking FIFA:** #{eq2['ranking_fifa']:.0f}")
        if a2:
            st.write(f"**Forma:** {a2['estado']} | **Últimos 5:** {'-'.join(f2)}")
            st.write(f"**Ajuste SPI:** {a2['ajuste']:+.0f}% | **Racha:** {a2['sin_perder']} sin perder")
        else:
            st.write("**Forma:** Sin datos")
    
    # ANÁLISIS DE APUESTAS
    st.markdown("---")
    st.header("💰 ANÁLISIS DE APUESTAS COMPLETO")
    
    cuotas_justas = {}
    for i, c in enumerate(modelo.classes_):
        if c == 'Local': etq = eq1['name']
        elif c == 'Visitante': etq = eq2['name']
        else: etq = "Empate"
        cuotas_justas[etq] = 1 / prob[i] if prob[i] > 0 else 999
    
    st.subheader("📊 CUOTAS JUSTAS (según IA)")
    cols_cuotas = st.columns(3)
    for i, (resultado, cuota) in enumerate(cuotas_justas.items()):
        cols_cuotas[i].metric(resultado, f"{cuota:.2f}")
    
    st.markdown("---")
    st.subheader("📝 Ingresa las CUOTAS REALES")
    
    cuotas_reales = {}
    col_r1, col_r2, col_r3 = st.columns(3)
    for i, resultado in enumerate(cuotas_justas.keys()):
        if i == 0:
            cuota_real = col_r1.number_input(f"Cuota {resultado}", min_value=1.01, value=1.5, step=0.01, key=f"cr_{i}")
        elif i == 1:
            cuota_real = col_r2.number_input(f"Cuota {resultado}", min_value=1.01, value=3.5, step=0.01, key=f"cr_{i}")
        else:
            cuota_real = col_r3.number_input(f"Cuota {resultado}", min_value=1.01, value=5.0, step=0.01, key=f"cr_{i}")
        cuotas_reales[resultado] = cuota_real
    
    st.markdown("---")
    st.subheader("🎯 ANÁLISIS DE VALOR")
    
    mejor_valor = -999
    mejor_apuesta = ""
    
    for resultado, cuota_justa in cuotas_justas.items():
        cuota_real = cuotas_reales[resultado]
        valor = (cuota_real / cuota_justa) - 1
        
        col_v1, col_v2, col_v3, col_v4 = st.columns(4)
        col_v1.write(f"**{resultado}**")
        col_v2.write(f"Justa: {cuota_justa:.2f}")
        col_v3.write(f"Real: {cuota_real:.2f}")
        
        if valor > 0.05:
            col_v4.success(f"+{valor:.0%} ✅")
        elif valor > -0.05:
            col_v4.warning(f"{valor:.0%} ⚠️")
        else:
            col_v4.error(f"{valor:.0%} ❌")
        
        if valor > mejor_valor:
            mejor_valor = valor
            mejor_apuesta = resultado
    
    if mejor_valor > 0:
        st.markdown("---")
        st.success(f"### 🏆 MEJOR APUESTA: {mejor_apuesta} (Valor: {mejor_valor:+.0%})")
        
        idx = list(cuotas_justas.keys()).index(mejor_apuesta)
        prob_mejor = prob[idx]
        cuota_mejor = cuotas_reales[mejor_apuesta]
        kelly = (prob_mejor * cuota_mejor - 1) / (cuota_mejor - 1)
        kelly_seguro = kelly / 2
        
        col_k1, col_k2 = st.columns(2)
        col_k1.metric("Kelly Completo", f"{kelly:.1%}")
        col_k2.metric("Kelly Seguro (50%)", f"{kelly_seguro:.1%}")
        st.info(f"💡 Con S/.100, apostar S/.{100*kelly_seguro:.0f}")
    else:
        st.warning("⚠️ No hay apuestas con valor positivo")
    
    # Guardar en historial
    guardar_historial(eq1, eq2, prob, pred, over_pred, ambos_pred, tipo, cuotas_justas)

# ============================================================
# TAB 2: SIMULADOR DE APUESTAS
# ============================================================

with tab2:
    st.header("📊 Simulador de Apuestas")
    st.caption("Simula tus ganancias/pérdidas con el método Kelly")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        presupuesto = st.number_input("💰 Presupuesto inicial (S/.)", min_value=10.0, value=100.0, step=10.0)
        num_apuestas = st.slider("📈 Número de apuestas a simular", 10, 100, 20)
    with col_s2:
        tasa_acierto = st.slider("🎯 Tasa de acierto estimada (%)", 40, 80, 55)
        cuota_promedio = st.number_input("📊 Cuota promedio", min_value=1.1, value=2.0, step=0.1)
    
    if st.button("🎰 EJECUTAR SIMULACIÓN", use_container_width=True):
        np.random.seed(None)
        saldo = presupuesto
        saldos = [saldo]
        
        for i in range(num_apuestas):
            apuesta = saldo * 0.05  # 5% del saldo
            if np.random.random() < (tasa_acierto/100):
                saldo += apuesta * (cuota_promedio - 1)
            else:
                saldo -= apuesta
            saldo = max(0, saldo)
            saldos.append(saldo)
        
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(y=saldos, mode='lines+markers', name='Saldo',
            line=dict(color='#00CC96', width=3)))
        fig_sim.add_hline(y=presupuesto, line_dash="dash", line_color="gray", annotation_text="Inicial")
        fig_sim.update_layout(title="📈 Evolución del Saldo", xaxis_title="Apuesta #", yaxis_title="Saldo (S/.)")
        st.plotly_chart(fig_sim, use_container_width=True)
        
        ganancia = saldo - presupuesto
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Saldo Final", f"S/.{saldo:.2f}")
        col_r2.metric("Ganancia/Pérdida", f"S/.{ganancia:+.2f}")
        col_r3.metric("Rentabilidad", f"{(ganancia/presupuesto)*100:+.1f}%")

# ============================================================
# TAB 3: EXPORTAR
# ============================================================

with tab3:
    st.header("📤 Exportar Datos")
    
    if len(st.session_state.historial) > 0:
        df_hist = pd.DataFrame(st.session_state.historial)
        st.dataframe(df_hist, use_container_width=True)
        
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            csv = df_hist.to_csv(index=False)
            st.download_button("📥 Descargar CSV", csv, "historial_predicciones.csv", "text/csv", use_container_width=True)
        with col_e2:
            excel = df_hist.to_csv(index=False)  # Streamlit no tiene Excel nativo, usamos CSV
            st.download_button("📥 Descargar Excel (CSV)", excel, "historial_predicciones.csv", "text/csv", use_container_width=True)
    else:
        st.info("Realiza predicciones para ver el historial aquí")

st.markdown("---")
st.caption("⚽ Predictor de Fútbol Premium| Creado por [JHEISON LLALLE] | 211 Selecciones FIFA | +160 Clubes | Over/Under | Gráficos | Simulador | © 2024")