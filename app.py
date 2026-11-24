"""Tasador de camisetas: deploy en Streamlit (82.20 Análisis Predictivo Avanzado, TP2).

Flujo: foto -> control de calidad -> modelo CV (SigLIP fine-tuneado, probabilidades calibradas)
       -> formulario pre-llenado (el usuario confirma o corrige) -> precio de referencia de MWS.

Correr local:   streamlit run app.py
Pesos: si no están en modelo/modelo_fp16.safetensors, se bajan del repo de Hugging Face
       indicado en los secretos (HF_REPO y, si el repo es privado, HF_TOKEN).
"""
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from tasador import cargar_modelo, calidad, predecir

AQUI = Path(__file__).parent
CARPETA_MODELO = AQUI / "modelo"
PESOS = CARPETA_MODELO / "modelo_fp16.safetensors"
CONDICIONES = ["Usada + firmada", "Usada", "Emitida + firmada", "Emitida", "Solo firmada"]
AYUDA_CONDICION = ("Usada: se usó en un partido. Emitida: se preparó para un jugador pero no se usó. "
                   "Firmada: tiene la firma del jugador.")

st.set_page_config(page_title="Tasador de camisetas", page_icon="👕", layout="centered")
torch.set_num_threads(2)


@st.cache_resource(show_spinner="Cargando el modelo (solo la primera vez, ~1 minuto)…")
def modelo():
    if not PESOS.exists():
        from huggingface_hub import hf_hub_download
        repo = st.secrets["HF_REPO"]
        token = st.secrets.get("HF_TOKEN")
        hf_hub_download(repo_id=repo, filename=PESOS.name, local_dir=CARPETA_MODELO, token=token)
    return cargar_modelo(CARPETA_MODELO)


@st.cache_data
def precios():
    return pd.read_csv(CARPETA_MODELO / "precios_referencia.csv")


def precio_referencia(club, temporada, condicion):
    """El grupo más específico con datos: club+temporada+condición, si no club+condición, si no club+temporada."""
    t = precios()
    for nivel, filtro in [("club+temporada+condicion", (t.club == club) & (t.temporada == temporada) & (t.condicion == condicion)),
                          ("club+condicion", (t.club == club) & (t.condicion == condicion)),
                          ("club+temporada", (t.club == club) & (t.temporada == temporada))]:
        fila = t[filtro & (t.nivel == nivel)]
        if len(fila):
            return fila.iloc[0]
    return None


m, cfg = modelo()
clubes = list(dict.fromkeys(c.split(" | ")[0] for c in cfg["clases"]))
temporadas = sorted({c.split(" | ")[1] for c in cfg["clases"]})

# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("Alcance")
    st.write(f"**Clubes:** {', '.join(clubes)}")
    st.write(f"**Temporadas:** {', '.join(temporadas)}")
    mt = cfg["metricas_test"]
    st.header("Cómo le fue en test")
    st.write(f"- Club correcto: **{mt['club']:.0%}**\n- Club y temporada (top-1): **{mt['top1']:.0%}**\n"
             f"- Respuesta correcta entre las 3 opciones: **{mt['top3']:.0%}**")
    st.caption("Fotos de usuarios que el modelo no vio al entrenar. TP2 de 82.20 Análisis Predictivo Avanzado (ITBA). "
               "Uso académico.")

st.title("👕 Tasador de camisetas")
st.write("Sacale una foto de frente a la camiseta. El modelo reconoce el club y la temporada y te pre-llena el "
         "formulario; vos confirmás o corregís y te mostramos un precio de referencia.")

# ---------------------------------------------------------------- 1. foto
tab_subir, tab_camara = st.tabs(["📁 Subir foto", "📷 Sacar foto"])
with tab_subir:
    archivo = st.file_uploader("Foto de la camiseta", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
with tab_camara:
    captura = st.camera_input("Foto de la camiseta", label_visibility="collapsed")
fuente = captura or archivo
if fuente is None:
    st.stop()

foto = Image.open(fuente)
st.image(foto, width=320)

# ---------------------------------------------------------------- 2. control de calidad
problemas = calidad(foto, cfg)
if problemas:
    for p in problemas:
        st.warning(p, icon="⚠️")
    if not st.checkbox("Usar esta foto igual"):
        st.stop()

# ---------------------------------------------------------------- 3. predicción
pred = predecir(m, cfg, foto)
st.subheader("Lo que ve el modelo")
for i, op in enumerate(pred["top3"], 1):
    st.progress(op["prob"], text=f"{i}. {op['club']} · {op['temporada']} — {op['prob']:.0%}")
if pred["club"] is None:
    st.info("El modelo no está seguro. Elegí el club y la temporada; las 3 opciones de arriba son las más probables.")
elif pred["temporada"] is None:
    st.info(f"El club parece **{pred['club']}** ({pred['confianza_club']:.0%}), pero la temporada no está clara: confirmala.")

# ---------------------------------------------------------------- 4. formulario
orden_clubes = sorted(clubes, key=lambda c: -pred["p_club"][c])
with st.form("formulario"):
    st.subheader("Datos de la camiseta")
    club = st.selectbox("Club", orden_clubes, index=orden_clubes.index(pred["club"]) if pred["club"] else None,
                        placeholder="Elegí el club")
    temporada = st.selectbox("Temporada", temporadas,
                             index=temporadas.index(pred["temporada"]) if pred["temporada"] else None,
                             placeholder="Elegí la temporada")
    condicion = st.selectbox("Condición", CONDICIONES, index=None, placeholder="¿Usada, emitida, firmada?",
                             help=AYUDA_CONDICION)
    c1, c2 = st.columns([3, 1])
    jugador = c1.text_input("Jugador (opcional)")
    dorsal = c2.text_input("Dorsal (opcional)")
    enviar = st.form_submit_button("Tasar", type="primary")

if not enviar:
    st.stop()
if not (club and temporada and condicion):
    st.error("Completá club, temporada y condición.")
    st.stop()

# ---------------------------------------------------------------- 5. precio de referencia
ref = precio_referencia(club, temporada, condicion)
st.subheader("Precio de referencia")
if ref is None:
    st.write("No hay suficientes subastas comparables en MWS para dar una referencia.")
else:
    st.metric("Mediana de subastas comparables en MWS", f"€ {ref.mediana:,.0f}".replace(",", "."))
    st.write(f"La mitad de esas camisetas se vendió entre **€ {ref.p25:,.0f} y € {ref.p75:,.0f}**".replace(",", ".")
             + f" (n = {ref.n} subastas, {ref.periodo}).")
    if ref.nivel != "club+temporada+condicion":
        st.caption(f"Pocas subastas de {club} {temporada} en esa condición: la referencia usa el grupo **{ref.nivel.replace('+', ' + ')}**.")
    st.caption("Es una referencia histórica por grupo, no la estimación del regresor: el precio real depende sobre todo "
               "del jugador y del partido" + (f" ({jugador})" if jugador else "") + ".")
modificado = (pred["club"] != club) or (pred["temporada"] != temporada)
st.caption("Corregiste la predicción del modelo." if modificado else "Usaste la predicción del modelo.")
