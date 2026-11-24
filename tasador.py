"""Lógica del tasador, separada de la interfaz (la usan app.py y los scripts de verificación).

- preparar(): el mismo preprocesamiento que el notebook (sección 7, pre_llenar_formulario):
  la foto se centra en un lienzo blanco cuadrado y se lleva a 224x224.
- calidad(): control previo a la predicción (resolución, nitidez, luz).
- predecir(): probabilidades calibradas con temperature scaling y armado del formulario.
"""
import json
from pathlib import Path

import numpy as np
import torch
import timm
from PIL import Image, ImageOps
from safetensors.torch import load_file

LADO = 224
MEAN = STD = (0.5, 0.5, 0.5)  # normalización de SigLIP (se guarda también en config.json)


def cuadrar(im: Image.Image) -> Image.Image:
    """Corrige la rotación EXIF del celular y centra la foto en un lienzo blanco cuadrado."""
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    s = max(w, h)
    lienzo = Image.new("RGB", (s, s), (255, 255, 255))
    lienzo.paste(im, ((s - w) // 2, (s - h) // 2))
    return lienzo


def preparar(im: Image.Image) -> torch.Tensor:
    x = np.asarray(cuadrar(im).resize((LADO, LADO), Image.BICUBIC), dtype=np.float32) / 255.0
    x = (x - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()


def nitidez(im: Image.Image) -> float:
    """Varianza del laplaciano con el lado mayor a 2x la resolución del modelo: baja si la foto está movida
    o es de muy baja resolución. Se mide sobre la foto sin el lienzo blanco (su borde sumaría nitidez falsa)."""
    im = ImageOps.exif_transpose(im).convert("L")
    w, h = im.size
    k = 2 * LADO / max(w, h)
    g = np.asarray(im.resize((max(3, round(w * k)), max(3, round(h * k))), Image.BILINEAR), dtype=np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def calidad(im: Image.Image, cfg: dict) -> list[str]:
    """Devuelve la lista de problemas de la foto (vacía si está bien)."""
    problemas = []
    w, h = ImageOps.exif_transpose(im).size
    if min(w, h) < cfg["lado_minimo_px"]:
        problemas.append(f"La foto es muy chica ({w}×{h} px). Sacala de nuevo o subí una de al menos {cfg['lado_minimo_px']} px de lado.")
    if nitidez(im) < cfg["umbral_nitidez"]:
        problemas.append("La foto parece movida o desenfocada. Apoyá el celular o acercate un poco y volvé a sacarla.")
    brillo = float(np.asarray(ImageOps.exif_transpose(im).convert("L"), dtype=np.float32).mean() / 255)
    if brillo < 0.12:
        problemas.append("La foto está muy oscura. Probá con más luz.")
    elif brillo > 0.95:
        problemas.append("La foto está quemada (demasiada luz). Probá sin flash o lejos de la ventana.")
    return problemas


def cargar_modelo(carpeta: Path):
    carpeta = Path(carpeta)
    cfg = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
    m = timm.create_model(cfg["backbone"], pretrained=False, num_classes=len(cfg["clases"]))
    estado = {k: v.float() for k, v in load_file(str(carpeta / "modelo_fp16.safetensors")).items()}
    m.load_state_dict(estado, strict=True)
    return m.eval(), cfg


@torch.no_grad()
def predecir(m, cfg: dict, im: Image.Image) -> dict:
    clases = cfg["clases"]
    p = (m(preparar(im).unsqueeze(0)) / cfg["temperatura"]).softmax(1)[0].numpy()
    club_de = np.array([c.split(" | ")[0] for c in clases])
    p_club = {c: float(p[club_de == c].sum()) for c in dict.fromkeys(club_de)}
    club = max(p_club, key=p_club.get)
    # temporada: la más probable dentro del club elegido
    idx_club = [i for i, c in enumerate(club_de) if c == club]
    i_temp = max(idx_club, key=lambda i: p[i])
    u = cfg["umbral_confianza"]
    top = np.argsort(-p)[:3]
    return {
        "club": club if p_club[club] >= u else None,
        "confianza_club": p_club[club],
        "temporada": clases[i_temp].split(" | ")[1] if p[i_temp] >= u else None,
        "confianza_temporada": float(p[i_temp]),
        "top3": [{"club": clases[i].split(" | ")[0], "temporada": clases[i].split(" | ")[1], "prob": float(p[i])} for i in top],
        "p_club": p_club,
    }
