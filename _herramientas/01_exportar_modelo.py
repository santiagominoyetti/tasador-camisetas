"""Exporta el checkpoint del notebook a un formato liviano para el deploy.

Entrada : cv/work/salidas/tasador_cv_siglip_ft.pt  (372 MB, fp32, con metadatos)
Salida  : deploy/modelo/modelo_fp16.safetensors    (~186 MB, solo pesos en fp16)
          deploy/modelo/config.json                 (clases, temperatura, umbrales, preprocesamiento)

Además verifica que el modelo exportado predice lo mismo que el original sobre las fotos del dataset
y calibra el umbral de nitidez del control de calidad de la app.
"""
import json, time, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, timm
from PIL import Image, ImageOps, ImageFilter
from safetensors.torch import save_file, load_file

TP2 = Path(r"C:\Users\santi\Documents\ITBA\PredictivoAvanzado\TP2")
CK = TP2 / "cv/work/salidas/tasador_cv_siglip_ft.pt"
DATA = TP2 / "cv/work/fka_camisetas_v2"
OUT = TP2 / "deploy/modelo"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(TP2 / "deploy"))
from tasador import preparar, nitidez  # mismo preprocesamiento que usa la app

ck = torch.load(CK, map_location="cpu", weights_only=False)
clases = ck["clases"]

# 1) pesos en fp16 (la mitad de tamaño; en la app se vuelven a fp32 para correr en CPU)
estado16 = {k: v.half().contiguous() if v.is_floating_point() else v.contiguous() for k, v in ck["estado"].items()}
save_file(estado16, str(OUT / "modelo_fp16.safetensors"))

# 2) verificación: fp32 original vs fp16 exportado
def armar(estado):
    m = timm.create_model(ck["backbone"], pretrained=False, num_classes=len(clases))
    m.load_state_dict({k: v.float() for k, v in estado.items()}, strict=True)
    return m.eval().cuda()

m32, m16 = armar(ck["estado"]), armar(load_file(str(OUT / "modelo_fp16.safetensors")))
meta = pd.read_csv(DATA / "meta.csv")
print(meta.decision.value_counts().to_dict())
fotos = meta[meta.decision == meta.decision.value_counts().index[0]]
X = torch.stack([preparar(Image.open(DATA / a)) for a in fotos.archivo])
with torch.no_grad():
    p32 = torch.cat([m32(X[i:i+64].cuda()).softmax(1).cpu() for i in range(0, len(X), 64)])
    p16 = torch.cat([m16(X[i:i+64].cuda()).softmax(1).cpu() for i in range(0, len(X), 64)])
acuerdo = (p32.argmax(1) == p16.argmax(1)).float().mean().item()
print(f"fotos: {len(X)} | misma clase top-1: {acuerdo:.4f} | máx. diferencia de probabilidad: {(p32-p16).abs().max():.4f}")

# 3) tiempo en CPU (lo que va a tener Streamlit Cloud)
mc = armar(load_file(str(OUT / "modelo_fp16.safetensors"))).cpu(); torch.set_num_threads(2)
with torch.no_grad():
    mc(X[:1]); t = time.time(); [mc(X[i:i+1]) for i in range(5)]
print(f"CPU, 2 hilos: {(time.time()-t)/5*1000:.0f} ms por foto")

# 4) umbral de nitidez: fotos del dataset vs las mismas fotos degradadas
def degradar(im, tipo):
    im = ImageOps.exif_transpose(im).convert("RGB")
    if tipo == "blur": return im.filter(ImageFilter.GaussianBlur(radius=max(im.size) / 224 * 2.5))
    if tipo == "baja": w, h = im.size; return im.resize((48, int(48 * h / w)), Image.BILINEAR).resize((w, h), Image.BILINEAR)
muestra = fotos.sample(min(300, len(fotos)), random_state=0)
nit = {"original": [], "blur": [], "baja": []}
for a in muestra.archivo:
    im = Image.open(DATA / a)
    nit["original"].append(nitidez(im)); nit["blur"].append(nitidez(degradar(im, "blur"))); nit["baja"].append(nitidez(degradar(im, "baja")))
q = {k: np.percentile(v, [1, 5, 50, 95, 99]).round(1).tolist() for k, v in nit.items()}
print("nitidez p1/p5/p50/p95/p99:", json.dumps(q))
# punto medio (geométrico) entre el p99 de las fotos degradadas y el p1 de las fotos reales
umbral_nitidez = float(np.sqrt(max(np.percentile(nit["blur"], 99), np.percentile(nit["baja"], 99)) * np.percentile(nit["original"], 1)))
print(f"umbral {umbral_nitidez:.1f}: rechaza {np.mean(np.array(nit['original']) < umbral_nitidez):.1%} de las fotos reales, "
      f"detecta {np.mean(np.array(nit['blur']) < umbral_nitidez):.0%} de las movidas y {np.mean(np.array(nit['baja']) < umbral_nitidez):.0%} de las de baja resolución")

cfg = {"backbone": ck["backbone"], "clases": clases, "mean": list(ck["mean"]), "std": list(ck["std"]),
       "temperatura": float(ck["temperatura"]), "umbral_confianza": float(ck["umbral"]),
       "umbral_nitidez": round(umbral_nitidez, 1), "lado_minimo_px": 224,
       "dataset": ck["dataset"], "dataset_sha256": ck["sha256"],
       "metricas_test": {k: round(float(v), 4) for k, v in ck["metricas_test"].items()},
       "verificacion_fp16": {"fotos": len(X), "acuerdo_top1": round(acuerdo, 4)}}
(OUT / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
print("tamaño pesos MB:", round((OUT / "modelo_fp16.safetensors").stat().st_size / 1e6, 1))
print(json.dumps({k: cfg[k] for k in ["temperatura", "umbral_confianza", "umbral_nitidez"]}))
