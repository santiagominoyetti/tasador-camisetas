# Tasador de camisetas — deploy

App de Streamlit del TP2 de 82.20 Análisis Predictivo Avanzado (ITBA).

**Qué hace:** foto de una camiseta → control de calidad de la foto → el modelo (SigLIP ViT-B/16 fine-tuneado)
reconoce club y temporada con probabilidades calibradas → formulario pre-llenado que el usuario confirma o corrige
→ precio de referencia (mediana de subastas comparables de MatchWornShirt).

**Alcance:** Arsenal, Chelsea, Liverpool, Manchester City, Manchester United y Tottenham; temporadas 2023-24 a 2025-26.

## Archivos

| Archivo | Qué es |
|---|---|
| `app.py` | Interfaz de Streamlit |
| `tasador.py` | Preprocesamiento, control de calidad y predicción (igual que la sección 7 del notebook) |
| `modelo/config.json` | Clases, temperatura (T = 0,54), umbrales y métricas de test |
| `modelo/precios_referencia.csv` | Solo agregados (n, cuartiles) de grupos con 5 subastas o más |
| `modelo/modelo_fp16.safetensors` | Pesos (186 MB). **No va a GitHub** (límite de 100 MB): se baja de Hugging Face |
| `requirements.txt` | Dependencias, con PyTorch solo CPU |
| `_herramientas/` | Scripts que generaron lo anterior y la prueba automática de la app |

## Correr en la PC

```bat
cd C:\Users\santi\Documents\ITBA\PredictivoAvanzado\TP2\deploy
C:\Users\santi\miniconda3\envs\tp2_cv\python.exe -m streamlit run app.py
```

## Subirlo a Streamlit Community Cloud

1. **Pesos a Hugging Face** (repo de modelo privado, gratis).
   Crear un token en https://huggingface.co/settings/tokens de tipo **Write** y correr:
   ```bat
   C:\Users\santi\miniconda3\envs\tp2_cv\python.exe _herramientas\04_subir_modelo_hf.py TU_USUARIO_HF/tasador-camisetas-cv
   ```
   Después crear un segundo token de tipo **Read** para la app.

2. **Código a GitHub.** Crear un repo vacío (por ejemplo `tasador-camisetas`) en https://github.com/new y desde esta carpeta:
   ```bat
   git init -b main
   git add .
   git commit -m "Tasador de camisetas: app de Streamlit"
   git remote add origin https://github.com/TU_USUARIO_GH/tasador-camisetas.git
   git push -u origin main
   ```
   El `.gitignore` deja afuera los pesos y la carpeta `_prueba` (tiene fotos de FKA, que no se redistribuyen).

3. **App en Streamlit.** En https://share.streamlit.io → *Create app* → repo `tasador-camisetas`, rama `main`, archivo `app.py`.
   En *Advanced settings*: Python **3.12** y en *Secrets* pegar:
   ```toml
   HF_REPO = "TU_USUARIO_HF/tasador-camisetas-cv"
   HF_TOKEN = "hf_...token Read..."
   ```
   *Deploy*. La primera vez tarda unos minutos (instala PyTorch y baja los 186 MB).

**Si falla la instalación:** mirar el log de *Manage app*. Si el problema es la versión de PyTorch,
sacar los `==` de `torch` y `torchvision` en `requirements.txt` y volver a hacer push.
La app se duerme después de 12 h sin visitas: abrirla un rato antes de presentar.

## Verificaciones hechas

- El modelo exportado en fp16 predice la misma clase que el checkpoint original en las 588 fotos del dataset (máxima diferencia de probabilidad 0,0014).
- En CPU con 2 hilos tarda ~0,3 s por foto.
- Control de nitidez: umbral 21 (varianza del laplaciano). Sobre 300 fotos del dataset no rechaza ninguna real
  y detecta el 100% de sus versiones movidas o de muy baja resolución.
- `_herramientas/03_probar_app.py` levanta la app y la recorre con un navegador headless (3 fotos de test, una movida y una chica).
