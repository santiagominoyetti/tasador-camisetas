"""Prueba de punta a punta de la app con un navegador headless (Playwright).

1. Elige fotos de TEST (usuarios que el modelo no vio) y arma una versión movida y una muy chica.
2. Levanta `streamlit run app.py` en el puerto 8599.
3. Sube cada foto, completa el formulario, toca "Tasar" y guarda capturas en _prueba/.
"""
import subprocess, sys, time, json
from pathlib import Path
import pandas as pd
from PIL import Image, ImageFilter
from playwright.sync_api import sync_playwright

DEPLOY = Path(__file__).resolve().parents[1]
TP2 = DEPLOY.parent
PRU = DEPLOY / "_prueba"; PRU.mkdir(exist_ok=True)
DATA = TP2 / "cv/work/fka_camisetas_v2"

part = pd.read_csv(TP2 / "cv/work/salidas/particion.csv")
te = part[(part.split == "test") & (part.origen != "catalogo")].sample(3, random_state=7)
fotos = []
for i, r in enumerate(te.itertuples()):
    im = Image.open(DATA / r.archivo).convert("RGB"); p = PRU / f"test_{i}.jpg"; im.save(p, quality=92)
    fotos.append((p, r.clase, "ok"))
im = Image.open(fotos[0][0]); im.filter(ImageFilter.GaussianBlur(radius=max(im.size) / 224 * 3)).save(PRU / "movida.jpg")
im.resize((150, int(150 * im.size[1] / im.size[0]))).save(PRU / "chica.jpg")
fotos += [(PRU / "movida.jpg", fotos[0][1], "calidad"), (PRU / "chica.jpg", fotos[0][1], "calidad")]

srv = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(DEPLOY / "app.py"), "--server.port", "8599",
                        "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=DEPLOY,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
resultados = []
try:
    time.sleep(6)
    with sync_playwright() as p:
        b = p.chromium.launch()
        for k, (foto, clase, tipo) in enumerate(fotos):
            pg = b.new_page(viewport={"width": 1100, "height": 2300})
            t0 = time.time()
            pg.goto("http://localhost:8599", timeout=60000)
            pg.get_by_text("Tasador de camisetas").first.wait_for(timeout=180000)
            pg.locator("input[type=file]").set_input_files(str(foto))
            if tipo == "calidad":
                pg.get_by_text("Usar esta foto igual").wait_for(timeout=60000)
                avisos = pg.locator("[data-testid=stAlert]").all_inner_texts()
                pg.screenshot(path=str(PRU / f"{k}_{foto.stem}_aviso.png"), full_page=True)
                pg.get_by_text("Usar esta foto igual").click()
            else:
                avisos = []
            pg.get_by_text("Lo que ve el modelo").wait_for(timeout=120000)
            pg.locator("[data-testid=stProgress]").nth(2).wait_for(timeout=60000)
            top = pg.locator("[data-testid=stProgress]").all_inner_texts()
            # condición: el selectbox de Streamlit es un combobox
            pg.get_by_placeholder("¿Usada, emitida, firmada?").click()
            pg.get_by_role("option", name="Usada + firmada", exact=True).click()
            # si el modelo no pre-llenó club o temporada, el "usuario" elige la opción 1 del top-3
            club1, temp1 = top[0].split(". ", 1)[1].split(" — ")[0].split(" · ")
            sin_club = pg.get_by_text("El modelo no está seguro").count() > 0
            sin_temp = sin_club or pg.get_by_text("la temporada no está clara").count() > 0
            for falta, ph, valor in [(sin_club, "Elegí el club", club1), (sin_temp, "Elegí la temporada", temp1)]:
                if falta:
                    pg.get_by_placeholder(ph).click(); pg.get_by_role("option", name=valor, exact=True).click()
            pg.get_by_role("button", name="Tasar").click()
            pg.locator("[data-testid=stMetricValue]").first.wait_for(timeout=60000)
            precio = pg.locator("[data-testid=stMetricValue]").all_inner_texts()
            pg.screenshot(path=str(PRU / f"{k}_{foto.stem}_tasado.png"), full_page=True)
            form = [t.replace("\n", ": ") for t in pg.locator("[data-testid=stSelectbox]").all_inner_texts()]
            rango = pg.get_by_text("La mitad de esas camisetas").all_inner_texts()
            resultados.append({"foto": foto.name, "real": clase, "top3": top, "avisos": avisos, "formulario": form,
                               "precio": precio, "rango": rango,
                               "segundos": round(time.time() - t0, 1)})
            pg.close()
        b.close()
finally:
    srv.terminate()
    log = srv.stdout.read().decode("utf-8", "replace")[-3000:]
(PRU / "resultados.json").write_text(json.dumps(resultados, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(resultados, indent=1, ensure_ascii=True))
if "Traceback" in log or "Error" in log:
    print("LOG DEL SERVIDOR:\n", log)
