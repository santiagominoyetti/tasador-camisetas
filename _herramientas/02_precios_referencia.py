"""Tabla de precios de referencia para la app: medianas de MWS por club, temporada y condición.

Solo se publican AGREGADOS (n, cuartiles) de grupos con al menos N_MIN subastas: ninguna fila del dataset
viaja a la app (GOBERNANZA_DATOS.md: no redistribuir los datos crudos).
Población: la recomendada en GOBERNANZA_DATOS.md §7 (vendida, Camiseta, bidding, precio >= 20 EUR).
"""
import json
from pathlib import Path
import pandas as pd

MWS = Path(r"C:\Users\santi\Documents\Camisetas\dataset\camisetas_mws_v3.1.csv")
OUT = Path(r"C:\Users\santi\Documents\ITBA\PredictivoAvanzado\TP2\deploy\modelo\precios_referencia.csv")
CFG = json.loads((OUT.parent / "config.json").read_text(encoding="utf-8"))
N_MIN = 5
# nombre en la clase del modelo (FKA) -> nombre en MWS
CLUB_MWS = {"Arsenal FC": "Arsenal", "Chelsea FC": "Chelsea", "Liverpool FC": "Liverpool",
            "Manchester City": "Manchester City", "Manchester United": "Manchester United",
            "Tottenham Hotspur": "Tottenham Hotspur"}
CONDICIONES = ["Usada + firmada", "Usada", "Emitida + firmada", "Emitida", "Solo firmada"]

cols = ["equipo", "temporada", "condicion", "tipo_articulo", "estado", "metodo_venta", "precio_eur", "fin_subasta"]
df = pd.read_csv(MWS, usecols=cols, low_memory=False)
pob = df[(df.estado == "vendida") & (df.tipo_articulo == "Camiseta") & (df.metodo_venta == "bidding") & (df.precio_eur >= 20)].copy()
inv = {v: k for k, v in CLUB_MWS.items()}
pob = pob[pob.equipo.isin(inv)]
pob["club"] = pob.equipo.map(inv)
pob["temporada"] = pob.temporada.astype(str).str.replace("/", "-", regex=False)
temporadas = sorted({c.split(" | ")[1] for c in CFG["clases"]})
pob = pob[pob.temporada.isin(temporadas) & pob.condicion.isin(CONDICIONES)]
print("subastas usadas:", len(pob)); print(pd.crosstab(pob.club, pob.temporada))

def resumen(g, nivel):
    r = g.precio_eur.agg(n="count", p25=lambda s: s.quantile(.25), mediana="median", p75=lambda s: s.quantile(.75)).reset_index()
    r["nivel"] = nivel
    return r

tabla = pd.concat([
    resumen(pob.groupby(["club", "temporada", "condicion"]), "club+temporada+condicion"),
    resumen(pob.groupby(["club", "condicion"]), "club+condicion"),
    resumen(pob.groupby(["club", "temporada"]), "club+temporada"),
], ignore_index=True)
tabla = tabla[tabla.n >= N_MIN]
for c in ["p25", "mediana", "p75"]:
    tabla[c] = tabla[c].round(0).astype(int)
tabla["periodo"] = f"{pob.fin_subasta.str[:7].min()} a {pob.fin_subasta.str[:7].max()}"
tabla.to_csv(OUT, index=False)
print(tabla[tabla.nivel == "club+temporada+condicion"].to_string(index=False))
print("filas:", len(tabla), "->", OUT)
