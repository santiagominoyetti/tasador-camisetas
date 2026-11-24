import torch, pandas as pd
ck = torch.load(r"C:\Users\santi\Documents\ITBA\PredictivoAvanzado\TP2\cv\work\salidas\tasador_cv_siglip_ft.pt", map_location="cpu", weights_only=False)
print({k: (type(v).__name__) for k, v in ck.items()})
print("backbone", ck["backbone"]); print("clases", ck["clases"]); print("mean", ck["mean"], "std", ck["std"])
print("umbral", ck["umbral"], "T", ck["temperatura"]); print("metricas", ck["metricas_test"])
print(list(ck["estado"].keys())[-8:])
cols = ["equipo","temporada","condicion","tipo_articulo","estado","metodo_venta","precio_eur","competicion","fin_subasta"]
df = pd.read_csv(r"C:\Users\santi\Documents\Camisetas\dataset\camisetas_mws_v3.1.csv", usecols=cols, low_memory=False)
pob = df[(df.estado=="vendida")&(df.tipo_articulo=="Camiseta")&(df.metodo_venta=="bidding")&(df.precio_eur>=20)]
print(len(pob))
print(pob.condicion.value_counts().head(12))
print(pob.equipo.value_counts().head(40))
print(pob.temporada.dropna().unique()[:30])
