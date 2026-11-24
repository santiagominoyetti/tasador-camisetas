"""Sube los pesos del modelo a un repo de modelo PRIVADO de Hugging Face.

Uso (desde la carpeta deploy):
    python _herramientas/04_subir_modelo_hf.py TU_USUARIO/tasador-camisetas-cv

Pide el token de Hugging Face (tipo "Write") si no está en la variable de entorno HF_TOKEN.
"""
import os, sys, getpass
from pathlib import Path
from huggingface_hub import HfApi

repo = sys.argv[1] if len(sys.argv) > 1 else input("Repo (usuario/nombre): ").strip()
token = os.environ.get("HF_TOKEN") or getpass.getpass("Token de Hugging Face (Write, no se muestra): ").strip()
pesos = Path(__file__).resolve().parents[1] / "modelo" / "modelo_fp16.safetensors"

api = HfApi(token=token)
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
api.upload_file(path_or_fileobj=str(pesos), path_in_repo=pesos.name, repo_id=repo, repo_type="model",
                commit_message="Pesos fp16 del tasador (SigLIP fine-tuneado, 18 clases)")
print(f"Listo: https://huggingface.co/{repo} (privado)")
