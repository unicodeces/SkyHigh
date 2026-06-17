import os
import subprocess
import sys
import urllib.request
import venv

BASE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(BASE, "weights")
VENV_DIR = os.path.join(BASE, "venv")
os.makedirs(WEIGHTS, exist_ok=True)

if os.name == "nt":
    PYTHON = sys.executable
else:
    if not os.path.exists(VENV_DIR):
        venv.create(VENV_DIR, with_pip=True)
    PYTHON = os.path.join(VENV_DIR, "bin", "python")

subprocess.run([PYTHON, "-m", "pip", "install", "--upgrade", "pip"])

subprocess.run([PYTHON, "-m", "pip", "install", "torch", "torchvision",
                 "--index-url", "https://download.pytorch.org/whl/cu121"])

subprocess.run([PYTHON, "-m", "pip", "install", "-r", os.path.join(BASE, "requirements.txt")])

MODELOS = {
    "RealESRGAN_x4plus_anime_6B.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}

for nome, url in MODELOS.items():
    destino = os.path.join(WEIGHTS, nome)
    if not os.path.exists(destino):
        urllib.request.urlretrieve(url, destino)

subprocess.run([PYTHON, os.path.join(BASE, "upscale.py")])
