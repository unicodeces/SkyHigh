import os
import subprocess
import sys
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(BASE, "weights")
os.makedirs(WEIGHTS, exist_ok=True)

subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision",
                 "--index-url", "https://download.pytorch.org/whl/cu121"])

subprocess.run([sys.executable, "-m", "pip", "install", "-r", os.path.join(BASE, "requirements.txt")])

MODELOS = {
    "RealESRGAN_x4plus_anime_6B.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}

for nome, url in MODELOS.items():
    destino = os.path.join(WEIGHTS, nome)
    if not os.path.exists(destino):
        urllib.request.urlretrieve(url, destino)

subprocess.run([sys.executable, os.path.join(BASE, "upscale.py")])