import os
import subprocess
import sys
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(BASE, "weights")
os.makedirs(WEIGHTS, exist_ok=True)


def instalar_vcredist():
    if os.name != "nt":
        return

    try:
        import winreg
        chaves = [
            r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
            r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
        ]
        for chave in chaves:
            try:
                reg = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, chave)
                winreg.CloseKey(reg)
                return
            except FileNotFoundError:
                continue
    except Exception:
        pass

    url = "https://aka.ms/vc14/vc_redist.x64.exe"
    destino = os.path.join(BASE, "vc_redist.x64.exe")

    try:
        urllib.request.urlretrieve(url, destino)
        subprocess.run([destino, "/install", "/quiet", "/norestart"], check=True)
    except Exception:
        pass
    finally:
        if os.path.exists(destino):
            os.remove(destino)


def corrigir_basicsr():
    try:
        import site
        for sp in site.getsitepackages():
            p = os.path.join(sp, "basicsr", "data", "degradations.py")
            if os.path.exists(p):
                t = open(p, "r", encoding="utf-8").read()
                open(p, "w", encoding="utf-8").write(t.replace(
                    "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
                    "from torchvision.transforms.functional import rgb_to_grayscale"
                ))
                break
    except Exception:
        pass


instalar_vcredist()

subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision",
                 "--index-url", "https://download.pytorch.org/whl/cu121"])

subprocess.run([sys.executable, "-m", "pip", "install", "-r", os.path.join(BASE, "requirements.txt")])

corrigir_basicsr()

MODELOS = {
    "RealESRGAN_x4plus_anime_6B.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}

for nome, url in MODELOS.items():
    destino = os.path.join(WEIGHTS, nome)
    if not os.path.exists(destino):
        urllib.request.urlretrieve(url, destino)

subprocess.run([sys.executable, os.path.join(BASE, "upscale.py")])
