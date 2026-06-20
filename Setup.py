import os
import subprocess
import sys
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(BASE, "weights")
os.makedirs(WEIGHTS, exist_ok=True)

WINDOWS = os.name == "nt"
VENV = os.path.join(BASE, "venv")
PYTHON = sys.executable if WINDOWS else os.path.join(VENV, "bin", "python")


def instalar_vcredist():
    try:
        import winreg
        for chave in [
            r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
            r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
        ]:
            try:
                winreg.CloseKey(winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, chave))
                return
            except FileNotFoundError:
                continue
    except Exception:
        pass

    destino = os.path.join(BASE, "vc_redist.x64.exe")
    try:
        urllib.request.urlretrieve("https://aka.ms/vc14/vc_redist.x64.exe", destino)
        subprocess.run([destino, "/install", "/quiet", "/norestart"], check=True)
    except Exception:
        pass
    finally:
        if os.path.exists(destino):
            os.remove(destino)


def criar_venv():
    if not os.path.exists(VENV):
        subprocess.run([sys.executable, "-m", "venv", VENV], check=True)


def corrigir_basicsr():
    try:
        result = subprocess.run(
            [PYTHON, "-c", "import site; print('\\n'.join(site.getsitepackages()))"],
            capture_output=True, text=True
        )
        for sp in result.stdout.strip().splitlines():
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


if WINDOWS:
    instalar_vcredist()
else:
    criar_venv()

subprocess.run([PYTHON, "-m", "pip", "install", "torch", "torchvision",
                "--index-url", "https://download.pytorch.org/whl/cu121"])

subprocess.run([PYTHON, "-m", "pip", "install", "-r", os.path.join(BASE, "requirements.txt")])

corrigir_basicsr()

for nome, url in {
    "RealESRGAN_x4plus_anime_6B.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}.items():
    destino = os.path.join(WEIGHTS, nome)
    if not os.path.exists(destino):
        urllib.request.urlretrieve(url, destino)

subprocess.run([PYTHON, os.path.join(BASE, "upscale.py")])
