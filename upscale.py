import contextlib
import io
import os
import sys
import urllib.request

import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

try:
    from tkinter import Tk, filedialog
    TKINTER_DISPONIVEL = True
except ImportError:
    TKINTER_DISPONIVEL = False

try:
    from colorama import init as colorama_init
    colorama_init()
except ImportError:
    pass


EXTENSOES_SUPORTADAS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif")
DIRETORIO_BASE = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_MODELOS = os.path.join(DIRETORIO_BASE, "weights")

MODELOS = {
    "anime": {
        "nome": "RealESRGAN_x4plus_anime_6B",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "num_block": 6,
    },
    "realistic": {
        "nome": "RealESRGAN_x4plus",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "num_block": 23,
    },
}

COR_AZUL = (40, 30, 200)
COR_ROXO = (130, 20, 160)


def interpolar_cor(cor_inicial, cor_final, fator):
    r = int(cor_inicial[0] + (cor_final[0] - cor_inicial[0]) * fator)
    g = int(cor_inicial[1] + (cor_final[1] - cor_inicial[1]) * fator)
    b = int(cor_inicial[2] + (cor_final[2] - cor_inicial[2]) * fator)
    return r, g, b


def codigo_ansi_rgb(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"


def renderizar_banner():
    linhas = [
        r" $$$$$$\  $$\                 $$\   $$\ $$\           $$\       ",
        r"$$  __$$\ $$ |                $$ |  $$ |\__|          $$ |      ",
        r"$$ /  \__|$$ |  $$\ $$\   $$\ $$ |  $$ |$$\  $$$$$$\  $$$$$$$\  ",
        r"\$$$$$$\  $$ | $$  |$$ |  $$ |$$$$$$$$ |$$ |$$  __$$\ $$  __$$\ ",
        r" \____$$\ $$$$$$  / $$ |  $$ |$$  __$$ |$$ |$$ /  $$ |$$ |  $$ |",
        r"$$\   $$ |$$  _$$<  $$ |  $$ |$$ |  $$ |$$ |$$ |  $$ |$$ |  $$ |",
        r"\$$$$$$  |$$ | \$$\ \$$$$$$$ |$$ |  $$ |$$ |\$$$$$$$ |$$ |  $$ |",
        r" \______/ \__|  \__| \____$$ |\__|  \__|\__| \____$$ |\__|  \__|",
        r"                    $$\   $$ |              $$\   $$ |          ",
        r"                    \$$$$$$  |              \$$$$$$  |          ",
        r"                     \______/                \______/           ",
    ]

    total_linhas = len(linhas)
    reset = "\033[0m"

    for indice, linha in enumerate(linhas):
        fator = indice / max(total_linhas - 1, 1)
        cor = interpolar_cor(COR_AZUL, COR_ROXO, fator)
        print(f"{codigo_ansi_rgb(*cor)}{linha}{reset}")


def renderizar_menu():
    reset = "\033[0m"
    cinza = "\033[38;2;130;130;140m"
    branco = "\033[38;2;230;230;235m"
    destaque_anime = "\033[38;2;90;60;200m"
    destaque_realistic = "\033[38;2;30;170;180m"

    print()
    print(f"  {destaque_anime}[1]{reset} {branco}Upscale Anime Photo{reset}")
    print(f"  {destaque_anime}[2]{reset} {branco}Upscale Anime Folder{reset}")
    print(f"  {destaque_realistic}[3]{reset} {branco}Upscale Realistic Photo{reset}")
    print(f"  {destaque_realistic}[4]{reset} {branco}Upscale Realistic Folder{reset}")
    print()
    print(f"  {cinza}[0]{reset} {cinza}Exit{reset}")
    print()


def log_info(mensagem):
    print(f"\033[38;2;90;140;220m[*]\033[0m {mensagem}")


def log_sucesso(mensagem):
    print(f"\033[38;2;90;220;140m[+]\033[0m {mensagem}")


def log_erro(mensagem):
    print(f"\033[38;2;220;90;90m[-]\033[0m {mensagem}")


def log_aviso(mensagem):
    print(f"\033[38;2;220;180;90m[!]\033[0m {mensagem}")


def caminho_modelo(categoria):
    info = MODELOS[categoria]
    return os.path.join(DIRETORIO_MODELOS, f"{info['nome']}.pth")


def garantir_modelo(categoria):
    os.makedirs(DIRETORIO_MODELOS, exist_ok=True)
    destino = caminho_modelo(categoria)
    info = MODELOS[categoria]

    if not os.path.exists(destino):
        log_info(f"Downloading model {info['nome']}...")
        urllib.request.urlretrieve(info["url"], destino)
        log_sucesso("Model downloaded successfully.")
    else:
        log_info(f"Model {info['nome']} already available locally.")

    return destino


def construir_motor(categoria, tile=0, dispositivo="cuda"):
    info = MODELOS[categoria]
    modelo_path = garantir_modelo(categoria)

    arquitetura = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=info["num_block"],
        num_grow_ch=32,
        scale=4,
    )

    motor = RealESRGANer(
        scale=4,
        model_path=modelo_path,
        model=arquitetura,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=(dispositivo == "cuda"),
        device=dispositivo,
    )
    return motor


def processar_imagem(caminho_entrada, caminho_saida, motor, fator_escala=4.0):
    imagem = cv2.imread(caminho_entrada, cv2.IMREAD_UNCHANGED)
    if imagem is None:
        log_erro(f"Failed to read: {caminho_entrada}")
        return False

    nome = os.path.basename(caminho_entrada)
    log_info(f"Processing {nome} ({imagem.shape[1]}x{imagem.shape[0]})")

    def enhance_silencioso(img, escala):
        with contextlib.redirect_stdout(io.StringIO()):
            return motor.enhance(img, outscale=escala)

    try:
        resultado, _ = enhance_silencioso(imagem, fator_escala)
    except RuntimeError as erro:
        if "out of memory" in str(erro).lower():
            log_aviso("Insufficient GPU memory. Retrying with tile=256...")
            torch.cuda.empty_cache()
            motor.tile = 256
            try:
                resultado, _ = enhance_silencioso(imagem, fator_escala)
            except RuntimeError as erro2:
                if "out of memory" in str(erro2).lower():
                    log_aviso("Still out of memory. Retrying with tile=128...")
                    torch.cuda.empty_cache()
                    motor.tile = 128
                    try:
                        resultado, _ = enhance_silencioso(imagem, fator_escala)
                    except RuntimeError as erro3:
                        if "out of memory" in str(erro3).lower():
                            log_erro("Insufficient GPU memory even with minimum tile. Try a smaller scale.")
                            return False
                        raise
                else:
                    raise
        else:
            raise

    cv2.imwrite(caminho_saida, resultado)
    log_sucesso(f"Saved: {caminho_saida} ({resultado.shape[1]}x{resultado.shape[0]})")
    return True


def display_grafico_disponivel():
    if not TKINTER_DISPONIVEL:
        return False
    if os.name == "nt":
        return True
    if sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def abrir_seletor_arquivo():
    if not display_grafico_disponivel():
        return input("  Image path: ").strip().strip('"')

    janela = Tk()
    janela.withdraw()
    janela.attributes("-topmost", True)
    caminho = filedialog.askopenfilename(
        title="Select image",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.tif *.gif")],
    )
    janela.destroy()
    return caminho


def abrir_seletor_pasta(titulo):
    if not display_grafico_disponivel():
        return input(f"  {titulo}: ").strip().strip('"')

    janela = Tk()
    janela.withdraw()
    janela.attributes("-topmost", True)
    caminho = filedialog.askdirectory(title=titulo)
    janela.destroy()
    return caminho


def escolher_escala():
    reset = "\033[0m"
    branco = "\033[38;2;230;230;235m"
    destaque = "\033[38;2;90;60;200m"

    print()
    print(f"  {destaque}[1]{reset} {branco}2x{reset}")
    print(f"  {destaque}[2]{reset} {branco}4x{reset}")
    print()

    opcoes = {"1": 2.0, "2": 4.0}
    while True:
        escolha = input("  > ").strip()
        if escolha in opcoes:
            return opcoes[escolha]
        log_erro("Invalid option. Choose 1 or 2.")


def detectar_gpu():
    if not torch.cuda.is_available():
        log_aviso("No NVIDIA GPU detected. Using CPU.")
        return "cpu", 0

    nome = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024 ** 3)

    if vram_gb >= 10:
        tile = 0
    elif vram_gb >= 6:
        tile = 512
    elif vram_gb >= 4:
        tile = 256
    else:
        tile = 128

    tile_str = "no tile" if tile == 0 else f"tile {tile}"
    log_info(f"GPU: {nome} ({vram_gb:.1f} GB VRAM) {tile_str}")
    return "cuda", tile



def executar_fluxo_arquivo_unico(categoria):
    log_info("Waiting for image selection...")
    entrada = abrir_seletor_arquivo()
    if not entrada:
        log_aviso("Operation cancelled.")
        return

    log_info("Waiting for output folder selection...")
    pasta_saida = abrir_seletor_pasta("Select output folder")
    if not pasta_saida:
        log_aviso("Operation cancelled.")
        return

    escala = escolher_escala()
    dispositivo, tile = detectar_gpu()

    sufixo = f"_{escala}x" if escala != int(escala) else f"_{int(escala)}x"
    nome_base, _ = os.path.splitext(os.path.basename(entrada))
    saida = os.path.join(pasta_saida, f"{nome_base}_upscaled{sufixo}.png")

    motor = construir_motor(categoria, tile=tile, dispositivo=dispositivo)
    processar_imagem(entrada, saida, motor, fator_escala=escala)


def executar_fluxo_pasta(categoria):
    log_info("Waiting for input folder selection...")
    pasta_entrada = abrir_seletor_pasta("Select folder with images")
    if not pasta_entrada:
        log_aviso("Operation cancelled.")
        return

    arquivos = [f for f in os.listdir(pasta_entrada) if f.lower().endswith(EXTENSOES_SUPORTADAS)]
    if not arquivos:
        log_aviso("No images found in the selected folder.")
        return

    log_info(f"{len(arquivos)} image(s) found.")

    log_info("Waiting for output folder selection...")
    pasta_saida = abrir_seletor_pasta("Select output folder")
    if not pasta_saida:
        log_aviso("Operation cancelled.")
        return

    escala = escolher_escala()
    dispositivo, tile = detectar_gpu()

    motor = construir_motor(categoria, tile=tile, dispositivo=dispositivo)

    total = len(arquivos)
    sucessos = 0
    for indice, nome_arquivo in enumerate(arquivos, start=1):
        entrada = os.path.join(pasta_entrada, nome_arquivo)
        nome_base, _ = os.path.splitext(nome_arquivo)
        sufixo = f"_{escala}x" if escala != int(escala) else f"_{int(escala)}x"
        saida = os.path.join(pasta_saida, f"{nome_base}_upscaled{sufixo}.png")

        print(f"\033[38;2;130;130;140m[{indice}/{total}]\033[0m", end=" ")
        if processar_imagem(entrada, saida, motor, fator_escala=escala):
            sucessos += 1

    print()
    log_sucesso(f"Done: {sucessos}/{total} image(s) processed.")


def limpar_tela():
    os.system("cls" if os.name == "nt" else "clear")


def main():
    while True:
        limpar_tela()
        renderizar_banner()
        renderizar_menu()

        escolha = input("  > ").strip()

        if escolha == "1":
            print()
            executar_fluxo_arquivo_unico("anime")
        elif escolha == "2":
            print()
            executar_fluxo_pasta("anime")
        elif escolha == "3":
            print()
            executar_fluxo_arquivo_unico("realistic")
        elif escolha == "4":
            print()
            executar_fluxo_pasta("realistic")
        elif escolha == "0":
            sys.exit(0)
        else:
            log_erro("Invalid option.")

        print()
        input("Press ENTER to return to the menu...")


if __name__ == "__main__":
    main()
