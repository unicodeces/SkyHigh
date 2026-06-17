# SkyHigh

SkyHigh is a command-line image upscaling tool powered by [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN). It increases image resolution and sharpness using GPU-accelerated AI super-resolution, with dedicated models for anime/illustration and realistic photos.

## Features

- AI-based super-resolution upscaling (4x) via Real-ESRGAN
- Dedicated models for **anime/illustration** and **realistic photos**
- Single image or full folder batch processing
- Native file picker on Windows, with automatic terminal fallback on Linux
- CUDA GPU acceleration with automatic CPU fallback
- Clean terminal interface with color-coded status output
- Supports PNG, JPG, JPEG, WEBP, BMP, TIFF, TIF and GIF

## Requirements

- Python 3.9 or higher
- NVIDIA GPU with CUDA support (recommended for best performance)
- Windows or Linux

## Installation

Clone the repository:

```bash
git clone https://github.com/unicodeces/SkyHigh.git
cd SkyHigh
```

Run the setup script. On Windows, it installs dependencies directly into your active Python environment. On Linux/macOS, it automatically creates a virtual environment (`venv/`) and installs everything inside it, avoiding conflicts with system-managed Python packages (PEP 668). Either way, it installs PyTorch with CUDA 12.1 support, installs all dependencies from `requirements.txt`, downloads the required Real-ESRGAN model weights, and launches the tool automatically:

```bash
python Setup.py
```

### Manual installation

If you prefer to install dependencies manually:

**Windows:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Linux/macOS (recommended: virtual environment):**

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> If your GPU uses a different CUDA version, adjust the index URL accordingly. See [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

On Linux, if you intend to use the graphical file picker, make sure `python3-tk` is installed:

```bash
sudo apt install python3-tk
```

If no graphical environment is detected, SkyHigh automatically falls back to terminal-based path input.

## Usage

If you ran `Setup.py`, the tool launches automatically after installation. To run it again later:

**Windows:**

```bash
python upscale.py
```

**Linux/macOS** (activate the virtual environment created by `Setup.py` first):

```bash
source venv/bin/activate
python upscale.py
```

You'll be presented with the following menu:

```
[1] Upscale Anime Photo
[2] Upscale Anime Folder
[3] Upscale Realistic Photo
[4] Upscale Realistic Folder

[0] Exit
```

| Option | Description |
|---|---|
| **1** | Upscales a single anime/illustration image |
| **2** | Batch upscales all supported images in a folder using the anime model |
| **3** | Upscales a single realistic photo |
| **4** | Batch upscales all supported images in a folder using the realistic model |

After selecting an option, you'll be prompted to choose the input file/folder and the output destination. Model weights are downloaded automatically on first use and cached locally in the `weights/` folder.

## How it works

SkyHigh loads the appropriate Real-ESRGAN architecture (`RealESRGAN_x4plus_anime_6B` for anime, `RealESRGAN_x4plus` for realistic photos) and runs inference on GPU when CUDA is available, falling back to CPU otherwise. Each processed image is upscaled 4x and saved to the chosen output folder with an `_upscaled` suffix.

## Project structure

```
SkyHigh/
├── Setup.py            # One-step installer and launcher
├── upscale.py          # Main application
├── requirements.txt    # Python dependencies
├── weights/            # Downloaded model weights (created automatically)
└── venv/                # Virtual environment on Linux/macOS (created automatically)
```

## Credits

Developed by **Daishinkan**

Powered by [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (Xintao Wang et al.)
