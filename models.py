"""All image-side models (PyTorch). Lazy-loaded so the app starts fast."""
import re
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from registry import REVERSE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)
WEIGHTS_URL = "https://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/{}.pth"
WATERMARK_TEXT = "Generated with CycleGAN Style Studio"


# ---------------- CycleGAN ResNet-9 generator (state-dict keys match the official repo) ----------------
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, bias=True), nn.InstanceNorm2d(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, bias=True), nn.InstanceNorm2d(dim))

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, ngf=64, n_blocks=9):
        super().__init__()
        m = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, 7, bias=True), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        for i in range(2):
            mult = 2 ** i
            m += [nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1, bias=True),
                  nn.InstanceNorm2d(ngf * mult * 2), nn.ReLU(True)]
        m += [ResnetBlock(ngf * 4) for _ in range(n_blocks)]          # indices 10..18
        for i in range(2):
            mult = 2 ** (2 - i)
            m += [nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3, stride=2, padding=1, output_padding=1, bias=True),
                  nn.InstanceNorm2d(ngf * mult // 2), nn.ReLU(True)]
        m += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, 7), nn.Tanh()]
        self.model = nn.Sequential(*m)

    def forward(self, x):
        return self.model(x)


BOTTLENECK_IDX = 18   # output of the last ResNet block: 256 channels at 1/4 resolution = the "latent code"
_GENS = {}


def load_generator(name):
    if name in _GENS:
        return _GENS[name]
    path = CKPT_DIR / f"{name}.pth"
    if not path.exists():
        urllib.request.urlretrieve(WEIGHTS_URL.format(name), path)
    sd = torch.load(path, map_location="cpu")
    sd = {k: v for k, v in sd.items() if not re.search(r"running_(mean|var)|num_batches_tracked", k)}
    net = ResnetGenerator()
    net.load_state_dict(sd)
    _GENS[name] = net.eval().to(DEVICE)
    return _GENS[name]


def fit(img, long_side):
    w, h = img.size
    s = long_side / max(w, h)
    nw, nh = (max(4, int(round(d * s / 4)) * 4) for d in (w, h))
    return img.resize((nw, nh), Image.LANCZOS)


def _to_tensor(img):
    x = np.array(img.convert("RGB"), dtype=np.float32) / 127.5 - 1
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def _to_pil(y):
    y = y[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(((y + 1) * 127.5).clip(0, 255).astype(np.uint8))


@torch.no_grad()
def cyclegan(img, name):
    return _to_pil(load_generator(name)(_to_tensor(img)))


@torch.no_grad()
def cyclegan_with_latent(img, name):
    """Return (output image, bottleneck feature map as numpy [C,h,w])."""
    net, grabbed = load_generator(name), {}
    h = net.model[BOTTLENECK_IDX].register_forward_hook(lambda m, i, o: grabbed.update(z=o[0].cpu().numpy()))
    try:
        out = _to_pil(net(_to_tensor(img)))
    finally:
        h.remove()
    return out, grabbed["z"]


def latent_pca_image(z, size):
    """Project the 256-d feature at each position onto its top-3 principal components -> RGB map."""
    c, h, w = z.shape
    f = z.reshape(c, -1).T
    f = f - f.mean(0)
    _, _, vt = np.linalg.svd(f, full_matrices=False)
    p = f @ vt[:3].T
    p = (p - p.min(0)) / (np.ptp(p, axis=0) + 1e-8)
    img = Image.fromarray((p.reshape(h, w, 3) * 255).astype(np.uint8))
    return img.resize(size, Image.NEAREST)


@torch.no_grad()
def reconstruct(original, translated, name):
    """Cycle-consistency demo: F(G(x)). Returns (reconstruction, mean L1 error in [0,1]) or (None, None)."""
    back = REVERSE.get(name)
    if back is None:
        return None, None
    rec = cyclegan(translated, back)
    a = np.asarray(original.convert("RGB"), dtype=np.float32) / 255
    b = np.asarray(rec.resize(original.size), dtype=np.float32) / 255
    return rec, float(np.abs(a - b).mean())


# ---------------- diffusion img2img + BLIP ----------------
_PIPE, _BLIP = {}, {}


def diffusion(img, prompt, strength):
    if "p" not in _PIPE:
        from diffusers import AutoPipelineForImage2Image
        p = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sd-turbo", torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32)
        _PIPE["p"] = p.to(DEVICE)
    strength = max(0.3, float(strength))   # steps * strength must be >= 1
    return _PIPE["p"](prompt=prompt, image=fit(img, 512), strength=strength,
                      guidance_scale=0.0, num_inference_steps=4).images[0]


def caption(img):
    if "m" not in _BLIP:
        from transformers import BlipForConditionalGeneration, BlipProcessor
        n = "Salesforce/blip-image-captioning-base"
        _BLIP["proc"] = BlipProcessor.from_pretrained(n)
        _BLIP["m"] = BlipForConditionalGeneration.from_pretrained(n).to(DEVICE).eval()
    inputs = _BLIP["proc"](img.convert("RGB"), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = _BLIP["m"].generate(**inputs, max_new_tokens=30)
    return _BLIP["proc"].decode(ids[0], skip_special_tokens=True)


# ---------------- watermark ----------------
def watermark(img, text=WATERMARK_TEXT):
    img = img.convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    size = max(12, img.width // 28)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:
        font = ImageFont.load_default()
    box = d.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    x, y = img.width - tw - 10, img.height - th - 10
    d.rectangle([x - 6, y - 4, x + tw + 6, y + th + 6], fill=(0, 0, 0, 90))
    d.text((x, y), text, font=font, fill=(255, 255, 255, 200))
    return Image.alpha_composite(img, layer).convert("RGB")
