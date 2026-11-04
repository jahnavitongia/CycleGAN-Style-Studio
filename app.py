"""CycleGAN Style Studio (P15) - Gradio app.
U1 router (zero-shot / few-shot / fine-tuned DistilBERT / keyword) | U2 CycleGAN + cycle-consistency
U3 diffusion + BLIP | U4 strength slider + gallery/ZIP | U5 watermark + copyright note | + guardrails"""
import json
import os
import tempfile
import zipfile

import gradio as gr
from PIL import Image

import guardrails
import models
import routers
from registry import REGISTRY, REVERSE, llm

METHODS = ["few_shot", "zero_shot", "bert", "keyword"]
BLOCKED = {"count": 0}


def explain_cycle_consistency(request, model):
    info = REGISTRY.get(model, {"desc": "an unpaired image translation"})
    prompt = (f"The user asked: '{request}'. We used the CycleGAN model '{model}' ({info['desc']}). "
              "Explain cycle-consistency in 120 words or fewer for a student, tied to this translation. "
              "Include the loss L_cyc = E[||F(G(x)) - x||_1] + E[||G(F(y)) - y||_1], where G maps X->Y and "
              "F maps Y->X, and say why it is needed when there are no paired training images.")
    out = llm([{"role": "user", "content": prompt}], max_tokens=300, temperature=0.4)
    return out or ("(No LLM key set.) Cycle-consistency forces F(G(x)) ≈ x and G(F(y)) ≈ y, so translating "
                   "there and back must recover the original. L_cyc = E||F(G(x)) − x||₁ + E||G(F(y)) − y||₁. "
                   "It replaces paired supervision, which does not exist for horse↔zebra or photo↔Monet.")


def gallery_view(state):
    return [(e["image"], e["caption"]) for e in state]


def empty(msg, state):
    return (msg, None, None, None, None, None, "", None, "", "", state, gallery_view(state))


def run(img, request, method, strength, compare, internals, state):
    state = list(state or [])
    if img is None:
        raise gr.Error("Upload a photo first.")
    if not request or not request.strip():
        raise gr.Error("Describe the translation you want.")

    # ---- Layer V: guardrails run BEFORE any model ----
    ok, why = guardrails.check_request(request)
    if ok:
        ok, why = guardrails.check_image(img)
    if not ok:
        BLOCKED["count"] += 1
        return empty(f"🚫 {why}", state)

    # ---- U1: routing ----
    probs, attn = None, None
    if method == "bert" and routers.bert_available():
        try:
            choice = routers.bert(request, with_details=True) | {"method": "bert"}
            probs, attn = choice.get("probs"), choice.get("attention")
        except Exception as e:
            print("bert details failed:", e)
            choice = routers.route(request, method)
    else:
        choice = routers.route(request, method)
    model = choice["model"]
    if model == "none":
        return empty(f"**Router ({choice['method']}):** no matching model. {choice['reason']}\n\n"
                     "Supported: Monet / Van Gogh / Cézanne / Ukiyo-e / horse↔zebra / painting→photo.", state)

    # ---- U2: CycleGAN (+ latent view and cycle reconstruction) ----
    base = models.fit(img.convert("RGB"), 320)
    cg, z = models.cyclegan_with_latent(base, model)
    rec_img, latent_img, l1_txt = None, None, ""
    if internals:
        latent_img = models.latent_pca_image(z, base.size)
        rec_img, l1 = models.reconstruct(base, cg, model)
        if rec_img is not None:
            pair = "trained pair" if model in ("horse2zebra", "zebra2horse") else "separately trained pair (approximate cycle)"
            l1_txt = f"Cycle reconstruction F(G(x)) via `{REVERSE[model]}` — mean L1 error vs original: **{l1:.3f}** ({pair})"
        else:
            l1_txt = "No reverse generator available for this model, so F(G(x)) cannot be shown."

    # ---- U4: strength = alpha blend (CycleGAN has no native strength) ----
    blended = Image.blend(base, cg, float(strength))
    cg_out = models.watermark(blended)                      # ---- U5 ----
    note = REGISTRY[model]["note"]
    info = (f"**Router ({choice['method']}):** `{model}` — {REGISTRY[model]['desc']}  \n_{choice['reason']}_")

    # ---- U3: diffusion comparison + BLIP ----
    diff_out, caps = None, ""
    if compare:
        try:
            diff_out = models.watermark(models.diffusion(img, REGISTRY[model]["prompt"], strength))
            caps = (f"**Original:** {models.caption(base)}  \n**CycleGAN:** {models.caption(cg)}  \n"
                    f"**Diffusion:** {models.caption(diff_out)}")
        except Exception as e:
            caps = f"Diffusion/BLIP failed: {e}"

    meta = dict(model=model, request=request, router=choice["method"], strength=float(strength), note=note)
    state.append(dict(image=cg_out, caption=f"{model} @ {strength:.2f}", **meta))
    if diff_out is not None:
        state.append(dict(image=diff_out, caption=f"sd-turbo ({model} prompt) @ {strength:.2f}", **(meta | {"model": "sd-turbo"})))

    return (info, probs, attn, cg_out, diff_out, rec_img, l1_txt, latent_img, caps, "⚠️ " + note,
            state, gallery_view(state))


def export_zip(state):
    if not state:
        raise gr.Error("Gallery is empty.")
    tmp = tempfile.mkdtemp()
    zpath = os.path.join(tmp, "style_studio_gallery.zip")
    meta = []
    with zipfile.ZipFile(zpath, "w") as z:
        for i, e in enumerate(state, 1):
            fn = f"{i:02d}_{e['model']}.png"
            p = os.path.join(tmp, fn)
            e["image"].save(p)
            z.write(p, fn)
            meta.append({k: v for k, v in e.items() if k != "image"} | {"file": fn})
        z.writestr("metadata.json", json.dumps(meta, indent=2))
    return zpath


def explain(request):
    if not request:
        raise gr.Error("Type a request first.")
    return explain_cycle_consistency(request, routers.route(request, "keyword")["model"])


ABOUT = """
### How this app fits Generative AI
**Generative vs discriminative.** A *discriminative* model learns p(y|x) (e.g. DistilBERT deciding which model a request needs).
A *generative* model learns to produce x itself (CycleGAN generators, SD-Turbo, the LLM's text). This app uses both.

**Lifecycle:** data → model choice → training/fine-tuning → evaluation → deployment → monitoring (see report diagram, `docs/lifecycle.png`).

**Prompt techniques compared:** zero-shot vs few-shot router (results in `results.md`).
**Fine-tuning vs prompting:** fine-tuned DistilBERT vs LLM prompts, same 30-request test set.
**GAN element:** CycleGAN with cycle-consistency loss. **Latent view:** PCA of the 256-channel bottleneck.
**Attention explanation:** DistilBERT last-layer [CLS] attention over request tokens.
**Safety:** rule-based request/image guardrails, visible watermark, public-domain-aware copyright notes. See `MODEL_CARD.md`.
"""

with gr.Blocks(title="CycleGAN Style Studio") as demo:
    gr.Markdown("# 🎨 CycleGAN Style Studio\nDescribe a translation in plain English, e.g. "
                "*“make this look like a Van Gogh painting”* or *“turn the horse into a zebra”*.")
    state = gr.State([])
    with gr.Tab("Studio"):
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Input photo")
                req = gr.Textbox(label="Your request", placeholder="Make this photo look like a Monet painting")
                method = gr.Dropdown(METHODS, value="few_shot", label="Router (U1): few_shot / zero_shot = LLM prompt · bert = fine-tuned DistilBERT")
                strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05,
                                     label="Strength (CycleGAN: blend with original · Diffusion: img2img strength)")
                compare = gr.Checkbox(label="Compare with diffusion img2img + BLIP captions (slow on CPU)", value=False)
                internals = gr.Checkbox(label="Show internals: cycle reconstruction F(G(x)) + latent PCA", value=True)
                with gr.Row():
                    go = gr.Button("Translate", variant="primary")
                    ex = gr.Button("Explain cycle-consistency")
            with gr.Column():
                info = gr.Markdown()
                probs = gr.Label(label="Router class probabilities (BERT only)", num_top_classes=4)
                attn = gr.HighlightedText(label="BERT attention: which words drove the decision", show_legend=False)
                out_cg = gr.Image(label="CycleGAN output")
                out_df = gr.Image(label="Diffusion output (optional)")
                caps = gr.Markdown()
                note = gr.Markdown()
        with gr.Row():
            rec = gr.Image(label="Reconstruction F(G(x))")
            lat = gr.Image(label="Latent PCA (bottleneck, 3 components → RGB)")
        l1 = gr.Markdown()
        expl = gr.Markdown()
        gal = gr.Gallery(label="Session gallery", columns=4, height="auto")
        with gr.Row():
            zbtn = gr.Button("Export gallery as ZIP")
            zfile = gr.File(label="Download")
    with gr.Tab("About"):
        gr.Markdown(ABOUT)
        if os.path.exists("docs/lifecycle.png"):
            gr.Image("docs/lifecycle.png", label="Generative AI lifecycle for this project", interactive=False)

    go.click(run, [inp, req, method, strength, compare, internals, state],
             [info, probs, attn, out_cg, out_df, rec, l1, lat, caps, note, state, gal])
    ex.click(explain, req, expl)
    zbtn.click(export_zip, state, zfile)

if __name__ == "__main__":
    demo.queue().launch()
