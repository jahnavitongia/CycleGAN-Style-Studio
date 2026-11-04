---
title: CycleGAN Style Studio
emoji: 🎨
colorFrom: indigo
colorTo: pink
sdk: gradio
app_file: app.py
pinned: false
---

# CycleGAN Style Studio (Generative AI CA, P15)

Plain-English image translation (photo↔painting, horse↔zebra) with pretrained CycleGAN, an LLM router
(zero-shot / few-shot) vs a fine-tuned DistilBERT router, diffusion + BLIP comparison, strength slider,
gallery/ZIP export, watermarking, guardrails and a model card.

## Files
| File | Purpose |
|---|---|
| `app.py` | Gradio app (all units) |
| `models.py` | CycleGAN generator, reconstruction, latent PCA, SD-Turbo, BLIP, watermark |
| `routers.py` | zero-shot, few-shot, DistilBERT, keyword routers |
| `train_bert_router.py` | fine-tunes DistilBERT into `bert_router/` |
| `eval_router.py` | accuracy / hallucination rate / latency → `results.md`, `failures.json` |
| `data.py` | router training templates + 30-request hand-written test set |
| `guardrails.py` | request and image safety checks (`python guardrails.py` self-test) |
| `make_lifecycle.py` | draws `docs/lifecycle.png` |
| `MODEL_CARD.md`, `REPORT_OUTLINE.md` | model card; report structure and demo script |

## Workflow
```bash
pip install -r requirements.txt
export LLM_API_KEY=...                      # Groq or Gemini key
export LLM_BASE_URL=https://api.groq.com/openai/v1   # default
export LLM_MODEL=llama-3.1-8b-instant                # default
python train_bert_router.py                 # a few minutes on CPU -> bert_router/
python eval_router.py                       # -> results.md, failures.json
python app.py                               # run locally
```
Gemini: `LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`, `LLM_MODEL=gemini-2.0-flash`.

## Deploy on Hugging Face Spaces
1. New Space (SDK Gradio), upload all files including `bert_router/` and `docs/lifecycle.png`
   (or push the model to the Hub and set `BERT_ROUTER_PATH=youruser/bert-router`).
2. Settings → Variables and secrets: add secret `LLM_API_KEY` (and `LLM_BASE_URL`, `LLM_MODEL`).
3. First run downloads CycleGAN weights, SD-Turbo and BLIP; keep "Compare" off for a fast first test.

## Notes
- CycleGAN has no native strength parameter: the slider alpha-blends with the original. Diffusion uses img2img `strength` (clamped ≥ 0.3).
- Reconstruction F(G(x)) is a true trained pair for horse↔zebra, only approximate for Monet.
- Without `LLM_API_KEY` or `bert_router/`, the app falls back to the keyword router and still works.
