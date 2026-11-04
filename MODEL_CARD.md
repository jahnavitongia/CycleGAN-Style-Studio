# Model Card: CycleGAN Style Studio (P15)

**Author:** <Your Name>, Roll No. 15 · **Course:** Generative AI (CA mini project) · **Date:** November 2026

## Intended use
Educational demo of unpaired image-to-image translation (photo ↔ painting, horse ↔ zebra) driven by a plain-English request.
**Not intended for:** impersonation, deepfakes, passing generated art off as an artist's original, or any commercial use
without checking every licence below.

## Components, licences and sources
| Component | Role | Licence / terms (VERIFY on the source page before submitting) |
|---|---|---|
| CycleGAN pretrained generators (`junyanz/pytorch-CycleGAN-and-pix2pix`) | Image translation | Code is BSD-licensed; check the repo for the weights' terms |
| `stabilityai/sd-turbo` | Diffusion comparison | Stability AI licence on the model page (restrictions on commercial use have applied) |
| `Salesforce/blip-image-captioning-base` | Captions | BSD-3-Clause (check model page) |
| `distilbert-base-uncased` (fine-tuned by me) | Request router | Apache-2.0 |
| LLM API (Groq / Gemini) | Router prompts, explanation | Provider terms of service |
| Gradio, PyTorch, Transformers, Diffusers | Libraries | Apache-2.0 / BSD |

## Training data (the part I trained)
DistilBERT router: 216 template-generated requests across 8 labels (7 models + `none`), see `data.py`.
Evaluated on 30 hand-written requests with **zero overlap** with training text. CycleGAN, SD-Turbo and BLIP are used as published; I did not retrain them.

## Measured results  *(paste from `results.md` after running `python eval_router.py`)*
| Router | Accuracy | In-scope acc | Hallucination rate | Median latency |
|---|---|---|---|---|
| zero_shot | _fill_ | _fill_ | _fill_ | _fill_ |
| few_shot | _fill_ | _fill_ | _fill_ | _fill_ |
| bert (fine-tuned) | _fill_ | _fill_ | _fill_ | _fill_ |
| keyword baseline | _fill_ | _fill_ | _fill_ | _fill_ |

Hallucination = an out-of-scope request (e.g. "make it look like Picasso") routed to a model instead of `none`.

## Known limitations and failure cases
- CycleGAN works best on images similar to its training data (horses/zebras in fields, landscapes). Portraits and indoor scenes often look wrong.
- Output is 256–320 px; fine detail is lost.
- Strength slider is an alpha blend with the original, not a true model parameter.
- Reconstruction F(G(x)) uses a separately trained reverse model for Monet, so it is only an approximate cycle.
- BLIP was trained on natural photos, so its captions of stylised images may be less accurate (check this in your own runs and cite an example).
- Router test set is small (30), so accuracy figures have wide uncertainty.
- **Failure case for report:** _paste one row from `failures.json` and one bad image._

## Bias, safety and IP
- **Guardrails:** `guardrails.py` blocks deepfake/face-swap, explicit, violent, hateful and forgery requests *before* any model runs (10/10 on its self-test; rule-based, so it can be evaded by rephrasing).
- **Watermark:** every output carries a visible "Generated with CycleGAN Style Studio" label.
- **Artist-style note:** shown under each output. Styles are limited to artists whose works are in the public domain (Monet d. 1926, Van Gogh d. 1890, Cézanne d. 1906) and Ukiyo-e prints. Living artists are not offered.
- **Bias:** CycleGAN datasets contain mostly Western landscapes and few people, so results on other scenes and on people of different appearance are less reliable. I did not measure image bias; my measured test is router hallucination (above).
- The app does not intentionally store user photos beyond the session gallery; Gradio/hosting may keep temporary cache files, so do not upload private images.
