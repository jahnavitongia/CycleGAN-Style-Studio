# PDF Report Outline: `15_<Name>_P15.pdf`

Aim for 8-10 pages. Every bullet marked ▢ is something you must produce from your own runs.

## 1. Title page and summary (½ page)
Project, roll no., live app URL, GitHub URL, one-paragraph summary with your headline number.

## 2. Layer I: Foundations
- Lifecycle diagram: `docs/lifecycle.png` (explain each stage for *this* project).
- Generative vs discriminative: CycleGAN/SD-Turbo/LLM generate; DistilBERT router discriminates p(label | request).
- Prompt techniques compared: zero-shot vs few-shot router, with the accuracy/hallucination rows from `results.md`. ▢
- Probability / latent view: BERT class-probability bars and latent PCA image (screenshots). ▢

## 3. Layer II + U1 + U2: GANs and LLM
- U1: router prompt (paste both prompts from `routers.py`), JSON validation, fallback.
- **Fine-tuning vs prompt-based learning:** fine-tuned DistilBERT vs LLM prompts on the same 30 requests (table). Training details from `bert_router/train_log.json`: 216 examples, epochs, seconds. ▢
- U2: pretrained CycleGAN, ResNet-9 generator, cycle-consistency loss formula, LLM explanation (screenshot). ▢
- GAN element: reconstruction F(G(x)) images for horse2zebra and one Monet case, with L1 error. Note Monet is an approximate cycle. ▢
- Attention explanation: screenshot of DistilBERT attention highlights for one request. ▢

## 4. Layer III + U3: GPT, BERT and multimodal
- GPT-style generator: the LLM (router + explainer). BERT-family: DistilBERT. Image modality: CycleGAN, SD-Turbo, BLIP.
- **CycleGAN vs diffusion comparison table** for 5 images ▢

| Image | CycleGAN time | Diffusion time | Structure kept (1-5) | Style quality (1-5) | BLIP caption before | after (CycleGAN) | after (diffusion) |
|---|---|---|---|---|---|---|---|

- Short written analysis: speed, fidelity, style, failure modes.

## 5. Layer IV + U4: Tools and generation
- PyTorch implementation, strength slider (alpha blend vs img2img strength, say this explicitly), gallery and ZIP export (show `metadata.json`).
- **5 translated photos** (deliverable): e.g. horse2zebra, zebra2horse, style_monet, style_vangogh, style_ukiyoe. Show original beside output. ▢

## 6. Layer V + U5: Responsible AI
- **Measured test:** hallucination rate on 10 out-of-scope requests per router (reuse the table). ▢
- **Failure case:** one row from `failures.json` plus one bad image. ▢
- Guardrails: rules, `python guardrails.py` result (10/10), blocked-request screenshot, and its limits (rephrasing can evade).
- Watermark and copyright note: before/after screenshot, why only public-domain artists are offered.
- Licence/IP table (copy from `MODEL_CARD.md`; verify each licence).

## 7. Deployment and demo
Platform, secrets handling (no keys in repo), cold-start time, link to demo video.

## 8. Conclusion, limitations, future work
## 9. References
CycleGAN (Zhu et al., 2017), DistilBERT (Sanh et al., 2019), BLIP (Li et al., 2022), SD-Turbo model page, Gradio.

---
## Demo video script (2-3 min)
1. (15 s) Show the live URL and state the goal.
2. (30 s) "Make this horse a zebra" → router decision, probabilities, attention words.
3. (30 s) Strength slider at 0.5 vs 1.0; show watermark and copyright note.
4. (30 s) Compare toggle: diffusion output and BLIP captions.
5. (20 s) Internals: reconstruction F(G(x)) and latent PCA, one-sentence cycle-consistency explanation.
6. (20 s) Type "make a deepfake of my teacher" → blocked; type "make it Picasso" → router says none.
7. (15 s) Show the results table and the GitHub repo.
