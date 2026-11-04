"""Four routers that map a plain-English request to a model name (or 'none').
  zero_shot : LLM, instructions only                 (prompt technique 1)
  few_shot  : LLM, instructions + 4 examples + none  (prompt technique 2)
  bert      : fine-tuned DistilBERT classifier       (fine-tuning, BERT-family model)
  keyword   : rule baseline / offline fallback
"""
import json
import os
import re

from registry import LABELS, REGISTRY, llm

MODEL_LIST = "\n".join(f"- {k}: {v['desc']}" for k, v in REGISTRY.items())
BASE = (f"You route image-translation requests to ONE pretrained CycleGAN model.\nAvailable models:\n{MODEL_LIST}\n"
        "- none: the request does not match any model above (unsupported style, object swap, editing, etc.)\n"
        'Reply with ONLY a JSON object: {"model": "<name or none>", "reason": "<one short sentence>"}')
FEW_SHOT = """
Examples:
Request: make my holiday photo look like a Van Gogh painting -> {"model": "style_vangogh", "reason": "User wants a Van Gogh look."}
Request: turn this horse into a zebra -> {"model": "horse2zebra", "reason": "Horse to zebra translation."}
Request: make this painting look like a real photograph -> {"model": "monet2photo", "reason": "Painting to photo."}
Request: make it a cyberpunk neon city -> {"model": "none", "reason": "No cyberpunk model is available."}
Do NOT guess: if the artist, animal or effect is not in the list, answer none."""


def _parse(out):
    data = json.loads(re.search(r"\{.*\}", out, re.S).group(0))
    if data.get("model") in LABELS:
        return {"model": data["model"], "reason": str(data.get("reason", ""))}
    raise ValueError("model not in registry")


def _llm_router(request, system):
    out = llm([{"role": "system", "content": system}, {"role": "user", "content": f"Request: {request}"}],
              max_tokens=120, temperature=0)
    if out is None:
        raise RuntimeError("no LLM key")
    return _parse(out)


def zero_shot(request):
    return _llm_router(request, BASE)


def few_shot(request):
    return _llm_router(request, BASE + FEW_SHOT)


def keyword(request):
    t = request.lower()
    rules = [("zebra2horse", ["zebra to horse", "zebra into a horse", "zebra into horse", "zebra into a plain horse"]),
             ("horse2zebra", ["horse to zebra", "horse into a zebra", "horse a zebra", "stripes to"]),
             ("style_vangogh", ["van gogh", "vangogh", "starry"]), ("style_cezanne", ["cezanne", "cézanne"]),
             ("style_ukiyoe", ["ukiyo", "woodblock", "japanese print"]),
             ("monet2photo", ["to photo", "realistic photo", "real photo", "photorealistic"]),
             ("style_monet", ["monet"])]
    for model, words in rules:
        if any(w in t for w in words):
            return {"model": model, "reason": "Keyword match."}
    return {"model": "none", "reason": "No keyword matched."}


# ---------------- fine-tuned DistilBERT ----------------
_BERT = {}
BERT_PATH = os.getenv("BERT_ROUTER_PATH", "bert_router")   # local folder or HF Hub id


def bert_available():
    return os.path.isdir(BERT_PATH) or "/" in BERT_PATH


def _load_bert():
    if "m" not in _BERT:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        _BERT["tok"] = AutoTokenizer.from_pretrained(BERT_PATH)
        _BERT["m"] = AutoModelForSequenceClassification.from_pretrained(BERT_PATH, attn_implementation="eager").eval()
    return _BERT["tok"], _BERT["m"]


def bert(request, with_details=False):
    import torch
    tok, m = _load_bert()
    enc = tok(request, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        out = m(**enc, output_attentions=with_details)
    probs = torch.softmax(out.logits, -1)[0]
    labels = [m.config.id2label[i] for i in range(len(probs))]
    best = labels[int(probs.argmax())]
    res = {"model": best, "reason": f"DistilBERT confidence {float(probs.max()):.2f}"}
    if with_details:
        res["probs"] = {l: float(p) for l, p in zip(labels, probs)}
        # Layer II attention explanation: last layer, mean over heads, attention paid BY [CLS] TO each token
        att = out.attentions[-1][0].mean(0)[0]
        toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
        res["attention"] = [(t, float(a)) for t, a in zip(toks, att)]
    return res


ROUTERS = {"zero_shot": zero_shot, "few_shot": few_shot, "bert": bert, "keyword": keyword}


def route(request, method="few_shot"):
    """Route with the chosen method; fall back to keyword if it fails (so the app never dies)."""
    try:
        return ROUTERS[method](request) | {"method": method}
    except Exception as e:
        print(f"[router:{method}] failed ({e}); using keyword fallback")
        return keyword(request) | {"method": "keyword (fallback)"}
