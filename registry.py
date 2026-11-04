"""Model registry + LLM helper. No heavy dependencies, so eval scripts can import it."""
import os

PD = ("Style learned from works by {a} (d. {y}), which are in the public domain. "
      "This is an AI-generated imitation, not an original work by the artist, "
      "and is not affiliated with or endorsed by them.")

REGISTRY = {
    "style_monet": dict(desc="photo -> Monet-style painting", prompt="an impressionist oil painting by Claude Monet",
                        note=PD.format(a="Claude Monet", y=1926)),
    "style_vangogh": dict(desc="photo -> Van Gogh-style painting",
                          prompt="a post-impressionist painting by Vincent van Gogh, swirling brushstrokes",
                          note=PD.format(a="Vincent van Gogh", y=1890)),
    "style_cezanne": dict(desc="photo -> Cezanne-style painting", prompt="a painting by Paul Cezanne",
                          note=PD.format(a="Paul Cezanne", y=1906)),
    "style_ukiyoe": dict(desc="photo -> Ukiyo-e Japanese woodblock print", prompt="a ukiyo-e Japanese woodblock print",
                         note="Style learned from historical Ukiyo-e prints (public domain). AI-generated imitation, "
                              "not an original work by any artist."),
    "monet2photo": dict(desc="Monet-style painting -> realistic photo", prompt="a realistic photograph",
                        note="Translation of a painting into a photo-like image. The source artwork remains the "
                             "property of its creator; check its copyright status before reuse."),
    "horse2zebra": dict(desc="horse -> zebra", prompt="a photo of a zebra",
                        note="AI-generated image. No artist style involved."),
    "zebra2horse": dict(desc="zebra -> horse", prompt="a photo of a horse",
                        note="AI-generated image. No artist style involved."),
}
LABELS = list(REGISTRY) + ["none"]          # 8 classes, "none" = unsupported request

# Reverse generator F for reconstruction F(G(x)).  True trained pair only for horse<->zebra;
# for Monet the two checkpoints were trained separately, so the cycle is approximate.
REVERSE = {"horse2zebra": "zebra2horse", "zebra2horse": "horse2zebra",
           "style_monet": "monet2photo", "monet2photo": "style_monet"}


def llm(messages, max_tokens=500, temperature=0.2):
    """OpenAI-compatible chat call (Groq by default, Gemini via LLM_BASE_URL). None if no key."""
    key = os.getenv("LLM_API_KEY")
    if not key:
        return None
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    r = client.chat.completions.create(model=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
                                       messages=messages, max_tokens=max(max_tokens, 1000), temperature=temperature)
    return r.choices[0].message.content
