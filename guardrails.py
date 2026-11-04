"""Safety guardrails (Layer V). Rule-based, fast, and testable without any model."""
import re

BLOCK_RULES = [
    ("deepfake / face swap / impersonation", r"deep\s?fake|face\s?swap|swap (the )?faces?|impersonat|pretend to be|saying something"),
    ("sexual or explicit content", r"\bnsfw\b|\bnude\b|\bnaked\b|undress|remove (her |his |their )?cloth|porn|explicit|sexual"),
    ("graphic violence", r"\bgore\b|graphic violence|mutilat|dismember|behead"),
    ("hate content", r"nazi|white power|racial slur|hate symbol"),
    ("misleading media", r"fake (news|evidence|id|passport|document)|forge"),
]
MAX_PIXELS = 4000 * 4000      # reject absurdly large uploads
MIN_SIDE = 32


def check_request(text: str):
    """Return (ok, reason). Blocks requests that match a risky category."""
    t = (text or "").lower()
    for reason, pattern in BLOCK_RULES:
        if re.search(pattern, t):
            return False, f"Blocked: request looks like **{reason}**. This studio only does style/animal translation."
    if len(t) > 400:
        return False, "Blocked: request is too long (max 400 characters)."
    return True, ""


def check_image(img):
    w, h = img.size
    if min(w, h) < MIN_SIDE:
        return False, "Image is too small (minimum 32 px per side)."
    if w * h > MAX_PIXELS:
        return False, "Image is too large (max 16 megapixels)."
    return True, ""


if __name__ == "__main__":  # quick self-test used for the report's guardrail table
    cases = [("make my photo look like Monet", True), ("turn this horse into a zebra", True),
             ("make a deepfake of my teacher", False), ("swap the faces in this photo", False),
             ("make her naked", False), ("remove her clothes", False), ("forge a passport photo", False),
             ("gory graphic violence scene", False), ("van gogh style please", True), ("turn this into ukiyo-e", True)]
    ok = 0
    for text, expected in cases:
        got = check_request(text)[0]
        ok += got == expected
        print("OK " if got == expected else "BAD", text, "->", "allowed" if got else "blocked")
    print(f"{ok}/{len(cases)} correct")
