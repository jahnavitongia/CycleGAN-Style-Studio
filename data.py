"""Router datasets.
TRAIN: generated from templates (for fine-tuning DistilBERT).
TEST : 30 hand-written requests (20 in-scope + 10 out-of-scope), phrased differently from the templates."""
import random

SUBJ = ["my photo", "this picture", "the image", "this photo", "my holiday snap", "this landscape", "the photo I uploaded"]
HORSE = ["this horse", "the horse in my photo", "the horse", "my horse picture", "these horses"]
ZEBRA = ["this zebra", "the zebra in my photo", "the zebra", "my zebra picture", "these zebras"]

TEMPLATES = {
    "style_monet": ["make {s} look like a Monet painting", "turn {s} into an impressionist painting by Monet",
                    "paint {s} in the style of Claude Monet", "Monet style for {s}", "give {s} a Monet water lily feel",
                    "convert {s} to Monet-style art", "i want {s} as a Monet impressionist oil painting"],
    "style_vangogh": ["make {s} look like a Van Gogh painting", "turn {s} into Starry Night style art",
                      "paint {s} like Vincent van Gogh", "Van Gogh style for {s}", "give {s} swirling Van Gogh brushstrokes",
                      "convert {s} to a Van Gogh style painting", "i want {s} as a van gogh post-impressionist piece"],
    "style_cezanne": ["make {s} look like a Cezanne painting", "turn {s} into a Paul Cezanne style artwork",
                      "paint {s} in the style of Cezanne", "Cezanne style for {s}", "give {s} a Cezanne look",
                      "convert {s} to Cezanne-style art", "i want {s} as a cezanne painting"],
    "style_ukiyoe": ["make {s} look like a Japanese woodblock print", "turn {s} into ukiyo-e art",
                     "convert {s} to a ukiyo-e print", "ukiyo-e style for {s}", "give {s} a Hokusai woodblock feel",
                     "redraw {s} as a traditional Japanese print", "i want {s} as an Edo period woodblock print"],
    "monet2photo": ["turn this painting into a realistic photo", "make this Monet painting look like a real photograph",
                    "convert this impressionist painting to a photo", "make the painting photorealistic",
                    "change this artwork into a realistic photograph", "painting to photo please",
                    "i want a real photo version of this painting"],
    "horse2zebra": ["turn {h} into a zebra", "make {h} a zebra", "convert {h} to a zebra", "horse to zebra",
                    "give {h} zebra stripes", "replace {h} with a zebra", "paint stripes on {h} so it becomes a zebra"],
    "zebra2horse": ["turn {z} into a horse", "make {z} a horse", "convert {z} to a horse", "zebra to horse",
                    "remove the stripes from {z} so it becomes a horse", "replace {z} with a horse",
                    "change {z} into a plain brown horse"],
    "none": ["make {s} look like a Picasso painting", "turn {s} into an anime scene", "make {s} a cyberpunk neon city",
             "convert {s} into a pencil sketch", "remove the background from {s}", "turn day into night in {s}",
             "add snow to {s}", "make {s} look like a Salvador Dali painting", "turn {s} into pixel art",
             "make {s} black and white", "upscale {s} to 4K", "turn {s} into a cartoon", "add a hat to the person in {s}",
             "turn {h} into a unicorn", "turn {z} into a giraffe", "turn my cat into a dog", "make {s} look like Warhol pop art",
             "blur the background of {s}", "write me a poem about {s}", "what is the capital of France",
             "make {s} look like a Rembrandt painting", "turn {s} into a watercolor", "swap the faces in {s}"],
}


def _fill(t, rng):
    return t.format(s=rng.choice(SUBJ), h=rng.choice(HORSE), z=rng.choice(ZEBRA))


def make_train(per_template=3, seed=0):
    rng = random.Random(seed)
    rows = []
    for label, temps in TEMPLATES.items():
        for t in temps:
            for _ in range(per_template):
                rows.append((_fill(t, rng), label))
    rng.shuffle(rows)
    return rows


TEST = [  # (request, expected label)
    ("Can you give my beach photo that dreamy Monet look?", "style_monet"),
    ("I'd love this garden shot to resemble an impressionist canvas by Claude Monet", "style_monet"),
    ("monet please", "style_monet"),
    ("Make the sky in my picture swirl like Starry Night", "style_vangogh"),
    ("Redo this street photo as if Vincent painted it", "style_vangogh"),
    ("van gogh vibes for my selfie background", "style_vangogh"),
    ("Could you render my fruit bowl photo like a Cezanne still life?", "style_cezanne"),
    ("cezanne style, thanks", "style_cezanne"),
    ("Make this mountain photo look like Hokusai's Great Wave era prints", "style_ukiyoe"),
    ("Edo-style woodblock version of my city skyline", "style_ukiyoe"),
    ("I want this to look like a traditional Japanese woodcut", "style_ukiyoe"),
    ("This painting is lovely, can you show what the scene would look like as a real photo?", "monet2photo"),
    ("de-paint this artwork into something camera-like", "monet2photo"),
    ("make this canvas look like it was photographed in real life", "monet2photo"),
    ("My horse should look like a zebra", "horse2zebra"),
    ("Add zebra stripes to the animal in the field", "horse2zebra"),
    ("horse -> zebra", "horse2zebra"),
    ("Can the zebra in this safari picture become a horse instead?", "zebra2horse"),
    ("strip the stripes off and show a horse", "zebra2horse"),
    ("zebra into horse", "zebra2horse"),
    # ---- out-of-scope: the correct answer is "none"; picking any model is a hallucination ----
    ("Make my photo look like a Picasso cubist painting", "none"),
    ("Turn this into an anime still", "none"),
    ("Give me a neon cyberpunk version of this street", "none"),
    ("Convert my dog photo into a cat", "none"),
    ("Could you remove the person standing behind me?", "none"),
    ("Make this a pencil sketch", "none"),
    ("Turn the horse into a giraffe", "none"),
    ("Make this look like a Dali melting-clocks painting", "none"),
    ("Change this daytime picture to midnight", "none"),
    ("Generate a deepfake of my teacher saying something funny", "none"),
]

if __name__ == "__main__":
    tr = make_train()
    print(len(tr), "train rows;", len(TEST), "test rows")
    from collections import Counter
    print(Counter(l for _, l in tr))
    train_set = {t for t, _ in tr}
    print("test/train overlap:", sum(t in train_set for t, _ in TEST))
