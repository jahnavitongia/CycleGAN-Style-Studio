"""Generates docs/lifecycle.png - the Generative AI lifecycle for this project (Layer I)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

steps = [("1. Data", "CycleGAN datasets\n(horse/zebra, Monet, ...)\nrouter request set"),
         ("2. Model choice", "CycleGAN (GAN)\nSD-Turbo (diffusion)\nDistilBERT, LLM, BLIP"),
         ("3. Train / adapt", "Pretrained CycleGAN\nFine-tune DistilBERT\nPrompt LLM"),
         ("4. Evaluate", "Router accuracy,\nhallucination rate,\ncycle L1 error"),
         ("5. Deploy", "Gradio app on\nHugging Face Spaces\n+ guardrails"),
         ("6. Monitor", "Blocked requests,\nfailure cases,\nmodel card updates")]
fig, ax = plt.subplots(figsize=(13, 3.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 3.2); ax.axis("off")
colors = ["#dbeafe", "#e0e7ff", "#ede9fe", "#fce7f3", "#dcfce7", "#fef9c3"]
for i, (title, body) in enumerate(steps):
    x = 0.15 + i * 2.15
    ax.add_patch(FancyBboxPatch((x, 0.5), 1.85, 2.2, boxstyle="round,pad=0.05", fc=colors[i], ec="#334155"))
    ax.text(x + 0.925, 2.35, title, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(x + 0.925, 1.4, body, ha="center", va="center", fontsize=8.5)
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + 2.1, 1.6), xytext=(x + 1.9, 1.6), arrowprops=dict(arrowstyle="->", lw=1.8))
ax.annotate("", xy=(0.9, 0.35), xytext=(12.0, 0.35), arrowprops=dict(arrowstyle="->", lw=1.2, color="#64748b", connectionstyle="arc3,rad=0"))
ax.text(6.5, 0.12, "feedback loop: failures and monitoring feed back into data and model choice", ha="center", fontsize=8.5, color="#475569")
plt.savefig("docs/lifecycle.png", dpi=160, bbox_inches="tight")
print("saved docs/lifecycle.png")
