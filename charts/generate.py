#!/usr/bin/env python3
"""Generate comparison charts for the local-llm-benchmarks repo.

Outputs 3 PNGs into the charts/ dir:
  1. decode_vs_prefill.png — grouped bar: decode + 12K cold prefill for all candidates
  2. vram_coresidence.png — co-residence VRAM analysis vs 21.6 GiB ceiling
  3. b10068_uplift.png     — before/after b10068 for the two models we have both numbers for
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})

# ─────────────────────────────────────────────────────────────
# Chart 1: Decode + Cold 12K prefill grouped bars
# ─────────────────────────────────────────────────────────────
# name, decode tok/s, cold 12K prefill tok/s, note
models_perf = [
    ("Ornith 9B + MTP\n(prod)",                    51.8,  896, "baseline"),
    ("Qwen 3.6-35B-A3B\nMTP UD-Q4_K_XL",            49.0,  798, "35B-A3B win"),
    ("Qwen 3.6-35B-A3B\nMTP UD-Q4_K_S",               37.7,  820, "tested, not adopted"),
    ("Qwen 3.6-35B-A3B\nMTP UD-IQ4_XS",              31.6,  776, "tested, not adopted"),
    ("Qwen 3.6-35B-A3B\nKimi Distilled IQ4_XS",     30.6,  904, "prefill king"),
    ("Qwen 3.6-35B-A3B\nClaude APEX-MTP Compact",   36.9,  763, "fits prod"),
    ("Gemma 4 26B-A4B\n+ MTP",                       53.0,  650, "reasoning fallback"),
    ("Qwen 3.6-35B-A3B\n(base, no MTP)",             31.1,  823, "superseded"),
    ("Qwen 3.6-27B\n(dense)",                        18.0,  374, "dense penalty"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
names = [m[0] for m in models_perf]
decode = [m[1] for m in models_perf]
prefill = [m[2] for m in models_perf]

# color: green for prod, teal for viable candidates, purple for reasoning, gray for others
def bar_color(m):
    if m[3] == "baseline":
        return "#2E7D32"
    if m[3] in ("35B-A3B win", "prefill king", "fits prod"):
        return "#00838F"
    if m[3] == "reasoning fallback":
        return "#6A1B9A"
    return "#9E9E9E"

colors = [bar_color(m) for m in models_perf]

# Decode chart
y_pos = np.arange(len(names))
bars1 = ax1.barh(y_pos, decode, color=colors, edgecolor="black", linewidth=0.4)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(names, fontsize=9)
ax1.invert_yaxis()
ax1.set_xlabel("Decode tok/s (single-stream)", fontweight="bold")
ax1.set_title("Decode throughput\n(higher is better)", fontweight="bold", pad=12)
ax1.axvline(x=51.8, color="#2E7D32", linestyle="--", alpha=0.5, linewidth=1)
ax1.text(52.5, len(names) - 0.3, "Ornith 9B baseline",
         color="#2E7D32", fontsize=8, va="bottom", fontweight="bold")
for b, v in zip(bars1, decode):
    ax1.text(v + 2, b.get_y() + b.get_height()/2, f"{v:.1f}",
             va="center", fontsize=8)
ax1.set_xlim(0, max(decode) * 1.15)
ax1.grid(axis="x", linestyle=":", alpha=0.4)

# Prefill chart
bars2 = ax2.barh(y_pos, prefill, color=colors, edgecolor="black", linewidth=0.4)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(names, fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel("Cold 12K prefill tok/s", fontweight="bold")
ax2.set_title("Cold prefill throughput\n(higher is better)", fontweight="bold", pad=12)
ax2.axvline(x=896, color="#2E7D32", linestyle="--", alpha=0.5, linewidth=1)
ax2.text(920, len(names) - 0.3, "Ornith 9B baseline",
         color="#2E7D32", fontsize=8, va="bottom", fontweight="bold")
for b, v in zip(bars2, prefill):
    ax2.text(v + 60, b.get_y() + b.get_height()/2, f"{v:,}",
             va="center", fontsize=8)
ax2.set_xlim(0, max(prefill) * 1.15)
ax2.grid(axis="x", linestyle=":", alpha=0.4)

fig.suptitle("B60 Pro (Battlemage, 24 GB) — Model bench comparison\n"
             "llama.cpp:sycl-f16 b10068 · isolated benches · 2026-07-19",
             fontsize=13, fontweight="bold", y=1.005)

# Legend
handles = [
    plt.Rectangle((0,0),1,1, fc="#2E7D32", ec="black", lw=0.4, label="Production baseline"),
    plt.Rectangle((0,0),1,1, fc="#00838F", ec="black", lw=0.4, label="Viable candidate"),
    plt.Rectangle((0,0),1,1, fc="#6A1B9A", ec="black", lw=0.4, label="Reasoning fallback"),
    plt.Rectangle((0,0),1,1, fc="#9E9E9E", ec="black", lw=0.4, label="Tested, not adopted"),
]
fig.legend(handles=handles, loc="lower center", ncol=4,
           bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig(OUT / "decode_vs_prefill.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote", OUT / "decode_vs_prefill.png")


# ─────────────────────────────────────────────────────────────
# Chart 2: VRAM co-residence analysis
# ─────────────────────────────────────────────────────────────
# name, isolated chat VRAM (GiB)
vram_models = [
    ("Ornith 9B + MTP\n(prod)",                    10.9),
    ("Qwen 3.6-35B-A3B\nClaude APEX-MTP",           19.4),
    ("Qwen 3.6-35B-A3B\nKimi IQ4_XS",               21.4),
    ("Qwen 3.6-35B-A3B\nMTP UD-Q4_K_XL",             24.4),
    ("Qwen 3.6-35B-A3B\nMTP UD-Q4_K_S",               24.1),
    ("Qwen 3.6-35B-A3B\nMTP UD-IQ4_XS",              21.1),
    ("Gemma 4 26B-A4B\n+ MTP",                       22.9),
    ("Qwen 3.6-27B\n(dense)",                        20.2),
]
OVERHEAD = 2.4  # embed 0.5 + rerank 0.5 + tei 1.4
CEILING = 24.0
BUDGET = CEILING - OVERHEAD  # 21.6

fig, ax = plt.subplots(figsize=(12, 6.5))
names_v = [m[0] for m in vram_models]
chat = [m[1] for m in vram_models]
overhead = [OVERHEAD] * len(vram_models)
y = np.arange(len(names_v))

# Bar chart, stacked
bars_chat = ax.barh(y, chat, color="#1E88E5", edgecolor="black",
                    linewidth=0.4, label="Chat model (isolated)")
bars_over = ax.barh(y, overhead, left=chat, color="#FFB300", edgecolor="black",
                    linewidth=0.4, label="Non-chat services (embed+rerank+TEI)")

# Ceiling + budget lines
ax.axvline(x=CEILING, color="#C62828", linestyle="-", linewidth=2, label=f"24 GiB card ceiling")
ax.axvline(x=BUDGET, color="#2E7D32", linestyle="--", linewidth=1.5,
           label=f"Safe chat-model budget ({BUDGET:.1f} GiB)")

# Labels on bars
for i, (c_val, o_val) in enumerate(zip(chat, overhead)):
    total = c_val + o_val
    color = "#C62828" if total > CEILING else "#2E7D32" if total < CEILING - 0.5 else "#F57C00"
    label = f"{total:.1f} GiB total"
    if total > CEILING:
        label += " ⚠"
    ax.text(total + 0.15, i, label, va="center", color=color,
            fontsize=9, fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels(names_v, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("VRAM (GiB)", fontweight="bold")
ax.set_xlim(0, CEILING * 1.15)
ax.legend(loc="lower right", framealpha=0.95, fontsize=9)
ax.grid(axis="x", linestyle=":", alpha=0.4)
ax.set_title("Co-residence VRAM analysis\n"
             "Bars past red = can't fit alongside prod embed/rerank/TEI at 32K ctx",
             fontweight="bold", pad=12, fontsize=13)

plt.tight_layout()
plt.savefig(OUT / "vram_coresidence.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote", OUT / "vram_coresidence.png")


# ─────────────────────────────────────────────────────────────
# Chart 3: b9948 → b10068 uplift on the two models we have both numbers for
# ─────────────────────────────────────────────────────────────
# metric, Ornith b9948, Ornith b10068, Gemma b9948, Gemma b10068
uplift_data = [
    ("Cold 12K\nprefill (tok/s)",       632,  896,  655,  650),
    ("Cold 12K\nwall time (s)",         22.8, 12.1, 21.5, 20.0),
    ("Peak decode\n(tok/s)",             50.0, 51.8, 50.0, 53.0),
    ("5K prefill\n(tok/s)",              830, 1213,  830,  971),
]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(uplift_data))
width = 0.2

ornith_before = [d[1] for d in uplift_data]
ornith_after = [d[2] for d in uplift_data]
gemma_before = [d[3] for d in uplift_data]
gemma_after = [d[4] for d in uplift_data]

# Instead of raw bars for 4 different metric units, plot as relative uplift %
def pct(before, after, invert=False):
    if invert:  # for wall time, lower is better; +% = improvement
        return (before - after) / before * 100
    return (after - before) / before * 100

ornith_pct = [
    pct(632, 896),
    pct(22.8, 12.1, invert=True),
    pct(50.0, 51.8),
    pct(830, 1213),
]
gemma_pct = [
    pct(655, 650),
    pct(21.5, 20.0, invert=True),
    pct(50.0, 53.0),
    pct(830, 971),
]

bars_o = ax.bar(x - width/2, ornith_pct, width, label="Ornith 9B (dense-GQA)",
                color="#2E7D32", edgecolor="black", linewidth=0.4)
bars_g = ax.bar(x + width/2, gemma_pct, width, label="Gemma 4 26B-A4B (MoE)",
                color="#6A1B9A", edgecolor="black", linewidth=0.4)

for b, v in zip(bars_o, ornith_pct):
    ax.text(b.get_x() + b.get_width()/2, v + 1.5 if v >= 0 else v - 3,
            f"{v:+.0f}%", ha="center", fontsize=9, fontweight="bold",
            color="#2E7D32")
for b, v in zip(bars_g, gemma_pct):
    ax.text(b.get_x() + b.get_width()/2, v + 1.5 if v >= 0 else v - 3,
            f"{v:+.0f}%", ha="center", fontsize=9, fontweight="bold",
            color="#6A1B9A")

ax.set_xticks(x)
ax.set_xticklabels([d[0] for d in uplift_data])
ax.axhline(y=0, color="black", linewidth=0.6)
ax.set_ylabel("Improvement over b9948 (%)", fontweight="bold")
ax.set_title("llama.cpp b9948 → b10068 uplift on B60\n"
             "XMX+oneDNN FA lifts dense-GQA (Ornith) much more than MoE (Gemma 4)",
             fontweight="bold", pad=12, fontsize=13)
ax.legend(fontsize=10, framealpha=0.95)
ax.grid(axis="y", linestyle=":", alpha=0.4)
ax.set_ylim(-5, max(ornith_pct + gemma_pct) * 1.25)

plt.tight_layout()
plt.savefig(OUT / "b10068_uplift.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote", OUT / "b10068_uplift.png")
