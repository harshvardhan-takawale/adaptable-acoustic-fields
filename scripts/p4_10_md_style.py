"""P4-10 meeting-day figure style (D211): 16:9, >= 14 pt, PNG 200 dpi + PDF + JSON sidecar; MP4 (H.264) + GIF for animations. Entity colours fixed (FIGURE_MANIFEST)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("outputs/p4_10/meeting")
COL = {"truth": "#111111", "R0": "#7a7a7a", "R1": "#2a78d6", "R1-S": "#1baf7a", "R1-M": "#c64f97", "RCP": "#eda100", "S-1": "#e8743b", "ignore": "#b00020", "KRR": "#8a6d3b"}
NAME = {"R0_120K": "R0", "R1_150K": "R1", "R1S_150K": "R1-S", "R1M_150K": "R1-M"}
plt.rcParams.update({"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14, "axes.edgecolor": "#888888",
                     "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})


def new(w=16.0, h=9.0):
    return plt.figure(figsize=(w, h))


def save(fig, name, meta):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / (name + ".png"), dpi=200)
    fig.savefig(OUT / (name + ".pdf"))
    (OUT / (name + ".json")).write_text(json.dumps(meta, indent=1))
    plt.close(fig)
    print("wrote", name, flush=True)


def outline(ax, verts, lw=2.2):
    v = [tuple(p) for p in verts] + [tuple(verts[0])]
    ax.plot([p[0] for p in v], [p[1] for p in v], color="black", lw=lw, solid_joinstyle="miter", zorder=6)
    ax.set_aspect("equal")
    ax.axis("off")


def star(ax, src, ms=17):
    ax.plot(src[0], src[1], "*", color="white", ms=ms, markeredgecolor="black", markeredgewidth=1.1, zorder=8)


def movie(frames, name, seconds=12):
    """frames: list of RGB uint8 arrays -> MP4 (H.264, yuv420p) and GIF"""
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image
    h, w = frames[0].shape[0] // 2 * 2, frames[0].shape[1] // 2 * 2
    fps = max(1.0, len(frames) / float(seconds))
    wr = imageio_ffmpeg.write_frames(str(OUT / (name + ".mp4")), (w, h), fps=fps, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=1, quality=8, output_params=["-r", "24"])
    wr.send(None)
    for f in frames:
        wr.send(np.ascontiguousarray(f[:h, :w]))
    wr.close()
    im = [Image.fromarray(f[:h, :w]).convert("P", palette=Image.ADAPTIVE) for f in frames]
    im[0].save(OUT / (name + ".gif"), save_all=True, append_images=im[1:], duration=int(1000 / fps), loop=0)
    print("wrote movie", name, len(frames), "frames", flush=True)
