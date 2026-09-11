"""Build 31_vlm_generation_gallery.ipynb: per treated site, the input sheets, the generated P10/P11
pictures from every model/layout/arm/band that exists, and the observed preview beside them.
Looking only, no numbers.  python build_nb31.py && jupyter nbconvert --execute --to notebook --inplace 31_vlm_generation_gallery.ipynb"""
import nbformat as nbf

nb = nbf.v4.new_notebook(); C = []
C.append(nbf.v4.new_markdown_cell("""# 31 — VLM direct generation: gallery

Plan `quirky-growing-wombat` (2026-09-08). Inputs are the collaborator's preview pictures
(notebooks 07/08), assembled into contact sheets by `vg_prepare.py`; outputs are whatever the model
returned (`vg_generate.py` on tinkercliffs), collected by `vg_collect.py`. **No evaluation here.**"""))
C.append(nbf.v4.new_code_cell("""import json, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import vg_lib as vg
P = pd.read_csv(vg.PROMPTS_CSV)
G = pd.read_csv(vg.VG / "vg_generated_index.csv") if (vg.VG / "vg_generated_index.csv").exists() else pd.DataFrame()
print(len(P), "prompts;", len(G), "generated outputs" if len(G) else "generated outputs (none yet)")
if len(G): print(G.groupby(["model","layout","sensor","arm","band","target"]).ok.sum().to_string())"""))
C.append(nbf.v4.new_markdown_cell("## 1. One full prompt per arm (band text on), as sent"))
C.append(nbf.v4.new_code_cell("""J = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(vg.PROMPTS_JSONL)}
for pid in ("s2_treatment_0001_shc_P11_band1", "s2_treatment_0001_scm_P11_band1"):
    print("=====", pid, "(persite layout) =====\\n" + J[pid]["prompt_persite"] + "\\n")"""))
C.append(nbf.v4.new_markdown_cell("## 2. Input sheets and outputs per treated site\n\nLeft: the SCM sheet for P11 (rows = TARGET then SIMILAR 1–5, columns = P09, P10). Then the observed P10 and P11 previews, then every generated picture that exists (title = model / layout / arm / band / target)."))
C.append(nbf.v4.new_code_cell("""def show_site(sensor, site):
    sheet = Image.open(vg.VG / P[(P.sensor==sensor)&(P.target_site==site)&(P.arm=="scm")&(P.target=="P11")&(P.band==0)].sheet_image.iloc[0])
    obs = {t: Image.open(vg.PREVIEWS / f"{site}_{sensor}_{t}.png") for t in ("P10","P11")}
    gen = G[(G.sensor==sensor)&(G.target_site==site)&(G.ok==True)] if len(G) else G
    n = 3 + len(gen)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3.4))
    axes[0].imshow(sheet); axes[0].set_title("SCM sheet P11"); axes[0].axis("off")
    for ax, t in zip(axes[1:3], ("P10","P11")):
        ax.imshow(obs[t]); ax.set_title(f"observed {t}"); ax.axis("off")
    for ax, r in zip(axes[3:], gen.itertuples()):
        ax.imshow(Image.open(vg.VG / r.file)); ax.set_title(f"{r.model}/{r.layout}\\n{r.arm} band{r.band} {r.target}", fontsize=8); ax.axis("off")
    fig.suptitle(f"{sensor} {site}"); plt.tight_layout(); plt.show()
for sensor in ("sentinel1", "sentinel2"):
    for site in sorted(P[P.sensor==sensor].target_site.unique()):
        show_site(sensor, site)"""))
nb["cells"] = C; nbf.write(nb, "31_vlm_generation_gallery.ipynb"); print("wrote 31_vlm_generation_gallery.ipynb")
