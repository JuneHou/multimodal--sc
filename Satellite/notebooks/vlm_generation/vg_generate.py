"""vg_generate — SERVER side (tinkercliffs, one A100). Reads vg_prompts.jsonl, loads one model once,
writes one PNG per prompt and a manifest line. Resumable: existing outputs are skipped.

    python vg_generate.py --model qwen  --layout persite --seed 0 [--limit 4]
    python vg_generate.py --model bagel --layout sheet   --seed 0 [--limit 4]

Models (plan): qwen = Qwen/Qwen-Image-Edit-2511 via diffusers.QwenImageEditPlusPipeline (bf16);
bagel = ByteDance-Seed/BAGEL-7B-MoT via the BAGEL repo's InterleaveInferencer (bf16).
Layouts: sheet = one contact-sheet image per prompt; persite = one image per site (TARGET first).
Qwen resizes every condition image to ~384x384 px for its vision encoder (diffusers source,
CONDITION_IMAGE_SIZE), so `persite` keeps more detail per parcel than one big sheet.
Outputs: outputs/<model>_<layout>/<prompt_id>_s<seed>.png ; manifest vg_outputs_<model>_<layout>.jsonl
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
MODEL_IDS = {"qwen": "Qwen/Qwen-Image-Edit-2511", "bagel": "ByteDance-Seed/BAGEL-7B-MoT"}
QWEN_SETTINGS = dict(true_cfg_scale=4.0, num_inference_steps=40, guidance_scale=1.0,
                     height=1024, width=1024, negative_prompt=" ")
BAGEL_SETTINGS = dict(cfg_text_scale=4.0, cfg_img_scale=2.0, cfg_interval=[0.0, 1.0], timestep_shift=3.0,
                      num_timesteps=50, cfg_renorm_min=0.0, cfg_renorm_type="text_channel",
                      image_shapes=(1024, 1024))


def load_prompts(path, layout):
    for line in open(path):
        p = json.loads(line)
        p["images"] = [p["sheet_image"]] if layout == "sheet" else p["persite_images"]
        p["prompt"] = p["prompt_sheet"] if layout == "sheet" else p["prompt_persite"]
        yield p


class Qwen:
    def __init__(self, path):
        import torch
        from diffusers import QwenImageEditPlusPipeline
        self.torch = torch
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(path, torch_dtype=torch.bfloat16).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.settings = QWEN_SETTINGS

    def __call__(self, images, prompt, seed):
        g = self.torch.Generator(device="cuda").manual_seed(seed)
        with self.torch.inference_mode():
            out = self.pipe(image=images, prompt=prompt, generator=g, num_images_per_prompt=1, **self.settings)
        return out.images[0]


class Bagel:
    """Loading follows the BAGEL repo's app.py (read 2026-09-08); the repo dir must be given."""
    def __init__(self, path, repo):
        import torch
        sys.path.insert(0, str(repo))
        from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
        from data.data_utils import add_special_tokens
        from data.transforms import ImageTransform
        from inferencer import InterleaveInferencer
        from modeling.autoencoder import load_ae
        from modeling.bagel import BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
        from modeling.qwen2 import Qwen2Tokenizer
        self.torch = torch
        llm_config = Qwen2Config.from_json_file(os.path.join(path, "llm_config.json"))
        llm_config.qk_norm = True; llm_config.tie_word_embeddings = False; llm_config.layer_module = "Qwen2MoTDecoderLayer"
        vit_config = SiglipVisionConfig.from_json_file(os.path.join(path, "vit_config.json"))
        vit_config.rope = False; vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
        vae_model, vae_config = load_ae(local_path=os.path.join(path, "ae.safetensors"))
        config = BagelConfig(visual_gen=True, visual_und=True, llm_config=llm_config, vit_config=vit_config,
                             vae_config=vae_config, vit_max_num_patch_per_side=70, connector_act="gelu_pytorch_tanh",
                             latent_patch_size=2, max_latent_size=64)
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
        tokenizer = Qwen2Tokenizer.from_pretrained(path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)
        device_map = infer_auto_device_map(model, max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
                                           no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"])
        same = ["language_model.model.embed_tokens", "time_embedder", "latent_pos_embed", "vae2llm", "llm2vae",
                "connector", "vit_pos_embed"]
        first = device_map.get("language_model.model.embed_tokens", 0)
        for k in same:
            if k in device_map:
                device_map[k] = first
        model = load_checkpoint_and_dispatch(model, checkpoint=os.path.join(path, "ema.safetensors"),
                                             device_map=device_map, offload_buffers=True, offload_folder="offload", dtype=torch.bfloat16,
                                             force_hooks=True).eval()
        self.inf = InterleaveInferencer(model=model, vae_model=vae_model, tokenizer=tokenizer,
                                        vae_transform=vae_transform, vit_transform=vit_transform,
                                        new_token_ids=new_token_ids)
        self.settings = BAGEL_SETTINGS

    def __call__(self, images, prompt, seed):
        self.torch.manual_seed(seed); self.torch.cuda.manual_seed_all(seed)
        with self.torch.inference_mode():
            if len(images) == 1:
                out = self.inf(image=images[0], text=prompt, think=False, **self.settings)["image"]
            else:
                res = self.inf.interleave_inference(list(images) + [prompt], think=False, **self.settings)
                out = [r for r in res if isinstance(r, Image.Image)][-1]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_IDS), required=True)
    ap.add_argument("--layout", choices=["sheet", "persite"], default="persite")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="first N prompts only (smoke test)")
    ap.add_argument("--prompts", default=str(HERE / "vg_prompts.jsonl"))
    ap.add_argument("--model-path", default=None, help="local model dir; default = HF id (needs cache/offline)")
    ap.add_argument("--bagel-repo", default=os.environ.get("BAGEL_REPO", str(HERE.parent / "Bagel")))
    a = ap.parse_args()
    out_dir = HERE / "outputs" / f"{a.model}_{a.layout}"; out_dir.mkdir(parents=True, exist_ok=True)
    manifest = HERE / f"vg_outputs_{a.model}_{a.layout}.jsonl"
    done = {json.loads(l)["prompt_id"] + f"_s{json.loads(l)['seed']}" for l in open(manifest)} if manifest.exists() else set()
    prompts = list(load_prompts(a.prompts, a.layout))
    if a.limit:
        keep = {}
        for p in prompts:                      # one prompt per (sensor, arm, band) cell, up to limit
            keep.setdefault((p["sensor"], p["arm"], p["band"]), p)
        prompts = list(keep.values())[:a.limit]
    todo = [p for p in prompts if f"{p['prompt_id']}_s{a.seed}" not in done]
    print(f"{len(prompts)} prompts, {len(todo)} to do, model {a.model} layout {a.layout} seed {a.seed}", flush=True)
    if not todo:
        return
    path = a.model_path or MODEL_IDS[a.model]
    t0 = time.time()
    gen = Qwen(path) if a.model == "qwen" else Bagel(path, a.bagel_repo)
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)
    with open(manifest, "a") as mf:
        for i, p in enumerate(todo):
            images = [Image.open(HERE / q).convert("RGB") for q in p["images"]]
            t1 = time.time(); rec = {"prompt_id": p["prompt_id"], "seed": a.seed, "model": a.model,
                                     "model_path": str(path), "layout": a.layout, "settings": gen.settings}
            try:
                img = gen(images, p["prompt"], a.seed)
                f = out_dir / f"{p['prompt_id']}_s{a.seed}.png"; img.save(f)
                rec.update(ok=True, file=str(f.relative_to(HERE)), size=list(img.size), seconds=round(time.time() - t1, 1))
            except Exception as e:                      # logged, never silent
                rec.update(ok=False, error=f"{type(e).__name__}: {e}", seconds=round(time.time() - t1, 1))
            mf.write(json.dumps(rec) + "\n"); mf.flush()
            print(f"[{i + 1}/{len(todo)}] {p['prompt_id']} {'ok' if rec['ok'] else 'FAILED'} {rec['seconds']}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
