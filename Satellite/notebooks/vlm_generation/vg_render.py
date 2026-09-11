"""vg_render — the collaborator's preview pictures, verbatim, plus tiles and contact sheets.

Pictures are NOT ours: `s2_preview` is notebook 07 `create_rgb_preview` (Sentinel-2 natural colour,
bands 3,2,1 = red,green,blue; ONE 2nd-98th percentile stretch over the finite values of the three
bands together; non-finite -> 0) and `s1_preview` is notebook 08 `create_sentinel1_preview` (VV, VH,
VV-VH as red, green, blue, EACH band 2nd-98th percentile stretched; non-finite -> 0), both from
`data/scripts/`. Only the file reading differs: they use rasterio band 1..3, we take the same bands
from the (H, W, C) array `panel_monthly.read_chip` returns.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def percentile_stretch(array, lower_percentile=2, upper_percentile=98):
    """Notebook 08 `percentile_stretch`, unchanged."""
    array = array.astype("float32")
    valid = array[np.isfinite(array)]
    if valid.size == 0:
        return np.zeros_like(array, dtype="float32")
    lower = np.nanpercentile(valid, lower_percentile)
    upper = np.nanpercentile(valid, upper_percentile)
    if upper <= lower:
        upper = lower + 1e-6
    stretched = np.clip((array - lower) / (upper - lower), 0, 1)
    stretched[~np.isfinite(stretched)] = 0
    return stretched


def s2_preview(chip):
    """Notebook 07 `create_rgb_preview` on an (H, W, 8) chip: bands B2, B3, B4 = blue, green, red."""
    blue, green, red = chip[..., 0], chip[..., 1], chip[..., 2]
    rgb = np.stack([red, green, blue], axis=-1).astype("float32")
    valid_values = rgb[np.isfinite(rgb)]
    if valid_values.size == 0:
        return None
    lower = np.nanpercentile(valid_values, 2)
    upper = np.nanpercentile(valid_values, 98)
    if upper <= lower:
        upper = lower + 1e-6
    stretched = np.clip((rgb - lower) / (upper - lower), 0, 1)
    stretched[~np.isfinite(stretched)] = 0
    return stretched


def s1_preview(chip):
    """Notebook 08 `create_sentinel1_preview` on an (H, W, 3) chip: VV, VH, VV-VH."""
    vv, vh, difference = chip[..., 0], chip[..., 1], chip[..., 2]
    return np.stack([percentile_stretch(vv), percentile_stretch(vh), percentile_stretch(difference)],
                    axis=-1)


PREVIEW = {"sentinel2": s2_preview, "sentinel1": s1_preview}


def to_image(arr01):
    """(H, W, 3) in [0, 1] -> PIL RGB uint8 (round half up, as imshow/savefig would display)."""
    return Image.fromarray(np.clip(np.rint(arr01 * 255), 0, 255).astype(np.uint8), "RGB")


def font(size=13):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def tile(img, label, scale=2, strip=22, pad=4):
    """Nearest-neighbour upsampled picture with a one-line label strip above it."""
    w, h = img.size
    up = img.resize((w * scale, h * scale), Image.NEAREST)
    out = Image.new("RGB", (w * scale, h * scale + strip), "white")
    ImageDraw.Draw(out).text((pad, 3), label, fill="black", font=font(13))
    out.paste(up, (0, strip))
    return out


def contact_sheet(rows, gap=8, margin=10):
    """rows = [(row_label, [(PIL image, period_label), ...]), ...] -> one PIL sheet.
    Tiles are labelled '<row_label> <period_label>'; rows = sites, columns = months in time order."""
    tiles = [[tile(im, f"{rl} {pl_}") for im, pl_ in items] for rl, items in rows]
    tw, th = tiles[0][0].size
    ncol = max(len(r) for r in tiles)
    W = margin * 2 + ncol * tw + (ncol - 1) * gap
    H = margin * 2 + len(tiles) * th + (len(tiles) - 1) * gap
    sheet = Image.new("RGB", (W, H), "white")
    for i, r in enumerate(tiles):
        for j, t in enumerate(r):
            sheet.paste(t, (margin + j * (tw + gap), margin + i * (th + gap)))
    return sheet
