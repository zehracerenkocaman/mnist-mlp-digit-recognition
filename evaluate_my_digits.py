"""
Kendi el yazısı okul numarası rakamlarını eğitilmiş MLP ile test eder.
Dosya adı formatı: a_b.png  (a = numaradaki sıra, b = rakamın gerçek değeri)

"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
HIDDEN_SIZES = [256, 128]  # train.py ile AYNI olmalı


def build_mlp(hidden_sizes):
    layers, in_features = [nn.Flatten()], 28 * 28
    for h in hidden_sizes:
        layers += [nn.Linear(in_features, h), nn.ReLU()]
        in_features = h
    layers.append(nn.Linear(in_features, 10))
    return nn.Sequential(*layers)


def preprocess(path):
    """Taranmış görüntüyü MNIST formatına getirir."""
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)

    # 1) MNIST: siyah arka plan, beyaz rakam -> kağıt açık renkliyse ters çevir
    if np.median(arr) > 127:
        arr = 255.0 - arr

    # 2) Arka plan seviyesini çıkar, [0,1]'e ger, hafif gürültüyü temizle
    arr = arr - np.median(arr)
    arr = np.clip(arr, 0, None)
    if arr.max() > 0:
        arr = arr / arr.max()
    arr[arr < 0.25] = 0.0

    # 3) Rakamın sınırlayıcı kutusuna kırp
    ys, xs = np.nonzero(arr)
    if len(ys) == 0:
        raise ValueError(f"{path}: rakam bulunamadı")
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # 4) En-boy oranını koruyarak uzun kenarı 20 piksele küçült (MNIST böyle hazırlanmış)
    h, w = arr.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    img = Image.fromarray((arr * 255).astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)

    # 5) 28x28 tuvale ortala
    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = np.asarray(img, dtype=np.float32) / 255.0

    # 6) Kütle merkezini (28x28'in merkezine) kaydır - MNIST'teki gibi
    total = canvas.sum()
    if total > 0:
        cy = (canvas.sum(1) * np.arange(28)).sum() / total
        cx = (canvas.sum(0) * np.arange(28)).sum() / total
        shift_y, shift_x = int(round(13.5 - cy)), int(round(13.5 - cx))
        shifted = np.zeros_like(canvas)
        ys0, ys1 = max(0, shift_y), min(28, 28 + shift_y)
        xs0, xs1 = max(0, shift_x), min(28, 28 + shift_x)
        shifted[ys0:ys1, xs0:xs1] = canvas[ys0 - shift_y:ys1 - shift_y, xs0 - shift_x:xs1 - shift_x]
        canvas = shifted
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="my_digits", help="a_b.png dosyalarının klasörü")
    ap.add_argument("--model", default="results/mlp_model.pt")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    model = build_mlp(HIDDEN_SIZES)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    files = sorted(
        (p for p in Path(args.images).iterdir()
         if p.suffix.lower() in {".png", ".jpg", ".jpeg"}),
        key=lambda p: int(p.stem.split("_")[0]),
    )
    if not files:
        raise SystemExit(f"{args.images} klasöründe görüntü bulunamadı")
    rows = []
    for f in files:
        pos, true = (int(v) for v in f.stem.split("_"))
        img = preprocess(f)
        x = torch.from_numpy((img - MNIST_MEAN) / MNIST_STD).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
        pred = int(probs.argmax())
        rows.append({"file": f.name, "position": pos, "true": true, "pred": pred,
                     "confidence": float(probs[pred]), "correct": pred == true,
                     "img": img})

    # ---- Rapor ----
    print(f"{'Dosya':<10}{'Gerçek':>8}{'Tahmin':>8}{'Güven':>9}  Sonuç")
    for r in rows:
        print(f"{r['file']:<10}{r['true']:>8}{r['pred']:>8}{r['confidence']:>9.2%}  "
              f"{'OK' if r['correct'] else 'YANLIŞ'}")

    n_correct = sum(r["correct"] for r in rows)
    acc = n_correct / len(rows)
    print(f"\nDoğruluk: {n_correct}/{len(rows)} = {acc:.2%}")
    print("Gerçek numara :", "".join(str(r["true"]) for r in rows))
    print("Okunan numara :", "".join(str(r["pred"]) for r in rows))

    with open(out / "kendi_el_yazim_sonuc.json", "w", encoding="utf-8") as fp:
        json.dump({"accuracy": acc,
                   "results": [{k: v for k, v in r.items() if k != "img"} for r in rows]},
                  fp, indent=2, ensure_ascii=False)

    # ---- Çıktı: işlenmiş 28x28 görüntüler + tahminler ----
    fig, axes = plt.subplots(1, len(rows), figsize=(1.6 * len(rows), 2.3))
    for ax, r in zip(np.atleast_1d(axes), rows):
        ax.imshow(r["img"], cmap="gray")
        ax.set_title(f"gerçek {r['true']}\ntahmin {r['pred']}\n{r['confidence']:.0%}",
                     color="green" if r["correct"] else "red", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out / "kendi_el_yazim_tahminler.png", dpi=150)
    print("Kaydedildi:", out / "kendi_el_yazim_tahminler.png")


if __name__ == "__main__":
    main()
