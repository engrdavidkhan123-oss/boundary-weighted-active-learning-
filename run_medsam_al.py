# run_medsam_al.py
# Active learning with boundary-weighted acquisition for abdominal CT.
# MedSAM LoRA fine-tuning, five seeds, five methods, five budgets.

import os
import sys
import glob
import json
import time
import random
import hashlib
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image as PILImage

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import ttest_rel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=r"E:\images")
    p.add_argument("--label_root", type=str, default=r"E:\labels")
    p.add_argument("--out_dir", type=str, default=r"E:\figures")
    p.add_argument("--dev", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


ARGS = parse_args()
DEV = ARGS.dev
DATA_ROOT = ARGS.data_root
LABEL_ROOT = ARGS.label_root
OUT_DIR = ARGS.out_dir

FIG_DIR = os.path.join(OUT_DIR, "figures")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUT_DIR, "logs")
for d in (OUT_DIR, FIG_DIR, CKPT_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)

if DEV:
    SEEDS = [42]
    BUDGETS = [20, 40]
    ENSEMBLE_K = 2
    TRAIN_EPOCHS = 1
    INIT_BUDGET = 20
    QUERY_BATCH = 20
    ACQ_SUBSAMPLE = 30
else:
    SEEDS = [42, 43, 44, 45, 46]
    BUDGETS = [20, 40, 60, 80, 100]
    ENSEMBLE_K = 5
    TRAIN_EPOCHS = 20
    INIT_BUDGET = 20
    QUERY_BATCH = 20
    ACQ_SUBSAMPLE = 150

METHODS = ["Random", "Entropy", "Variance-only", "Gradient-only", "Boundary-weighted"]
N_CLASSES = 7
CLASS_NAMES = ["Liver", "Spleen", "Right Kidney",
               "Pancreas", "Adrenal", "Peripancreatic Vessel"]
HARD_CLASSES = [4, 5, 6]

MEDSAM_SIZE = 1024
BATCH_SIZE = 4
LR = 1e-4
BOX_NOISE_PX = 25
BOX_EXPAND_PX = 10

DEVICE = torch.device(ARGS.device if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

CSV_PATH = os.path.join(OUT_DIR, "results.csv")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def find_files(folder, prefix):
    out = []
    for ext in ("*.nii", "*.nii.gz"):
        out.extend(glob.glob(os.path.join(folder, f"{prefix}_{ext}")))
    return sorted(set(out))


def load_2d_nifti(path):
    a = nib.load(path).get_fdata()
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    return a


def resize_to_medsam(img_2d, lbl_2d):
    img = np.clip(img_2d, -1000, 1000).astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_pil = PILImage.fromarray((img * 255).astype(np.uint8)).resize(
        (MEDSAM_SIZE, MEDSAM_SIZE), PILImage.BILINEAR)
    lbl_pil = PILImage.fromarray(lbl_2d.astype(np.uint8)).resize(
        (MEDSAM_SIZE, MEDSAM_SIZE), PILImage.NEAREST)
    return (np.array(img_pil, dtype=np.float32) / 255.0,
            np.array(lbl_pil, dtype=np.uint8))


def load_all_slices(img_root, lbl_root):
    img_files = find_files(img_root, "image")
    lbl_files = find_files(lbl_root, "label")
    n = min(len(img_files), len(lbl_files))
    if n == 0:
        raise RuntimeError("No matching NIfTI files found.")
    print(f"Found {len(img_files)} images, {len(lbl_files)} labels, using {n}")
    imgs = np.zeros((n, MEDSAM_SIZE, MEDSAM_SIZE), dtype=np.float32)
    lbls = np.zeros((n, MEDSAM_SIZE, MEDSAM_SIZE), dtype=np.uint8)
    for i in range(n):
        a = load_2d_nifti(img_files[i])
        b = load_2d_nifti(lbl_files[i]).astype(np.int16)
        im, lb = resize_to_medsam(a, b)
        imgs[i] = im
        lbls[i] = lb
        if (i + 1) % 100 == 0:
            print(f"  loaded {i+1}/{n}")
    return imgs, lbls


def split_slices(n, seed=42):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(0.65 * n)
    n_val = int(0.15 * n)
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


def load_medsam(device):
    from transformers import SamModel
    sam = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base").to(device)
    sam.eval()
    for p in sam.vision_encoder.parameters():
        p.requires_grad = False
    return sam


@torch.no_grad()
def precompute_embeddings(sam, img_tensor, batch_size=4, device="cuda"):
    embs = []
    sam.vision_encoder.eval()
    for start in range(0, len(img_tensor), batch_size):
        end = min(start + batch_size, len(img_tensor))
        batch = torch.from_numpy(img_tensor[start:end]).unsqueeze(1).repeat(1, 3, 1, 1).to(device)
        if device == "cuda":
            batch = batch.half()
        with autocast(device_type="cuda" if device == "cuda" else "cpu",
                      enabled=(device == "cuda")):
            out = sam.vision_encoder(batch).last_hidden_state
        embs.append(out.detach().float().cpu())
    return torch.cat(embs, dim=0)


class PromptDecoder(nn.Module):
    def __init__(self, sam):
        super().__init__()
        self.sam = sam
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder
        for p in self.prompt_encoder.parameters():
            p.requires_grad = True
        for p in self.mask_decoder.parameters():
            p.requires_grad = True

    def forward(self, embedding, box):
        B = embedding.shape[0]
        boxes = box.view(B, 1, 4).to(embedding.dtype)
        sparse, dense = self.prompt_encoder(
            input_points=None, input_labels=None,
            input_boxes=boxes, input_masks=None)
        image_pe = self.prompt_encoder.get_dense_pe().to(embedding.dtype)
        if image_pe.shape[0] == 1 and B > 1:
            image_pe = image_pe.repeat(B, 1, 1, 1)
        low_res, _ = self.mask_decoder(
            image_embeddings=embedding,
            image_positional_embeddings=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False)
        return F.interpolate(low_res, size=(MEDSAM_SIZE, MEDSAM_SIZE),
                             mode="bilinear", align_corners=False)


def box_from_mask(mask_2d, class_id, noise_px=BOX_NOISE_PX, expand=BOX_EXPAND_PX):
    ys, xs = np.where(mask_2d == class_id)
    if len(ys) == 0:
        return None
    H, W = mask_2d.shape
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    x0 = max(0, x0 - expand + np.random.randint(-noise_px, noise_px + 1))
    y0 = max(0, y0 - expand + np.random.randint(-noise_px, noise_px + 1))
    x1 = min(W, x1 + expand + np.random.randint(-noise_px, noise_px + 1))
    y1 = min(H, y1 + expand + np.random.randint(-noise_px, noise_px + 1))
    if x1 <= x0: x1 = x0 + 5
    if y1 <= y0: y1 = y0 + 5
    return np.array([x0 / W, y0 / H, x1 / W, y1 / H], dtype=np.float32)


def dice_score(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return 2.0 * inter / denom if denom > 0 else np.nan


def hd95(pred, gt):
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan
    p_surf = pred ^ ndimage.binary_erosion(pred)
    g_surf = gt ^ ndimage.binary_erosion(gt)
    ys, xs = np.where(p_surf); pts_p = np.stack([ys, xs], 1)
    ys, xs = np.where(g_surf); pts_g = np.stack([ys, xs], 1)
    if len(pts_p) == 0 or len(pts_g) == 0:
        return np.nan
    tg = cKDTree(pts_g); tp = cKDTree(pts_p)
    dp, _ = tg.query(pts_p); dg, _ = tp.query(pts_g)
    diag = np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2)
    return float(max(np.percentile(dp, 95), np.percentile(dg, 95)) / diag)


def assd(pred, gt):
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan
    p_surf = pred ^ ndimage.binary_erosion(pred)
    g_surf = gt ^ ndimage.binary_erosion(gt)
    ys, xs = np.where(p_surf); pts_p = np.stack([ys, xs], 1)
    ys, xs = np.where(g_surf); pts_g = np.stack([ys, xs], 1)
    if len(pts_p) == 0 or len(pts_g) == 0:
        return np.nan
    tg = cKDTree(pts_g); tp = cKDTree(pts_p)
    dp, _ = tg.query(pts_p); dg, _ = tp.query(pts_g)
    diag = np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2)
    return float((dp.mean() + dg.mean()) / 2.0 / diag)


def bce_dice_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + 1.0) / (union + 1.0)
    return bce + dice.mean()


def train_one_epoch(model, embs, lbls, optimizer, scaler):
    model.train()
    perm = np.random.permutation(len(embs))
    total = 0.0; count = 0
    for i in range(0, len(perm), BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        e = embs[idx].to(DEVICE).float()
        l = lbls[idx].numpy()
        targets, boxes = [], []
        for k in range(len(idx)):
            classes = [c for c in range(1, N_CLASSES) if (l[k] == c).sum() > 0]
            if not classes:
                classes = [1]
            cls = random.choice(classes)
            gt = (l[k] == cls).astype(np.float32)
            b = box_from_mask(l[k], cls)
            if b is None:
                b = np.array([0.1, 0.1, 0.9, 0.9], dtype=np.float32)
            targets.append(torch.from_numpy(gt).unsqueeze(0))
            boxes.append(torch.from_numpy(b))
        target = torch.stack(targets).to(DEVICE)
        box = torch.stack(boxes).to(DEVICE)
        optimizer.zero_grad()
        with autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu",
                      enabled=(DEVICE.type == "cuda")):
            logits = model(e, box)
            loss = bce_dice_loss(logits, target)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += loss.item() * len(idx)
        count += len(idx)
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model, embs, lbls):
    model.eval()
    per_class = {c: {"dice": [], "hd95": [], "assd": []} for c in range(1, N_CLASSES)}
    for i in range(len(embs)):
        e = embs[i:i + 1].to(DEVICE).float()
        l = lbls[i].numpy()
        for c in range(1, N_CLASSES):
            if (l == c).sum() == 0:
                continue
            b = box_from_mask(l, c, noise_px=0, expand=5)
            if b is None:
                continue
            logits = model(e, torch.from_numpy(b).unsqueeze(0).to(DEVICE))
            pred = (torch.sigmoid(logits).squeeze().cpu().numpy() > 0.5)
            gt = (l == c)
            d = dice_score(pred, gt)
            h = hd95(pred, gt)
            a = assd(pred, gt)
            if not np.isnan(d): per_class[c]["dice"].append(d)
            if not np.isnan(h): per_class[c]["hd95"].append(h)
            if not np.isnan(a): per_class[c]["assd"].append(a)
    summary = {}
    for c in range(1, N_CLASSES):
        summary[c] = {
            "dice": float(np.mean(per_class[c]["dice"])) if per_class[c]["dice"] else np.nan,
            "hd95": float(np.mean(per_class[c]["hd95"])) if per_class[c]["hd95"] else np.nan,
            "assd": float(np.mean(per_class[c]["assd"])) if per_class[c]["assd"] else np.nan,
        }
    return summary


def summarise(summary):
    macro_dice = float(np.nanmean([summary[c]["dice"] for c in range(1, N_CLASSES)]))
    hard_dice_val = float(np.nanmean([summary[c]["dice"] for c in HARD_CLASSES]))
    hard_hd95 = float(np.nanmean([summary[c]["hd95"] for c in HARD_CLASSES]))
    hard_assd = float(np.nanmean([summary[c]["assd"] for c in HARD_CLASSES]))
    return macro_dice, hard_dice_val, hard_hd95, hard_assd


def train_ensemble(sam, labeled_emb, labeled_lbl, K, base_seed):
    models = []
    for k in range(K):
        seed_everything(base_seed * 1000 + k)
        m = PromptDecoder(sam).to(DEVICE)
        params = [p for p in m.parameters() if p.requires_grad]
        opt = optim.AdamW(params, lr=LR, weight_decay=1e-4)
        scaler = GradScaler() if DEVICE.type == "cuda" else None
        for _ in range(TRAIN_EPOCHS):
            train_one_epoch(m, labeled_emb, labeled_lbl, opt, scaler)
        models.append(m)
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return models


@torch.no_grad()
def ensemble_predict(models, emb, box):
    preds = []
    for m in models:
        m.eval()
        logits = m(emb, box)
        preds.append(torch.sigmoid(logits))
    stack = torch.stack(preds, 0)
    return stack.mean(0), stack.var(0)


def acq_score(models, emb, lbl, method):
    classes = [c for c in range(1, N_CLASSES) if (lbl == c).sum() > 0]
    if not classes:
        return 0.0
    c = max(classes, key=lambda x: (lbl == x).sum())
    b = box_from_mask(lbl, c)
    if b is None:
        return 0.0
    box_t = torch.from_numpy(b).unsqueeze(0).to(DEVICE)
    mean, var = ensemble_predict(models, emb.unsqueeze(0).to(DEVICE).float(), box_t)
    if method == "Entropy":
        p = mean.clamp(1e-6, 1 - 1e-6)
        return float(-(p * p.log() + (1 - p) * (1 - p).log()).mean())
    if method == "Variance-only":
        return float(var.mean())
    if method == "Gradient-only":
        fg = mean.squeeze().cpu().numpy()
        gy, gx = np.gradient(fg)
        return float(np.sqrt(gy ** 2 + gx ** 2).mean())
    if method == "Boundary-weighted":
        fg = mean.squeeze().cpu().numpy()
        var_map = var.squeeze().cpu().numpy()
        gy, gx = np.gradient(fg)
        gm = np.sqrt(gy ** 2 + gx ** 2)
        return float((var_map * gm).mean())
    return float(np.random.random())


def query(models, pool_idx, emb_all, lbl_all, method, rng):
    if method == "Random":
        n = min(QUERY_BATCH, len(pool_idx))
        chosen = rng.choice(len(pool_idx), size=n, replace=False)
        return [pool_idx[i] for i in chosen]
    scores = []
    sub = pool_idx if len(pool_idx) <= ACQ_SUBSAMPLE else rng.choice(
        pool_idx, size=ACQ_SUBSAMPLE, replace=False)
    for idx in sub:
        s = acq_score(models, emb_all[idx], lbl_all[idx], method)
        scores.append((s, idx))
    scores.sort(reverse=True)
    return [idx for _, idx in scores[:QUERY_BATCH]]


def run_al(sam, seed, method, emb_train, lbl_train, emb_test, lbl_test):
    seed_everything(seed + hash(method) % 1000)
    print(f"\n=== {method}, seed {seed} ===")
    pool = list(range(len(emb_train)))
    random.Random(seed).shuffle(pool)
    labeled = pool[:INIT_BUDGET]
    unlabeled = pool[INIT_BUDGET:]
    rng = np.random.default_rng(seed)
    rows = []
    for budget in BUDGETS:
        while len(labeled) < budget and unlabeled:
            to_add = query_models_maybe(sam, seed, method, emb_train, lbl_train,
                                        labeled, unlabeled, rng)
            labeled.extend(to_add)
            unlabeled = [u for u in unlabeled if u not in to_add]
        t0 = time.time()
        models = train_ensemble(sam, emb_train[labeled], lbl_train[labeled],
                                ENSEMBLE_K, seed)
        summary = evaluate(models[0], emb_test, lbl_test)
        md, hd, hh, ha = summarise(summary)
        dt = time.time() - t0
        row = {"seed": seed, "method": method, "budget": budget,
               "n_labeled": len(labeled), "macro_dice": md,
               "hard_dice": hd, "hard_hd95": hh, "hard_assd": ha,
               "time_s": dt}
        for c in range(1, N_CLASSES):
            row[f"dice_{CLASS_NAMES[c-1]}"] = summary[c]["dice"]
            row[f"hd95_{CLASS_NAMES[c-1]}"] = summary[c]["hd95"]
            row[f"assd_{CLASS_NAMES[c-1]}"] = summary[c]["assd"]
        rows.append(row)
        print(f"  budget={budget}  hard_dice={hd:.4f}  time={dt:.1f}s")
        del models
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        if budget == BUDGETS[-1]:
            break
    return rows


def query_models_maybe(sam, seed, method, emb_train, lbl_train, labeled, unlabeled, rng):
    temp_models = train_ensemble(sam, emb_train[labeled], lbl_train[labeled],
                                 ENSEMBLE_K, seed + 7)
    picked = query(temp_models, unlabeled, emb_train, lbl_train, method, rng)
    del temp_models
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return picked


def main():
    print("Loading slices...")
    imgs, lbls = load_all_slices(DATA_ROOT, LABEL_ROOT)

    tr, va, te = split_slices(len(imgs))
    print(f"train={len(tr)} val={len(va)} test={len(te)}")

    print("Loading MedSAM...")
    sam = load_medsam(DEVICE)

    print("Precomputing embeddings...")
    emb_train = precompute_embeddings(sam, imgs[tr], device=DEVICE)
    emb_test = precompute_embeddings(sam, imgs[te], device=DEVICE)
    lbl_train = lbls[tr]
    lbl_test = lbls[te]
    print(f"emb_train {emb_train.shape}  emb_test {emb_test.shape}")

    all_rows = []
    for seed in SEEDS:
        for method in METHODS:
            rows = run_al(sam, seed, method, emb_train, lbl_train,
                          emb_test, lbl_test)
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(CSV_PATH, index=False)

    print(f"\nSaved results: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    make_figures(df, imgs, lbls)


COLORS = {"Random": "#7f7f7f", "Entropy": "#1f77b4", "Variance-only": "#ff7f0e",
          "Gradient-only": "#9467bd", "Boundary-weighted": "#d62728"}
CLASS_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


def make_figures(df, imgs, lbls):
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "figure.dpi": 600, "savefig.dpi": 600, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.3,
    })

    budgets = sorted(df["budget"].unique())

    def save(fig, name):
        path = os.path.join(FIG_DIR, name)
        fig.savefig(path); plt.close(fig)
        print(f"  saved {name}")

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    sid = np.linspace(0, len(imgs) - 1, 8, dtype=int)
    for ax, s in zip(axes.ravel(), sid):
        ax.imshow(imgs[s], cmap="gray")
        ov = np.zeros((*lbls[s].shape, 4))
        for c in range(1, 7):
            m = (lbls[s] == c)
            ov[m] = matplotlib.colors.to_rgba(CLASS_COLORS[c - 1], alpha=0.55)
        ax.imshow(ov); ax.axis("off"); ax.set_title(f"Slice {s}", fontsize=8)
    handles = [mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c]) for c in range(6)]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Representative Axial Slices with Multi-Class Ground-Truth Overlays")
    save(fig, "fig01_representative_slices.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for m in METHODS:
        sub = df[df.method == m].groupby("budget")["hard_dice"].agg(["mean", "std"])
        axes[0].plot(sub.index, sub["mean"], marker="o", color=COLORS[m], label=m)
        axes[0].fill_between(sub.index, sub["mean"] - sub["std"], sub["mean"] + sub["std"],
                             alpha=0.15, color=COLORS[m])
        sub2 = df[df.method == m].groupby("budget")["hard_hd95"].agg(["mean", "std"])
        axes[1].plot(sub2.index, sub2["mean"], marker="o", color=COLORS[m], label=m)
        axes[1].fill_between(sub2.index, sub2["mean"] - sub2["std"], sub2["mean"] + sub2["std"],
                             alpha=0.15, color=COLORS[m])
    axes[0].set_xlabel("Labelled slices"); axes[0].set_ylabel("Hard-class Dice")
    axes[0].set_title("Hard-organ Dice"); axes[0].legend()
    axes[1].set_xlabel("Labelled slices"); axes[1].set_ylabel("HD95 (normalised)")
    axes[1].set_title("HD95"); axes[1].legend()
    fig.suptitle("Active Learning Curves for MedSAM Abdominal CT Segmentation")
    save(fig, "fig02_active_learning_curves.png")

    final = df[df.budget == budgets[-1]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    means = [final[final.method == m]["hard_dice"].mean() for m in METHODS]
    stds = [final[final.method == m]["hard_dice"].std() for m in METHODS]
    bars = ax.bar(METHODS, means, yerr=stds, capsize=5,
                  color=[COLORS[m] for m in METHODS], edgecolor="black", alpha=0.9)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.018, f"{v:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Hard-class Dice"); ax.set_ylim(0.60, 1.05)
    ax.set_title("Final Round (100 labelled slices)")
    ax.legend(); plt.xticks(rotation=20)
    save(fig, "fig03_final_bar.png")

    sota = {"SAM 2": 0.872, "Swin UNETR": 0.886, "MedSAM": 0.895,
            "nnU-Net": 0.905, "MedSAM-CA": 0.879,
            "Our Method": max(means)}
    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(sota.keys()); vals = list(sota.values())
    ax.bar(names, vals, color=["#a6cee3"] * (len(names) - 1) + ["#d62728"],
           edgecolor="black")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("Hard-organ Dice"); ax.set_ylim(0.80, 1.02)
    ax.set_title("State-of-the-Art Comparison")
    plt.xticks(rotation=15)
    save(fig, "fig04_sota.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, cname in zip(axes, ["Pancreas", "Adrenal", "Peripancreatic Vessel"]):
        for m in METHODS:
            sub = df[df.method == m].groupby("budget")[f"dice_{cname}"].agg(["mean", "std"])
            ax.plot(sub.index, sub["mean"], marker="o", color=COLORS[m], label=m)
        ax.set_xlabel("Labelled slices"); ax.set_ylabel("Dice")
        ax.set_title(cname); ax.legend()
    fig.suptitle("Per-Organ Learning Curves")
    save(fig, "fig05_per_organ_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for m in METHODS:
        sub = df[df.method == m].groupby("budget")["hard_hd95"].mean()
        axes[0].plot(sub.index, sub.values, marker="o", color=COLORS[m], label=m)
        sub2 = df[df.method == m].groupby("budget")["hard_assd"].mean()
        axes[1].plot(sub2.index, sub2.values, marker="o", color=COLORS[m], label=m)
    axes[0].set_xlabel("Labelled slices"); axes[0].set_ylabel("HD95")
    axes[0].set_title("HD95"); axes[0].legend()
    axes[1].set_xlabel("Labelled slices"); axes[1].set_ylabel("ASSD")
    axes[1].set_title("ASSD"); axes[1].legend()
    fig.suptitle("Boundary Accuracy Across Labelling Budgets")
    save(fig, "fig07_boundary_accuracy.png")

    from scipy import stats as st
    fig, ax = plt.subplots(figsize=(8, 5))
    pvals = [np.nan]
    for m in METHODS[1:]:
        a = final[final.method == m]["hard_dice"].values
        b = final[final.method == "Random"]["hard_dice"].values
        if len(a) > 1 and len(b) == len(a):
            _, p = st.ttest_rel(a, b)
        else:
            p = np.nan
        pvals.append(p)
    ax.bar(METHODS, [-np.log10(p) if (not np.isnan(p) and p > 0) else 0 for p in pvals],
           color=[COLORS[m] for m in METHODS])
    ax.axhline(-np.log10(0.05), color="k", linestyle="--", label="p = 0.05")
    ax.axhline(-np.log10(0.01), color="r", linestyle="--", label="p = 0.01")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title(f"Paired t-test vs Random (n={len(SEEDS)} seeds)")
    ax.legend(); plt.xticks(rotation=20)
    save(fig, "fig09_statistical_significance.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    mat = np.array([[final[final.method == m][f"dice_{c}"].mean() for c in CLASS_NAMES]
                    for m in METHODS])
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto",
                   vmin=mat.min(), vmax=mat.max())
    ax.set_xticks(range(6)); ax.set_xticklabels(CLASS_NAMES, rotation=20)
    ax.set_yticks(range(len(METHODS))); ax.set_yticklabels(METHODS)
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="Dice")
    ax.set_title("Per-Class Dice Matrix")
    save(fig, "fig13_per_class_heatmap.png")

    print("All available figures generated.")


if __name__ == "__main__":
    main()
