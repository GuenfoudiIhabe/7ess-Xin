#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import sys, traceback, warnings
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib as _mpl
import matplotlib.pyplot as _plt
import numpy as _np
import pandas as _pd
import seaborn as _sns
import torch as _torch
from sklearn.metrics import classification_report, confusion_matrix
from tokenizers import Tokenizer
from tqdm.auto import tqdm
from wordcloud import WordCloud

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

_CSV_CANDIDATES = [
    PROJ_ROOT / "Sentiment140.csv",
    PROJ_ROOT / "data" / "Sentiment140.csv",
    PROJ_ROOT / "data" / "sentiment140" / "Sentiment140.csv",
    PROJ_ROOT / "dataset" / "Sentiment140.csv",
]
DEFAULT_CSV = next((p for p in _CSV_CANDIDATES if p.exists()), _CSV_CANDIDATES[0])
DEFAULT_MODEL_DIR = PROJ_ROOT / "model_output" / "V6_optimized"

_OUT = Path(__file__).with_suffix("").parent / "qualitative_images"
_OUT.mkdir(parents=True, exist_ok=True)

_BATCH = 64
_PREVIEW = 20

_pd.set_option("display.max_colwidth", None)
_pd.set_option("display.width", 140)
_sns.set_context("talk")
warnings.filterwarnings("ignore", category=FutureWarning)


def _cmap(name="rocket_r", lut=256):
    try:
        return _mpl.colormaps.get_cmap(name).resampled(lut)
    except AttributeError:
        return _plt.get_cmap(name, lut)


def _strip(tok: str) -> str:
    t = tok.lstrip("Ġ▁")
    return t or "<BOS>"


def _tokenizer(model_dir: Path) -> Tokenizer:
    for f in ("tokenizer.pkl", "tokenizer.json"):
        p = model_dir / f
        if p.exists():
            return Tokenizer.from_file(str(p))
    raise FileNotFoundError("tokenizer missing")


def _load(csv: Path) -> Tuple[List[str], List[int]]:
    cols = ["target", "id", "date", "flag", "user", "text"]
    df = _pd.read_csv(csv, names=cols, header=None, encoding="latin-1")
    df = df[df.target.isin([0, 4])]
    return df.text.astype(str).tolist(), (df.target // 4).tolist()


def _encode(tok: Tokenizer, sents: Iterable[str], max_len: int) -> _torch.Tensor:
    pad = 0
    rows = [tok.encode(s).ids[:max_len] for s in sents]
    rows = [r + [pad] * (max_len - len(r)) for r in rows]
    return _torch.tensor(rows, dtype=_torch.long)


def _infer(net, tok, texts, max_len, dev):
    out: List[_np.ndarray] = []
    with _torch.no_grad():
        for i in range(0, len(texts), _BATCH):
            chunk = texts[i : i + _BATCH]
            out.append(net(_encode(tok, chunk, max_len).to(dev)).cpu().numpy())
    return _np.vstack(out)


def _conf(y_true, y_pred):
    raw = confusion_matrix(y_true, y_pred)
    pct = confusion_matrix(y_true, y_pred, normalize="true") * 100
    lab = _np.asarray([f"{p:.1f}%\n{c:,}" for p, c in zip(pct.flatten(), raw.flatten())]).reshape(2, 2)
    _sns.heatmap(pct, annot=lab, fmt="", cmap="Blues", xticklabels=["neg", "pos"], yticklabels=["neg", "pos"])
    _plt.tight_layout()
    _plt.savefig(_OUT / "confusion_matrix.png")
    _plt.close()


def _collapse(t):
    if isinstance(t, list):
        t = _torch.stack(t)
    while t.ndim > 2:
        t = t.mean(0)
    return t.detach().cpu().numpy()


def _attention(net, tok, sent, max_len, dev, layer=0):
    enc = tok.encode(sent.strip())
    ids = enc.ids[:max_len] + [0] * (max_len - len(enc.ids[:max_len]))
    _ = net(_torch.tensor([ids], device=dev))
    att = _collapse(net.get_attention_weights()[layer])  # type: ignore
    toks = [_strip(t) for t in enc.tokens[:_PREVIEW]]
    k = len(toks)
    imp = att[:k, :k].mean(0)
    keep = [i for i, t in enumerate(toks) if t != "<BOS>"]
    toks = [toks[i] for i in keep]
    att = att[keep][:, keep]
    imp = imp[keep]
    _sns.heatmap(att, cmap="rocket_r", vmin=0, vmax=0.4, xticklabels=toks, yticklabels=toks)
    _plt.xticks(rotation=90)
    _plt.tight_layout()
    _plt.savefig(_OUT / "attention_sentence.png")
    _plt.close()
    (_pd.DataFrame({"token": toks, "importance": imp})
     .plot.bar(x="token", y="importance", legend=False, figsize=(8, 2.5)))
    _plt.tight_layout()
    _plt.savefig(_OUT / "token_importance.png")
    _plt.close()
    for t, w in sorted(zip(toks, imp), key=lambda x: -x[1])[:10]:
        print(f"{t:<15}{w:.3f}")


def _wc(texts, name):
    if texts:
        WordCloud(width=640, height=400, background_color="white").generate(" ".join(texts)).to_file(_OUT / name)


def _net(model_dir: Path, dev):
    ckpt = _torch.load(model_dir / "best_model.pt", map_location=dev)
    state, cfg = ckpt["model"], ckpt["config"]
    tok = _tokenizer(model_dir)
    from models.transformer import EmotionAnalysisModel
    net = EmotionAnalysisModel(
        vocab_size=len(tok.get_vocab()),
        emb_dim=cfg["emb_dim"],
        stack_depth=cfg["stack_depth"],
        attn_heads=cfg["attn_heads"],
        ff_expansion=cfg["ff_expansion"] // 2,
        max_len=cfg["max_len"],
        dropout=cfg["dropout"],
    ).to(dev)
    net.load_state_dict(state, strict=False)
    net.eval()
    return net, tok, cfg


def full_eval(model_dir: Path, csv: Path, dev):
    net, tok, cfg = _net(model_dir, dev)
    texts, labs = _load(csv)
    logits = _infer(net, tok, texts, cfg["max_len"], dev)
    preds = logits.argmax(1)
    print(classification_report(labs, preds, target_names=["neg", "pos"], digits=4))
    _conf(labs, preds)
    df = _pd.DataFrame({"text": texts, "true": labs, "pred": preds})
    bad = df[df.true != df.pred].sample(1, random_state=0).iloc[0]
    for lbl, tag in [(1, "pos"), (0, "neg")]:
        _wc(df[(df.true == lbl) & (df.pred == lbl)].text.tolist(), f"wc_{tag}_correct.png")
        _wc(df[(df.true == lbl) & (df.pred != df.true)].text.tolist(), f"wc_{tag}_wrong.png")
    _attention(net, tok, bad.text, cfg["max_len"], dev)


def single_eval(model_dir: Path, sentence: str, dev):
    net, tok, cfg = _net(model_dir, dev)
    ids = tok.encode(sentence.strip()).ids[: cfg["max_len"]]
    ids += [0] * (cfg["max_len"] - len(ids))
    pred = net(_torch.tensor([ids], device=dev)).argmax().item()
    print(["NEGATIVE", "POSITIVE"][pred])
    _attention(net, tok, sentence, cfg["max_len"], dev)


def main():
    dev = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--data_path", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--sent", help="single sentence")
    args = ap.parse_args()
    try:
        if args.sent:
            single_eval(args.model_dir, args.sent, dev)
        else:
            full_eval(args.model_dir, args.data_path, dev)
    except Exception:
        print("error", file=sys.stderr)
        traceback.print_exc()
        print(_dt.datetime.now().isoformat())


if __name__ == "__main__":
    main()
