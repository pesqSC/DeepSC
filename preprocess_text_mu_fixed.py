"""Strictly aligned multilingual preprocessing for DeepSC."""
from __future__ import annotations
import argparse, json, pickle, random, re, unicodedata
from collections import Counter
from itertools import zip_longest
from pathlib import Path

try:
    from w3lib.html import remove_tags
except ImportError:
    def remove_tags(text):
        return re.sub(r"<[^>]+>", " ", text)

SPECIAL_TOKENS = ["<PAD>", "<START>", "<END>", "<UNK>", "<EN>", "<PT>", "<ES>", "<FR>"]
LANG_TOKEN = {"en":"<EN>", "pt":"<PT>", "es":"<ES>", "fr":"<FR>"}

def normalize_text(text):
    text = unicodedata.normalize("NFC", remove_tags(text)).lower().strip()
    text = re.sub(r"([.!?,;:()])", r" \1 ", text)
    text = text.replace("_", " ")
    text = re.sub(r"[^\w\s.!?,;:()'\-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()

def count_lines(path):
    with open(path, encoding="utf-8", errors="strict") as f:
        return sum(1 for _ in f)

def iter_parallel(src_path, trg_path):
    ns, nt = count_lines(src_path), count_lines(trg_path)
    if ns != nt:
        raise ValueError(f"Line-count mismatch: {src_path}={ns}, {trg_path}={nt}")
    with open(src_path, encoding="utf-8", errors="strict") as fs, open(trg_path, encoding="utf-8", errors="strict") as ft:
        for i, (s, t) in enumerate(zip_longest(fs, ft), 1):
            if s is None or t is None:
                raise RuntimeError(f"Unexpected mismatch at line {i}")
            yield i, s, t

def valid(text, min_len, max_length):
    n = len(text.split())
    return min_len <= n <= max_length - 3

def load_pairs(name, src_path, trg_path, min_len, max_length, max_ratio):
    pairs, stats = [], {"dataset":name,"raw":0,"kept":0,"empty":0,"src_len":0,"trg_len":0,"ratio":0}
    for line_no, raw_s, raw_t in iter_parallel(src_path, trg_path):
        stats["raw"] += 1
        s, t = normalize_text(raw_s), normalize_text(raw_t)
        if not s or not t:
            stats["empty"] += 1; continue
        if not valid(s, min_len, max_length):
            stats["src_len"] += 1; continue
        if not valid(t, min_len, max_length):
            stats["trg_len"] += 1; continue
        if max_ratio > 0:
            a, b = len(s.split()), len(t.split())
            if max(a,b)/max(min(a,b),1) > max_ratio:
                stats["ratio"] += 1; continue
        pairs.append((s,t,line_no)); stats["kept"] += 1
    return pairs, stats

def build_vocab(all_texts, min_count):
    c = Counter()
    for text in all_texts: c.update(text.split())
    token_to_idx = {tok:i for i,tok in enumerate(SPECIAL_TOKENS)}
    for tok, count in sorted(c.items(), key=lambda x:(-x[1], x[0])):
        if count >= min_count and tok not in token_to_idx:
            token_to_idx[tok] = len(token_to_idx)
    idx_to_token = [tok for tok,_ in sorted(token_to_idx.items(), key=lambda x:x[1])]
    return {"token_to_idx":token_to_idx,"idx_to_token":idx_to_token,"special_tokens":SPECIAL_TOKENS}

def encode(text, vocab, lang, max_length):
    unk = vocab["<UNK>"]
    ids = [vocab["<START>"], vocab[LANG_TOKEN[lang]]]
    ids += [vocab.get(tok, unk) for tok in text.split()]
    ids += [vocab["<END>"]]
    if len(ids) > max_length:
        raise ValueError(f"Encoded length {len(ids)} > {max_length}")
    return ids

def split_data(data, meta, ratio, seed):
    idx = list(range(len(data))); random.Random(seed).shuffle(idx)
    cut = int(len(idx)*ratio)
    tr, te = idx[:cut], idx[cut:]
    return [data[i] for i in tr], [data[i] for i in te], [meta[i] for i in tr], [meta[i] for i in te]

def save_json(obj, path):
    with open(path,"w",encoding="utf-8") as f: json.dump(obj,f,ensure_ascii=False,indent=2)

def save_pkl(obj, path):
    with open(path,"wb") as f: pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="data/train/europarl")
    for lang in ("pt","es","fr"):
        p.add_argument(f"--en-{lang}-source", required=True)
        p.add_argument(f"--en-{lang}-target", required=True)
    p.add_argument("--min-len", type=int, default=2)
    p.add_argument("--max-length", type=int, default=30)
    p.add_argument("--min-count", type=int, default=2)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-length-ratio", type=float, default=3.0)
    p.add_argument("--inspection-samples", type=int, default=50)
    a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)

    paths = {
        "en_pt":(a.en_pt_source,a.en_pt_target,"pt"),
        "en_es":(a.en_es_source,a.en_es_target,"es"),
        "en_fr":(a.en_fr_source,a.en_fr_target,"fr"),
    }
    cleaned, report = {}, {"config":vars(a),"datasets":{}}
    for name,(src,trg,lang) in paths.items():
        pairs, stats = load_pairs(name,src,trg,a.min_len,a.max_length,a.max_length_ratio)
        if not pairs: raise RuntimeError(f"No valid pairs for {name}")
        cleaned[name] = pairs; report["datasets"][name] = stats
        print(name, stats)

    all_texts = [x for pairs in cleaned.values() for s,t,_ in pairs for x in (s,t)]
    vocab_obj = build_vocab(all_texts, a.min_count)
    vocab = vocab_obj["token_to_idx"]
    save_json(vocab_obj, out/"vocab_multilingual.json")

    datasets, metas = {}, {}
    for name,pairs in cleaned.items():
        lang = paths[name][2]
        datasets[name] = [(encode(s,vocab,"en",a.max_length), encode(t,vocab,lang,a.max_length)) for s,t,_ in pairs]
        metas[name] = [{"line":ln,"source":s,"target":t} for s,t,ln in pairs]

    english = list(dict.fromkeys(s for pairs in cleaned.values() for s,_,_ in pairs))
    datasets["en_en"] = [(encode(s,vocab,"en",a.max_length), encode(s,vocab,"en",a.max_length)) for s in english]
    metas["en_en"] = [{"source":s,"target":s} for s in english]

    for offset,name in enumerate(("en_en","en_pt","en_es","en_fr")):
        tr,te,mtr,mte = split_data(datasets[name], metas[name], a.train_ratio, a.seed+offset)
        save_pkl(tr, out/f"train_{name}.pkl"); save_pkl(te, out/f"test_{name}.pkl")
        rng = random.Random(a.seed+offset)
        sample = rng.sample(mtr, min(a.inspection_samples,len(mtr))) if mtr else []
        save_json(sample, out/f"inspection_{name}.json")
        report["datasets"].setdefault(name,{})
        report["datasets"][name].update({"total":len(datasets[name]),"train":len(tr),"test":len(te)})
        print(f"{name}: train={len(tr)} test={len(te)}")

    report["vocab_size"] = len(vocab)
    save_json(report, out/"preprocessing_report.json")
    print("Done. Inspect inspection_en_pt.json before training.")

if __name__ == "__main__":
    main()
