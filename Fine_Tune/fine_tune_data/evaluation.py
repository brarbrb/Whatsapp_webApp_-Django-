# eval_bbt.py
# Evaluate a fine-tuned LLaMA on BBT next-utterance data with 3 metrics:
# 1) Perplexity on target-only tokens
# 2) Embedding-centroid affinity (no training)
# 3) Stylometric distance using Jensen–Shannon Divergence

import argparse, json, math, os, re
from collections import defaultdict
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional deps for metrics 2 and 3
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon


# -----------------------------
# Utils
# -----------------------------
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_jsonl(path, max_samples=None):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                data.append(json.loads(line))
            if max_samples and len(data) >= max_samples:
                break
    return data


def batchify(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs]


# -----------------------------
# 1) Embedding centroid affinity
# -----------------------------
def centroid_affinity(real_lines: Dict[str, List[str]], gen_lines: Dict[str, List[str]], emb_model_name="all-mpnet-base-v2"):
    st = SentenceTransformer(emb_model_name)

    def mean_embed(lines):
        if not lines:
            return np.zeros((st.get_sentence_embedding_dimension(),), dtype=np.float32)
        return np.mean(st.encode(lines, convert_to_numpy=True, show_progress_bar=False), axis=0)

    centroids = {char: mean_embed(lines) for char, lines in real_lines.items()}
    results = []
    for char, lines in gen_lines.items():
        for line in lines:
            emb = st.encode([line], convert_to_numpy=True, show_progress_bar=False)[0]
            sims = {c: float(cosine_similarity([emb], [centroids[c]])[0, 0]) for c in centroids}
            best = max(sims, key=sims.get)
            results.append({"speaker": char, "predicted_as": best, "cosine_to_true": sims.get(char, 0.0)})

    df = pd.DataFrame(results) if results else pd.DataFrame(columns=["speaker","predicted_as","cosine_to_true"])
    acc = float((df["speaker"] == df["predicted_as"]).mean()) if not df.empty else 0.0
    confusion = pd.crosstab(df["speaker"], df["predicted_as"], normalize="index") if not df.empty else pd.DataFrame()
    per_char_cos = df.groupby("speaker")["cosine_to_true"].mean() if not df.empty else pd.Series(dtype=float)
    return acc, per_char_cos, confusion, df


# -----------------------------
# 2) Stylometric JSD
# -----------------------------
_feat_regex = {
    "first_person": re.compile(r"\b(I|me|my|mine)\b", re.I),
    "modal_verbs": re.compile(r"\b(can|could|should|might|must|may|would|will)\b", re.I),
    "laughter": re.compile(r"\b(ha|haha|lol)\b", re.I),
    "discourse": re.compile(r"\b(well|so|actually|anyway|like)\b", re.I),
}

def extract_features(lines: List[str]):
    rows = []
    for text in lines:
        t = (text or "").strip()
        words = t.split()
        n = len(words)
        if n == 0:
            continue
        rows.append({
            "tokens_per_line": n,
            "punct_q": t.count("?") / n,
            "punct_ex": t.count("!") / n,
            "first_person_ratio": len(_feat_regex["first_person"].findall(t)) / n,
            "modal_verbs": len(_feat_regex["modal_verbs"].findall(t)) / n,
            "laughter_ratio": len(_feat_regex["laughter"].findall(t)) / n,
            "uppercase_ratio": sum(c.isupper() for c in t) / max(1, len(t)),
            "discourse_markers": len(_feat_regex["discourse"].findall(t)) / n,
        })
    if not rows:
        # return zeros with the expected keys
        return {k: 0.0 for k in [
            "tokens_per_line","punct_q","punct_ex","first_person_ratio",
            "modal_verbs","laughter_ratio","uppercase_ratio","discourse_markers"
        ]}
    df = pd.DataFrame(rows)
    return df.mean().to_dict()

def jsd_between_dicts(d1, d2):
    keys = sorted(set(d1) | set(d2))
    p = np.array([d1.get(k, 0.0) for k in keys], dtype=np.float64)
    q = np.array([d2.get(k, 0.0) for k in keys], dtype=np.float64)
    p = p / (p.sum() + 1e-8)
    q = q / (q.sum() + 1e-8)
    return float(jensenshannon(p, q))  # 0..1

def stylometric_jsd(real_lines: Dict[str, List[str]], gen_lines: Dict[str, List[str]]):
    rows = []
    for char in sorted(set(list(real_lines.keys()) + list(gen_lines.keys()))):
        rf = extract_features(real_lines.get(char, []))
        gf = extract_features(gen_lines.get(char, []))
        dist = jsd_between_dicts(rf, gf)
        diffs = {f"{k}_absdiff": abs(rf[k] - gf[k]) for k in rf}
        rows.append({"character": char, "JSD": dist, **diffs})
    df = pd.DataFrame(rows)
    df["JSD%"] = (df["JSD"] * 100).round(2)
    overall = float(df["JSD"].mean()) if not df.empty else 0.0
    return overall, df


# -----------------------------
# Generation helper
# -----------------------------
@torch.inference_mode()
def generate_responses(model, tokenizer, items, device, max_new_tokens=48, top_p=0.92, temperature=0.7, batch_size=4, max_length=1024):
    gens = []

     # ✅ open the file ONCE, in append mode (or write mode if you want overwrite)
    f = open("gens_orig_llama.jsonl", "w", encoding="utf-8")

    for batch in tqdm(list(batchify(items, batch_size)), desc="Generate"):
        prompts = [x["prompt"] for x in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        out = model.generate(
            **enc,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
        for i, x in enumerate(batch):
            inp_len = enc["input_ids"][i].size(0)
            full = out[i]
            gen_ids = full[inp_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            
            row = {
                "ep": x.get("ep"),
                "scene": x.get("scene"),
                "target_speaker": x.get("target_speaker"),
                "prompt": x["prompt"],
                "gold": x["target"],
                "gen": text
            }

            # ✅ save **line-by-line** immediately
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()   # force real-time write (important!)
    
            gens.append(row)

    f.close()   # ✅ close file 
    return gens


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="/Users/ransela/merged", help="Path to merged HF model folder")
    ap.add_argument("--test_path", type=str, default="bbt_test.jsonl", help="Path to JSONL with fields: prompt, target, target_speaker")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_samples", type=int, default=None, help="Limit test samples for quick runs")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--emb_model", type=str, default="all-MiniLM-L6-v2")
    args = ap.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    # Load test items
    test_path = '/Users/ransela/Desktop/data_science_degree/4th_year/spring/Data Analysis and Visualization Lab/project/Whatsapp_webApp_-Django-/fine_tune_data/bbt_test_cleaned.jsonl'
    raw = read_jsonl(test_path, max_samples=args.max_samples)
    # Normalize expected fields
    items = []
    for r in raw:
        items.append({
            "prompt": r["prompt"],
            "target": r["target"],
            "target_speaker": r.get("target_speaker") or r.get("speaker") or "UNKNOWN",
            "ep": r.get("ep"),
            "scene": r.get("scene") or r.get("scene_idx")
        })

    summary_rows = []

    # Load model and tokenizer
    for model in ["original","Fine_tune"]:
        if model == "original":
            args.model_dir = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            path_to_gen = "gens_orig_llama.jsonl"
        else:
            args.model_dir = "/Users/ransela/merged"
            path_to_gen = "gens_fine_tune.jsonl"  

        print(f"Loading model from {args.model_dir}")
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32


        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token


        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        

        # also update the model config so generate() uses the correct pad token
        # (important for decoder-only LLMs)
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch_dtype)
        model.config.pad_token_id = tokenizer.pad_token_id
        
        model.to(device)
        model.eval()

        # 1) Generate responses then centroid affinity
        if os.path.exists(path_to_gen):
            print("Loading cached generations...")
            gens = []
            with open(path_to_gen, "r", encoding="utf-8") as f:
                for line in f:
                    gens.append(json.loads(line))
        else:
            print("Generating responses (first time)...")
            gens = generate_responses(model, tokenizer, items, device,
                                max_new_tokens=args.max_new_tokens,
                                batch_size=args.batch_size, max_length=args.max_length)

        # Build corpora per character
        real_lines = defaultdict(list)
        gen_lines = defaultdict(list)
        for g in gens:
            spk = g["target_speaker"]
            real_lines[spk].append(g["gold"])
            gen_lines[spk].append(g["gen"] if g["gen"] else "")

        acc, per_char_cos, confusion, df_aff = centroid_affinity(real_lines, gen_lines, emb_model_name=args.emb_model)
        print(f"\nCentroid affinity accuracy: {acc*100:.2f}%")
        if len(per_char_cos) > 0:
            print("\nMean cosine to true centroid per character:")
            print(per_char_cos.round(3))
        if not confusion.empty:
            print("\nConfusion matrix (row normalized):")
            print(confusion.round(2))

        # 2) Stylometric JSD
        jsd_overall, df_jsd = stylometric_jsd(real_lines, gen_lines)
        print(f"\nStylometric JSD (lower is better). Overall: {jsd_overall:.3f}")
        print(df_jsd.round(3))

        # Save the row for this model
        summary_rows.append({
            "centroid_affinity_acc": acc,
            "jsd_overall": jsd_overall
        })

    # After loop, save final summary
    pd.DataFrame(summary_rows).to_csv("eval_summary.csv", index=False)

    print("\nSaved:")
    print(" - eval_summary.csv")


if __name__ == "__main__":
    main()