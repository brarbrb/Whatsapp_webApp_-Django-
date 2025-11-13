import json, re
from collections import Counter

INPUT = '/Users/ransela/Desktop/data_science_degree/4th_year/spring/Data Analysis and Visualization Lab/project/Whatsapp_webApp_-Django-/gens_orig_llama.jsonl'
#OUTPUT = "/home/student/Whatsapp_webApp_-Django-/fine_tune_data/bbt_test_cleaned.jsonl"
#gens_path = "/home/student/Whatsapp_webApp_-Django-/gens_cache_cleaned.jsonl"
#OUTPUT = "/home/student/Whatsapp_webApp_-Django-/gens_cache_cleaned.jsonl"

SMALL_VALID_SPEAKERS = [
    "Sheldon",
    "Leonard",
    "Penny",
    "Howard",
    "Raj",
    "Amy",
    "Bernadette"
]
BIG_VALID_SPEAKERS = [
    "Sheldon",
    "Leonard",
    "Penny",
    "Howard",
    "Raj",
    "Amy",
    "Bernadette",
    "Stuart",
    "Leslie",
    "Emily",
    "Wil",
    "Barry",
    "Kripke",
    "Bert",
    "Arthur",
    "Dave",
    "Mike",
    "Wyatt",
    "Beverly",
    "Gablehouser",
    "Alex",
    "Lucy",
    "Zack",
    "Steph",
    "Koothrappali",
    "Ramona",
    "Rostenkowski",
    "Janine",
    "Hawking",
    "Petrescu",
]
# Only domain knowledge here (nicknames, surnames). No typos.
ALIAS_MAP = {
    # Surnames
    "cooper": "Sheldon",
    "hofstadter": "Leonard",
    "wolowitz": "Howard",
    "koothrappali": "Raj",
    "fowler": "Amy",
    "rostenkowski": "Bernadette",
    # Common in-world nicknames
    "lenny": "Leonard",
    "howie": "Howard",
    "shelly": "Sheldon",
    "bernie": "Bernadette",
    "rajish": "Raj",   # if appears in some transcripts
    "rajesh": "Raj",
}

VALID_LOWER = [c.lower() for c in SMALL_VALID_SPEAKERS]

def normalize_speaker(s: str) -> str:
    if not s:
        return ""
    t = s.strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[()\[\]{}]+", "", t)           # remove any brackets anywhere
    t = re.sub(r"[:;,.!?]+$", "", t).strip()    # remove trailing punctuation
    low = t.lower()

    # 1) exact alias match
    if low in ALIAS_MAP:
        return ALIAS_MAP[low]

    # 2) substring catch for typos like "sheldon)" or "penny:"
    for name_low, canon in zip(VALID_LOWER, SMALL_VALID_SPEAKERS):
        if name_low in low:
            return canon

    return t

def is_valid_exact(s: str) -> bool:
    # after normalization this should be an exact canonical name
    return s in SMALL_VALID_SPEAKERS

rows = []
with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        # normalize first
        norm = normalize_speaker(obj.get("target_speaker", ""))
        obj["target_speaker"] = norm
        rows.append(obj)

# split after normalization
good = [r for r in rows if is_valid_exact(r.get("target_speaker", ""))]
bad  = [r for r in rows if not is_valid_exact(r.get("target_speaker", ""))]

print(f"Total rows: {len(rows)}")
print(f" Valid rows: {len(good)}")
print(f" Invalid rows: {len(bad)}")

# histogram on normalized speakers (all rows so you can inspect bad labels too)
speaker_hist = Counter(r.get("target_speaker", "") for r in rows)
print("\nSpeaker histogram (normalized):")
for spk, count in sorted(speaker_hist.items(), key=lambda x: (-x[1], x[0])):
    print(f"{spk:12s} {count}")

# peek a few bad labels to see what's left
if bad:
    sample = Counter(r.get("target_speaker", "") for r in bad).most_common(15)
    print("\nTop bad labels after normalization:")
    for spk, cnt in sample:
        print(f"{spk:12s} {cnt}")

# # write cleaned file with only valid rows
# with open(OUTPUT, "w", encoding="utf-8") as f:
#     for r in good:
#         f.write(json.dumps(r, ensure_ascii=False) + "\n")

# print(f"\nCleaned file saved: {OUTPUT}")