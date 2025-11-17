import csv
import json
import re

input_csv = "big_bang_theory.csv"
output_json = "big_bang_theory.json"


def parse_episode_name(full_name):
    match = re.search(r"Series\s+(\d+)\s+Episode\s+(\d+)\s+–\s+(.*)", full_name)
    if match:
        season_num = match.group(1).zfill(2)
        episode_num = match.group(2).zfill(2)
        title = match.group(3).strip()
        return f"S{season_num}E{episode_num}", title
    return None, None


episodes = {}

with open(r'C:/Users/User/Downloads/1_10_seasons_tbbt.csv', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    current_scene = []
    current_episode = None

    for row in reader:
        if len(row) != 3:
            continue

        episode_name, dialogue, speaker = row
        ep_key, ep_title = parse_episode_name(episode_name)
        if not ep_key:
            continue

        if ep_key not in episodes:
            episodes[ep_key] = {
                "Episode Title": ep_title,
                "Script": []
            }
            current_episode = ep_key
            current_scene = []

        if speaker.strip().lower() == "scene":
            if current_scene:
                episodes[current_episode]["Script"].append(current_scene)
                current_scene = []
            continue

        current_scene.append({
            "Speaker": speaker.strip(),
            "Text": dialogue.strip()
        })

    if current_scene and current_episode:
        episodes[current_episode]["Script"].append(current_scene)

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(episodes, f, indent=2, ensure_ascii=False)

print(f"✅ JSON saved to {output_json}")
