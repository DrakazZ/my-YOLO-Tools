import os
from tqdm import tqdm

# Paths
old_labels_dir = "A:/stage25/model/labels/trainV1"   # original label folder
new_labels_dir = "A:/stage25/model/labels/train"  # new output folder

os.makedirs(new_labels_dir, exist_ok=True)

# Old ID -> New ID mapping
id_map = {
    0: 0,  # bon_unchecked → unchecked
    1: 0,  # moyen_unchecked → unchecked
    2: 0,  # non_unchecked → unchecked
    3: 1,  # bon_checked → checked
    4: 1,  # moyen_checked → checked
    5: 1   # non_checked → checked
}

for file in tqdm(os.listdir(old_labels_dir)):
    if not file.endswith(".txt"):
        continue

    with open(os.path.join(old_labels_dir, file), "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        old_id = int(parts[0])
        new_id = id_map[old_id]
        new_lines.append(" ".join([str(new_id)] + parts[1:]))

    with open(os.path.join(new_labels_dir, file), "w") as f:
        f.write("\n".join(new_lines))
