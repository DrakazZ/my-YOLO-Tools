from pathlib import Path
from collections import Counter

labels_path = Path("A:/stage25/qrdb/dataset/labels/train")
counter = Counter()

for label_file in labels_path.glob("*.txt"):
    with open(label_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            class_id = line.strip().split()[0]
            counter[class_id] += 1

print("Class counts:", counter)
