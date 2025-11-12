from pathlib import Path

labels_path = Path("A:/stage25/qrdb/dataset/labels/train")
for label_file in labels_path.glob("*.txt"):
    with open(label_file) as f:
        lines = f.readlines()
    if any(line.split()[0] in {"bon", "moyen", "non"} for line in lines):
        print(f"Deleting: {label_file}")
        label_file.unlink()
