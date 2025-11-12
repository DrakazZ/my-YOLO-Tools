# split_annotations.py
import os
from pathlib import Path

def split_yolo_annotations(annotations_file, output_dir):
    """Split single annotation file into individual YOLO txt files per image"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    current_file = None
    current_content = []
    
    with open(annotations_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Detect a new annotation block from "File: ..."
            if line.startswith('File: '):
                # Save the previous file's annotations if any
                if current_file and current_content:
                    output_path = output_dir / current_file
                    with open(output_path, 'w') as out_f:
                        out_f.write('\n'.join(current_content) + '\n')
                    print(f"Created: {output_path}")
                
                # Start new file
                current_file = Path(line.replace('File: ', '')).name
                current_content = []
            
            elif line and not line.startswith('File:'):
                # This is annotation data
                current_content.append(line)
        
        # Save the last file
        if current_file and current_content:
            output_path = output_dir / current_file
            with open(output_path, 'w') as out_f:
                out_f.write('\n'.join(current_content) + '\n')
            print(f"Created: {output_path}")

# Example usage:
split_yolo_annotations('yolo_annotationsV8.txt', 'A:/stage25/model/labels/val')
