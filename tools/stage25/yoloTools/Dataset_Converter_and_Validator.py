#!/usr/bin/env python3
"""
Dataset Converter and Validator
Converts HTML-generated exam images to YOLO format and validates the dataset
"""

import os
import cv2
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import argparse

class DatasetConverter:
    """Convert HTML-generated images to proper YOLO dataset"""
    
    def __init__(self):
        self.class_names = [
            'bon_unchecked', 'moyen_unchecked', 'non_unchecked',
            'bon_checked', 'moyen_checked', 'non_checked'
        ]
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}
        
        # Checkbox detection parameters (adjust based on your HTML generator)
        self.checkbox_size_mm = 10
        self.grading_section_left_mm = 5
        self.grading_section_top_mm = 80
        self.mm_to_pixel = 3.78  # 300 DPI conversion
        
    def extract_checkboxes_from_image(self, image_path, num_questions=6):
        """
        Extract checkbox locations from exam image using computer vision
        This serves as a backup if manual annotations aren't available
        """
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error loading image: {image_path}")
            return []
        
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate expected checkbox positions based on HTML layout
        checkboxes = []
        checkbox_size_px = int(self.checkbox_size_mm * self.mm_to_pixel)
        section_left_px = int(self.grading_section_left_mm * self.mm_to_pixel)
        section_top_px = int(self.grading_section_top_mm * self.mm_to_pixel)
        
        # For each question, create checkbox entries
        for q in range(num_questions):
            question_y_offset = q * int(20 * self.mm_to_pixel)  # 20mm spacing
            
            # Bon, Moyen, Non checkboxes for each question
            for i, grade_type in enumerate(['bon', 'moyen', 'non']):
                checkbox_y = section_top_px + question_y_offset + (i * int(10 * self.mm_to_pixel))
                checkbox_x = section_left_px + int(2 * self.mm_to_pixel)
                
                # Check if this checkbox is marked (simple intensity check)
                roi = gray[
                    max(0, checkbox_y - checkbox_size_px//2):min(height, checkbox_y + checkbox_size_px//2),
                    max(0, checkbox_x - checkbox_size_px//2):min(width, checkbox_x + checkbox_size_px//2)
                ]
                
                if roi.size > 0:
                    avg_intensity = np.mean(roi)
                    is_checked = avg_intensity < 200  # Dark indicates marking
                    
                    class_name = f"{grade_type}_{'checked' if is_checked else 'unchecked'}"
                    class_id = self.class_to_id[class_name]
                    
                    # Convert to YOLO format (normalized)
                    center_x = checkbox_x / width
                    center_y = checkbox_y / height
                    bbox_width = checkbox_size_px / width
                    bbox_height = checkbox_size_px / height
                    
                    checkboxes.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'bbox': [center_x, center_y, bbox_width, bbox_height],
                        'confidence': 0.8 if is_checked else 0.9  # Higher conf for unchecked (easier to detect)
                    })
        
        return checkboxes
    
    def convert_image_folder(self, input_dir, output_dir, auto_detect=True):
        """Convert folder of exam images to YOLO dataset"""
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output structure
        for split in ['images', 'labels']:
            for subset in ['train', 'val']:
                (output_path / split / subset).mkdir(parents=True, exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"Found {len(image_files)} images in {input_dir}")
        
        # Split train/val (80/20)
        split_idx = int(len(image_files) * 0.8)
        train_images = image_files[:split_idx]
        val_images = image_files[split_idx:]
        
        stats = {'train': defaultdict(int), 'val': defaultdict(int)}
        
        # Process training images
        print("Processing training images...")
        for i, img_file in enumerate(train_images):
            self._process_image(img_file, output_path, 'train', auto_detect, stats['train'])
            if i % 20 == 0:
                print(f"  Processed {i+1}/{len(train_images)} training images")
        
        # Process validation images
        print("Processing validation images...")
        for i, img_file in enumerate(val_images):
            self._process_image(img_file, output_path, 'val', auto_detect, stats['val'])
            if i % 10 == 0:
                print(f"  Processed {i+1}/{len(val_images)} validation images")
        
        # Create dataset.yaml
        self._create_dataset_yaml(output_path)
        
        # Print statistics
        self._print_dataset_stats(stats)
        
        return output_path
    
    def _process_image(self, img_file, output_path, split, auto_detect, stats):
        """Process a single image"""
        
        # Copy image
        dst_img = output_path / 'images' / split / img_file.name
        import shutil
        shutil.copy2(img_file, dst_img)
        
        # Generate or find labels
        label_file = img_file.with_suffix('.txt')
        checkboxes = []
        
        if label_file.exists():
            # Read existing labels
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        bbox = [float(x) for x in parts[1:5]]
                        checkboxes.append({
                            'class_id': class_id,
                            'class_name': self.class_names[class_id],
                            'bbox': bbox
                        })
        elif auto_detect:
            # Auto-detect checkboxes
            checkboxes = self.extract_checkboxes_from_image(img_file)
        
        # Save labels
        dst_label = output_path / 'labels' / split / img_file.with_suffix('.txt').name
        with open(dst_label, 'w') as f:
            for cb in checkboxes:
                bbox = cb['bbox']
                f.write(f"{cb['class_id']} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                stats[cb['class_name']] += 1
    
    def _create_dataset_yaml(self, output_path):
        """Create dataset.yaml configuration"""
        
        config = {
            'path': str(output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        import yaml
        with open(output_path / 'dataset.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def _print_dataset_stats(self, stats):
        """Print dataset statistics"""
        
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        
        total_train = sum(stats['train'].values())
        total_val = sum(stats['val'].values())
        
        print(f"Total training samples: {total_train}")
        print(f"Total validation samples: {total_val}")
        print(f"Total samples: {total_train + total_val}")
        
        print(f"\nClass distribution:")
        print(f"{'Class':<20} {'Train':<8} {'Val':<8} {'Total':<8} {'%':<8}")
        print("-" * 50)
        
        for class_name in self.class_names:
            train_count = stats['train'][class_name]
            val_count = stats['val'][class_name]
            total_count = train_count + val_count
            percentage = (total_count / (total_train + total_val)) * 100 if total_train + total_val > 0 else 0
            
            print(f"{class_name:<20} {train_count:<8} {val_count:<8} {total_count:<8} {percentage:<8.1f}")

class DatasetValidator:
    """Validate and visualize YOLO dataset"""
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.class_names = [
            'bon_unchecked', 'moyen_unchecked', 'non_unchecked',
            'bon_checked', 'moyen_checked', 'non_checked'
        ]
    
    def validate_dataset(self):
        """Validate dataset structure and content"""
        
        print("VALIDATING DATASET")
        print("="*40)
        
        errors = []
        warnings = []
        
        # Check directory structure
        required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
        for dir_path in required_dirs:
            full_path = self.dataset_path / dir_path
            if not full_path.exists():
                errors.append(f"Missing directory: {dir_path}")
        
        # Check dataset.yaml
        yaml_path = self.dataset_path / 'dataset.yaml'
        if not yaml_path.exists():
            errors.append("Missing dataset.yaml file")
        
        # Check image-label pairs
        for split in ['train', 'val']:
            img_dir = self.dataset_path / 'images' / split
            lbl_dir = self.dataset_path / 'labels' / split
            
            if img_dir.exists() and lbl_dir.exists():
                img_files = set(f.stem for f in img_dir.glob('*.jpg'))
                lbl_files = set(f.stem for f in lbl_dir.glob('*.txt'))
                
                missing_labels = img_files - lbl_files
                missing_images = lbl_files - img_files
                
                if missing_labels:
                    warnings.append(f"{split}: {len(missing_labels)} images without labels")
                if missing_images:
                    warnings.append(f"{split}: {len(missing_images)} labels without images")
        
        # Validate annotation format
        self._validate_annotations()
        
        # Print results
        if errors:
            print("❌ ERRORS FOUND:")
            for error in errors:
                print(f"  - {error}")
        
        if warnings:
            print("⚠️ WARNINGS:")
            for warning in warnings:
                print(f"  - {warning}")
        
        if not errors and not warnings:
            print("✅ Dataset validation passed!")
        
        return len(errors) == 0
    
    def _validate_annotations(self):
        """Validate annotation format and content"""
        
        for split in ['train', 'val']:
            lbl_dir = self.dataset_path / 'labels' / split
            if not lbl_dir.exists():
                continue
            
            for lbl_file in lbl_dir.glob('*.txt'):
                with open(lbl_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        parts = line.strip().split()
                        if len(parts) != 5:
                            print(f"❌ {lbl_file.name}:{line_num}: Expected 5 values, got {len(parts)}")
                            continue
                        
                        try:
                            class_id = int(parts[0])
                            bbox = [float(x) for x in parts[1:5]]
                            
                            # Validate class ID
                            if class_id < 0 or class_id >= len(self.class_names):
                                print(f"❌ {lbl_file.name}:{line_num}: Invalid class ID {class_id}")
                            
                            # Validate bbox (should be normalized 0-1)
                            for i, val in enumerate(bbox):
                                if val < 0 or val > 1:
                                    coord_names = ['center_x', 'center_y', 'width', 'height']
                                    print(f"❌ {lbl_file.name}:{line_num}: {coord_names[i]} out of range: {val}")
                        
                        except ValueError as e:
                            print(f"❌ {lbl_file.name}:{line_num}: Parse error: {e}")
    
    def visualize_samples(self, num_samples=6, save_path=None):
        """Visualize annotated samples"""
        
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = ['red', 'orange', 'blue', 'green', 'purple', 'brown']
        
        # Get sample images
        img_dir = self.dataset_path / 'images' / 'train'
        lbl_dir = self.dataset_path / 'labels' / 'train'
        
        img_files = list(img_dir.glob('*.jpg'))[:num_samples]
        
        for i, img_file in enumerate(img_files):
            if i >= num_samples:
                break
            
            # Load image
            image = cv2.imread(str(img_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            
            # Load labels
            lbl_file = lbl_dir / f"{img_file.stem}.txt"
            
            axes[i].imshow(image)
            axes[i].set_title(f"{img_file.name}")
            axes[i].axis('off')
            
            if lbl_file.exists():
                with open(lbl_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            cx, cy, w, h = [float(x) for x in parts[1:5]]
                            
                            # Convert to pixel coordinates
                            x = (cx - w/2) * width
                            y = (cy - h/2) * height
                            w_px = w * width
                            h_px = h * height
                            
                            # Draw bounding box
                            rect = patches.Rectangle(
                                (x, y), w_px, h_px,
                                linewidth=2, edgecolor=colors[class_id],
                                facecolor='none'
                            )
                            axes[i].add_patch(rect)
                            
                            # Add label
                            axes[i].text(
                                x, y-5, self.class_names[class_id],
                                color=colors[class_id], fontsize=8,
                                bbox=dict(boxstyle="round,pad=0.3", 
                                         facecolor='white', alpha=0.7)
                            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def analyze_class_distribution(self, save_path=None):
        """Analyze and visualize class distribution"""
        
        class_counts = {'train': Counter(), 'val': Counter()}
        
        for split in ['train', 'val']:
            lbl_dir = self.dataset_path / 'labels' / split
            if not lbl_dir.exists():
                continue
            
            for lbl_file in lbl_dir.glob('*.txt'):
                with open(lbl_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            class_counts[split][class_id] += 1
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training distribution
        train_counts = [class_counts['train'][i] for i in range(len(self.class_names))]
        ax1.bar(self.class_names, train_counts, color='skyblue', alpha=0.7)
        ax1.set_title('Training Set Class Distribution')
        ax1.set_ylabel('Number of Instances')
        ax1.tick_params(axis='x', rotation=45)
        
        # Validation distribution
        val_counts = [class_counts['val'][i] for i in range(len(self.class_names))]
        ax2.bar(self.class_names, val_counts, color='lightcoral', alpha=0.7)
        ax2.set_title('Validation Set Class Distribution')
        ax2.set_ylabel('Number of Instances')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Class distribution plot saved to {save_path}")
        
        plt.show()
        
        # Print statistics
        print("\nCLASS DISTRIBUTION ANALYSIS")
        print("="*50)
        print(f"{'Class':<20} {'Train':<8} {'Val':<8} {'Total':<8} {'Balance':<10}")
        print("-" * 50)
        
        total_train = sum(train_counts)
        total_val = sum(val_counts)
        
        for i, class_name in enumerate(self.class_names):
            train_count = train_counts[i]
            val_count = val_counts[i]
            total_count = train_count + val_count
            
            # Calculate balance score (how close to uniform distribution)
            expected = (total_train + total_val) / len(self.class_names)
            balance = min(total_count / expected, expected / total_count) if expected > 0 else 0
            
            print(f"{class_name:<20} {train_count:<8} {val_count:<8} {total_count:<8} {balance:<10.2f}")
        
        # Recommendations
        min_count = min(train_counts + val_counts)
        max_count = max(train_counts + val_counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\nImbalance ratio: {imbalance_ratio:.2f}")
        if imbalance_ratio > 3:
            print("⚠️ High class imbalance detected! Consider:")
            print("  - Generating more samples for underrepresented classes")
            print("  - Using class weights in training")
            print("  - Applying data augmentation")
        elif imbalance_ratio > 1.5:
            print("⚠️ Moderate class imbalance. Monitor training performance.")
        else:
            print("✅ Classes are well balanced!")

def main():
    """Main function with CLI interface"""
    
    parser = argparse.ArgumentParser(description='Dataset Converter and Validator for YOLO Checkbox Detection')
    parser.add_argument('command', choices=['convert', 'validate', 'visualize', 'analyze'], 
                       help='Command to execute')
    parser.add_argument('--input', '-i', required=False, 
                       help='Input directory (for convert command)')
    parser.add_argument('--output', '-o', required=False,
                       help='Output directory (for convert command)')
    parser.add_argument('--dataset', '-d', required=False,
                       help='Dataset directory (for validate/visualize/analyze commands)')
    parser.add_argument('--auto-detect', action='store_true',
                       help='Auto-detect checkboxes if labels not found')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save visualization plots')
    
    args = parser.parse_args()
    
    if args.command == 'convert':
        if not args.input or not args.output:
            print("Error: --input and --output are required for convert command")
            return
        
        converter = DatasetConverter()
        output_path = converter.convert_image_folder(
            args.input, args.output, args.auto_detect
        )
        print(f"\n✅ Dataset converted successfully!")
        print(f"Output directory: {output_path}")
        print(f"Next steps:")
        print(f"  1. Review the generated annotations")
        print(f"  2. Run validation: python {__file__} validate --dataset {output_path}")
        print(f"  3. Start training with the improved script")
    
    elif args.command == 'validate':
        if not args.dataset:
            print("Error: --dataset is required for validate command")
            return
        
        validator = DatasetValidator(args.dataset)
        is_valid = validator.validate_dataset()
        
        if is_valid:
            print("\n✅ Dataset is ready for training!")
        else:
            print("\n❌ Please fix the errors before training.")
    
    elif args.command == 'visualize':
        if not args.dataset:
            print("Error: --dataset is required for visualize command")
            return
        
        validator = DatasetValidator(args.dataset)
        save_path = 'dataset_samples.png' if args.save_plots else None
        validator.visualize_samples(save_path=save_path)
    
    elif args.command == 'analyze':
        if not args.dataset:
            print("Error: --dataset is required for analyze command")
            return
        
        validator = DatasetValidator(args.dataset)
        save_path = 'class_distribution.png' if args.save_plots else None
        validator.analyze_class_distribution(save_path=save_path)

if __name__ == "__main__":
    main()