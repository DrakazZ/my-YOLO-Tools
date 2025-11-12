#!/usr/bin/env python3
import torch
from ultralytics import YOLO

def train_checkbox_model():
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        device = 'cuda'
    else:
        print("No GPU detected, using CPU")
        device = 'cpu'
    
    # Load pre-trained YOLOv8
    model = YOLO('yolov8s.pt')
    
    print("Starting optimized checkbox training...")

    results = model.train(
        data='dataset.yaml',
        epochs=200,
        imgsz=640,
        batch=-1,  # auto-adjust to GPU memory
        device=device,  # explicitly specify device
        warmup_epochs=3,    # stabilize early learning
        
        # Optimized for small objects
        lr0=0.005,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        
        box=7.5,
        cls=1.0,
        dfl=1.5,
        
        # Augmentation tuned for small fixed-position objects
        degrees=2.0,
        translate=0.05,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,   # reduced to avoid context loss
        mixup=0.1,
        
        optimizer='AdamW',
        patience=50,
        save=True,
        save_period=10,   # save extra checkpoints
        project='runs/detect',
        name='checkbox_optimized',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        seed=42,
        deterministic=True,
        val=True,
        plots=True,
        amp=True,  # Automatic Mixed Precision for faster GPU training
        cache=True  # Cache images in RAM/disk for faster loading
    )
    
    print(f"\nTraining completed!")
    print(f"Best model: {results.save_dir}/weights/best.pt")
    print(f"mAP@0.5: {results.box.map50:.3f}")
    
    return results

if __name__ == "__main__":
    train_checkbox_model()