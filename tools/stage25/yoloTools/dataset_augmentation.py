import cv2
import os
import glob
import albumentations as A

# Paths
images_dir = "W:/competition/model/images/val"
labels_dir = "W:/competition/model/labels/val"
output_images_dir = "W:/competition/model/images/val"
output_labels_dir = "W:/competition/model/labels/val"

os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# CHECKBOX-SAFE augmentations only
transform = A.Compose([
    # Safe brightness/contrast changes
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    
    # Very gentle gamma adjustment
    A.RandomGamma(gamma_limit=(80, 120), p=0.4),
    
    # Slight hue/saturation (helps with different scanner colors)
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    
    # Light noise to simulate scanning artifacts
    A.GaussNoise(var_limit=(5.0, 15.0), p=0.3),
    
    # Very subtle sharpening/blurring
    A.OneOf([
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.8, 1.2), p=0.5),
        A.MotionBlur(blur_limit=3, p=0.5)
    ], p=0.2),
    
    # REMOVED: HorizontalFlip (destroys checkmarks)
    # REMOVED: Rotation (distorts checkmarks) 
    # REMOVED: RandomScale (makes marks too small)
    # REMOVED: Heavy blur (removes fine details)
    
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Reduce augmentations per image since we're being more conservative
num_aug_per_image = 5

# Loop over all images
for image_path in glob.glob(os.path.join(images_dir, "*.jpg")):
    filename = os.path.basename(image_path)
    label_path = os.path.join(labels_dir, filename.replace(".jpg", ".txt"))
    
    # Read image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    # Read YOLO labels
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    bboxes = []
    class_labels = []
    for line in lines:
        cls, x, y, w, h = map(float, line.strip().split())
        bboxes.append([x, y, w, h])
        class_labels.append(int(cls))
    
    # Generate augmentations
    for i in range(num_aug_per_image):
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']
        
        # Save augmented image
        new_image_name = f"{filename[:-4]}_aug{i}.jpg"
        cv2.imwrite(os.path.join(output_images_dir, new_image_name), aug_img)
        
        # Save updated YOLO labels
        new_label_name = f"{filename[:-4]}_aug{i}.txt"
        with open(os.path.join(output_labels_dir, new_label_name), "w") as f:
            for bbox, cls in zip(aug_bboxes, class_labels):
                f.write(f"{cls} {' '.join(map(str, bbox))}\n")

print("✅ Checkbox-safe augmentation completed!")