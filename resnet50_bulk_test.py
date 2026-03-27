import os
import random
import torch
import sys
from PIL import Image

# Project Path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

import config
from model import get_model
from data_loader import get_eval_transforms

def get_random_sample(test_dir, total_count=50):
    all_images = []
    classes = ["Autistic", "Non_Autistic"]
    per_class = total_count // 2
    
    for cls in classes:
        cls_path = os.path.join(test_dir, cls)
        if not os.path.exists(cls_path):
            continue
        images = [os.path.join(cls_path, img) for img in os.listdir(cls_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Take up to 25 from each class
        if len(images) < per_class:
            selected = images
        else:
            selected = random.sample(images, per_class)
            
        for img in selected:
            all_images.append((img, cls))
    
    random.shuffle(all_images)
    return all_images

def main():
    print("=" * 80)
    print(f"{'BULK TESTING: ResNet50 (50 Random Images)':^80}")
    print("=" * 80)
    
    test_sample = get_random_sample(config.TEST_DIR, total_count=50)
    if not test_sample:
        print("No images found in test directory!")
        return
        
    transform = get_eval_transforms()
    
    # Load ResNet50
    model_name = "resnet50"
    print(f"Loading {model_name}...")
    model = get_model(model_name, config.NUM_CLASSES, freeze=True)
    model_path = os.path.join(config.MODELS_DIR, f"{model_name}_best.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: Weights not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    
    correct = 0
    total = len(test_sample)
    misclassified = []
    
    print(f"Testing {total} images...")
    
    with torch.no_grad():
        for i, (img_path, true_label) in enumerate(test_sample):
            image = Image.open(img_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(config.DEVICE)
            output = model(tensor)
            _, predicted = torch.max(output, 1)
            pred_label = config.CLASS_NAMES[predicted.item()]
            
            if pred_label == true_label:
                correct += 1
            else:
                misclassified.append({
                    "image": os.path.basename(img_path),
                    "true": true_label,
                    "pred": pred_label
                })
            
            # Simple progress update
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{total}...")

    accuracy = (correct / total) * 100
    
    print("\n" + "=" * 80)
    print(f"RESULTS SUMMARY")
    print("-" * 80)
    print(f"  Total Images: {total}")
    print(f"  Correct:      {correct}")
    print(f"  Incorrect:    {len(misclassified)}")
    print(f"  Batch Accuracy: {accuracy:.2f}%")
    print("=" * 80)
    
    if misclassified:
        print("\nMISCLASSIFIED IMAGES (Error Analysis):")
        print("-" * 60)
        print(f"{'IMAGE':<30} | {'TRUE':<12} | {'PREDICTED':<12}")
        print("-" * 60)
        for error in misclassified:
            print(f"{error['image']:<30} | {error['true']:<12} | {error['pred']:<12}")
        print("-" * 60)

if __name__ == "__main__":
    main()
