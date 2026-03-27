import os
import random
import torch
import sys
from PIL import Image

# Force UTF-8 for Windows emoji support
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Project Path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_DIR)

import config
from model import get_model
from data_loader import get_eval_transforms

def get_random_images(test_dir, num_per_class=5):
    all_images = []
    classes = ["Autistic", "Non_Autistic"]
    
    for cls in classes:
        cls_path = os.path.join(test_dir, cls)
        if not os.path.exists(cls_path):
            continue
        images = [os.path.join(cls_path, img) for img in os.listdir(cls_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(images) < num_per_class:
            selected = images
        else:
            selected = random.sample(images, num_per_class)
        for img in selected:
            all_images.append((img, cls))
    
    random.shuffle(all_images)
    return all_images

def main():
    print("=" * 120)
    header = f"{'IMAGE':<30} | {'TRUE':<12} | {'RESNET18':<12} | {'EFF-B0':<12} | {'DENSE121':<12} | {'RESNET50':<12} | {'MATCH'}"
    print(header)
    print("-" * 120)
    
    test_images = get_random_images(config.TEST_DIR)
    if not test_images:
        print("No images found in test directory!")
        return
        
    transform = get_eval_transforms()
    
    # Store predictions
    results = {img_path: {"true": true_label, "preds": {}} for img_path, true_label in test_images}
    
    for model_name in config.MODEL_LIST:
        print(f"Inference with {model_name}...")
        model = get_model(model_name, config.NUM_CLASSES, freeze=True)
        model_path = os.path.join(config.MODELS_DIR, f"{model_name}_best.pth")
        
        if not os.path.exists(model_path):
            print(f"Skipping {model_name}: weights not found at {model_path}")
            continue
            
        try:
            model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
            model.eval()
            
            with torch.no_grad():
                for img_path, _ in test_images:
                    image = Image.open(img_path).convert("RGB")
                    tensor = transform(image).unsqueeze(0).to(config.DEVICE)
                    output = model(tensor)
                    _, predicted = torch.max(output, 1)
                    pred_label = config.CLASS_NAMES[predicted.item()]
                    results[img_path]["preds"][model_name] = pred_label
        except Exception as e:
            print(f"Error with {model_name}: {e}")

    print("\nFOLLOW-UP RANDOM PREDICTIONS TABLE\n")
    print("=" * 120)
    print(header)
    print("-" * 120)
    
    for img_path, data in results.items():
        fname = os.path.basename(img_path)[:30]
        t = data["true"]
        p18 = data["preds"].get("resnet18", "N/A")
        peb = data["preds"].get("efficientnet_b0", "N/A")
        pd1 = data["preds"].get("densenet121", "N/A")
        p50 = data["preds"].get("resnet50", "N/A")
        
        preds_list = [p18, peb, pd1, p50]
        valid_preds = [p for p in preds_list if p != "N/A"]
        match = "YES" if valid_preds and all(p == t for p in valid_preds) else "NO"
        
        print(f"{fname:<30} | {t:<12} | {p18:<12} | {peb:<12} | {pd1:<12} | {p50:<12} | {match}")

    print("=" * 120)

if __name__ == "__main__":
    main()
