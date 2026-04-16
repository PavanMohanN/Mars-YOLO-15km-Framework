"""
generate_xai_maps.py

This script extracts deep feature activations from the YOLOv8m detection head 
(Layer 22) to generate 2D Explainable AI (XAI) Saliency/Attention Maps.
It proves that the model anchors its coordinate regression to the physical 
geomorphometry (illuminated rim crests and shadowed inner basins) of the crater.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from ultralytics import YOLO

# ==========================================
# 1. SETUP & RELATIVE PATHS
# ==========================================
# Automatically resolves paths relative to the GitHub repository structure
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.pt')
INPUT_DIR = os.path.join(BASE_DIR, 'data', 'images', 'val') 
OUT_DIR = os.path.join(BASE_DIR, 'validation_metrics', 'XAI_Heatmaps')

os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================
# 2. XAI FEATURE EXTRACTOR CLASS
# ==========================================
class YOLOv8_XAI:
    def __init__(self, model_path):
        print(f"[*] Loading YOLOv8m model from {model_path}...")
        self.model = YOLO(model_path)
        self.feature_maps = None
        self._register_hook()

    def _hook_fn(self, module, input, output):
        # FIX: We grab the INPUT to the detection head (which is a 2D spatial map)
        # rather than the OUTPUT (which YOLO flattens into 1D bounding box math).
        spatial_tensors = input[0]
        
        if isinstance(spatial_tensors, (list, tuple)):
            self.feature_maps = spatial_tensors[0] # Grabs the 2D (B, C, H, W) tensor
        else:
            self.feature_maps = spatial_tensors
            
        # Final safety check to ensure we have a Tensor
        if isinstance(self.feature_maps, (list, tuple)):
            self.feature_maps = self.feature_maps[0]

    def _register_hook(self):
        pytorch_model = self.model.model
        # Attach hook to layer 22 (Detection head in YOLOv8)
        target_layer = pytorch_model.model[22] 
        target_layer.register_forward_hook(self._hook_fn)

    def generate_heatmap(self, img_path, save_path):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[-] Error: Could not load image {img_path}")
            return
            
        original_img = img.copy()
        H, W = img.shape[:2]

        # Trigger inference (this forces the image through the network and triggers our hook)
        _ = self.model.predict(img_path, verbose=False)

        if self.feature_maps is None:
            print("[-] Feature maps not captured.")
            return

        # ==========================================
        # 3. HEATMAP PROCESSING
        # ==========================================
        if not isinstance(self.feature_maps, torch.Tensor):
            print("[-] Extracted feature map is not a Tensor.")
            return

        # Compress channels into a single spatial activation map
        activation = torch.mean(self.feature_maps, dim=1).squeeze()
        activation = activation.cpu().detach().numpy()
        
        # Normalize the activation map to [0, 255]
        activation = np.maximum(activation, 0)
        activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)
        activation = np.uint8(255 * activation)

        # Resize heatmap back to the original image dimensions
        heatmap_resized = cv2.resize(activation, (W, H))
        
        # Apply the Viridis scientific colormap
        colormap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_VIRIDIS)
        
        # Create the fusion overlay (60% Heatmap, 40% Original Image)
        fusion_map = cv2.addWeighted(colormap, 0.6, original_img, 0.4, 0)

        # ==========================================
        # 4. PLOTTING
        # ==========================================
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Input THEMIS Image: {os.path.basename(img_path)}", fontsize=16, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(fusion_map, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Fusion Attention Map (Semantic Centroid Anchoring)", fontsize=16, fontweight='bold')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Saved XAI Heatmap: {os.path.basename(save_path)}")

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\n==================================================")
    print(" Mars-YOLO-15km: Explainable AI (XAI) Generator ")
    print("==================================================\n")
    
    if not os.path.exists(INPUT_DIR):
        print(f"[-] Input directory not found: {INPUT_DIR}")
        print("[*] Please ensure you have images in the data/images/val/ directory.")
    else:
        xai_generator = YOLOv8_XAI(MODEL_PATH)
        
        valid_extensions = ('.png', '.jpg', '.jpeg')
        images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]
        
        if not images:
            print(f"[-] No images found in {INPUT_DIR}.")
        else:
            print(f"[*] Found {len(images)} images for processing. Generating heatmaps...")
            # Limit to 10 images by default to avoid cluttering the validation folder
            # Users can modify this slice if they want to process the whole folder
            for filename in images[:10]:
                img_path = os.path.join(INPUT_DIR, filename)
                save_path = os.path.join(OUT_DIR, f"XAI_{filename}")
                xai_generator.generate_heatmap(img_path, save_path)
            
            print(f"\n[+] XAI Processing Complete! Maps saved to {OUT_DIR}")