import os
import pickle
import base64
import io
import cv2

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

print("🚀 Initializing model and device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

backbone_arch = models.wide_resnet50_2(weights=None)
feature_extractor = nn.Sequential(*list(backbone_arch.children())[:-2]).to(device)

# Load the pre-trained weights
try:
    feature_extractor.load_state_dict(torch.load("universal_feature_extractor.pth", map_location=device))
    feature_extractor.eval()
    print("✅ Universal feature extractor loaded and set to evaluation mode.")
except FileNotFoundError:
    print("❌ ERROR: 'universal_feature_extractor.pth' not found.")
    print("Please place the model file in the same directory as app.py")
except Exception as e:
    print(f"❌ ERROR: Failed to load model: {e}")

# Define transforms
inference_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Helper Classes & Functions ---
class GoldenSampleDataset(Dataset):
    """Custom dataset to load images from in-memory file streams."""
    def __init__(self, file_streams, transform):
        self.file_streams = file_streams
        self.transform = transform
    
    def __len__(self):
        return len(self.file_streams)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.file_streams[idx]).convert("RGB")
            return self.transform(image)
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            # Return a dummy tensor if an image fails
            return torch.zeros((3, 224, 224))

def greedy_coreset_subsampling(feature_vectors, percentage=0.01):
    """Implements greedy coreset subsampling from the notebook."""
    n_samples = int(len(feature_vectors) * percentage)
    if n_samples == 0: n_samples = 1
    print(f"🧠 Starting coreset subsampling to select {n_samples} representative features...")
    if isinstance(feature_vectors, torch.Tensor):
        feature_vectors = feature_vectors.cpu().numpy()

    coreset_indices = [np.random.randint(len(feature_vectors))]
    min_distances = np.linalg.norm(feature_vectors - feature_vectors[coreset_indices[0]], axis=1)
    
    for i in range(1, n_samples):
        if i % 200 == 0:
            print(f"   ... selected {i}/{n_samples} features")
        next_idx = np.argmax(min_distances)
        coreset_indices.append(next_idx)
        new_distances = np.linalg.norm(feature_vectors - feature_vectors[next_idx], axis=1)
        min_distances = np.minimum(min_distances, new_distances)
        
    print(f"✅ Coreset subsampling complete.")
    return feature_vectors[coreset_indices]

# --- Image Conversion Utilities ---
def pil_to_base64(pil_image):
    """Converts a PIL Image to a base64 data URI."""
    with io.BytesIO() as buf:
        pil_image.save(buf, format="PNG")
        img_bytes = buf.getvalue()
    
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"

def heatmap_to_base64(anomaly_map):
    """Converts a numpy anomaly map to a JET colormap base64 data URI."""
    norm_map = (anomaly_map - np.min(anomaly_map)) / (np.max(anomaly_map) - np.min(anomaly_map) + 1e-6)
    norm_map = (norm_map * 255).astype(np.uint8)

    heatmap_img = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.png', heatmap_rgb)
    
    base64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"


# --- API Endpoints ---
@app.route('/setup', methods=['POST'])
def setup_product():
    """
    Endpoint to generate and save a product-specific coreset.
    Expects 'product_name' and 'files' in form data.
    """
    print("\n--- Received request on /setup ---")
    try:
        product_name = request.form['product_name']
        files = request.files.getlist('files')
        
        if not product_name:
            return jsonify({"error": "Product name is required"}), 400
        if not files:
            return jsonify({"error": "At least one golden sample image is required"}), 400

        print(f"Processing setup for product: '{product_name}' with {len(files)} images.")

        # Read file streams into a list to be used by the dataset
        file_streams = [io.BytesIO(file.read()) for file in files]
        
        # 1. Dataset Preparation
        golden_dataset = GoldenSampleDataset(file_streams, inference_transform)
        golden_loader = DataLoader(golden_dataset, batch_size=32, shuffle=False)
        print(f"Prepared {len(golden_dataset)} golden samples.")

        # 2. Feature Extraction
        features = {}
        def get_features_hook(name):
            def hook(model, input, output):
                features[name] = output
            return hook
        
        hook_layer2 = feature_extractor[5].register_forward_hook(get_features_hook('layer2'))
        hook_layer3 = feature_extractor[6].register_forward_hook(get_features_hook('layer3'))

        memory_bank = []
        print("🔍 Extracting features from golden samples...")
        with torch.no_grad():
            for images in golden_loader:
                images = images.to(device)
                _ = feature_extractor(images)
                
                layer2_features = features['layer2']
                layer3_features = features['layer3']
                
                upsampled_layer3 = torch.nn.functional.interpolate(
                    layer3_features, size=layer2_features.shape[2:], mode='bilinear', align_corners=False
                )
                combined_features = torch.cat((layer2_features, upsampled_layer3), dim=1)
                patch_embeddings = combined_features.permute(0, 2, 3, 1).flatten(0, 2).cpu().numpy()
                memory_bank.append(patch_embeddings)
        
        # Remove hooks
        hook_layer2.remove()
        hook_layer3.remove()

        memory_bank = np.concatenate(memory_bank, axis=0)
        print(f"✅ Memory bank created with {memory_bank.shape[0]} feature vectors.")

        # 3. Coreset Subsampling
        coreset = greedy_coreset_subsampling(memory_bank, percentage=0.01)
        
        # 4. Save Coreset
        coreset_filename = f"{product_name}_coreset.pkl"
        with open(coreset_filename, "wb") as f:
            pickle.dump(coreset, f)
        
        print(f"💾 Product-specific memory coreset saved to {coreset_filename}")

        return jsonify({
            "success": True,
            "productName": product_name,
            "coresetSize": coreset.shape[0],
            "message": f"Successfully generated coreset for {product_name}."
        })

    except Exception as e:
        print(f"❌ Error in /setup: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Close file streams
        if 'file_streams' in locals():
            for stream in file_streams:
                stream.close()

@app.route('/inference', methods=['POST'])
def run_inference():
    """
    Endpoint to run anomaly detection on a test image.
    Expects 'product_name' and 'file' in form data.
    """
    print("\n--- Received request on /inference ---")
    try:
        product_name = request.form['product_name']
        file = request.files['file']

        if not product_name:
            return jsonify({"error": "Product name is required"}), 400
        if not file:
            return jsonify({"error": "A test image is required"}), 400

        print(f"Running inference for product: '{product_name}'")

        # 1. Load Product-Specific Coreset
        coreset_filename = f"{product_name}_coreset.pkl"
        if not os.path.exists(coreset_filename):
            return jsonify({"error": f"Coreset for product '{product_name}' not found. Please run setup first."}), 404
        
        with open(coreset_filename, "rb") as f:
            coreset = pickle.load(f)
        print(f"✅ Loaded coreset '{coreset_filename}' with {coreset.shape[0]} features.")

        # 2. Prepare Nearest-Neighbor Search
        nn_searcher = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(coreset)
        print("✅ NN search algorithm ready.")

        # 3. Load and Preprocess Test Image
        image_pil = Image.open(file.stream).convert("RGB")
        test_tensor = inference_transform(image_pil).unsqueeze(0).to(device)

        # 4. Feature Extraction
        features = {}
        def get_features_hook(name):
            def hook(model, input, output):
                features[name] = output
            return hook
        
        hook_layer2 = feature_extractor[5].register_forward_hook(get_features_hook('layer2'))
        hook_layer3 = feature_extractor[6].register_forward_hook(get_features_hook('layer3'))

        with torch.no_grad():
            _ = feature_extractor(test_tensor)
            layer2_features = features['layer2']
            layer3_features = features['layer3']
            
            upsampled_layer3 = torch.nn.functional.interpolate(
                layer3_features, size=layer2_features.shape[2:], mode='bilinear', align_corners=False
            )
            combined_features = torch.cat((layer2_features, upsampled_layer3), dim=1)
            patch_embeddings = combined_features.permute(0, 2, 3, 1).flatten(0, 2).cpu().numpy()

        hook_layer2.remove()
        hook_layer3.remove()
        
        # 5. Nearest-Neighbor Search
        distances, _ = nn_searcher.kneighbors(patch_embeddings)
        patch_scores = distances.flatten()

        # 6. Anomaly Map Generation
        feature_map_size = combined_features.shape[2:]
        anomaly_map_low_res = patch_scores.reshape(feature_map_size)
        
        # Upsample to original image size
        anomaly_map_high_res = torch.nn.functional.interpolate(
            torch.tensor(anomaly_map_low_res).unsqueeze(0).unsqueeze(0),
            size=image_pil.size[::-1], # (height, width)
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()
        
        # 7. Image-Level Scoring
        image_level_score = np.max(patch_scores)
        IMAGE_THRESHOLD = 3.5  # From inference notebook
        decision = "ANOMALOUS 🔴" if image_level_score > IMAGE_THRESHOLD else "NORMAL 🟢"
        print(f"Inference complete. Score: {image_level_score:.4f}. Decision: {decision}")

        # 8. Convert images to Base64
        original_image_b64 = pil_to_base64(image_pil)
        heatmap_image_b64 = heatmap_to_base64(anomaly_map_high_res)

        return jsonify({
            "decision": decision,
            "score": f"{image_level_score:.4f}",
            "original_image_b64": original_image_b64,
            "heatmap_image_b64": heatmap_image_b64
        })

    except Exception as e:
        print(f"❌ Error in /inference: {e}")
        return jsonify({"error": str(e)}), 500

# --- Run the App ---
if __name__ == '__main__':
    print("Starting Flask server on http://localhost:5001")
    print("Ensure 'universal_feature_extractor.pth' is in this directory.")
    app.run(debug=False, port=5001, host='0.0.0.0')