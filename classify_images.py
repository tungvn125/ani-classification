import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download
import onnxruntime as ort
from rich.progress import Progress

# ================= CONFIGURATION =================
# Folder containing images to check
IMAGE_FOLDER = ''

# Path to save the output JSON
OUTPUT_JSON = 'output.json'

# Threshold for including a tag (standard for WD1.4 models is ~0.35)
TAG_THRESHOLD = 0.35

# Model Repository ID
REPO_ID = "SmilingWolf/wd-vit-tagger-v3"
# =================================================

def load_model():
    """Download and load the ONNX model and tags."""
    print(f"Loading model from {REPO_ID}...")
    
    # Download model.onnx
    model_path = hf_hub_download(repo_id=REPO_ID, filename="model.onnx")
    
    # Download selected_tags.csv
    tags_path = hf_hub_download(repo_id=REPO_ID, filename="selected_tags.csv")
    
    # Load tags
    print("Loading tags...")
    tags_df = pd.read_csv(tags_path)
    tag_names = tags_df['name'].tolist()
    
    # Load ONNX session
    # Use CUDAExecutionProvider if available, else CPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        providers = ['CPUExecutionProvider']
    
    print(f"Starting Inference Session with providers: {providers}")
    session = ort.InferenceSession(model_path, providers=providers)
    
    return session, tag_names

def preprocess_image(image_path, target_size=448):
    """
    Load and preprocess an image for the WD1.4 Tagger.
    Model expects: BGR, channel-last, pixels 0-255 (float32).
    """
    try:
        img = Image.open(image_path).convert('RGB')
        
        # Resize with padding or direct resize? 
        # Standard WD14 implementations usually just force resize with BICUBIC.
        img = img.resize((target_size, target_size), Image.BICUBIC)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # RGB to BGR (Standard for these ONNX models)
        img_array = img_array[:, :, ::-1]
        
        # Add batch dimension: (1, 448, 448, 3)
        input_tensor = np.expand_dims(img_array, axis=0)
        
        return input_tensor
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_folder(folder_path, output_path, threshold):
    session, tag_names = load_model()
    input_name = session.get_inputs()[0].name
    
    results = {}
    
    # Get list of image files
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    
    print(f"Found {len(files)} images to process.")
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing images... ", total=len(files))

        for filename in files:
            #print("\033[31mĐây là văn bản màu đỏ!\033[0m")
            #print("\033[1m\033[32mĐây là văn bản màu xanh lá cây và in đậm!\033[0m")
            #print("\033[33mĐây là văn bản màu vàng!\033[0m")

            print(f"\033[33mDEBUG: Processing {filename}\033[0m")
            progress.update(task, description=f"[cyan]Processing...")
            file_path = os.path.join(folder_path, filename)
            
            input_tensor = preprocess_image(file_path)
            
            if input_tensor is None:
                progress.update(task, advance=1)
                continue
            
            # Run inference
            probs = session.run(None, {input_name: input_tensor})[0][0]
            
            # Parse results
            image_results = {
                "ratings": {},
                "tags": {}
            }
            
            # The first 4 tags are usually ratings (general, sensitive, questionable, explicit)
            # However, checking the CSV structure is safer. 
            # Usually WD1.4 tags CSV has a 'category' column: 9=rating, 0=general, 4=character
            # But simply mapping index -> tag name is sufficient for the output.
            
            # Let's extract tags above threshold
            for idx, probability in enumerate(probs):
                tag_name = tag_names[idx]
                
                # Simple heuristic: Rating tags often start with "rating:" or occur at the start
                if tag_name.startswith("rating:"):
                    image_results["ratings"][tag_name.replace("rating:", "")] = float(probability)
                elif probability > threshold:
                    image_results["tags"][tag_name] = float(probability)
            
            # Sort tags by confidence
            image_results["tags"] = dict(sorted(image_results["tags"].items(), key=lambda item: item[1], reverse=True))
    
            results[filename] = image_results

            progress.update(task, advance=1)
            print(f"\033[1m\033[32mDEBUG: Finished {filename}\033[0m")

    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    print(f"Done! Results saved to {output_path}")

if __name__ == "__main__":
    if IMAGE_FOLDER == '':
        print("Error: IMAGE_FOLDER is not set. Please set the path to the folder containing images.")
    else:
        if not os.path.exists(IMAGE_FOLDER):
            print(f"Error: Folder '{IMAGE_FOLDER}' does not exist.")
        else:
            process_folder(IMAGE_FOLDER, OUTPUT_JSON, TAG_THRESHOLD)
