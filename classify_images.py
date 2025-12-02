import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download
import onnxruntime as ort
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Optional

from database import ImageDatabase

# ================= CONFIGURATION =================
CONFIG_FILE = 'config.json'
BATCH_SIZE = 8 # Adjust based on your VRAM/RAM
# =================================================

class ClassificationResult:
    """Holds the classification results for a single image."""
    def __init__(self, filepath: str, tags: Dict[str, float]):
        self.filepath = filepath
        self.tags = tags
        self.confidence_score = self._get_overall_confidence(tags)

    def _get_overall_confidence(self, tags: Dict[str, float]) -> float:
        """Calculate an overall confidence score (e.g., average of top N tags, or max)."""
        # Simple average of all non-rating tags
        non_rating_tags = [
            conf for tag, conf in tags.items() if not tag.startswith("rating:")
        ]
        return sum(non_rating_tags) / len(non_rating_tags) if non_rating_tags else 0.0

    def get_display_tags(self, top_n: int = 5) -> str:
        """Returns a string of the top N tags."""
        sorted_tags = sorted(
            [ (tag, conf) for tag, conf in self.tags.items() if not tag.startswith("rating:") ],
            key=lambda item: item[1],
            reverse=True
        )
        return ", ".join([tag for tag, _ in sorted_tags[:top_n]])

class Classifier:
    def __init__(self, config_file: str = CONFIG_FILE):
        self.config = self._load_config(config_file)
        self.session: Optional[ort.InferenceSession] = None
        self.tag_names: List[str] = []
        self.db: Optional[ImageDatabase] = None

    def _load_config(self, config_file: str) -> Dict[str, any]:
        """Load configuration from a JSON file."""
        default_config = {
            "IMAGE_FOLDER": "", "DB_PATH": "images.db", "TAG_THRESHOLD": 0.35,
            "REPO_ID": "SmilingWolf/wd-vit-tagger-v3"
        }
        if Path(config_file).exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            config = default_config.copy()
            config.update(user_config)
            return config
        else:
            # Create config file with defaults if it doesn't exist
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4)
            return default_config

    def load_model(self):
        """Download and load the ONNX model and tags."""
        print(f"Loading model from {self.config['REPO_ID']}...")
        model_path = hf_hub_download(repo_id=self.config['REPO_ID'], filename="model.onnx")
        tags_path = hf_hub_download(repo_id=self.config['REPO_ID'], filename="selected_tags.csv")
        
        print("Loading tags...")
        tags_df = pd.read_csv(tags_path)
        self.tag_names = tags_df['name'].tolist()
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            print("CUDAExecutionProvider not found. Using CPU.")
            providers = ['CPUExecutionProvider']
        
        print(f"Starting Inference Session with providers: {providers}")
        self.session = ort.InferenceSession(model_path, providers=providers)

    def preprocess_image(self, image_path: Path, target_size: int = 448) -> Optional[np.ndarray]:
        """Load and preprocess a single image for batching."""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((target_size, target_size), Image.BICUBIC)
            img_array = np.array(img, dtype=np.float32)
            return img_array[:, :, ::-1]  # RGB to BGR
        except Exception as e:
            print(f"Error processing {image_path}: {e}") # This print should be handled by GUI
            return None

    def classify_images_in_folder(self, 
                                  folder_path: Path, 
                                  progress_callback: Optional[Callable[[int, int, str], None]] = None,
                                  batch_size: int = BATCH_SIZE) -> List[ClassificationResult]:
        """
        Processes images in batches, classifies them, saves results to DB, and yields ClassificationResult.
        """
        if not self.session:
            self.load_model()
        if not self.db:
            self.db = ImageDatabase(self.config['DB_PATH'])

        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        
        all_files_in_folder = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
        files_to_process = [
            f for f in all_files_in_folder if not self.db.is_image_classified(str(folder_path / f))
        ]
        
        total_new_images = len(files_to_process)
        if total_new_images == 0:
            print("No new images to classify.") # This print should be handled by GUI
            self.db.close()
            return []

        classified_results: List[ClassificationResult] = []
        input_name = self.session.get_inputs()[0].name
        
        current_processed = 0
        for i in range(0, total_new_images, batch_size):
            batch_filenames = files_to_process[i:i + batch_size]
            batch_input = []
            valid_paths_in_batch = []

            for filename in batch_filenames:
                file_path = folder_path / filename
                img_array = self.preprocess_image(file_path)
                if img_array is not None:
                    batch_input.append(img_array)
                    valid_paths_in_batch.append(file_path)
            
            if not batch_input:
                current_processed += len(batch_filenames)
                if progress_callback:
                    progress_callback(current_processed, total_new_images, "Processing...")
                continue

            input_tensor = np.stack(batch_input, axis=0)
            all_probs = self.session.run(None, {input_name: input_tensor})[0]

            for idx, probs in enumerate(all_probs):
                filepath_to_save = str(valid_paths_in_batch[idx])
                tags_with_confidence = {}
                
                # Extract tags above threshold and all rating tags
                for tag_idx, probability in enumerate(probs):
                    tag_name = self.tag_names[tag_idx]
                    if tag_name.startswith("rating:") or probability > self.config['TAG_THRESHOLD']:
                        tags_with_confidence[tag_name] = float(probability)
                
                if tags_with_confidence:
                    self.db.add_classification_data(filepath_to_save, tags_with_confidence)
                    classified_results.append(
                        ClassificationResult(filepath_to_save, tags_with_confidence)
                    )
            
            current_processed += len(batch_filenames)
            if progress_callback:
                progress_callback(current_processed, total_new_images, 
                                  f"Classifying {current_processed}/{total_new_images} images...")

        self.db.close()
        return classified_results

if __name__ == "__main__":
    # Example usage for CLI - this part will be mostly ignored by GUI
    classifier = Classifier()
    config = classifier.config # Access config after classifier is initialized
    
    image_folder_path = Path(config['IMAGE_FOLDER'])
    
    if not image_folder_path.is_dir():
        print("\033[93mWarning: IMAGE_FOLDER is not set or is not a valid directory.\033[0m")
        while True:
            new_path_str = input("Please enter the path to your image folder: ")
            image_folder_path = Path(new_path_str)
            if image_folder_path.is_dir():
                config['IMAGE_FOLDER'] = str(image_folder_path)
                with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=4)
                break
            else:
                print(f"Error: Folder '{new_path_str}' does not exist or is not a directory.")

    def cli_progress_callback(current, total, message):
        print(f"CLI Progress: {message} ({current}/{total})")

    results = classifier.classify_images_in_folder(
        image_folder_path, 
        progress_callback=cli_progress_callback
    )
    print(f"Done! Classified {len(results)} new images.")

