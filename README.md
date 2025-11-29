# AI-Powered Anime Image Tagger and Browser

A two-part system for automatically tagging anime images and browsing them in a terminal-based interface. This project uses a pre-trained AI model to classify images and a Textual TUI to provide a fast and efficient way to search and preview them by their tags.

## Features

- **Automatic Image Tagging**: Uses the `SmilingWolf/wd-vit-tagger-v3` model from Hugging Face to generate descriptive tags for your anime images.
- **Terminal-Based UI**: A fast, mouse-aware TUI built with Textual for searching and browsing.
- **Real-time Search**: Filters images instantly as you type.
- **Tag-based Filtering**: Find images by combining multiple tags (e.g., `1girl, red_hair, sword`).
- **In-Terminal Previews**: Displays image previews directly in the terminal using ASCII/block characters if 3 or fewer results are found.
- **Tag Autocompletion**: Suggests tags as you type to speed up searching.

## How It Works

1.  **Classification (`classify_images.py`)**: This script processes a folder of images. For each image, it generates tags and confidence scores using the ONNX model and saves the results into `output.json`.
2.  **TUI (`app.py`)**: This application reads `output.json` and provides the search interface. If `output.json` doesn't exist, it will automatically run the classification script first.

## Requirements

- Python 3.10+
- The following Python packages:
  - `textual`
  - `pillow`
  - `numpy`
  - `pandas`
  - `huggingface-hub`
  - `onnxruntime`
  - `rich`


## Installation


1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tungvn125/wall-classification.git
    cd wall-classification
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Initial Run & Configuration:**
    Run either script to generate the configuration file.
    ```bash
    python app.py
    ```
    - The script will detect that `config.json` is missing and create one for you with default settings.
    - It will then prompt you to enter the path to your image folder.

2.  **Edit `config.json`:**
    Open the newly created `config.json` file. You might want to adjust some parameters in there.
    ```json
    {
        "IMAGE_FOLDER": "/path/to/your/images",
        "OUTPUT_JSON": "output.json",
        "TAG_THRESHOLD": 0.35,
        "REPO_ID": "SmilingWolf/wd-vit-tagger-v3"
    }
    ```

3.  **Run the Application:**
    Simply run the `app.py` script:
    ```bash
    python app.py
    ```
    - The first time you run it with a valid image folder, the app will pause to download the model and classify your images. This may take a long time depending on your internet speed, the number of images, and your hardware.
    - On subsequent runs, the app will start immediately by loading the saved `output.json`.

4.  **Alternative (Manual Classification):**
    For large image collections, you can run the classification script separately first.
    ```bash
    python classify_images.py
    ```
    Once it's done, you can run the TUI:
    ```bash
    python app.py
    ```

## Future Plans

- Create a graphical user interface (GUI) for a more visual experience.

## Credits

-   **Image Tagging Model**: `SmilingWolf/wd-vit-tagger-v3` from [SmilingWolf](https://huggingface.co/SmilingWolf) on [HuggingFace](https://huggingface.co)

## License

This project is licensed under the MIT License.
