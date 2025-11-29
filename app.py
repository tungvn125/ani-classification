import os
import json
from PIL import Image
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Header, Footer, Input, Static, Label
from textual.reactive import reactive
from textual.suggester import SuggestFromList
from rich.text import Text
from rich.style import Style

# ================= CONFIGURATION =================
# Must match the folder used in the previous script
IMAGE_FOLDER = ''
JSON_FILE = 'output.json'
# =================================================

class ImagePreview(Static):
    """Widget to display an image converted to ASCII/Blocks."""
    def set_image(self, image_path, filename):
        try:
            # Open image
            with Image.open(image_path) as img:
                # Resize for terminal (maintain aspect ratio roughly)
                # Terminal characters are about twice as tall as they are wide
                # So we resize width to 40-60 chars, and height appropriately
                base_width = 40
                w_percent = (base_width / float(img.size[0]))
                h_size = int((float(img.size[1]) * float(w_percent)) * 0.5)
                img = img.resize((base_width, h_size), Image.Resampling.NEAREST).convert("RGB")

                # Build Rich Text object
                ascii_art = Text()
                ascii_art.append(f"{filename}\n", style="bold underline yellow")
                
                pixels = img.load()
                width, height = img.size
                
                for y in range(height):
                    for x in range(width):
                        r, g, b = pixels[x, y]
                        # Use a block character with the pixel color as background
                        ascii_art.append("  ", style=Style(bgcolor=f"rgb({r},{g},{b})"))
                    ascii_art.append("\n")
                
                self.update(ascii_art)
        except Exception as e:
            self.update(f"Error loading {filename}: {e}")

class MultiTagSuggester(SuggestFromList):
    """Custom suggester to handle comma-separated tags."""
    async def get_suggestion(self, value: str) -> str | None:
        if not value:
            return None
        
        # Split by comma to find the last tag being typed
        parts = value.rsplit(',', 1)
        current_typing = parts[-1].strip()
        
        # If user just typed a comma and space, don't suggest yet
        if not current_typing:
            return None

        # Find match for the last part
        for suggestion in self._suggestions:
            if suggestion.startswith(current_typing):
                # Reconstruct the full string with the new suggestion
                prefix = parts[0] + ", " if len(parts) > 1 else ""
                return prefix + suggestion + ", "
        
        return None

class TagSearchApp(App):
    """The TUI Application."""
    # transparent background    
    CSS = """
    Screen {
        layout: vertical;
        background: transparent;
    }

    #main_container {
        height: 1fr;
        border: #BAABB0;
        background: transparent;
    }

    #left_pane {
        width: 50%;
        height: 100%;
        border-right: #DDBFC7;
        overflow-y: scroll;
        padding: 1;
    }

    #right_pane {
        width: 50%;
        height: 100%;
        overflow-y: scroll;
        padding: 1;
    }

    .result_item {
        margin-bottom: 1;
        background: #333844;
        padding: 1;
    }

    .filename {
        color: yellow;
        text-style: bold;
    }
    
    .tags {
        color: grey;
    }

    Input {
        dock: bottom;
        border: heavy $accent;
    }
    """

    # State variables
    all_data = {}
    all_tags = set()

    def on_mount(self) -> None:
        """Load JSON data when app starts."""
        if not os.path.exists(JSON_FILE):
            # Suspend the TUI to show the progress from the classification script
            with self.suspend():
                from classify_images import process_folder
                process_folder(IMAGE_FOLDER, JSON_FILE, threshold=0.35)
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            self.all_data = json.load(f)

        # Collect all unique tags for autocomplete
        for file_data in self.all_data.values():
            for tag in file_data.get('tags', {}).keys():
                self.all_tags.add(tag)
        
        # Setup autocomplete
        input_widget = self.query_one(Input)
        input_widget.suggester = MultiTagSuggester(list(self.all_tags), case_sensitive=True)
        input_widget.focus()

    def compose(self) -> ComposeResult:
        
        with Container(id="main_container"):
            with Horizontal():
                # Left Side: Text Results
                with Vertical(id="left_pane"):
                    yield Static("Type tags below to search...", id="results_list")
                
                # Right Side: Image Previews
                with Vertical(id="right_pane"):
                    yield Static("Previews appear here when results <= 3", id="preview_area")

        yield Input(placeholder="Search tags (e.g., 1girl, dark, rain)... (Press TAB to complete)", id="search_box")

    async def on_input_changed(self, event: Input.Changed) -> None:
        """Handle text input changes in real-time."""
        search_text = event.value.lower()
        
        # Get containers
        results_container = self.query_one("#left_pane")
        preview_container = self.query_one("#right_pane")
        
        # Clear containers (Must await this!)
        await results_container.remove_children()
        await preview_container.remove_children()

        if not search_text.strip():
            await results_container.mount(Static("Please enter tags."))
            return

        # Split input into required tags (comma separated)
        required_tags = [t.strip() for t in search_text.split(',') if t.strip()]
        
        matched_files = []

        # FILTER LOGIC
        for filename, data in self.all_data.items():
            file_tags = data.get('tags', {})
            # Check if ALL required tags are present in the image's tags
            if all(req in file_tags for req in required_tags):
                matched_files.append(filename)

        # 1. UPDATE LEFT PANE (Text List)
        if not matched_files:
            # FIX: Use [red]...[/red] markup instead of style="red"
            await results_container.mount(Static("[red]No matches found.[/red]"))
        else:
            new_widgets = []
            for fname in matched_files:
                # Format tags for display (top 10 tags)
                tags = self.all_data[fname]['tags']
                tag_str = ", ".join(list(tags.keys())[:10]) + "..."
                
                # Create text widget
                info = f"[{fname}]\n- tags: {tag_str}"
                new_widgets.append(Static(info, classes="result_item"))
            
            # Mount all text results at once
            await results_container.mount(*new_widgets)

        # 2. UPDATE RIGHT PANE (Preview) - Only if <= 3 matches
        if 0 < len(matched_files) <= 3:
            for fname in matched_files:
                full_path = os.path.join(IMAGE_FOLDER, fname)
                if os.path.exists(full_path):
                    img_widget = ImagePreview()
                    await preview_container.mount(img_widget)
                    img_widget.set_image(full_path, fname)
                else:
                    # FIX: Use markup
                    await preview_container.mount(Static(f"[red]Image file missing: {fname}[/red]"))
        elif len(matched_files) > 3:
            # FIX: Use markup
            await preview_container.mount(Static(f"[blue]Found {len(matched_files)} images.\nRefine search to <= 3 to see previews.[/blue]"))
if __name__ == "__main__":
    if IMAGE_FOLDER == '':
        print("Error: IMAGE_FOLDER is not set. Please set the path to the folder containing images.")
    else:
        app = TagSearchApp()
        app.run()
