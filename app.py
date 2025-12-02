import os
import json
import subprocess
from pathlib import Path
from PIL import Image
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Header, Footer, Input, Static, Button
from textual.reactive import reactive
from textual.suggester import SuggestFromList
from rich.text import Text
from rich.style import Style

from database import ImageDatabase

CONFIG_FILE = 'config.json'

def load_config(config_file):
    """Load configuration, creating it if it doesn't exist."""
    default_config = {
        "IMAGE_FOLDER": "",
        "DB_PATH": "images.db",
        "TAG_THRESHOLD": 0.35,
        "REPO_ID": "SmilingWolf/wd-vit-tagger-v3"
    }
    if Path(config_file).exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
        config = default_config.copy()
        config.update(user_config)
        return config
    else:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=4)
        return default_config

class ImagePreview(Static):
    """Widget to display an image converted to ASCII/Blocks."""
    def set_image(self, image_path, filename):
        try:
            with Image.open(image_path) as img:
                base_width = 80
                w_percent = (base_width / float(img.size[0]))
                h_size = int((float(img.size[1]) * float(w_percent)) * 0.5)
                img = img.resize((base_width, h_size), Image.Resampling.NEAREST).convert("RGB")
                
                ascii_art = Text()
                ascii_art.append(f"{filename}\n", style="bold underline yellow")
                pixels = img.load()
                width, height = img.size
                
                for y in range(height):
                    for x in range(width):
                        r, g, b = pixels[x, y]
                        ascii_art.append(" ", style=Style(bgcolor=f"rgb({r},{g},{b})"))
                    ascii_art.append("\n")
                self.update(ascii_art)
        except Exception as e:
            self.update(f"Error loading {filename}: {e}")

class MultiTagSuggester(SuggestFromList):
    """Custom suggester for comma-separated tags."""
    async def get_suggestion(self, value: str) -> str | None:
        if not value: return None
        parts = value.rsplit(',', 1)
        current_typing = parts[-1].strip()
        if not current_typing: return None
        for suggestion in self._suggestions:
            if suggestion.startswith(current_typing):
                prefix = parts[0] + ", " if len(parts) > 1 else ""
                return prefix + suggestion
        return None

class SelectableResult(Button):
    """A selectable search result button."""
    def __init__(self, *, filename: str, tags_str: str, score: float | None = None, **kwargs) -> None:
        self.filename = filename
        self.tags_str = tags_str
        self.score = score
        
        score_style = "[green]" if (score or 0) > 0.5 else "[yellow]" if (score or 0) > 0.25 else "[red]"
        score_text = f"{score_style}({self.score:.2f})[/]\n" if self.score is not None else ""
        
        label = f"[yellow b]{self.filename}[/yellow b]\n{score_text}- tags: {self.tags_str}"
        
        # Pass the constructed label to the parent Button's constructor
        super().__init__(label=label, **kwargs)

class TagSearchApp(App):
    """The TUI Application."""
    CSS = """
    Screen { layout: vertical; }
    #main_container { height: 1fr; }
    #left_pane { width: 40%; height: 100%; border-right: solid #888; overflow-y: scroll; padding: 1; }
    #right_pane { width: 60%; height: 100%; overflow-y: scroll; padding: 1; }
    #back_button { width: 100%; display: none; margin-bottom: 1; background: #555; }
    SelectableResult { width: 100%; height: auto; padding: 1; margin-bottom: 1; text-align: left; background: #333844; }
    SelectableResult:hover { background: #444a59; }
    Input { dock: bottom; border: heavy $accent; }
    """

    # -- State --
    all_data = {}
    all_tags = []
    app_mode = reactive('search')
    last_search_results = reactive([])
    last_search_query = reactive("")

    def __init__(self, image_folder, db_path, **kwargs):
        super().__init__(**kwargs)
        self.image_folder = Path(image_folder)
        self.db_path = db_path

    def on_mount(self) -> None:
        """Load data from database when app starts."""
        results_container = self.query_one("#results_list_container")
        db_file = Path(self.db_path)

        # Mount a temporary status message that we can update
        status_message = Static("Loading...")
        results_container.mount(status_message)

        if not db_file.exists() or db_file.stat().st_size == 0:
            status_message.update("[yellow]DB not found. Running classification...[/yellow]")
            with self.suspend():
                print("Running classification script, this may take a while...")
                subprocess.run(["python", "classify_images.py"], check=True)
                print("Classification finished. Starting UI...")

        db = ImageDatabase(self.db_path)
        raw_data = db.get_all_data()
        self.all_data = {os.path.basename(p): tags for p, tags in raw_data.items()}
        self.all_tags = db.get_all_tags()
        db.close()
        
        input_widget = self.query_one(Input)
        input_widget.suggester = MultiTagSuggester(self.all_tags, case_sensitive=False)

        # Now that everything is loaded, update the status message to the initial prompt.
        # This widget will be removed automatically when the user starts typing.
        status_message.update("Type tags to search...")
        
        input_widget.focus()

    def compose(self) -> ComposeResult:
        with Container(id="main_container"):
            with Horizontal():
                with Vertical(id="left_pane"):
                    yield Button("⬅ Back to Search", id="back_button")
                    yield Container(id="results_list_container")
                with Vertical(id="right_pane"):
                    yield Static("Previews appear here.", id="preview_area")
        yield Input(placeholder="Search: 1girl, red_hair... | Click a result to find similar", id="search_box")

    async def _update_results_list(self, results):
        results_container = self.query_one("#results_list_container")
        preview_container = self.query_one("#preview_area")
        await results_container.remove_children()
        await preview_container.remove_children()

        if not results:
            await results_container.mount(Static("[red]No matches found.[/red]"))
            return

        filenames = [r[0] if isinstance(r, tuple) else r for r in results]
        
        new_widgets = []
        for res in results:
            score = None
            if isinstance(res, tuple):
                fname, score = res
            else:
                fname = res
            
            tags = self.all_data.get(fname, {})
            tag_str = ", ".join(list(tags.keys())[:10]) + "..."
            new_widgets.append(SelectableResult(filename=fname, tags_str=tag_str, score=score))
        
        await results_container.mount(*new_widgets)

        if 0 < len(filenames) <= 5:
            for fname in filenames:
                full_path = self.image_folder / fname
                if full_path.exists():
                    img_widget = ImagePreview()
                    await preview_container.mount(img_widget)
                    img_widget.set_image(full_path, fname)
        elif len(filenames) > 5:
            await preview_container.mount(Static(f"[blue]{len(filenames)} results. Refine search for previews.[/blue]"))

    def watch_app_mode(self, new_mode: str) -> None:
        """Show/hide back button based on mode."""
        self.query_one("#back_button").display = (new_mode == 'similarity')

    async def on_input_changed(self, event: Input.Changed) -> None:
        """Handle real-time tag search."""
        if self.app_mode == 'similarity': return
        
        search_text = event.value.lower()
        self.last_search_query = search_text
        
        if not search_text.strip():
            await self.query_one("#results_list_container").remove_children()
            self.last_search_results = []
            return

        required_tags = [t.strip() for t in search_text.split(',') if t.strip()]
        matched_files = [
            fname for fname, ftags in self.all_data.items() 
            if all(req in ftags for req in required_tags)
        ]
        self.last_search_results = matched_files
        await self._update_results_list(matched_files)
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle clicks on result items or the back button."""
        if event.button.id == "back_button":
            self.app_mode = "search"
            self.query_one(Input).value = self.last_search_query
            await self._update_results_list(self.last_search_results)
        
        elif isinstance(event.button, SelectableResult):
            self.app_mode = "similarity"
            filename = event.button.filename
            full_path = str(self.image_folder / filename)
            
            self.query_one(Input).value = f"Similar to: {filename}"

            db = ImageDatabase(self.db_path)
            similar_images = db.find_similar_by_tags(full_path, limit=50)
            db.close()
            
            await self._update_results_list(similar_images)

if __name__ == "__main__":
    config = load_config(CONFIG_FILE)
    image_folder = config['IMAGE_FOLDER']
    db_path = config['DB_PATH']

    if not image_folder or not Path(image_folder).is_dir():
        print("Image folder not set.")
        while True:
            new_path = input("Enter path to your image folder: ")
            if Path(new_path).is_dir():
                image_folder = new_path
                config['IMAGE_FOLDER'] = new_path
                with open(CONFIG_FILE, 'w', encoding='utf-8') as f: json.dump(config, f, indent=4)
                break
            else: print(f"Error: '{new_path}' not found.")
    
    app = TagSearchApp(image_folder=image_folder, db_path=db_path)
    app.run()
