import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QLineEdit, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QScrollArea, QFrame, QDialog, QMessageBox, QMenuBar, QMenu
)
from PyQt6.QtGui import QPixmap, QIcon, QAction
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# Import the refactored Classifier from classify_images.py
from classify_images import Classifier, ClassificationResult
from database import ImageDatabase # Import ImageDatabase for search functions

class ClassificationWorker(QThread):
    """
    Worker thread to run the image classification in the background.
    Emits signals for progress updates and individual classification results.
    """
    progress_updated = pyqtSignal(int, int, str) # current, total, message
    result_ready = pyqtSignal(ClassificationResult)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, classifier: Classifier, folder_path: Path):
        super().__init__()
        self.classifier = classifier
        self.folder_path = folder_path

    def run(self):
        try:
            # Load model if not already loaded (will only happen once per app run)
            if not self.classifier.session:
                self.progress_updated.emit(0, 1, "Loading AI model...")
                self.classifier.load_model()
            
            # Start classification
            results = self.classifier.classify_images_in_folder(
                self.folder_path,
                progress_callback=self._emit_progress
            )
            
            for res in results:
                self.result_ready.emit(res)
            
            self.finished.emit()
        except Exception as e:
            error_message = f"Classification Error: {e}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_message)
            self.finished.emit() # Ensure finished signal is always emitted

    def _emit_progress(self, current: int, total: int, message: str):
        self.progress_updated.emit(current, total, message)

class ImageClassifierGUI(QWidget):
    def __init__(self):
        super().__init__()
        # Initialize core attributes BEFORE initUI() to ensure they exist
        self.classifier = Classifier()
        self.db = ImageDatabase(self.classifier.config['DB_PATH'])
        self.current_worker: Optional[ClassificationWorker] = None
        self.all_classified_images: List[ClassificationResult] = []
        self.filtered_images: List[ClassificationResult] = []
        self.last_search_query: str = ""
        self.app_mode: str = "search"

        self.setWindowTitle("AI-Powered Anime Image Classifier")
        self.setGeometry(100, 100, 1200, 800) # Larger window for better layout
        
        self.initUI() # Now call initUI() after attributes are initialized
        
        self.load_existing_data() # Load data from DB on startup
        self.apply_stylesheet("dark") # Default to dark theme
        self.update_status("Ready")

    def initUI(self):
        # --- Main Vertical Layout ---
        main_v_layout = QVBoxLayout()

        # --- Menu Bar ---
        menu_bar = QMenuBar(self)
        self.theme_menu = QMenu("Theme", self)
        
        dark_mode_action = QAction("Dark Mode", self)
        dark_mode_action.triggered.connect(lambda: self.apply_stylesheet("dark"))
        self.theme_menu.addAction(dark_mode_action)

        light_mode_action = QAction("Light Mode", self)
        light_mode_action.triggered.connect(lambda: self.apply_stylesheet("light"))
        self.theme_menu.addAction(light_mode_action)
        
        menu_bar.addMenu(self.theme_menu)
        main_v_layout.setMenuBar(menu_bar) # QWidget doesn't have a menu bar directly, add to layout

        # --- Horizontal Layout for Content ---
        content_h_layout = QHBoxLayout()

        # --- Left Pane (Controls and Results) ---
        left_pane_layout = QVBoxLayout()

        # 1. Search Bar
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search tags (e.g., 1girl, dark, rain)...")
        self.search_input.textChanged.connect(self.filter_results_by_tags)
        left_pane_layout.addWidget(self.search_input)

        # 2. File/Directory Selection
        file_selection_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Select image or directory to classify...")
        self.path_input.setReadOnly(True)
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.select_path)
        file_selection_layout.addWidget(self.path_input)
        file_selection_layout.addWidget(self.browse_button)
        left_pane_layout.addLayout(file_selection_layout)

        # 3. Classification and Action Buttons
        button_layout = QHBoxLayout()
        self.classify_button = QPushButton("Start Classification")
        self.classify_button.clicked.connect(self.start_classification)
        self.classify_button.setEnabled(False) # Disable until path is selected
        button_layout.addWidget(self.classify_button)

        self.find_similar_button = QPushButton("Find Similar")
        self.find_similar_button.clicked.connect(self.find_similar_images)
        self.find_similar_button.setEnabled(False) # Disable until a row is selected
        button_layout.addWidget(self.find_similar_button)

        self.back_button = QPushButton("Back to Search")
        self.back_button.clicked.connect(self.back_to_search)
        self.back_button.setVisible(False) # Hidden by default
        button_layout.addWidget(self.back_button)

        left_pane_layout.addLayout(button_layout)

        # 4. Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        left_pane_layout.addWidget(self.progress_bar)

        # 5. Results Table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Image File Path", "Tags", "Confidence"])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers) # Make table read-only
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.itemSelectionChanged.connect(self.handle_table_selection_change)
        left_pane_layout.addWidget(self.results_table)

        content_h_layout.addLayout(left_pane_layout)

        # --- Right Pane (Image Preview) ---
        right_pane_layout = QVBoxLayout()
        self.image_preview_label = QLabel("No image selected for preview")
        self.image_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview_label.setFrameShape(QFrame.Shape.Box)
        self.image_preview_label.setScaledContents(False)
        self.image_preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.image_preview_label)
        right_pane_layout.addWidget(scroll_area)

        content_h_layout.addLayout(right_pane_layout)
        
        main_v_layout.addLayout(content_h_layout)

        # --- Status Bar ---
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("status_label") # For QSS styling
        self.status_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        main_v_layout.addWidget(self.status_label)
        
        self.setLayout(main_v_layout)
        
        # Initial population of the table
        self._populate_results_table(self.all_classified_images)
        self.handle_table_selection_change() # Update button states initially

    def apply_stylesheet(self, theme: str):
        qss_path = Path(f"{theme}_theme.qss")
        if qss_path.exists():
            with open(qss_path, "r") as f:
                self.setStyleSheet(f.read())
            self.current_theme = theme
            self.update_status(f"Switched to {theme} theme.")
        else:
            self.update_status(f"Error: {theme}_theme.qss not found.")

    def load_existing_data(self):
        self.update_status("Loading existing classified images from database...")
        db_path = self.classifier.config['DB_PATH']
        # Classifier already has a db instance, use it
        all_data_raw = self.db.get_all_data()
        
        for filepath_str, tags_dict in all_data_raw.items():
            self.all_classified_images.append(ClassificationResult(filepath_str, tags_dict))
        
        self.filtered_images = list(self.all_classified_images) # Initially all images are filtered
        self.update_status(f"Loaded {len(self.all_classified_images)} existing images.")

    def _populate_results_table(self, images_to_display: List[ClassificationResult], with_score: bool = False):
        self.results_table.setRowCount(0) # Clear existing rows
        
        for result in images_to_display:
            row_position = self.results_table.rowCount()
            self.results_table.insertRow(row_position)
            
            filepath_item = QTableWidgetItem(result.filepath)
            tags_item = QTableWidgetItem(result.get_display_tags())
            
            if with_score and hasattr(result, 'similarity_score') and result.similarity_score is not None:
                confidence_item = QTableWidgetItem(f"{result.similarity_score:.2f}")
            else:
                confidence_item = QTableWidgetItem(f"{result.confidence_score:.2f}")

            self.results_table.setItem(row_position, 0, filepath_item)
            self.results_table.setItem(row_position, 1, tags_item)
            self.results_table.setItem(row_position, 2, confidence_item)

        self.results_table.resizeColumnsToContents()

    def handle_table_selection_change(self):
        selected_rows = self.results_table.selectionModel().selectedRows()
        self.find_similar_button.setEnabled(len(selected_rows) > 0 and self.app_mode == "search")
        self.display_image_preview() # Also update preview

    def filter_results_by_tags(self, search_text: str):
        self.last_search_query = search_text
        self.app_mode = "search"
        self.back_button.setVisible(False)
        self.find_similar_button.setVisible(True)

        if not search_text.strip():
            self.filtered_images = list(self.all_classified_images)
        else:
            required_tags = [t.strip().lower() for t in search_text.split(',') if t.strip()]
            self.filtered_images = [
                img_res for img_res in self.all_classified_images
                if all(req_tag in {tag.lower() for tag in img_res.tags.keys()} for req_tag in required_tags)
            ]
        
        self._populate_results_table(self.filtered_images)
        self.handle_table_selection_change() # Update button states and preview

    def find_similar_images(self):
        selected_rows = self.results_table.selectionModel().selectedRows()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        source_filepath = self.results_table.item(row, 0).text()
        
        self.app_mode = "similar"
        self.find_similar_button.setVisible(False)
        self.back_button.setVisible(True)
        self.update_status(f"Finding similar images to {Path(source_filepath).name}...")

        similar_results_raw = self.db.find_similar_by_tags(source_filepath, limit=50)
        similar_images_results: List[ClassificationResult] = []
        for filepath_str, similarity_score in similar_results_raw:
            # Find the full ClassificationResult object for the similar image
            original_result = next((res for res in self.all_classified_images if res.filepath == filepath_str), None)
            if original_result:
                # Create a temporary result to hold the similarity score for display
                temp_result = ClassificationResult(original_result.filepath, original_result.tags)
                temp_result.similarity_score = similarity_score # Add similarity_score attribute
                similar_images_results.append(temp_result)
        
        self._populate_results_table(similar_images_results, with_score=True)
        self.update_status(f"Found {len(similar_images_results)} similar images.")
        self.results_table.selectRow(0) # Select first similar image
        self.handle_table_selection_change() # Update preview

    def back_to_search(self):
        self.app_mode = "search"
        self.back_button.setVisible(False)
        self.find_similar_button.setVisible(True)
        self.search_input.setText(self.last_search_query) # This will trigger filter_results_by_tags
        self.update_status("Returned to tag search.")

    def select_path(self):
        # File dialog setup
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.webp *.bmp);;Directories (*)")
        
        if file_dialog.exec() == QDialog.DialogCode.Accepted:
            selected_paths = file_dialog.selectedFiles()
            if selected_paths:
                # Determine if a single file or a directory is selected
                first_path = Path(selected_paths[0])
                if first_path.is_file() and len(selected_paths) == 1:
                    # If a single file, classify only that file
                    self.current_selection_path = first_path.parent # Classifier expects a folder path
                    self.single_image_to_process = first_path # Store the specific image
                    self.path_input.setText(str(first_path))
                elif first_path.is_dir() or len(selected_paths) > 1:
                    # If a directory or multiple files, classify the entire directory
                    self.current_selection_path = first_path.parent if first_path.is_file() else first_path
                    self.single_image_to_process = None
                    self.path_input.setText(str(self.current_selection_path))
                
                self.classify_button.setEnabled(True)
                self.update_status(f"Selected: {self.current_selection_path}")
            else:
                self.classify_button.setEnabled(False)
                self.update_status("No path selected.")

    def start_classification(self):
        if not hasattr(self, 'current_selection_path') or not self.current_selection_path.exists():
            QMessageBox.warning(self, "Input Error", "Please select a valid image file or directory.")
            return

        # Disable UI elements during classification
        self.classify_button.setEnabled(False)
        self.path_input.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.search_input.setEnabled(False)
        self.find_similar_button.setEnabled(False)
        self.back_button.setEnabled(False)
        self.results_table.setRowCount(0) # Clear previous results
        self.progress_bar.setValue(0)
        self.update_status("Classification started...")
        self.image_preview_label.clear()
        self.image_preview_label.setText("No image selected for preview")

        # Initialize and start the worker thread
        self.current_worker = ClassificationWorker(self.classifier, self.current_selection_path)
        self.current_worker.progress_updated.connect(self.update_progress)
        self.current_worker.result_ready.connect(self.add_new_classification_result)
        self.current_worker.finished.connect(self.classification_finished)
        self.current_worker.error_occurred.connect(self.handle_classification_error)
        self.current_worker.start()

    def add_new_classification_result(self, result: ClassificationResult):
        self.all_classified_images.append(result)
        # Update filtered_images if in search mode and the new result matches current filter
        if self.app_mode == "search":
            search_text = self.search_input.text()
            if not search_text.strip():
                self.filtered_images.append(result)
            else:
                required_tags = [t.strip().lower() for t in search_text.split(',') if t.strip()]
                if all(req_tag in {tag.lower() for tag in result.tags.keys()} for req_tag in required_tags):
                    self.filtered_images.append(result)
        
        self._populate_results_table(self.filtered_images)

    def display_image_preview(self):
        selected_rows = self.results_table.selectionModel().selectedRows()
        if not selected_rows:
            self.image_preview_label.clear()
            self.image_preview_label.setText("No image selected for preview")
            return

        row = selected_rows[0].row()
        filepath = self.results_table.item(row, 0).text()
        
        try:
            pixmap = QPixmap(filepath)
            if pixmap.isNull():
                raise ValueError("Could not load image.")
            
            # Scale pixmap to fit the label, maintaining aspect ratio
            max_size = self.image_preview_label.size()
            scaled_pixmap = pixmap.scaled(max_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image_preview_label.setPixmap(scaled_pixmap)
            self.image_preview_label.setText("") # Clear "No image selected" text
        except Exception as e:
            self.image_preview_label.clear()
            self.image_preview_label.setText(f"Error loading preview: {e}")
            self.update_status(f"Error loading preview for {filepath}: {e}")

    def update_progress(self, current: int, total: int, message: str):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.update_status(message)

    def classification_finished(self):
        # Re-enable UI elements
        self.classify_button.setEnabled(True)
        self.path_input.setEnabled(True)
        self.browse_button.setEnabled(True)
        self.search_input.setEnabled(True)
        self.find_similar_button.setEnabled(self.results_table.selectionModel().hasSelection() and self.app_mode == "search")
        self.back_button.setEnabled(self.app_mode == "similar")
        
        self.update_status("Classification complete.")
        self.current_worker = None # Clear worker reference

    def handle_classification_error(self, message: str):
        QMessageBox.critical(self, "Classification Error", message)
        self.classification_finished() # Reset UI state

    def update_status(self, message: str):
        self.status_label.setText(message)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icon.png")) # Optional: add an icon
    gui = ImageClassifierGUI()
    gui.show()
    sys.exit(app.exec())
