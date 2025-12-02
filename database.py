import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Tuple

class ImageDatabase:
    def __init__(self, db_path: str):
        """
        Initializes the database connection.
        
        Args:
            db_path (str): The path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        """
        Creates the necessary tables if they don't exist.
        - images: Stores image file paths.
        - tags: Stores all unique tags.
        - image_tags: Links images and tags, storing the confidence score.
        """
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY,
                filepath TEXT NOT NULL UNIQUE
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_tags (
                image_id INTEGER,
                tag_id INTEGER,
                confidence REAL,
                FOREIGN KEY (image_id) REFERENCES images (id),
                FOREIGN KEY (tag_id) REFERENCES tags (id),
                PRIMARY KEY (image_id, tag_id)
            )
        ''')
        self.conn.commit()

    def add_classification_data(self, filepath: str, tags_with_confidence: Dict[str, float]):
        """
        Adds classification data for a single image to the database.

        Args:
            filepath (str): The path to the image file.
            tags_with_confidence (Dict[str, float]): A dictionary of tags and their confidence scores.
        """
        # Add image and get its ID
        self.cursor.execute("INSERT OR IGNORE INTO images (filepath) VALUES (?)", (filepath,))
        self.cursor.execute("SELECT id FROM images WHERE filepath = ?", (filepath,))
        image_id_result = self.cursor.fetchone()
        if not image_id_result:
            return # Should not happen
        image_id = image_id_result[0]
        
        # Prepare tag data
        tags_to_insert = [(tag,) for tag in tags_with_confidence.keys()]
        self.cursor.executemany("INSERT OR IGNORE INTO tags (name) VALUES (?)", tags_to_insert)

        # Get tag IDs
        tag_names = list(tags_with_confidence.keys())
        placeholders = ','.join('?' for _ in tag_names)
        self.cursor.execute(f"SELECT id, name FROM tags WHERE name IN ({placeholders})", tag_names)
        tag_id_map = {name: id for id, name in self.cursor.fetchall()}

        # Prepare image_tags data
        image_tags_to_insert = [
            (image_id, tag_id_map[tag], confidence)
            for tag, confidence in tags_with_confidence.items()
        ]
        
        self.cursor.executemany(
            "INSERT OR REPLACE INTO image_tags (image_id, tag_id, confidence) VALUES (?, ?, ?)",
            image_tags_to_insert
        )
        self.conn.commit()

    def get_all_data(self) -> Dict[str, Dict[str, float]]:
        """
        Retrieves all image data from the database.

        Returns:
            Dict[str, Dict[str, float]]: A dictionary where keys are file paths and
                                         values are dictionaries of tags and confidences.
        """
        query = """
            SELECT i.filepath, t.name, it.confidence
            FROM images i
            JOIN image_tags it ON i.id = it.image_id
            JOIN tags t ON it.tag_id = t.id
        """
        self.cursor.execute(query)
        
        data = {}
        for filepath, tag, confidence in self.cursor.fetchall():
            if filepath not in data:
                data[filepath] = {}
            data[filepath][tag] = confidence
        return data
        
    def get_all_tags(self) -> List[str]:
        """
        Retrieves a list of all unique tags from the database.
        
        Returns:
            List[str]: A list of tag names.
        """
        self.cursor.execute("SELECT name FROM tags ORDER BY name")
        return [row[0] for row in self.cursor.fetchall()]

    def is_image_classified(self, filepath: str) -> bool:
        """
        Checks if an image has already been classified and exists in the database.

        Args:
            filepath (str): The path to the image file.

        Returns:
            bool: True if the image is in the database, False otherwise.
        """
        self.cursor.execute("SELECT 1 FROM images WHERE filepath = ?", (filepath,))
        return self.cursor.fetchone() is not None

    def find_similar_by_tags(self, source_filepath: str, limit: int = 20) -> List[Tuple[str, float]]:
        """
        Finds images with similar tags to the source image, based on Jaccard similarity.

        Args:
            source_filepath (str): The file path of the image to find similars for.
            limit (int): The maximum number of similar images to return.

        Returns:
            A list of tuples, each containing (filepath, jaccard_similarity_score).
            The list is sorted by similarity in descending order.
        """
        # 1. Get the source image's ID
        self.cursor.execute("SELECT id FROM images WHERE filepath = ?", (source_filepath,))
        source_id_res = self.cursor.fetchone()
        if not source_id_res:
            return []
        source_id = source_id_res[0]

        # 2. Get the count of tags for the source image
        self.cursor.execute("SELECT COUNT(tag_id) FROM image_tags WHERE image_id = ?", (source_id,))
        source_tag_count_res = self.cursor.fetchone()
        source_tag_count = source_tag_count_res[0] if source_tag_count_res else 0
        if source_tag_count == 0:
            return []

        # 3. Find intersection and calculate Jaccard similarity
        # The Jaccard similarity is |A intersect B| / |A union B|
        # |A union B| = |A| + |B| - |A intersect B|
        query = f"""
            SELECT
                other_images.filepath,
                -- Jaccard Similarity Calculation
                CAST(COUNT(common_tags.tag_id) AS REAL) / (
                    {source_tag_count} + other_image_tag_counts.tag_count - COUNT(common_tags.tag_id)
                ) AS jaccard_similarity
            FROM image_tags AS source_tags
            -- Join to find images that share at least one tag
            INNER JOIN image_tags AS common_tags ON source_tags.tag_id = common_tags.tag_id
            -- Get the filepaths for the 'other' images
            INNER JOIN images AS other_images ON common_tags.image_id = other_images.id
            -- Pre-calculate tag counts for all 'other' images
            INNER JOIN (
                SELECT image_id, COUNT(tag_id) as tag_count
                FROM image_tags
                GROUP BY image_id
            ) AS other_image_tag_counts ON common_tags.image_id = other_image_tag_counts.image_id
            WHERE
                source_tags.image_id = ? -- Filter for our source image's tags
                AND common_tags.image_id != ? -- Exclude the source image itself
            GROUP BY
                common_tags.image_id
            HAVING
                jaccard_similarity > 0.50 -- Ensure there's some similarity
            ORDER BY
                jaccard_similarity DESC
            LIMIT ?;
        """
        self.cursor.execute(query, (source_id, source_id, limit))
        return self.cursor.fetchall()

    def close(self):
        """Closes the database connection."""
        self.conn.close()

if __name__ == '__main__':
    # Example usage and testing
    db = ImageDatabase('test.db')
    
    # Clean up previous test runs if file exists
    if Path('test.db').exists():
        db.cursor.execute("DROP TABLE IF EXISTS image_tags")
        db.cursor.execute("DROP TABLE IF EXISTS images")
        db.cursor.execute("DROP TABLE IF EXISTS tags")
        db.conn.commit()
        db._create_tables()

    # Test data
    test_data = {
        "/path/to/image1.jpg": {"1girl": 1, "red_hair": 1, "smile": 1, "cat_ears": 1},
        "/path/to/image2.png": {"1boy": 1, "blue_hair": 1, "sword": 1},
        "/path/to/image3.webp": {"1girl": 1, "red_hair": 1, "smile": 1, "book": 1}, # Very similar to 1
        "/path/to/image4.jpeg": {"1girl": 1, "red_hair": 1, "tree": 1}, # Somewhat similar to 1
        "/path/to/image5.gif": {"1boy": 1, "green_hair": 1, "gun": 1}, # Not similar to 1
    }
    for path, tags in test_data.items():
        db.add_classification_data(path, tags)
    print("Added initial data.")

    # Retrieve all data
    all_data = db.get_all_data()
    print("\nAll data from DB:")
    print(json.dumps(all_data, indent=2))
    
    # Test Similarity Search
    print("\n--- Testing Similarity Search for image1.jpg ---")
    similar_images = db.find_similar_by_tags("/path/to/image1.jpg")
    print("Found similar images:")
    for path, score in similar_images:
        print(f"  - Path: {path}, Similarity: {score:.4f}")

    # Expected output for image1 (4 tags):
    # image3 (3 common tags): intersection=3, union=4+4-3=5, score=3/5=0.6
    # image4 (2 common tags): intersection=2, union=4+3-2=5, score=2/5=0.4

    db.close()

    # Clean up test db
    Path('test.db').unlink()
    print("\nCleaned up test.db")
