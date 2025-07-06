import json
import pandas as pd
import logging
from typing import Optional
from ..registry import CLEANER  # Import the CLEANER registry

@CLEANER.register_module()
class RedditDataCleaner:
    """
    Class for cleaning and parsing Reddit JSON data (posts and comments).
    """

    def __init__(self):
        """
        Initialize the RedditDataCleaner with logging.
        """
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def parse_reddit_json(self, json_file_path: str) -> Optional[pd.DataFrame]:
        """
        Parses Reddit JSON data into a single DataFrame.

        Args:
            json_file_path (str): Path to the JSON file containing Reddit data.

        Returns:
            pd.DataFrame: A DataFrame containing the cleaned Reddit data.
        """
        try:
            self.logger.info(f"Parsing Reddit JSON file: {json_file_path}")
            with open(json_file_path, 'r') as file:
                data = json.load(file)

            # Initialize a list to store rows
            rows = []

            # Iterate through posts
            for post in data:
                # Add the post itself
                rows.append({
                    "type": "post",
                    "post_id": post["id"],
                    "title": post["title"],
                    "author": post["author"],
                    "body": post["selftext"],
                    "created_timestamp": post["created_timestamp"],
                    "score": post["score"],
                    "num_comments": post["num_comments"],
                    "comment_id": None,
                    "comment_author": None,
                    "comment_body": None,
                    "comment_created_timestamp": None,
                    "comment_score": None,
                    "permalink": post["permalink"],
                    "subreddit": post["subreddit"]
                })

                # Add comments for the post
                for comment in post.get("comments", []):
                    rows.append({
                        "type": "comment",
                        "post_id": post["id"],
                        "title": None,
                        "author": None,
                        "body": None,
                        "created_timestamp": None,
                        "score": None,
                        "num_comments": None,
                        "comment_id": comment["id"],
                        "comment_author": comment["author"],
                        "comment_body": comment["body"],
                        "comment_created_timestamp": comment["created_timestamp"],
                        "comment_score": comment["score"],
                        "permalink": post["permalink"],
                        "subreddit": post["subreddit"]
                    })

            # Convert rows to a DataFrame
            df = pd.DataFrame(rows)

            self.logger.info(f"Successfully parsed Reddit JSON file: {json_file_path}")
            return df

        except Exception as e:
            self.logger.error(f"Error parsing Reddit JSON file {json_file_path}: {e}")
            return None