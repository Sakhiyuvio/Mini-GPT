import os
import requests
import re 
import logging

# BOOKS to be parsed for dataset, from Project Grutenberg
books_dir = {
    "The Time Machine": 35, 
    "A Princess of Mars": 62,
    "The War of the Worlds": 36,
    "Frankenstein": 84, 
    "Flatland: A Romance of Many Dimensions": 201, 
}

class Dataset:
    def __init__(self, output_path: str):
        self.output_path = output_path 
        self.logger = logging.Logger("Dataset:")
        logging.basicConfig(level=logging.INFO)
    
    def get_book_txt(self, book_id: int): 
        http_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        try:
            self.logger.info(f"Downloading text file for book ID{book_id}")
            response = requests.get(http_url)
            response.raise_for_status()
            return response.text
        except:
            self.logger.info("Failed to retrieve book txt via http request!")
            self.logger.info("Retrying!")
            return self.get_book_txt(book_id)

    def clean_txt(self, book_text):
        # Remove unnecessary text like headers and footers
        header_s = re.search(r"\*\*\* START OF.*?\*\*\*", book_text, re.IGNORECASE)
        header_f = re.search(r"\*\*\* END OF.*?\*\*\*", book_text, re.IGNORECASE)

        if header_s and header_f: # if these headers are found, 
            # start text from end of start to start of end!
            book_text = book_text[header_s.end(): header_f.start()]
        
        return book_text.strip()

    def build_corpus(self):
        corpus = ""
        for title, book_id in books_dir.items():
            curr_book_txt = self.get_book_txt(book_id)
            cleaned_book_txt = self.clean_txt(curr_book_txt)
            # add to corpus for data processing
            corpus += f"\n\n### {title}\n\n{cleaned_book_txt}\n\n"

        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(corpus)
            
        self.logger.info(f"Corpus written to {self.output_path}")

        

            

