import logging
import torch

from dataset_curation import Dataset

"""
Making auto-regressive language model from scratch! 
Play around with it! Currently trained with ~ 1 MB - 3MB of data, using sci-fi books for contextualization.
"""

# Work-flow: 

# Get dataset for training and inference

# Tokenizer, simple encoder decoder lambda function 

# Model class f-pass, losses, and back-prop

# training proc

# testing proc 

class MiniGPT:
    def __init__(self, output_path: str):
        self.logger = logging.Logger("Mini-GPT:")
        logging.basicConfig(logging.INFO)
        self.logger.info("Enjoy this small language model, Mini-GPT!")
        self.output_path = output_path

    def read_corpus(self):
        # init dataset class for corpus init
        dataset = Dataset(self.output_path)
        dataset.build_corpus()

        # data for LM is prepared!, read the txt file
        with open(self.output_path, "r", encoding="utf-8") as f:
            corpus_text = f.read()
        return corpus_text

    def tokenizer(self):
        # encode 

        # decode

