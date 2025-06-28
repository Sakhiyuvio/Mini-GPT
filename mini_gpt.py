import logging
import torch

from corpus_builder import Dataset

"""
Making auto-regressive language model from scratch! 
Play around with it! Currently trained with ~ 1 MB - 3MB of data, using sci-fi books for contextualization.
"""

# Work-flow: 

# Get dataset for training and inference

# Tokenizer, simple encoder decoder, via idx enumeration

# Model class f-pass, losses, and back-prop

# training proc

# testing proc 

class Tokenizer:
    def __init__(self, corpus_text: str):
        characters = sorted(list(set(corpus_text)))
        self.vocab_size = len(characters)
        self.string_to_idx = {ch: i for i, ch in enumerate(characters)}
        self.idx_to_string = {i: ch for i, ch in enumerate(characters)}
        
    def encoder(self, text: str):
        encode = [self.string_to_idx[ch] for ch in text]
        return encode
    
    def decoder(self, token: list[int]):
        decode = "".join(self.idx_to_string[idx] for idx in token)
        return decode

class MiniGPT:
    def __init__(self, output_path: str):
        self.logger = logging.Logger("Mini-GPT:")
        logging.basicConfig(logging.INFO)
        self.logger.info("Enjoy this small language model, Mini-GPT!")
        self.output_path = output_path
        self.window_size = 10  # context window size for the model, number of tokens to look back
        self.batch_size = 4 # batch size for training in parallel

    def pipeline(self):
        # read corpus
        corpus = self.read_corpus()

        # init tokenizer
        tokenizer = Tokenizer(corpus)
        self.logger.info(f"Vocab size: {tokenizer.vocab_size}")
        self.logger.info("Tokenizer initialized.")

        # encode corpus
        corpus_encoded = torch.tensor(tokenizer.encoder(corpus), dtype=torch.long)
        self.logger.info("Corpus encoded.")
        
        # split dataset: 0.9 train, 0.1 test
        all = len(corpus_encoded)
        train_size = int(0.9 * all)
        train_corpus_data = corpus_encoded[:train_size]
        test_corpus_data = corpus_encoded[train_size:]

        # init model architecture: 10 context tokens 


    def read_corpus(self):
        # init dataset class for corpus init
        dataset = Dataset(self.output_path)
        dataset.build_corpus()

        # data for LM is prepared!, read the txt file
        with open(self.output_path, "r", encoding="utf-8") as f:
            corpus_text = f.read()
        return corpus_text
