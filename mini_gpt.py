import logging
import torch

from corpus_builder import Dataset
from enum import Enum
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

class DataType(Enum):
    TRAIN = "train"
    TEST = "test"

    def __str__(self):
        return self.value

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
        self.window_size = 32  # context window size for the model, number of tokens to look back
        self.batch_size = 16 # batch size for training in parallel

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
        self.train_corpus_data = corpus_encoded[:train_size]
        self.test_corpus_data = corpus_encoded[train_size:]

        # create batches for training
        self.logger.info("Creating batches for training...")
        x_train, y_train = self.create_batch(DataType.TRAIN)
        x_val, y_val = self.create_batch(DataType.TEST)

    def read_corpus(self):
        # init dataset class for corpus init
        dataset = Dataset(self.output_path)
        dataset.build_corpus()

        # data for LM is prepared!, read the txt file
        with open(self.output_path, "r", encoding="utf-8") as f:
            corpus_text = f.read()
        return corpus_text

    def create_batch(self, data_type: DataType):
        data = self.train_corpus_data if data_type == DataType.TRAIN else self.test_corpus_data

        torch.manual_seed(42) # Hitch-hiker's guide to the galaxy reference lol

        # randomly pick indices to process data in mini batches, pick # batch size of window chunks
        random_idx = torch.randint(len(data) - self.window_size, (self.batch_size,))
        input_to_nn = torch.stack([data[i:i + self.window_size] for i in random_idx])
        output_from_nn = torch.stack([data[i+1: i + 1 + self.window_size] for i in random_idx])
        self.logger.info(f"Batch created for {data_type} data with shape: {input_to_nn.shape}, {output_from_nn.shape}")
        return input_to_nn, output_from_nn

