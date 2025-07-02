import argparse
import logging
import os
import torch

from corpus_builder import Dataset
from dataclasses import dataclass
from enum import Enum
from torch import nn
from typing import List
"""
Making auto-regressive language model from scratch! 
Play around with it! Currently trained with ~ 1.5 MB of data, using sci-fi books for contextualization.
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
    
@dataclass
class MiniGPTParams:
    window_size: int = 32  # context window size for the model, number of tokens to look back
    batch_size: int = 16  # batch size for training in parallel
    embedding_dim: int = 32  # dimension of the embedding vector
    
class Tokenizer:
    def __init__(self, corpus_text: str):
        self.characters = sorted(list(set(corpus_text)))
        self.vocab_size = len(self.characters)
        self.string_to_idx = {ch: i for i, ch in enumerate(self.characters)}
        self.idx_to_string = {i: ch for i, ch in enumerate(self.characters)}
        
    def encoder(self, text: str):
        encode = [self.string_to_idx[ch] for ch in text]
        return encode
    
    def decoder(self, token: List[int]):
        decode = "".join(self.idx_to_string[idx] for idx in token)
        return decode

class MiniGPTModel(nn.Module):
    def __init__(self, vocab_size: int, window_size: int, embedding_dim: int):
        super(MiniGPTModel, self).__init__()
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        # vector embeddings for contextualization and indexing
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim) 
        self.position_embedding = nn.Embedding(window_size, embedding_dim)

        # q, k, and v
        self.query = nn.Linear(embedding_dim, embedding_dim)  # Query vector
        self.key = nn.Linear(embedding_dim, embedding_dim)  # Key vector
        self.value = nn.Linear(embedding_dim, embedding_dim)  # Value vector

        # linear layer for output
        self.linear_head = nn.Linear(embedding_dim, vocab_size) # output layer to predict next token
        
    def forward(self, x: torch.Tensor):
        # Perform embeddings
        token_embedded = self.token_embedding(x)  # Token embeddings

        # x_embedded is of shape (batch_size, window_size, embedding_dim)
        position_idx = self.position_embedding(torch.arange(self.window_size, device=x.device))  # Position embeddings
        # Sum of embeddings, go through multi-headed attention,
        token_embedded += position_idx.unsqueeze(0)  # Add position embeddings to token embeddings

        # Input to transformer: token_embedded, has size (batch_size, window_size, embedding_dim)

        # Transformers, start with self-attention. TO-DO. 
        q = self.query(token_embedded)  # Query vector (batch_size, window_size, embedding_dim)
        k = self.key(token_embedded)  # Key vector (batch_size, window_size, embedding_dim)
        weight = q @ k.transpose(-2, -1) # Scaled dot-product attention (batch_size, window_size, window_size)

        # Normalize the attention weights
        causal_tril_matrix = torch.tril(torch.ones(self.window_size, self.window_size))  # Lower triangular matrix for masking
        weight = weight.masked_fill(causal_tril_matrix == 0, float('-inf'))  # Apply causal mask
        weight = torch.softmax(weight, dim=-1)  # Softmax to get attention weights, this normalizes each row for self-attention

        v = self.value(token_embedded)  # Value vector (batch_size, window_size, embedding_dim)
        attention_output = weight @ v  # Attention output (batch_size, window_size, embedding_dim)  

        # Finally output layer to predict next token via linear head 

        # Loss function 

        # Return the logits and the loss results

        pass # Placeholder
        
class MiniGPT:
    def __init__(self, output_path: str, MiniGPTParams: MiniGPTParams = MiniGPTParams()):
        self.logger = logging.getLogger("Mini-GPT:")
        self.logger.info("Enjoy this small language model, Mini-GPT!")
        self.output_path = output_path
        self.window_size = MiniGPTParams.window_size # context window size for the model, number of tokens to look back
        self.batch_size = MiniGPTParams.batch_size # batch size for training in parallel
        self.embedding_dim = MiniGPTParams.embedding_dim # dimension of the embedding vector

    def pipeline(self):
        # read corpus
        corpus = self.read_corpus()

        # init tokenizer
        tokenizer = Tokenizer(corpus)
        self.logger.info(f"Vocab size: {tokenizer.vocab_size}")
        self.logger.info(f"Characters: {tokenizer.characters[:]}")  # print all characters
        self.logger.info("Tokenizer initialized.")

        # # encode corpus
        corpus_encoded = torch.tensor(tokenizer.encoder(corpus), dtype=torch.long)
        self.logger.info("Corpus encoded.")
        self.logger.info(f"Encoded corpus length: {len(corpus_encoded)} tokens.")
        self.logger.info(f"First 100 tokens of the corpus: {corpus_encoded[:100]}")
        
        # split dataset: 0.9 train, 0.1 test
        self.logger.info("Splitting dataset into training and testing sets...")
        all = len(corpus_encoded)
        train_size = int(0.9 * all)
        self.train_corpus_data = corpus_encoded[:train_size]
        self.test_corpus_data = corpus_encoded[train_size:]

        # create batches for training
        self.logger.info("Creating batches for training...")
        x_train, y_train = self.create_batch(DataType.TRAIN)
        x_val, y_val = self.create_batch(DataType.TEST)

    def read_corpus(self):
        self.logger.info(f"Reading corpus from {self.output_path}...")
        if not os.path.exists(self.output_path):
            self.logger.info("Corpus file does not exist, building corpus...")
            dataset = Dataset(self.output_path)
            dataset.build_corpus()
        with open(self.output_path, "r", encoding="utf-8") as f:
            # read corpus text
            corpus_text = f.read()
        return corpus_text

    def create_batch(self, data_type: DataType):
        data = self.train_corpus_data if data_type == DataType.TRAIN else self.test_corpus_data

        torch.manual_seed(42) # Hitch-hiker's guide to the galaxy special number 

        # randomly pick indices to process data in mini batches, pick # batch size of window chunks
        random_idx = torch.randint(len(data) - self.window_size, (self.batch_size,))
        input_to_nn = torch.stack([data[i:i + self.window_size] for i in random_idx])
        output_from_nn = torch.stack([data[i+1: i + 1 + self.window_size] for i in random_idx])
        self.logger.info(f"Batch created for {data_type} data with shape: {input_to_nn.shape}, {output_from_nn.shape}")
        return input_to_nn, output_from_nn

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Mini-GPT: A small language model training pipeline.")
    parser.add_argument("--output_path", type=str, default="corpus.txt", help="Path to the corpus text file.")
    args = parser.parse_args()
    mini_gpt_params = MiniGPTParams()
    mini_gpt = MiniGPT(args.output_path, MiniGPTParams=mini_gpt_params)
    mini_gpt.pipeline()
    mini_gpt.logger.info("Mini-GPT pipeline completed successfully!")
