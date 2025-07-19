import argparse
import logging
import os
import torch
import time 

from corpus_builder import Dataset
from dataclasses import dataclass
from enum import Enum
from torch import nn
from torch.optim import AdamW
from typing import List, Optional
"""
Making auto-regressive language model from scratch! 
Play around with it! Currently trained with ~ 1.5 MB of data, using sci-fi books for contextualization.
"""

class DataType(Enum):
    TRAIN = "train"
    TEST = "test"
    INFERENCE = "inference"

    def __str__(self):
        return self.value
    
@dataclass
class MiniGPTParams:
    window_size: int = 32  # context window size for the model, number of tokens to look back
    batch_size: int = 16  # batch size for training in parallel
    embedding_dim: int = 32  # dimension of the embedding vector
    hidden_dim: int = 128  # dimension of the hidden layer in feed-forward network
    num_heads: int = 4 # number of attention heads in multi-headed attention
    num_layers: int = 4  # number of layers in the transformer model
    learning_rate: float = 3e-2  # learning rate for the optimizer
    num_epochs: int = 1 #10  # number of epochs to train the model
    mode: DataType = DataType.TRAIN  # mode of operation, train or test
    
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
    
class LayerNormalization(nn.Module):
    def __init__(self, embedding_dim: int, eps = 1e-5):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embedding_dim)) # weights
        self.beta = nn.Parameter(torch.zeros(embedding_dim)) # biases

    def forward(self, x: torch.Tensor):
        x = nn.Functional.layer_norm(x, (x.size(-1),), self.gamma, self.beta, self.eps)  # Apply layer normalization
        return x  # Return normalized tensor

class FeedForwardNN(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super(FeedForwardNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.1)  # Dropout layer for regularization
        self.linear1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.embedding_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.dropout(x)  # Apply dropout to the input tensor
        x = self.linear1(x)  # Linear transformation to hidden layer (batch_size, window_size, hidden_dim)
        x = self.activation(x)  # Apply activation function (ReLU)
        x = self.linear2(x)  # Linear transformation back to embedding dimension (batch_size, window_size, embedding_dim)
        return x 

class MultiHeadedAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, window_size: int, masking: bool = False):
        super(MultiHeadedAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.self_attention_dim = embedding_dim // num_heads  # Dimension for each head
        self.dropout = nn.Dropout(0.1)  # Dropout layer for regularization
        self.q_k_v = nn.Linear(self.embedding_dim, 3 * self.embedding_dim)
        self.mha = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.masking = masking

    def forward(self, x: torch.Tensor):
        # x is of shape (batch_size, window_size, embedding_dim)
        x = self.dropout(x)  # Apply dropout to the input tensor
        q_k_v_proj = self.q_k_v(x)  # Project input to query, key, and value vectors (batch_size, window_size, 3 * embedding_dim)
        q, k, v = q_k_v_proj.chunk(3, dim=-1)  # Split into query, key, and value vectors (batch_size, window_size, embedding_dim)

        # Need to reshape the vectors to account for multiple heads! 
        q, k, v = q.view(q.size(0), q.size(1), self.num_heads, self.self_attention_dim), k.view(k.size(0), k.size(1), self.num_heads, self.self_attention_dim), v.view(v.size(0), v.size(1), self.num_heads, self.self_attention_dim)
        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)  # Rearrange to (batch_size, num_heads, window_size, self_attention_dim)
        weight = q @ k.transpose(-2, -1) / (self.self_attention_dim ** 0.5) # Scaled dot-product attention (batch_size, num_heads, window_size, window_size)

        if self.masking:
            weight_matrix = torch.tril(torch.ones(self.window_size, self.window_size, device=x.device)).unsqueeze(0).unsqueeze(0)  # Causal mask (1, 1, window_size, window_size)
            weight = weight.masked_fill(weight_matrix == 0, float('-inf'))  # Apply causal mask

        weight = torch.softmax(weight, dim=-1)  # Softmax to get attention weights, (batch_size, num_heads, window_size, window_size)
        attention_outputs = weight @ v  # Attention output (batch_size, num_heads, window_size, embedding_dim//heads)
        attention_outputs = attention_outputs.permute(0, 2, 1, 3)  # Rearrange to (batch_size, window_size, num_heads, embedding_dim//heads)
        attention_outputs = attention_outputs.reshape(attention_outputs.size(0), attention_outputs.size(1), -1)  # Flatten to (batch_size, window_size, embedding_dim)  
        return self.mha(attention_outputs)

class TransformerBlock(nn.Module):
    def __init__(self, cfg: MiniGPTParams = MiniGPTParams()):
        super(TransformerBlock, self).__init__()
        self.embedding_dim = cfg.embedding_dim
        self.hidden_dim = cfg.hidden_dim
        self.num_heads = cfg.num_heads
        self.window_size = cfg.window_size
        self.mha = MultiHeadedAttention(self.embedding_dim, self.num_heads, self.window_size, masking=False)
        self.masked_mha = MultiHeadedAttention(self.embedding_dim, self.num_heads, self.window_size, masking=True)
        self.ffn = FeedForwardNN(self.embedding_dim, self.hidden_dim)
        self.ln_1 = LayerNormalization(self.embedding_dim)
        self.ln_2 = LayerNormalization(self.embedding_dim)
        self.ln_3 = LayerNormalization(self.embedding_dim)

    def forward(self, x: torch.Tensor):
        # x is of shape (batch_size, window_size, embedding_dim)
        # Apply residual connections and layer normalization per sub-layers!
        x = self.masked_mha(x + self.ln_1(x))
        x = self.mha(x + self.ln_2(x))  # Multi-head attention
        x = self.ffn(x + self.ln_3(x))  # Feed-forward network
        return x

class MiniGPTModel(nn.Module):
    def __init__(self, vocab_size: int, cfg: MiniGPTParams  = MiniGPTParams()):
        super(MiniGPTModel, self).__init__()
        self.vocab_size = vocab_size
        self.window_size = cfg.window_size
        self.embedding_dim = cfg.embedding_dim
        self.num_layers = cfg.num_layers
        # vector embeddings for contextualization and indexing
        self.token_embedding = nn.Embedding(vocab_size, self.embedding_dim) 
        self.position_embedding = nn.Embedding(self.window_size, self.embedding_dim)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        # linear layer for output
        self.linear_head = nn.Linear(self.embedding_dim, vocab_size) # output layer to predict next token
        
    def forward(self, input_token: torch.Tensor, targets: Optional[torch.Tensor] = None, inference: bool = False):
        # input token is of shape, (batch_size, window_size)
        # Perform embeddings
        output_vect = self.token_embedding(input_token) # Token embeddings (batch_size, window_size, embedding_dim)
        position_idx = self.position_embedding(torch.arange(self.window_size, device=input_token.device))  # Position embeddings
        # Sum of embeddings, go through multi-headed attention,
        output_vect += position_idx.unsqueeze(0)  # Add position embeddings to token embeddings

        # Sequentially pass through transformer blocks 
        for t_block in self.transformer_blocks:
            output_vect = t_block(output_vect)  # Pass through transformer block

        # Finally output layer to predict next token via linear head 
        transformer_output = self.linear_head(output_vect)  # Output logits (batch_size, window_size, vocab_size)

        # Apply softmax for probabilities and predictions that spans over that window size
        logits = torch.softmax(transformer_output, dim=-1)  # Convert logits to probabilities (batch_size, window_size, vocab_size)
        loss = None  # Initialize loss variable, in case of inference

        if not inference:
            if targets is None:
                raise ValueError("Targets must be provided for training or validation purposes.")
            logits = logits.reshape(-1, self.vocab_size)  # Reshape to (batch_size * window_size, vocab_size) for easier processing
            # Calculate loss if targets are provided
            targets = targets.reshape(-1) # shape (batch_size * window_size)
            loss = nn.CrossEntropyLoss()(logits, targets)  # Cross-entropy loss for classification task
            logits.reshape(-1, self.window_size, self.vocab_size) # Reshape logits back to (batch_size, window_size, vocab_size)

        # Return the logits (and loss for training)
        return logits, loss 

    def train(self, input_tokens: torch.Tensor, targets: torch.Tensor, learning_rate: float = 3e-2):
        # training process of one single batch 
        self.train()  # Set the model to training mode
        optimizer = AdamW(self.parameters(), lr=learning_rate)  # Adam optimizer for training
        optimizer.zero_grad()  # Zero the gradients
        _, loss = self.forward(input_tokens, targets) # Forward pass through the model
        loss.backward()  # Backward pass to compute gradients
        optimizer.step()  # Update model parameters
        return loss.item()  # Return the loss value for monitoring

    def validate(self, input_tokens: torch.Tensor, targets: torch.Tensor):
        # validation process of one single batch
        self.eval() # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation for validation
            _, loss = self.forward(input_tokens, targets)  # Forward pass through the model
        return loss.item()  # Return the loss value for monitoring

    def generate(self, input_tokens: torch.Tensor, max_sequence: int = 50):
        self.eval() # Set the model to evaluation mode
        output_tokens = input_tokens.clone()  # Start with the input tokens
        for i in range(max_sequence):
            # FF pass through the model
            logits, _ = self.forward(input_tokens, inference=True)  # Get logits for the input tokens
            # Sample from the distribution to get the next token
            next_token = torch.multinomial(logits[:, -1, :], num_samples=1)  # Sample next token based on probabilities
            # Append the next token to the input sequence
            output_tokens = torch.cat((output_tokens, next_token.unsqueeze(1)), dim=1)  # Concatenate the next token to the input sequence
        
        return output_tokens  # Return the generated sequence of tokens
        
class MiniGPT:
    def __init__(self, output_path: str, corpus_path: str, cfg: MiniGPTParams = MiniGPTParams()):
        self.logger = logging.getLogger("Mini-GPT:")
        self.logger.info("Enjoy this small language model, Mini-GPT!")
        self.output_path = os.path.expanduser(output_path)
        self.corpus_path = os.path.expanduser(corpus_path)
        self.window_size = cfg.window_size # context window size for the model, number of tokens to look back
        self.batch_size = cfg.batch_size # batch size for training in parallel
        self.embedding_dim = cfg.embedding_dim # dimension of the embedding vector
        self.lr = cfg.learning_rate # learning rate for the optimizer
        self.epochs = cfg.num_epochs # number of epochs for training
        self.mode = cfg.mode # mode of operation, either 'train' or 'inference'

    def pipeline(self):
        # Check if output path and corpus path exist, if not create them
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            self.logger.info(f"Output path {self.output_path} created.")
        
        if not os.path.exists(self.corpus_path):
            os.makedirs(self.corpus_path)
            self.logger.info(f"Corpus path {self.corpus_path} created.")
        
        # read corpus
        corpus = self.read_corpus()

        # init tokenizer
        tokenizer = Tokenizer(corpus)
        self.logger.info(f"Vocab size: {tokenizer.vocab_size}")
        self.logger.info(f"Characters: {tokenizer.characters[:]}")  # print all characters
        self.logger.info("Tokenizer initialized.")

        # encode corpus
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

        # Initialize the model
        mini_gpt = MiniGPTModel(
            vocab_size=tokenizer.vocab_size,
            cfg=MiniGPTParams(),
        )
        self.logger.info("MiniGPT model initialized.")

        total_params = sum(p.numel() for p in mini_gpt.parameters())
        print(f"Total parameters: {total_params:,}")

        return # check param numbers

        # Train the model
        if self.mode.value == "train" or self.mode.value == "test":
            self.logger.info("Starting training and evaluation of the MiniGPT model...")
            start_time = time.time()
            for epoch in range(self.epochs):
                for batch in range(len(self.train_corpus_data) // self.batch_size):
                    # create batches for training
                    self.logger.info("Creating batches for training...")
                    x_train, y_train = self.create_batch(DataType.TRAIN)
                    x_val, y_val = self.create_batch(DataType.TEST)
                    loss_train = mini_gpt.train(x_train, y_train, learning_rate = self.lr)  # Train the model for 10 epochs, 16 batches at a time
                    loss_val = mini_gpt.validate(x_val, y_val)  # Validate the model
                    self.logger.info(f"Epoch {epoch + 1}, Batch {batch + 1}, Training loss: {loss_train}, Validation loss {loss_val}")  # Log the training loss
            end_time = time.time()
            self.logger.info(f"Training completed in {end_time - start_time:.2f} seconds.")
            self.logger.info("Training & Eval completed!")

        else:
            # Use the model to generate text - Inference mode 
            self.logger.info("Generating text with the MiniGPT model...")
            output_tokens = mini_gpt.generate(input_tokens=x_val, max_sequence=50)  # Generate text from the validation set

            # Decode
            decoded_output = tokenizer.decoder(output_tokens.tolist()[0])  # Decode the generated tokens back to text
            self.logger.info(f"Text generated by MiniGPT, here's a sample: {decoded_output[:20]}...")

            # Save to output path
            generated_output_file_path = os.path.join(self.output_path, "generated_text.txt")
            
            with open(generated_output_file_path, "w", encoding="utf-8") as f:
                f.write(decoded_output)
            
            self.logger.info(f"Generated text saved to {generated_output_file_path}")
        
    def read_corpus(self):
        self.logger.info(f"Reading corpus from {self.corpus_path}...")
        if not os.path.exists(self.corpus_path):
            self.logger.info("Corpus file does not exist, building corpus...")
            dataset = Dataset(self.corpus_pathh)
            dataset.build_corpus()
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            # read corpus text
            corpus_text = f.read()
        return corpus_text

    def create_batch(self, data_type: DataType):
        data = self.train_corpus_data if data_type == DataType.TRAIN else self.test_corpus_data

        torch.manual_seed(42) # Hitch-hiker's guide to the galaxy special number 

        # randomly pick indices to process data in mini batches, pick # batch size of window chunks
        random_idx = torch.randint(len(data) - (self.window_size + 1), (self.batch_size,))
        input_to_nn = torch.stack([data[i:i + self.window_size] for i in random_idx])
        output_from_nn = torch.stack([data[i+1: i + 1 + self.window_size] for i in random_idx])
        self.logger.info(f"Batch created for {data_type} data with shape: {input_to_nn.shape}, {output_from_nn.shape}")
        return input_to_nn, output_from_nn

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Mini-GPT: A small language model training pipeline.")
    parser.add_argument("--output_path", type=str, default="output", help="Path to save the output files.")
    parser.add_argument("--corpus_path", type=str, default="corpus.txt", help="Path to the corpus text file.")
    parser.add_argument("--inference", action = "store_true", help="Run in inference mode to generate text from the trained model.")
    args = parser.parse_args()
    mini_gpt_params = MiniGPTParams() if not args.inference else MiniGPTParams(mode = DataType.INFERENCE)
    mini_gpt = MiniGPT(args.output_path, args.corpus_path, cfg=mini_gpt_params)
    mini_gpt.pipeline()
    mini_gpt.logger.info("Mini-GPT pipeline completed successfully!")
