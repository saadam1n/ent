import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import policy
import dataset
import tokenizers

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available! Terminating...")

bctransformer = policy.TransformerPolicy(
    num_transformer_layers=6, 
    num_attention_heads=8, 
    context_length=160, 
    embedding_dim=512, 
    head_embedding_dim=64, 
    num_embeddings=100
).to("cuda")

bc_dataset = dataset.BehaviorColoningDataset(
    # replace this line to change tokenizer
    tokenizer=tokenizers.DefaultTokenizer(),
    history_size=4
)




