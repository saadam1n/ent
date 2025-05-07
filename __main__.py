import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

import policy
import dataset
import tokenizers

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available! Terminating...")

bc_tokenizer = tokenizers.DefaultTokenizer()

bc_dataset = dataset.BehaviorColoningDataset(
    # replace this line to change tokenizer
    tokenizer=bc_tokenizer,
    history_size=4
)

bc_dataloader = DataLoader(
    bc_dataset,
    batch_size=512,
    shuffle=True
)

bc_transformer = policy.TransformerPolicy(
    num_transformer_layers=6, 
    num_attention_heads=8, 
    context_length=140, 
    embedding_dim=512, 
    head_embedding_dim=64, 
    num_embeddings=bc_tokenizer.token_count
).to("cuda")

optimizer = torch.optim.Adam(bc_transformer.parameters(), lr=0.0001)

# loop over the dataset multiple times
bc_dataset.tokenization = True
for epoch_idx in range(128):
    print(f"Processing epoch idx {epoch_idx}")

    running_loss = 0.0
    for i, data in enumerate(bc_dataloader, 0):
        print(f"\tProcessing batch {i}")

        pre, token_ids = data

        optimizer.zero_grad()

        loss = bc_transformer(token_ids)
        loss.backward()
        optimizer.step()

        print(f"\t\tLoss was {loss}")

        running_loss += loss.item()

    print(f"Running loss was {running_loss}")

print('Finished Training')




