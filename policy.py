import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

import math

class TransformerFeedForward(nn.Module):
    """
    A double layer perceptron network with a skip connection. Automatically performs layer norm on the input.
    """
    def __init__(self, embedding_dim):
        super(TransformerFeedForward, self).__init__()

        self.embedding_dim = embedding_dim

        self.ffn = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.embedding_dim * 4),
            nn.GELU(),
            nn.Linear(self.embedding_dim * 4, self.embedding_dim)
        )

    def forward(self, x):
        return self.ffn(x) + x

class MultiHeadCasualAttention(nn.Module):
    def __init__(self, embedding_dim, num_attention_heads, head_embedding_dim, manual_attention=False):
        super(MultiHeadCasualAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.head_embedding_dim = head_embedding_dim

        self.total_head_dim = num_attention_heads * head_embedding_dim

        # pre-normalization
        self.pre_norm = nn.LayerNorm(self.embedding_dim)

        # linear layers to project our tokens to Q, K, and V
        self.qkv_linear = nn.Linear(self.embedding_dim, 3 * self.total_head_dim, bias=False)
        self.attention_linear = nn.Linear(self.total_head_dim, self.embedding_dim, bias=False)

        self.manual_attention = manual_attention

    """
    Returns (transformed tokens, K, V). Does not maintain a KV cache.
    """
    def forward(self, tokens):
        # nn.Linear gives us (N, L, D_embedding_dim) -> (N, L, D_total_head_dim)
        # we want to reformat it as (N, num_attention_heads, L, D_head_embedding_dim)

        N, L, _ = tokens.shape

        ln_tokens = self.pre_norm(tokens)

        qkv = self.qkv_linear(ln_tokens).view(N, L, 3, self.num_attention_heads, self.head_embedding_dim).permute((2, 0, 3, 1, 4))

        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        if self.manual_attention:
            qkT = torch.matmul(q, k.transpose(2, 3))
            masked_qkT = qkT + torch.triu(torch.ones_like(qkT) * -999.0, diagonal=1)

            attention_scores = F.softmax(masked_qkT / math.sqrt(self.head_embedding_dim), dim=3)
            weighted_values = torch.matmul(attention_scores, v)
        else:
            weighted_values = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        aggregated_tokens = weighted_values.permute((0, 2, 1, 3)).reshape(N, L, self.total_head_dim)
        attention_output = self.attention_linear(aggregated_tokens) + tokens

        return (attention_output, k, v)
    
class Transformer(nn.Module):
    def __init__(self, embedding_dim, num_attention_heads, head_embedding_dim, aggresive_checkpointing=False):
        super(Transformer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.head_embedding_dim = head_embedding_dim

        self.attention = MultiHeadCasualAttention(
            embedding_dim=self.embedding_dim, 
            num_attention_heads=self.num_attention_heads, 
            head_embedding_dim=self.head_embedding_dim
        )

        self.token_mlp = TransformerFeedForward(embedding_dim=self.embedding_dim)

        self.aggresive_checkpointing = aggresive_checkpointing

    def forward(self, tokens):
        attention_output, _, _ = checkpoint.checkpoint(self.attention, tokens, use_reentrant=False) if self.aggresive_checkpointing else self.attention(tokens)

        token_mlp_output = checkpoint.checkpoint(self.token_mlp, attention_output, use_reentrant=False) if self.aggresive_checkpointing else self.token_mlp(attention_output)

        return token_mlp_output

class TransformerPolicy(nn.Module):
    """
    A BC transformer policy. Given some token IDs, it attempts to reconstruct those token IDs.
    Please note that our transformer policy assumes that tokens are selected ahead of time. 
    Note that this policy has no concept of a "termination" token. 
    It tries to predict whatever you give, while ignoring -1 tokens in gradient calculation.
    """
    def __init__(self, num_transformer_layers, num_attention_heads, context_length, embedding_dim, head_embedding_dim, num_embeddings):
        super(TransformerPolicy, self).__init__()

        self.num_transformer_layers = num_transformer_layers
        self.num_attention_heads = num_attention_heads
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.head_embedding_dim = head_embedding_dim
        self.num_embeddings = num_embeddings

        self.embedding_matrix = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim)
        self.positional_encoding = nn.Embedding(num_embeddings=self.context_length, embedding_dim=self.embedding_dim)

        self.transformer_layers = nn.Sequential(
            *[
                Transformer(embedding_dim=self.embedding_dim, num_attention_heads=self.num_attention_heads, head_embedding_dim=self.head_embedding_dim)
                for _ in range(self.num_transformer_layers)
            ]
        )

        self.seq_start_token = nn.Embedding(num_embeddings=1, embedding_dim=self.embedding_dim)
        self.decode_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.temperature = 1.0

    """
    Takes in Token IDs. If in evaluation mode, returns softmax probabilities of the next token. Otherwise, returns the cross-entropy loss.
    """
    def forward(self, token_ids):
        N, _ = token_ids.shape

        token_embeddings = checkpoint.checkpoint(self.fetch_embeddings, token_ids, use_reentrant=False)
        latent_next_tokens = checkpoint.checkpoint_sequential(self.transformer_layers, segments=self.num_transformer_layers // 2, input=token_embeddings, use_reentrant=False)
        
        model_output = None
        if self.training:
            model_output = checkpoint.checkpoint(self.calc_next_token_loss, latent_next_tokens, token_ids, use_reentrant=False)
        else:
            model_output = checkpoint.checkpoint(self.calc_next_token_dist, latent_next_tokens, use_reentrant=False)

        return model_output
    
    """
    Takes in a list of tokens. Returns the embeddings.
    """
    def fetch_embeddings(self, token_ids):
        # we want to replace all padding indices
        cleaned_token_ids = torch.where(token_ids != -1, token_ids, 0)
        text_embedding = self.embedding_matrix(cleaned_token_ids) + self.positional_encoding.weight
        start_embedding = self.seq_start_token.weight.view(1, 1, -1).expand(text_embedding.shape[0], -1, -1)

        return torch.cat((start_embedding, text_embedding), dim=1)
    
    def calc_next_token_logits(self, latent_next_tokens):
        return torch.matmul(self.decode_layer_norm(latent_next_tokens), self.embedding_matrix.weight.transpose(0, 1)) / self.temperature
    
    def calc_next_token_dist(self, latent_next_tokens):
        return F.softmax(self.calc_next_token_logits(latent_next_tokens), dim=2)

    def calc_next_token_loss(self, latent_next_tokens, token_ids):
        # the last token doesn't have a next token to predict, so we can slice that out
        latent_next_tokens = latent_next_tokens[:, :-1, :]
        
        N, L, _ = latent_next_tokens.shape

        flattened_logits = self.calc_next_token_logits(latent_next_tokens).view(N * L, -1)
    
        # we do not have to shift the sequence here do tho the sequence start token
        # "<S>  I       like   to   go   to      school <E>" (what the LLM sees as its input) ->
        # "I    like    to     go   to   school  <E>" (what it outputs after slicing out last token predictions)
        # thus we can directly take in our input sequence and use it for cross-entropy loss
        flattened_tokens = token_ids.flatten()

        return F.cross_entropy(input=flattened_logits, target=flattened_tokens, ignore_index=-1)
    
    def set_temperature(self, temp):
        self.temperature = temp