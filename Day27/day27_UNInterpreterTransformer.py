# UNInterpreterTransformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size must be divisible by heads"

        self.values = nn.Linear(self.head_dim, embed_size, bias=False)
        self.keys = nn.Linear(self.head_dim, embed_size, bias=False)
        self.queries = nn.Linear(self.head_dim, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embeddings into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        out = out.reshape(N, query_len, self.embed_size)

        return self.fc_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size)
        )

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        return self.norm2(forward + x)

class SimpleTransformer(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_size, heads):
        super(SimpleTransformer, self).__init__()
        self.encoder_embedding = nn.Embedding(input_vocab_size, embed_size)
        self.decoder_embedding = nn.Embedding(output_vocab_size, embed_size)
        self.encoder = TransformerBlock(embed_size, heads)
        self.decoder = TransformerBlock(embed_size, heads)
        self.fc_out = nn.Linear(embed_size, output_vocab_size)

    def forward(self, src, trg, src_mask, trg_mask):
        enc_embed = self.encoder_embedding(src)
        enc_out = self.encoder(enc_embed, enc_embed, enc_embed, src_mask)

        dec_embed = self.decoder_embedding(trg)
        dec_out = self.decoder(enc_out, enc_out, dec_embed, trg_mask)

        out = self.fc_out(dec_out)
        return out
