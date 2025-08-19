import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, max_len=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))

        # Set batch_first=True for transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        # Add batch dimension if needed
        if src.dim() == 1:
            src = src.unsqueeze(0)
        if tgt.dim() == 1:
            tgt = tgt.unsqueeze(0)
            
        src_emb = self.embedding(src) + self.positional_encoding[:, :src.size(1)]
        tgt_emb = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1)]

        memory = self.encoder(src_emb)
        output = self.decoder(tgt_emb, memory)
        return self.fc(output)