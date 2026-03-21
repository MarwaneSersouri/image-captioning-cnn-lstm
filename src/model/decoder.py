import torch
import torch.nn as nn

class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout):
        super(LSTMDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embedding(captions))
        features = features.unsqueeze(1)
        inputs = torch.cat((features, embeddings), dim=1)
        lstm_out, _ = self.lstm(inputs)
        outputs = self.fc(lstm_out)
        return outputs

    def generate(self, feature, vocab, max_len=20):
        result = []
        states = None

        feature = feature.unsqueeze(0).unsqueeze(0)   # (1, 1, embed_size)
        lstm_out, states = self.lstm(feature, states)

        input_word = torch.tensor(
            [[vocab.stoi["<start>"]]],
            device=feature.device
        )

        for _ in range(max_len):
            emb = self.embedding(input_word)          # (1, 1, embed_size)
            lstm_out, states = self.lstm(emb, states)
            output = self.fc(lstm_out.squeeze(1))     # (1, vocab_size)

            predicted = output.argmax(dim=1)          # (1,)
            word = vocab.itos[predicted.item()]

            if word == "<end>":
                break

            if word not in ["<pad>", "<start>"]:
                result.append(word)

            input_word = predicted.unsqueeze(1)       # (1, 1)

        return " ".join(result)