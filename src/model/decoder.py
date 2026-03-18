import torch
import torch.nn as nn

class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout):
        super(LSTMDecoder, self).__init__()

        # Embedding : convertit les indices de mots en vecteurs
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # LSTM : génère les mots un par un
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Couche finale : projette vers le vocabulaire
        self.fc = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions):
        # features : vecteur image (batch_size, embed_size)
        # captions : indices des mots (batch_size, seq_len)

        # Embedder les captions
        embeddings = self.dropout(self.embedding(captions))

        # Ajouter le vecteur image au début de la séquence
        # (batch_size, 1, embed_size)
        features = features.unsqueeze(1)

        # Concaténer image + captions
        # (batch_size, seq_len+1, embed_size)
        inputs = torch.cat((features, embeddings), dim=1)

        # Passer dans le LSTM
        lstm_out, _ = self.lstm(inputs)

        # Projeter vers le vocabulaire
        outputs = self.fc(lstm_out)

        return outputs

    def generate(self, feature, vocab, max_len=20):
        """Génère une caption pour une image (inférence)."""
        result = []

        # Etat initial
        states = None
        input_word = torch.tensor([vocab.stoi["<start>"]]).unsqueeze(0)  # (1, 1)

        # Utiliser le vecteur image comme premier input
        feature = feature.unsqueeze(0).unsqueeze(0)  # (1, 1, embed_size)
        lstm_out, states = self.lstm(feature, states)

        for _ in range(max_len):
            # Embedder le mot actuel
            emb = self.dropout(self.embedding(input_word))  # (1, 1, embed_size)

            # Passer dans le LSTM
            lstm_out, states = self.lstm(emb, states)

            # Prédire le prochain mot
            output = self.fc(lstm_out.squeeze(1))  # (1, vocab_size)
            predicted = output.argmax(dim=1)       # indice du mot prédit

            word = vocab.itos[predicted.item()]
            if word == "<end>":
                break

            result.append(word)
            input_word = predicted.unsqueeze(0)

        return " ".join(result)
