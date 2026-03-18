import torch.nn as nn
from encoder import CNNEncoder
from decoder import LSTMDecoder

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout):
        super(ImageCaptioningModel, self).__init__()

        # L'encoder transforme l'image en vecteur
        self.encoder = CNNEncoder(embed_size)

        # Le decoder génère la caption mot par mot
        self.decoder = LSTMDecoder(embed_size, hidden_size, vocab_size, num_layers, dropout)

    def forward(self, images, captions):
        # 1. Extraire les features de l'image
        features = self.encoder(images)

        # 2. Générer la caption à partir des features
        outputs = self.decoder(features, captions)

        return outputs