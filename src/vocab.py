import re
from collections import Counter

class Vocabulary:
    def __init__(self, freq_threshold=5):
        # Tokens spéciaux
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold  # ignorer les mots trop rares

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        # Mettre en minuscules et enlever la ponctuation
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        return text.split()

    def build_vocab(self, captions):
        # Compter tous les mots dans toutes les captions
        counter = Counter()
        for caption in captions:
            tokens = self.tokenize(caption)
            counter.update(tokens)

        # Ajouter les mots suffisamment fréquents
        idx = len(self.itos)
        for word, freq in counter.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        # Convertir une phrase en liste d'indices
        tokens = self.tokenize(text)
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]