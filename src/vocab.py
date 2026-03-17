from collections import Counter


class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq

        self.pad_token = "<pad>"
        self.start_token = "<start>"
        self.end_token = "<end>"
        self.unk_token = "<unk>"

        self.stoi = {
            self.pad_token: 0,
            self.start_token: 1,
            self.end_token: 2,
            self.unk_token: 3,
        }
        self.itos = {idx: token for token, idx in self.stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def tokenize(self, text):
        text = text.lower().strip()
        text = text.replace(".", "")
        text = text.replace(",", "")
        text = text.replace("!", "")
        text = text.replace("?", "")
        text = text.replace(";", "")
        text = text.replace(":", "")
        text = text.replace('"', "")
        text = text.replace("'", "")
        tokens = text.split()
        return tokens

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()

        for sentence in sentence_list:
            tokens = self.tokenize(sentence)
            frequencies.update(tokens)

        idx = len(self.stoi)

        for word, freq in frequencies.items():
            if freq >= self.min_freq and word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokens = self.tokenize(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi[self.unk_token]
            for token in tokens
        ]

    def encode_caption(self, text):
        return (
            [self.stoi[self.start_token]]
            + self.numericalize(text)
            + [self.stoi[self.end_token]]
        )

    def decode_indices(self, indices):
        words = []

        for idx in indices:
            word = self.itos.get(idx, self.unk_token)

            if word == self.start_token or word == self.pad_token:
                continue
            if word == self.end_token:
                break

            words.append(word)

        return " ".join(words)