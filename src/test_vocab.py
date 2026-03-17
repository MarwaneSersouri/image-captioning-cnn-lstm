from vocab import Vocabulary

sentences = [
    "A dog runs in the grass.",
    "A child plays with a dog.",
    "The dog is running fast."
]

vocab = Vocabulary(min_freq=1)
vocab.build_vocabulary(sentences)

print("Taille vocab :", len(vocab))
print("stoi :", vocab.stoi)
encoded = vocab.encode_caption("A dog runs fast.")
print("Encoded :", encoded)
print("Decoded :", vocab.decode_indices(encoded))