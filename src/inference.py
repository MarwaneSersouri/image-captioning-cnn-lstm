import os
import torch
from PIL import Image
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {
            0: "<pad>",
            1: "<start>",
            2: "<end>",
            3: "<unk>"
        }
        self.stoi = {
            "<pad>": 0,
            "<start>": 1,
            "<end>": 2,
            "<unk>": 3
        }
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def tokenizer(self, text):
        return text.lower().strip().split()

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<unk>"]
            for token in tokenized_text
        ]


class Flickr8kDataset(Dataset):
    def __init__(self, img_dir, captions_file, train_file, freq_threshold=5, transform=None):
        self.img_dir = img_dir
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.imgs = []
        self.captions = []

        # 1) Lire les noms d'images du split train
        self.split_images = set()
        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                img_name = line.strip()
                if not img_name:
                    continue
                img_name = os.path.basename(img_name)
                self.split_images.add(img_name)

        # 2) Lire les captions Flickr30k version Kaggle (CSV: image_name,comment)
        with open(captions_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # ignorer header ou lignes vides
                if not line or line.startswith("image_name"):
                    continue

                # couper seulement à la première virgule
                parts = line.split(",", 1)
                if len(parts) < 2:
                    continue

                img_name = os.path.basename(parts[0].strip())
                caption = parts[1].strip().strip('"')

                if img_name in self.split_images:
                    self.imgs.append(img_name)
                    self.captions.append(caption)

        # sécurité
        if len(self.imgs) == 0:
            print("⚠️ Aucun sample trouvé")
            print("img_dir =", img_dir)
            print("captions_file =", captions_file)
            print("train_file =", train_file)
            print("nb split_images =", len(self.split_images))

        # 3) vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_path = os.path.join(self.img_dir, self.imgs[index])

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        numericalized_caption = [self.vocab.stoi["<start>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<end>"])

        return image, torch.tensor(numericalized_caption, dtype=torch.long)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)

        captions = [item[1] for item in batch]
        lengths = [len(cap) for cap in captions]

        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)

        return imgs, captions, lengths


def collate_fn(batch):
    pad_idx = 0  # <pad>
    return MyCollate(pad_idx)(batch)