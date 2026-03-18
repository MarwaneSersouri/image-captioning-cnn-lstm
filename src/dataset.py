import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from vocab import Vocabulary

class Flickr8kDataset(Dataset):
    def __init__(self, img_dir, captions_file, train_file, vocab=None, freq_threshold=5):
        self.img_dir = img_dir

        # Transformation appliquée à chaque image
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),       # redimensionner en 224x224
            transforms.ToTensor(),               # convertir en tensor
            transforms.Normalize(                # normaliser les pixels
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Charger la liste des images d'entraînement
        with open(train_file, "r") as f:
            train_images = set(line.strip() for line in f.readlines())

        # Charger les captions
        self.imgs = []
        self.captions = []
        with open(captions_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                img_name = parts[0].split("#")[0]
                caption = parts[1]
                if img_name in train_images:
                    self.imgs.append(img_name)
                    self.captions.append(caption)

        # Construire le vocabulaire
        if vocab is None:
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocabulary(self.captions)
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Charger et transformer l'image
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Convertir la caption en indices
        caption = self.captions[idx]
        tokens = [self.vocab.stoi["<start>"]]
        tokens += self.vocab.numericalize(caption)
        tokens += [self.vocab.stoi["<end>"]]
        caption_tensor = torch.tensor(tokens, dtype=torch.long)

        return image, caption_tensor


def collate_fn(batch):
    """Permet de regrouper des captions de longueurs différentes en un batch."""
    imgs, captions = zip(*batch)
    imgs = torch.stack(imgs, 0)

    # Padder les captions pour qu'elles aient toutes la même longueur
    lengths = [len(cap) for cap in captions]
    max_len = max(lengths)
    padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded[i, :len(cap)] = cap

    return imgs, padded, torch.tensor(lengths)