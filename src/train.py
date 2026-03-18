import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import sys
import os

sys.path.append(os.path.dirname(__file__))
from dataset import Flickr8kDataset, collate_fn
from model.captioning_model import ImageCaptioningModel

def train():
    with open("configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de : {device}")

    dataset = Flickr8kDataset(
        img_dir=config["data"]["img_dir"],
        captions_file=config["data"]["captions_file"],
        train_file=config["data"]["train_file"],
        freq_threshold=config["training"]["freq_threshold"]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )

    vocab_size = len(dataset.vocab)
    print(f"Taille du vocabulaire : {vocab_size}")

    model = ImageCaptioningModel(
        embed_size=config["model"]["embed_size"],
        hidden_size=config["model"]["hidden_size"],
        vocab_size=vocab_size,
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    num_epochs = config["training"]["num_epochs"]
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (imgs, captions, lengths) in enumerate(dataloader):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:, :-1])
            outputs = outputs[:, 1:, :]

            targets = captions[:, 1:].reshape(-1)
            outputs = outputs.reshape(-1, vocab_size)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss moyenne: {avg_loss:.4f}")

        os.makedirs(config["output"]["model_dir"], exist_ok=True)
        torch.save(model.state_dict(), f"{config['output']['model_dir']}/model_epoch_{epoch+1}.pth")

    print("Entrainement termine !")

if __name__ == "__main__":
    train()


