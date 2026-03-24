import os
import sys
import yaml
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))

from dataset import Flickr8kDataset, collate_fn
from model.captioning_model import ImageCaptioningModel


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.1, 0.1)


def train():
    with open("configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Utilisation de : {device}")

    dataset = Flickr8kDataset(
        img_dir=config["data"]["img_dir"],
        captions_file=config["data"]["captions_file"],
        train_file=config["data"]["train_file"],
        freq_threshold=config["training"]["freq_threshold"]
    )

    print(f"Taille du dataset : {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        drop_last=True
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

    model.apply(init_weights)

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<pad>"])

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=1e-4
    )

    output_dir = config["output"]["model_dir"]
    os.makedirs(output_dir, exist_ok=True)

    best_model_path = os.path.join(output_dir, "best_model.pth")
    checkpoint_path = os.path.join(output_dir, "checkpoint.pth")

    num_epochs = config["training"]["num_epochs"]
    best_loss = float("inf")
    start_epoch = 0

    # Reprise auto si un checkpoint existe
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint trouvé : {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]

        print(f"Reprise à partir de l'époque {start_epoch + 1}")
        print(f"Meilleure loss connue : {best_loss:.4f}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (imgs, captions, lengths) in enumerate(dataloader):
            imgs = imgs.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)

            outputs = model(imgs, captions[:, :-1])
            outputs = outputs[:, 1:, :]

            targets = captions[:, 1:]
            loss = criterion(
                outputs.reshape(-1, vocab_size),
                targets.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Batch [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss moyenne: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"🌟 Nouveau meilleur modèle sauvegardé : {best_model_path} (Loss: {best_loss:.4f})")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
        }, checkpoint_path)

        print(f"Checkpoint sauvegardé : {checkpoint_path}")

    print("Entraînement terminé !")


if __name__ == "__main__":
    train()