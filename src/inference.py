import os
import sys
import yaml
import torch
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt


sys.path.append(os.path.dirname(__file__))

from dataset import Flickr8kDataset
from model.captioning_model import ImageCaptioningModel


def load_config():
    with open("configs/base.yaml", "r") as f:
        return yaml.safe_load(f)


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_vocab(config):
    train_dataset = Flickr8kDataset(
        img_dir=config["data"]["img_dir"],
        captions_file=config["data"]["captions_file"],
        train_file=config["data"]["train_file"],
        freq_threshold=config["training"]["freq_threshold"]
    )
    return train_dataset.vocab


def load_model(config, vocab, model_path, device):
    model = ImageCaptioningModel(
        embed_size=config["model"]["embed_size"],
        hidden_size=config["model"]["hidden_size"],
        vocab_size=len(vocab),
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"]
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_caption(image_path, model, vocab, device):
    image = Image.open(image_path).convert("RGB")
    transform = get_transform()
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = model.encoder(image).squeeze(0)
        caption = model.decoder.generate(feature, vocab)

    return caption


def get_test_image_path(config, index=0):
    with open(config["data"]["test_file"], "r") as f:
        test_images = [line.strip() for line in f if line.strip()]
    image_name = test_images[index]
    return os.path.join(config["data"]["img_dir"], image_name), image_name


def get_ground_truth_captions(captions_file, image_name):
    refs = []
    with open(captions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                continue

            img_id, caption = parts
            img_name = img_id.split("#")[0].strip()

            if img_name == image_name.strip():
                refs.append(caption.strip())

    return refs


def inspect_feature(image_path, model, device):
    image = Image.open(image_path).convert("RGB")
    transform = get_transform()
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = model.encoder(image).squeeze(0).cpu()

    print("Feature shape :", feature.shape)
    print("Feature mean  :", feature.mean().item())
    print("Feature std   :", feature.std().item())
    print("Feature first5:", feature[:5].tolist())

if __name__ == "__main__":
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "/Users/marwanesersouri/Desktop/image-captioning-cnn-lstm/outputs/checkpoints/model_epoch_30.pth"

    vocab = load_vocab(config)
    model = load_model(config, vocab, model_path, device)
    for idx in [0, 1, 2]:
        image_path, image_name = get_test_image_path(config, index=idx)
        print("\n" + "=" * 60)
        print("Image :", image_name)
        inspect_feature(image_path, model, device)