# Image Captioning with CNN-LSTM

Projet de génération automatique de descriptions d’images avec :

- un CNN encoder codé/entrainé from scratch
- un LSTM decoder pour générer les captions et les descriptions d'images

## Project structure

- `src/model/encoder.py` : CNN encoder
- `src/model/decoder.py` : LSTM decoder
- `src/model/captioning_model.py` : modèle complet
- `src/dataset.py` : dataset et dataloader
- `src/vocab.py` : vocabulaire et tokenisation
- `src/train.py` : entraînement
- `src/evaluate.py` : évaluation
- `src/inference.py` : génération de captions
- `configs/base.yaml` : configuration

## Dataset

Dataset prévu : Flickr8k

## Goal

Construire un système d'image captioning en PyTorch avec un pipeline complet :
image -> CNN encoder -> feature vector -> LSTM decoder -> caption