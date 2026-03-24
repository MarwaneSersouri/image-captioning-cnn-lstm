import os
import random

# Définir une seed pour que la séparation soit reproductible
random.seed(42)

# Chemins vers tes dossiers Flickr30k (vérifie la majuscule de 'Images')
img_dir = "data/Flickr30k/Images" 
train_file = "data/Flickr30k/trainImages.txt"
dev_file = "data/Flickr30k/devImages.txt"
test_file = "data/Flickr30k/testImages.txt"

# 1. Lister toutes les images (.jpg) du dossier
images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg'))]

# 2. Mélanger la liste aléatoirement
random.shuffle(images)

# 3. Définir la répartition (Classique pour Flickr30k : 29k / 1k / ~1.7k)
num_train = 29000
num_dev = 1000

train_imgs = images[:num_train]
dev_imgs = images[num_train : num_train + num_dev]
test_imgs = images[num_train + num_dev :]

# 4. Fonction pour écrire les listes dans les fichiers textes
def write_to_file(filepath, img_list):
    with open(filepath, 'w') as f:
        for img in img_list:
            f.write(f"{img}\n")

# 5. Création des fichiers
write_to_file(train_file, train_imgs)
write_to_file(dev_file, dev_imgs)
write_to_file(test_file, test_imgs)

print(" Fichiers de séparation créés")
print(f" -> Entraînement : {len(train_imgs)} images")
print(f" -> Validation   : {len(dev_imgs)} images")
print(f" -> Test         : {len(test_imgs)} images")