import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, embed_size):
        super(CNNEncoder, self).__init__()

        # ---- Architecture CNN from scratch ----
        self.cnn = nn.Sequential(

            # Bloc 1 : 3 canaux (RGB) -> 32 filtres, image 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Bloc 2 : 32 -> 64 filtres, image 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Bloc 3 : 64 -> 128 filtres, image 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Bloc 4 : 128 -> 256 filtres, image 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Bloc 5 : 256 -> 512 filtres
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # -> 512 x 4 x 4
        )

        # Couche linéaire : aplatir et projeter vers embed_size
        self.fc = nn.Sequential(
            nn.Flatten(),                        # 512*4*4 = 8192
            nn.Linear(512 * 4 * 4, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, embed_size)          # -> vecteur final
        )

    def forward(self, images):
        features = self.cnn(images)   # extraire les features
        features = self.fc(features)  # projeter vers embed_size
        return features