import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    CNN simples pra classificação binária:
    - 3 blocos convolucionais [Conv3x3 -> BatchNorm -> ReLU -> MaxPool]
    - Global Average Pooling
    - Cabeça de classificação: Dropout -> FC(128->64->1)
    Saída é logit (antes do sigmoid)
    """
    def __init__(self, p_drop: float = 0.2):
        super().__init__()

        # camadas convolucionais pra extrair features
        self.features = nn.Sequential(
            # bloco 1: RGB (3 canais) -> 32 feature maps
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2),  # 224->112

            # bloco 2: 32 -> 64 canais
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),  # 112->56

            # bloco 3: 64 -> 128 canais
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),  # 56->28
        )

        # faz média em cada feature map
        self.gap = nn.AdaptiveAvgPool2d(1)

        # classificador final
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Linear(64, 1)  # output é logit
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)