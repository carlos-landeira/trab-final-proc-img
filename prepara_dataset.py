from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

IM_SIZE = 224

# transform pra treino: com data augmentation
train_tf = transforms.Compose([
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# transform pra validação/teste: só resize e normalização
eval_tf = transforms.Compose([
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class DeepFakeDataset(Dataset):
    # dataset simples que lê imagens de data/{split}/{fake,real}
    # label 0 = fake, label 1 = real
    def __init__(self, root, split, transform):
        self.items = []
        self.transform = transform

        # percorre fake (0) e real (1)
        for label, cls in enumerate(["fake", "real"]):
            d = os.path.join(root, split, cls)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                p = os.path.join(d, f)
                if os.path.isfile(p):
                    self.items.append((p, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            # fallback se a imagem não abrir
            img = Image.new("RGB", (IM_SIZE, IM_SIZE), (0, 0, 0))
        return self.transform(img), y