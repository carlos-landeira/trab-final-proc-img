import argparse
import os

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from model_baseline import SimpleCNN
from prepara_dataset import eval_tf  # usa o mesmo transform de validação/teste

# Grad-CAM aplicado na última camada convolucional da SimpleCNN

class SimpleCNNGradCAM:
    def __init__(self, model, device):
        self.model = model
        self.device = device

        # pega a última conv da sequência self.features
        # features = [Conv,BN,ReLU,Pool, Conv,BN,ReLU,Pool, Conv,BN,ReLU,Pool]
        # a última Conv tá no índice 8
        self.target_layer = self.model.features[8]

        self.activations = None
        self.gradients = None

        # registra hooks pra capturar ativações e gradientes
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        # captura ativações: [1, C, H, W]
        self.activations = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        # captura gradientes: [1, C, H, W]
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_logit=None):
        self.model.zero_grad()

        logits = self.model(input_tensor)
        if class_logit is None:
            class_logit = logits[0, 0]

        class_logit.backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations

        weights = grads.mean(dim=(2, 3))[0]  # média espacial

        # cria cam no mesmo device das ativações (importante pro MPS)
        cam = torch.zeros(
            acts.shape[2:],
            dtype=acts.dtype,
            device=acts.device,
        )

        for c, w in enumerate(weights):
            cam += w * acts[0, c, :, :]

        cam = torch.relu(cam)

        cam_np = cam.cpu().numpy()
        cam_np -= cam_np.min()
        cam_np /= (cam_np.max() + 1e-8)

        return cam_np


def load_image_as_tensor(path, device):
    img = Image.open(path).convert("RGB")
    tens = eval_tf(img).unsqueeze(0)  # [1,3,H,W]
    return img, tens.to(device)


def overlay_cam_on_image(img_pil, cam_np, out_path, alpha=0.4):
    # redimensiona o heatmap pro tamanho original da imagem
    cam_img = Image.fromarray(np.uint8(cam_np * 255))
    cam_img = cam_img.resize(img_pil.size, resample=Image.BILINEAR)
    cam_resized = np.array(cam_img)

    # monta o overlay usando matplotlib
    plt.figure(figsize=(5,5))
    plt.imshow(img_pil)
    plt.imshow(cam_resized, cmap="jet", alpha=alpha)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()
    print("Grad-CAM salvo em:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Imagem de entrada (.jpg/.png)")
    ap.add_argument("--ckpt", default="runs/best_model.pth", help="Checkpoint do modelo")
    ap.add_argument("--out", default="gradcam_overlay.png", help="Nome do arquivo de saída")
    args = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Device:", device)

    # carrega o modelo treinado
    model = SimpleCNN().to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # carrega a imagem
    img_pil, x = load_image_as_tensor(args.input, device)

    # cria o Grad-CAM
    gc = SimpleCNNGradCAM(model, device)

    # gera o mapa de ativação
    cam_np = gc.generate(x)

    # aplica overlay e salva
    overlay_cam_on_image(img_pil, cam_np, args.out)


if __name__ == "__main__":
    main()
