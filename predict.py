import argparse, os, torch
from PIL import Image
from torchvision import transforms
from prepara_dataset import eval_tf
from model_baseline import SimpleCNN

def load_image(path):
    # carrega imagem e aplica o mesmo transform de teste
    img = Image.open(path).convert("RGB")
    return eval_tf(img).unsqueeze(0)

@torch.no_grad()
def predict_one(model, device, path, thresh=0.5):
    # faz predição pra uma imagem
    x = load_image(path).to(device)
    logit = model(x)
    prob = torch.sigmoid(logit).item()
    label = "real" if prob >= thresh else "fake"
    return prob, label

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="imagem ou pasta")
    ap.add_argument("--ckpt", default="runs/best_model.pth", help="checkpoint .pth")
    ap.add_argument("--thresh", type=float, default=0.5, help="limiar 'real'")
    args = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # carrega o modelo
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # se for pasta, processa todos os arquivos
    if os.path.isdir(args.input):
        files = [os.path.join(args.input, f) for f in os.listdir(args.input)
                 if os.path.isfile(os.path.join(args.input, f))]
        for f in sorted(files):
            try:
                prob, label = predict_one(model, device, f, args.thresh)
                print(f"{f} -> {label} (p={prob:.3f})")
            except Exception as e:
                print(f"{f} -> erro: {e}")
    else:
        # processa uma imagem só
        prob, label = predict_one(model, device, args.input, args.thresh)
        print(f"{args.input} -> {label} (p={prob:.3f})")

if __name__ == "__main__":
    main()