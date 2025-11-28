import os, csv, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

from prepara_dataset import DeepFakeDataset, train_tf, eval_tf, IM_SIZE
from model_baseline import SimpleCNN
from utils_metrics import counts_to_metrics, confusion_counts, print_metrics, plot_series, plot_confusion_matrix, plot_roc_pr

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# usa GPU da Apple se tiver, senão CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

torch.set_float32_matmul_precision("high")
print("Device:", device)

DATA_ROOT = "data"
runs_dir = "runs"
os.makedirs(runs_dir, exist_ok=True)
csv_path = os.path.join(runs_dir, "metrics.csv")
BEST_PATH = os.path.join(runs_dir, "best_model.pth")

# carrega os datasets
train_ds = DeepFakeDataset(DATA_ROOT, "train", transform=train_tf)
valid_ds = DeepFakeDataset(DATA_ROOT, "valid", transform=eval_tf)
test_ds  = DeepFakeDataset(DATA_ROOT, "test",  transform=eval_tf)
print(f"Train: {len(train_ds)} | Valid: {len(valid_ds)} | Test: {len(test_ds)}")

def label_counts(ds):
    c = {0:0, 1:0}
    for _, y in ds.items: 
        c[y] += 1
    return c

# conta fake/real pra calcular o peso da loss
counts = label_counts(train_ds)
neg, pos = counts[0], counts[1]

pos_weight = torch.tensor([(neg / max(1, pos)) if pos > 0 else 1.0], device=device)
print("pos_weight:", float(pos_weight.item()))

BATCH_SIZE = 16
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# testa se tá carregando direito
imgs, ys = next(iter(train_loader))
print("Batch imgs:", imgs.shape, "| dtype:", imgs.dtype)
print("Batch ys:", ys.shape, "| únicos:", ys.unique())
assert imgs.shape[1:] == (3, IM_SIZE, IM_SIZE)

model = SimpleCNN().to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# reduz LR se a validação não melhorar
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=1
)

EPOCHS = 30
THRESH = 0.54
best_val = float("inf")

# cria CSV pra guardar métricas
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["epoch","train_loss","val_loss","val_acc","val_prec","val_rec","val_f1"])

@torch.no_grad()
def evaluate(loader, tag="VAL", threshold=0.5):
    # avalia o modelo e calcula as métricas
    model.eval()
    total_loss, n = 0.0, 0
    tp = fp = fn = tn = 0

    for imgs, ys in loader:
        imgs = imgs.to(device)
        ys_t = ys.to(device=device, dtype=torch.float32).unsqueeze(1)

        logits = model(imgs)
        loss = criterion(logits, ys_t)

        total_loss += float(loss.item()) * imgs.size(0)
        n += imgs.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        _tp, _fp, _fn, _tn = confusion_counts(preds, ys_t)
        tp += _tp; fp += _fp; fn += _fn; tn += _tn

    avg_loss = total_loss / max(1, n)
    acc, prec, rec, f1 = counts_to_metrics(tp, fp, fn, tn)
    print_metrics(tag, avg_loss, acc, prec, rec, f1)
    return avg_loss, acc, prec, rec, f1

def train_one_epoch(epoch):
    # treina uma época
    model.train()
    running_loss, seen = 0.0, 0

    pbar = tqdm(train_loader, desc=f"ép {epoch:02d}")
    for imgs, ys in pbar:
        imgs = imgs.to(device)
        ys_t = ys.to(device=device, dtype=torch.float32).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        # usa mixed precision no MPS
        with autocast(device_type=device.type, enabled=(device.type == "mps")):
            logits = model(imgs)
            loss = criterion(logits, ys_t)

        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * imgs.size(0)
        seen += imgs.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(1, seen)

@torch.no_grad()
def best_thresh_by_f1(loader):
    # busca o melhor threshold testando vários valores
    model.eval()
    all_logits, all_ys = [], []

    for imgs, ys in loader:
        imgs = imgs.to(device)
        ys_t = ys.to(device=device, dtype=torch.float32).unsqueeze(1)
        all_logits.append(model(imgs).cpu())
        all_ys.append(ys_t.cpu())

    logits = torch.cat(all_logits, 0)
    ys = torch.cat(all_ys, 0)
    probs = torch.sigmoid(logits)

    best_t, best_f1 = 0.5, 0.0

    # testa de 0.30 até 0.70
    for t in torch.linspace(0.30, 0.70, steps=41):
        preds = (probs >= t).float()
        tp = int(((preds == 1) & (ys == 1)).sum())
        tn = int(((preds == 0) & (ys == 0)).sum())
        fp = int(((preds == 1) & (ys == 0)).sum())
        fn = int(((preds == 0) & (ys == 1)).sum())

        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)

    return best_t, best_f1

# loop de treino
ep_hist, tr_loss_hist, va_loss_hist, acc_hist, prec_hist, rec_hist, f1_hist = [], [], [], [], [], [], []

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(epoch)

    val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
        valid_loader, tag=f"VAL   ep{epoch:02d}", threshold=THRESH
    )

    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow([epoch, train_loss, val_loss, val_acc, val_prec, val_rec, val_f1])

    # guarda histórico pra plotar depois
    ep_hist.append(epoch); tr_loss_hist.append(train_loss); va_loss_hist.append(val_loss)
    acc_hist.append(val_acc); prec_hist.append(val_prec); rec_hist.append(val_rec); f1_hist.append(val_f1)

    # salva se melhorou
    if val_loss < best_val - 1e-6:
        best_val = val_loss
        torch.save(model.state_dict(), BEST_PATH)
        print(f"  ↳ checkpoint salvo: {BEST_PATH}")

    # ajusta learning rate se necessário
    prev_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_loss)
    new_lr = optimizer.param_groups[0]['lr']
    if new_lr < prev_lr:
        print(f"  ↳ LR reduzido: {prev_lr:.2e} → {new_lr:.2e}")

# gera gráficos de evolução
plot_series(
    ep_hist, tr_loss_hist, va_loss_hist,
    acc_hist, prec_hist, rec_hist, f1_hist,
    out_png=os.path.join(runs_dir, "fig_evolucao.png")
)

# busca melhor threshold
print("\n[Ajuste de limiar por F1 na validação]")
best_t, best_f1 = best_thresh_by_f1(valid_loader)
print(f"THRESH* = {best_t:.2f} (val_F1={best_f1:.4f})")

# avalia no teste
print("\n[INFO] Carregando melhor modelo e avaliando no TEST...")
model.load_state_dict(torch.load(BEST_PATH, map_location=device))
model.to(device)

test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(
    test_loader, tag="TEST  best", threshold=best_t
)

@torch.no_grad()
def collect_probs(loader):
    model.eval()
    probs_list, ys_list = [], []
    for imgs, ys in loader:
        imgs = imgs.to(device)
        ys_t = ys.to(device=device, dtype=torch.float32).unsqueeze(1)
        probs = torch.sigmoid(model(imgs))
        probs_list.append(probs.cpu())
        ys_list.append(ys_t.cpu())
    return torch.cat(probs_list, 0).numpy().ravel(), torch.cat(ys_list, 0).numpy().ravel()

# coleta probabilidades pra fazer as curvas
probs_test, ys_test = collect_probs(test_loader)

preds_test = (torch.from_numpy(probs_test) >= best_t).int()
ys_test_t  = torch.from_numpy(ys_test).int()

tp, fp, fn, tn = confusion_counts(preds_test, ys_test_t)

plot_confusion_matrix(tp, fp, fn, tn, out_png=os.path.join(runs_dir, "fig_confusion.png"))

plot_roc_pr(
    ys_test, probs_test,
    out_roc_png=os.path.join(runs_dir, "fig_roc.png"),
    out_pr_png=os.path.join(runs_dir, "fig_pr.png")
)

# salva resumo final
with open(os.path.join(runs_dir, "summary.txt"), "w") as f:
    f.write(
        f"TEST (THRESH*={best_t:.2f})\n"
        f"loss={test_loss:.4f} acc={test_acc:.4f} prec={test_prec:.4f} rec={test_rec:.4f} f1={test_f1:.4f}\n"
        f"CM: TP={tp} FP={fp} FN={fn} TN={tn}\n"
    )