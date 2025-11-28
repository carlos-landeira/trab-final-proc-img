import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np

@torch.no_grad()
def counts_to_metrics(tp, fp, fn, tn):
    # calcula métricas a partir dos contadores da matriz de confusão
    total = tp + fp + fn + tn
    acc  = (tp + tn) / max(1, total)
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    f1   = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return acc, prec, rec, f1

@torch.no_grad()
def confusion_counts(preds, ys):
    # conta TP, FP, FN, TN
    tp = int(((preds == 1) & (ys == 1)).sum().item())
    tn = int(((preds == 0) & (ys == 0)).sum().item())
    fp = int(((preds == 1) & (ys == 0)).sum().item())
    fn = int(((preds == 0) & (ys == 1)).sum().item())
    return tp, fp, fn, tn

def print_metrics(tag, loss, acc, prec, rec, f1):
    # printa as métricas numa linha só
    print(f"[{tag}] loss={loss:.4f} | acc={acc:.4f} | prec={prec:.4f} | rec={rec:.4f} | f1={f1:.4f}")

def plot_series(epochs, train_loss, val_loss, acc, prec, rec, f1, out_png):
    # plota evolução das métricas ao longo do treino
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss,   label="val_loss")
    plt.plot(epochs, acc,        label="val_acc")
    plt.plot(epochs, prec,       label="val_prec")
    plt.plot(epochs, rec,        label="val_rec")
    plt.plot(epochs, f1,         label="val_f1")
    plt.xlabel("Época")
    plt.ylabel("Valor")
    plt.title("Evolução por época")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_confusion_matrix(tp, fp, fn, tn, out_png):
    # plota a matriz de confusão
    cm = np.array([[tn, fp],[fn, tp]])

    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j], ha="center", va="center", color="black")

    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["fake","real"])
    ax.set_yticklabels(["fake","real"])
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    ax.set_title("Matriz de Confusão")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def plot_roc_pr(y_true_np, prob_np, out_roc_png, out_pr_png):
    # plota curvas ROC e Precision-Recall
    fpr, tpr, _ = roc_curve(y_true_np, prob_np)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],"--", alpha=0.5)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_roc_png, dpi=150)
    plt.close()

    prec, rec, _ = precision_recall_curve(y_true_np, prob_np)
    plt.figure(figsize=(5,4))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.tight_layout()
    plt.savefig(out_pr_png, dpi=150)
    plt.close()