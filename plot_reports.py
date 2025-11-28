import csv, os, matplotlib.pyplot as plt

def read_csv(csv_path):
    # lê o CSV e retorna lista de dicionários
    rows = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def main():
    csv_path = "runs/metrics.csv"
    out_png = "runs/fig_evolucao_from_csv.png"

    if not os.path.exists(csv_path):
        print("CSV não encontrado:", csv_path)
        return

    rows = read_csv(csv_path)

    # converte as colunas pra poder plotar
    ep   = [int(r["epoch"]) for r in rows]
    tr   = [float(r["train_loss"]) for r in rows]
    va   = [float(r["val_loss"])   for r in rows]
    acc  = [float(r["val_acc"])    for r in rows]
    prec = [float(r["val_prec"])   for r in rows]
    rec  = [float(r["val_rec"])    for r in rows]
    f1   = [float(r["val_f1"])     for r in rows]

    plt.figure(figsize=(8,5))
    plt.plot(ep, tr, label="train_loss")
    plt.plot(ep, va, label="val_loss")
    plt.plot(ep, acc, label="val_acc")
    plt.plot(ep, prec, label="val_prec")
    plt.plot(ep, rec, label="val_rec")
    plt.plot(ep, f1, label="val_f1")

    plt.xlabel("Época")
    plt.ylabel("Valor")
    plt.title("Evolução (CSV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print("salvo:", out_png)

if __name__ == "__main__":
    main()