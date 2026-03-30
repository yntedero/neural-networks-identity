# Uloha 1 - Identita
# Neuronova siet 4-2-4 s Sigmoid aktivaciou.
#
# Autor: Ostapchuk

import torch
import torch.nn as nn
import numpy as np
import os

# seed - aby vysledky boli vzdy rovnake
torch.manual_seed(42)
np.random.seed(42)

print("PyTorch verzia:", torch.__version__)

# neuronova siet 4-2-4
# vstup 4 neurony -> skryta vrstva 2 neurony -> vystup 4 neurony
# sigmoid dava hodnoty medzi 0 a 1

class IdentityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 2)
        self.output = nn.Linear(2, 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x


print("Siet pripravena")

# funkcia na trenovanie siete
# v kazdej epoche sa nahodne zamiesaju vektory
# pre kazdy vektor sa spocita chyba (SSE) a upravia vahy

def train(model, data, epochs, lr, print_every=500):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        poradie = torch.randperm(len(data))
        celkova_chyba = 0.0

        for i in poradie:
            x = data[i]
            vystup = model(x)
            chyba = torch.sum((vystup - x) ** 2)  # SSE
            celkova_chyba += chyba.item()

            optimizer.zero_grad()
            chyba.backward()
            optimizer.step()

        if epoch % print_every == 0 or epoch == epochs:
            print(f"Epocha {epoch}: chyba = {celkova_chyba:.4f}")

    return celkova_chyba

# testovanie siete
# accuracy = kolko bitov sedi po zaokruhleni vystupu
# [AI] koncept reliability som nasiel cez AI - meria ci su vystupy
# jasne (blizko 0 alebo 1) a nie niekde v strede okolo 0.5

def test(model, data, epsilon=0.2):
    model.eval()
    spravne = 0
    celk_acc = 0
    celk_rel = 0

    print("\nVysledky testu:")

    with torch.no_grad():
        for x in data:
            vystup = model(x)
            zaokr = torch.round(vystup)
            chyba = torch.sum((vystup - x) ** 2).item()

            # kolko bitov je spravnych
            acc = (zaokr == x).sum().item() / len(x) * 100
            celk_acc += acc

            # kolko hodnot je blizko 0 alebo 1
            rel_ok = ((vystup < epsilon) | (vystup > 1 - epsilon)).sum().item()
            rel = rel_ok / len(x) * 100
            celk_rel += rel

            if (zaokr == x).all():
                spravne += 1

            inp = ' '.join([str(int(v)) for v in x])
            resp = ' '.join([f'{v:.2f}' for v in vystup])
            print(f"  {inp}  ->  {resp}   chyba={chyba:.3f}  acc={acc:.0f}%")

    n = len(data)
    print(f"\nSpravne vektory: {spravne}/{n}")
    print(f"Priemerna accuracy: {celk_acc / n:.1f}%")
    print(f"Priemerna reliability: {celk_rel / n:.1f}%")

    model.train()
    return spravne, celk_acc / n, celk_rel / n


# ============================================================
# Poduloha 1: Identita pre 5 vektorov
# Siet sa uci reprodukovat 5 vybranych binarnych vektorov.
# ============================================================

# 5 binarnych vektorov - vstup = vystup (identita)
data_5 = torch.tensor([
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 0, 0]
], dtype=torch.float32)

print("Data (5 vektorov):")
print(data_5)

# Experiment 1.1 - zaklad, lr=0.5, 1000 epoch
print("=== Experiment 1.1 (lr=0.5) ===\n")

torch.manual_seed(42)
model_1_1 = IdentityNet()
train(model_1_1, data_5, epochs=1000, lr=0.5, print_every=200)
s_1_1, a_1_1, r_1_1 = test(model_1_1, data_5)

# Experiment 1.2 - vyssie lr=2.0, viac epoch
print("=== Experiment 1.2 (lr=2.0) ===\n")

torch.manual_seed(42)
model_1_2 = IdentityNet()
train(model_1_2, data_5, epochs=3000, lr=2.0, print_every=500)
s_1_2, a_1_2, r_1_2 = test(model_1_2, data_5)

# Experiment 1.3 - krokovy learning rate
# [AI] napad pouzit krokovy lr som nasiel s pomocou AI
# princip: zaciname s velkym lr (rychle ucenie) a postupne
# ho znizujeme (presnejsie doladenie na konci)

print("=== Experiment 1.3 (krokovy lr) ===\n")

torch.manual_seed(42)
model_1_3 = IdentityNet()
print("--- faza 1: lr=2.0 ---")
train(model_1_3, data_5, epochs=1000, lr=2.0, print_every=500)
print("\n--- faza 2: lr=0.5 ---")
train(model_1_3, data_5, epochs=1000, lr=0.5, print_every=500)
print("\n--- faza 3: lr=0.05 ---")
train(model_1_3, data_5, epochs=500, lr=0.05, print_every=500)
s_1_3, a_1_3, r_1_3 = test(model_1_3, data_5)

# porovnanie experimentov - poduloha 1
print("=" * 50)
print("POROVNANIE - PODULOHA 1 (5 vektorov)")
print("=" * 50)
print(f"Exp 1.1 (lr=0.5):    spravne={s_1_1}/5  acc={a_1_1:.1f}%  rel={r_1_1:.1f}%")
print(f"Exp 1.2 (lr=2.0):    spravne={s_1_2}/5  acc={a_1_2:.1f}%  rel={r_1_2:.1f}%")
print(f"Exp 1.3 (step lr):   spravne={s_1_3}/5  acc={a_1_3:.1f}%  rel={r_1_3:.1f}%")


# ============================================================
# Poduloha 2: Identita pre 16 vektorov
# Siet sa uci vsetkych 16 moznych 4-bitovych vektorov.
# Je to tazsie, lebo 2 skryte neurony musia zakodovat 16 roznych vzorov.
# ============================================================

# vygenerovanie vsetkych 16 binarnych vektorov
# [AI] bitovy posun (>> a &) som si nasiel pomocou AI
# funguje to tak ze z cisla napr. 13 (binarne 1101) dostanem [1, 1, 0, 1]
# >> posunie bity doprava, & 1 zoberie len posledny bit

data_16 = []
for i in range(16):
    vektor = [(i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1]
    data_16.append(vektor)

data_16 = torch.tensor(data_16, dtype=torch.float32)

print("Data (16 vektorov):")
for v in data_16:
    print(f"  [{int(v[0])} {int(v[1])} {int(v[2])} {int(v[3])}]")

# Experiment 2.1 - zaklad, lr=0.5, 5000 epoch
print("=== Experiment 2.1 (lr=0.5) ===\n")

torch.manual_seed(42)
model_2_1 = IdentityNet()
train(model_2_1, data_16, epochs=5000, lr=0.5, print_every=1000)
s_2_1, a_2_1, r_2_1 = test(model_2_1, data_16)

# Experiment 2.2 - lr=2.0, 10000 epoch
print("=== Experiment 2.2 (lr=2.0) ===\n")

torch.manual_seed(42)
model_2_2 = IdentityNet()
train(model_2_2, data_16, epochs=10000, lr=2.0, print_every=2000)
s_2_2, a_2_2, r_2_2 = test(model_2_2, data_16)

# Experiment 2.3 - krokovy lr, rovnaky princip ako v 1.3
print("=== Experiment 2.3 (krokovy lr) ===\n")

torch.manual_seed(42)
model_2_3 = IdentityNet()
print("--- faza 1: lr=2.0 ---")
train(model_2_3, data_16, epochs=5000, lr=2.0, print_every=2000)
print("\n--- faza 2: lr=0.5 ---")
train(model_2_3, data_16, epochs=3000, lr=0.5, print_every=2000)
print("\n--- faza 3: lr=0.05 ---")
train(model_2_3, data_16, epochs=2000, lr=0.05, print_every=2000)
s_2_3, a_2_3, r_2_3 = test(model_2_3, data_16)

# porovnanie experimentov - poduloha 2
print("=" * 50)
print("POROVNANIE - PODULOHA 2 (16 vektorov)")
print("=" * 50)
print(f"Exp 2.1 (lr=0.5):    spravne={s_2_1}/16  acc={a_2_1:.1f}%  rel={r_2_1:.1f}%")
print(f"Exp 2.2 (lr=2.0):    spravne={s_2_2}/16  acc={a_2_2:.1f}%  rel={r_2_2:.1f}%")
print(f"Exp 2.3 (step lr):   spravne={s_2_3}/16  acc={a_2_3:.1f}%  rel={r_2_3:.1f}%")

# ulozenie modelov
os.makedirs('models', exist_ok=True)

torch.save(model_1_1.state_dict(), 'models/poduloha1_exp1.pth')
torch.save(model_1_2.state_dict(), 'models/poduloha1_exp2.pth')
torch.save(model_1_3.state_dict(), 'models/poduloha1_exp3.pth')
torch.save(model_2_1.state_dict(), 'models/poduloha2_exp1.pth')
torch.save(model_2_2.state_dict(), 'models/poduloha2_exp2.pth')
torch.save(model_2_3.state_dict(), 'models/poduloha2_exp3.pth')
print("Modely ulozene")

# Zhrnutie
#
# Poduloha 1 (5 vektorov) - siet zvladne 100% accuracy, je to lahsia uloha.
# Krokovy lr dava najlepsiu reliability.
#
# Poduloha 2 (16 vektorov) - tazsie, 2 skryte neurony musia zakodovat 16 vzorov.
# Krokovy lr opat funguje najlepsie.
#
# Co som zistil:
# - velky learning rate = rychle ucenie ale moze byt nestabilne
# - maly learning rate = pomale ale stabilne
# - krokovy lr (velky -> maly) je najlepsia strategia
