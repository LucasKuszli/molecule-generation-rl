"""Sample and display 20 HIV-active molecules with P(active) from the GNN.

Shows batches of 20 in a loop. Close the window to see the next batch.
Continues until all actives have been shown.
"""

from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

from rsrc.gnn.predictor import HIVActivityPredictor

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts" / "gnn" / "seed_42" / "best_gnn.pt"

data_path = ROOT / "data" / "training_data" / "hiv.csv"
df = pd.read_csv(data_path)
actives = df[df["label"] == 1].sample(frac=1).reset_index(drop=True)

print(f"Total molecules: {len(df)}")
print(f"HIV actives: {len(actives)}")

predictor = HIVActivityPredictor(str(CHECKPOINT))

batch_size = 20
seen = 0

while seen < len(actives):
    batch = actives.iloc[seen : seen + batch_size]
    mols, legends = [], []
    for smi in batch["SMILES"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        _, prob = predictor.predict(smi)
        if prob is None:
            continue
        mols.append(mol)
        legends.append(f"P(active)={prob:.3f}")

    seen += len(batch)

    if not mols:
        continue

    print(f"Showing {seen}/{len(actives)} ({len(mols)} in this batch)")

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=5,
        subImgSize=(300, 300),
        legends=legends,
    )
    # matplotlib blocks until the window is closed
    fig, ax = plt.subplots(figsize=(15, 12))
    ax.imshow(img)
    ax.set_axis_off()
    fig.suptitle(f"HIV Actives - batch {seen}/{len(actives)}")
    plt.tight_layout()
    plt.show()  # blocks until window is closed

print("All actives shown.")
