"""Display HIV-inactive molecules ranked by highest P(active) from the GNN.

Shows batches of 20 in a loop, sorted from highest to lowest predicted
probability. Close the window to see the next batch.
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
inactives = df[df["label"] == 0].copy()

print(f"Total molecules: {len(df)}")
print(f"HIV inactives: {len(inactives)}")
print("Scoring all inactives (this may take a moment)...")

predictor = HIVActivityPredictor(str(CHECKPOINT))

probs = []
for smi in inactives["SMILES"]:
    _, prob = predictor.predict(smi)
    probs.append(prob if prob is not None else 0.0)

inactives["pred_prob"] = probs
inactives = inactives.sort_values("pred_prob", ascending=False).reset_index(drop=True)

print(f"Top P(active) among inactives: {inactives['pred_prob'].iloc[0]:.4f}")

batch_size = 20
seen = 0

while seen < len(inactives):
    batch = inactives.iloc[seen : seen + batch_size]
    mols, legends = [], []
    for _, row in batch.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            continue
        mols.append(mol)
        legends.append(f"P(active)={row['pred_prob']:.3f}")

    seen += len(batch)

    if not mols:
        continue

    print(f"Showing {seen}/{len(inactives)} ({len(mols)} in this batch)")

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=5,
        subImgSize=(300, 300),
        legends=legends,
    )
    fig, ax = plt.subplots(figsize=(15, 12))
    ax.imshow(img)
    ax.set_axis_off()
    fig.suptitle(f"HIV Inactives (by highest P(active)) - {seen}/{len(inactives)}")
    plt.tight_layout()
    plt.show()

print("All inactives shown.")
