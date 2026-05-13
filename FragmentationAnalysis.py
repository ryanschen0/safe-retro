import safe as sf
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from collections import Counter
import pandas as pd

# Open Sample
lines = open("/data/ryanschen/safe-retro/USPTO_SAFE_preprocessed/src-train.txt").readlines()

n_frags_list = []
single_frag = 0
large_frag = 0  # Fragments with >20 heavy atoms, meaning majoritarily unsplit aromatic sytems

for line in lines[:10000]:   # sample first 10k
    safe_str = line.strip()
    frags = safe_str.split('.')
    n = len(frags)
    n_frags_list.append(n)
    if n == 1:
        single_frag += 1
    # check for large unsplit fragments
    for frag in frags:
        try:
            mol = Chem.MolFromSmiles(frag.replace('%10','').replace('%11',''))
            if mol and mol.GetNumHeavyAtoms() > 20:
                large_frag += 1
        except:
            pass

print(f"Single-fragment molecules: {single_frag/100:.1f}%")
print(f"Fragment count distribution: {Counter(n_frags_list).most_common(10)}")
print(f"Large unsplit fragments found: {large_frag}")