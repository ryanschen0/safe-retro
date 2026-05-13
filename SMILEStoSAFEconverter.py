"""
SMILEStoSAFEconverter.py
SMILES to SAFE conversion for USPTO-480k reactions with round-trip validation.
"""

import safe
import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

REACTANT_SEP = "~"
SMILES_COL = "rxn_smiles"


# Utilities

def canonical_smiles(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"RDKit failed to parse: {smi}")
    return Chem.MolToSmiles(mol)


def strip_atom_map(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Cannot parse: {smi}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def load_uspto_splits(train_path: str, val_path: str, test_path: str) -> pd.DataFrame:
    train = pd.read_csv(train_path)
    val   = pd.read_csv(val_path)
    test  = pd.read_csv(test_path)

    train["split"] = "train"
    val["split"]   = "val"
    test["split"]  = "test"

    df = pd.concat([train, val, test], ignore_index=True)
    print(f"Loaded {len(train)} train / {len(val)} val / {len(test)} test = {len(df)} total rows")
    return df

# Core conversion

from safe._exception import SAFEFragmentationError

def _encode_molecule_group(smiles_group: str) -> str:
    mols = smiles_group.split(".")
    encoded = []
    for m in mols:
        try:
            encoded.append(safe.encode(m))
        except SAFEFragmentationError:
            # molecule too simple to fragment — keep as plain SMILES
            encoded.append(m)
    return REACTANT_SEP.join(encoded)


def _decode_molecule_group(safe_group: str) -> str:
    pieces = safe_group.split(REACTANT_SEP)
    decoded = [canonical_smiles(safe.decode(p)) for p in pieces]
    return ".".join(decoded)


def smiles_to_safe(rxn_smiles: str) -> str:
    if ">>" in rxn_smiles:
        parts = rxn_smiles.split(">>")
        if len(parts) != 2:
            raise ValueError(f"Malformed reaction SMILES: {rxn_smiles}")
        reactants, products = parts
        return f"{_encode_molecule_group(reactants)}>>{_encode_molecule_group(products)}"
    return _encode_molecule_group(rxn_smiles)


def safe_to_smiles(rxn_safe: str) -> str:
    if ">>" in rxn_safe:
        parts = rxn_safe.split(">>")
        if len(parts) != 2:
            raise ValueError(f"Malformed SAFE reaction: {rxn_safe}")
        reactants, products = parts
        return f"{_decode_molecule_group(reactants)}>>{_decode_molecule_group(products)}"
    return _decode_molecule_group(rxn_safe)


# SAFE to SMILES to SAFE round-trip validation
def _canonical_group(group: str) -> str:
    return ".".join(strip_atom_map(m) for m in group.split("."))


def _canonical_reaction(rxn_smiles: str) -> str:
    if ">>" in rxn_smiles:
        r, p = rxn_smiles.split(">>")
        return f"{_canonical_group(r)}>>{_canonical_group(p)}"
    return _canonical_group(rxn_smiles)


def round_trip_ok(rxn_smiles: str) -> bool:
    try:
        original = _canonical_reaction(rxn_smiles)
        recovered = safe_to_smiles(smiles_to_safe(rxn_smiles))
        return original == recovered
    except Exception:
        return False

# Dataset Validation

def validate_dataset(df: pd.DataFrame, smiles_column: str = SMILES_COL):
    n = len(df)
    encode_fail, decode_fail, mismatch, success = [], [], [], 0

    for i, rxn in enumerate(tqdm(df[smiles_column], total=n, desc="Validating")):
        try:
            safe_str = smiles_to_safe(rxn)
        except Exception as e:
            encode_fail.append((i, rxn, repr(e)))
            continue

        try:
            recovered = safe_to_smiles(safe_str)
        except Exception as e:
            decode_fail.append((i, rxn, safe_str, repr(e)))
            continue

        try:
            if _canonical_reaction(rxn) == recovered:
                success += 1
            else:
                mismatch.append((i, rxn, recovered))
        except Exception as e:
            encode_fail.append((i, rxn, f"canonicalization: {e!r}"))

    print(f"\nTotal rows:       {n}")
    print(f"Round-trip OK:    {success} ({100*success/n:.2f}%)")
    print(f"Encode failures:  {len(encode_fail)}")
    print(f"Decode failures:  {len(decode_fail)}")
    print(f"Mismatches:       {len(mismatch)}")

    return {
        "n": n,
        "success": success,
        "encode_fail": encode_fail,
        "decode_fail": decode_fail,
        "mismatch": mismatch,
    }

# Parallelization

from multiprocessing import Pool, cpu_count

def _safe_encode_one(rxn):
    """Top-level helper for multiprocessing — must be picklable."""
    try:
        return smiles_to_safe(rxn) if pd.notna(rxn) else None
    except Exception:
        return None


def _validate_one(args):
    """Worker for parallel validation. Returns (status, payload)."""
    i, rxn = args
    try:
        safe_str = smiles_to_safe(rxn)
    except Exception as e:
        return ("encode_fail", (i, rxn, repr(e)))

    try:
        recovered = safe_to_smiles(safe_str)
    except Exception as e:
        return ("decode_fail", (i, rxn, safe_str, repr(e)))

    try:
        if _canonical_reaction(rxn) == recovered:
            return ("success", None)
        return ("mismatch", (i, rxn, recovered))
    except Exception as e:
        return ("encode_fail", (i, rxn, f"canonicalization: {e!r}"))


def validate_dataset_parallel(df: pd.DataFrame, smiles_column: str = SMILES_COL, n_workers: int = None):
    n_workers = n_workers or max(1, cpu_count() - 1)
    n = len(df)
    encode_fail, decode_fail, mismatch, success = [], [], [], 0

    args_iter = list(enumerate(df[smiles_column]))

    with Pool(n_workers) as pool:
        for status, payload in tqdm(
            pool.imap_unordered(_validate_one, args_iter, chunksize=64),
            total=n, desc=f"Validating ({n_workers} workers)"
        ):
            if status == "success":
                success += 1
            elif status == "encode_fail":
                encode_fail.append(payload)
            elif status == "decode_fail":
                decode_fail.append(payload)
            elif status == "mismatch":
                mismatch.append(payload)

    print(f"\nTotal rows:       {n}")
    print(f"Round-trip OK:    {success} ({100*success/n:.2f}%)")
    print(f"Encode failures:  {len(encode_fail)}")
    print(f"Decode failures:  {len(decode_fail)}")
    print(f"Mismatches:       {len(mismatch)}")

    return {"n": n, "success": success,
            "encode_fail": encode_fail, "decode_fail": decode_fail, "mismatch": mismatch}


def convert_split_parallel(subset: pd.DataFrame, smiles_column: str, n_workers: int = None):
    n_workers = n_workers or max(1, cpu_count() - 1)
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(_safe_encode_one, subset[smiles_column], chunksize=64),
            total=len(subset), desc=f"Converting ({n_workers} workers)"
        ))
    return results


# Entry point

if __name__ == "__main__":
    df = load_uspto_splits(
        train_path="Data/train.csv",
        val_path="Data/val.csv",
        test_path="Data/test.csv",
    )
    df["rxn_smiles"] = df["precursors"] + ">>" + df["products"]
    print("\nColumn names:", df.columns.tolist())
    results = validate_dataset_parallel(df)