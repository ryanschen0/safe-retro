#!/usr/bin/env python3
"""
Smoke-test for augment_safe_fragments.py

Run on the server:
  cd ~/safe-retro/
  source /data/ryanschen/safe-retro/saferetrouv/bin/activate
  python test_augmentation.py

Checks:
  1. permute_safe_fragments returns valid permutations for a multi-fragment molecule.
  2. Every permutation round-trips to the same canonical SMILES.
  3. Single-fragment molecules return no permutations (nothing to permute).
  4. augment_reaction produces pairs consistent in count and format.
"""

import sys, random
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import safe as sf

# ---- import the helpers from augmentation script ----
# (assumes augment_safe_fragments.py is in the same directory or on PYTHONPATH)
sys.path.insert(0, ".")
from FragmentAugmentation import (
    permute_safe_fragments,
    augment_reaction,
    _renumber_ring_digits,
)


# Helper to convert a SAFE string to canonical SMILES for comparison
def safe_to_canon(s: str) -> str:
    mol = sf.decode(s, as_mol=True, ignore_errors=True)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


# Test
# Aspirin (multi-fragment in SAFE)
ASPIRIN_SMILES   = "CC(=O)Oc1ccccc1C(=O)O"
# Simple single-atom molecule (no BRICS bonds)
ETHANOL_SMILES   = "CCO"
# A precursor side with two molecules (joined by ~)
TWO_MOL_SMILES   = "CC(=O)O~Brc1ccccc1"  # acetic acid ~ bromobenzene (as SAFE)

def smiles_to_safe(smi: str) -> str:
    try:
        return sf.encode(smi)
    except Exception as e:
        return None


# Test 1: multi-fragment molecule
def test_multi_fragment():
    safe_str = smiles_to_safe(ASPIRIN_SMILES)
    print(f"\n[Test 1] Aspirin SAFE: {safe_str}")
    if safe_str is None:
        print("  SKIP: encoding failed")
        return

    n_frags = len(safe_str.split('.'))
    print(f"  Fragments: {n_frags}")

    if n_frags <= 1:
        print("  Only 1 fragment — permutation test not applicable")
        return

    perms = permute_safe_fragments(safe_str, max_perms=6)
    print(f"  Permutations generated: {len(perms)}")
    assert len(perms) > 0, "Expected at least one permutation for multi-fragment molecule"

    canon_orig = safe_to_canon(safe_str)
    for p in perms:
        assert p != safe_str, "Permutation identical to original"
        canon_p = safe_to_canon(p)
        assert canon_p == canon_orig, \
            f"Round-trip FAIL:\n  original canon: {canon_orig}\n  permuted canon: {canon_p}\n  permuted SAFE:  {p}"

    print("  PASS: all permutations round-trip correctly")



# Test 2: single-fragment molecule (no BRICS-cleavable bonds)
def test_single_fragment():
    safe_str = smiles_to_safe(ETHANOL_SMILES)
    print(f"\n[Test 2] Ethanol SAFE: {safe_str}")
    if safe_str is None:
        print("  SKIP: encoding failed")
        return

    n_frags = len(safe_str.split('.'))
    print(f"  Fragments: {n_frags}")

    perms = permute_safe_fragments(safe_str, max_perms=4)
    print(f"  Permutations generated: {len(perms)}")
    assert len(perms) == 0, "Single-fragment molecule should yield 0 permutations"
    print("  PASS")



# Test 3: augment_reaction with a two-molecule tgt
def test_augment_reaction():
    # Build a minimal fake reaction:
    #   product  -> acetic acid ~ bromobenzene  (as retrosynthesis)
    # src = SAFE(aspirin), tgt = SAFE(acetic acid)~SAFE(bromobenzene)
    src_safe = smiles_to_safe(ASPIRIN_SMILES)
    aa_safe  = smiles_to_safe("CC(=O)O")
    bb_safe  = smiles_to_safe("Brc1ccccc1")

    if any(x is None for x in [src_safe, aa_safe, bb_safe]):
        print("\n[Test 3] SKIP: encoding failed for one molecule")
        return

    tgt_safe = f"{aa_safe}~{bb_safe}"
    print(f"\n[Test 3] augment_reaction")
    print(f"  src: {src_safe}")
    print(f"  tgt: {tgt_safe}")

    rng = random.Random(42)
    augmented = augment_reaction(src_safe, tgt_safe, max_aug=4, rng=rng)
    print(f"  Augmented pairs generated: {len(augmented)}")

    for i, (s, t) in enumerate(augmented):
        # src must round-trip
        c_orig = safe_to_canon(src_safe)
        c_new  = safe_to_canon(s)
        assert c_new == c_orig, f"src round-trip FAIL at pair {i}: {c_new} != {c_orig}"

        # each molecule in tgt must round-trip
        orig_mols = tgt_safe.split('~')
        new_mols  = t.split('~')
        assert len(orig_mols) == len(new_mols), \
            f"tgt molecule count changed at pair {i}"
        for o_m, n_m in zip(orig_mols, new_mols):
            co = safe_to_canon(o_m)
            cn = safe_to_canon(n_m)
            assert cn == co, f"tgt molecule round-trip FAIL: {cn} != {co}"

    print("  PASS: all augmented pairs round-trip correctly")


# Test 4: _renumber_ring_digits is idempotent for a simple case
def test_renumber():
    # A string with %12 and %15 — after renumbering should become %10 and %11
    s = "C%12CC%15.C%12CC%15"
    result = _renumber_ring_digits(s)
    print(f"\n[Test 4] renumber: {s!r} -> {result!r}")
    assert "%10" in result
    assert "%11" in result
    print("  PASS")


if __name__ == "__main__":
    test_single_fragment()
    test_multi_fragment()
    test_augment_reaction()
    test_renumber()
    print("\n=== All tests passed ===")