import argparse
import random
import re
import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path
 
try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("RDKit not found. Activate your environment: source /data/ryanschen/safe-retro/saferetrouv/bin/activate")

try:
    import safe as sf
except ImportError:
    sys.exit("safe-mol not found. Install with: uv pip install safe-mol")
 
#SAFE Fragment Utilities

def _renumber_ring_digits(safe_str: str) -> str:
    # Re-number all ring-closure digits in a SAFE string so they start from %10. Needed after fragment permutation because the original ring digits may collide or leave gaps.
    # Find all %NN tokens in order of appearance
    pattern = re.compile(r'%(\d{2})')
    seen = {}     # Old Digit -> New Digit
    counter = [10]
 
    def replacer(m):
        old = m.group(1)
        if old not in seen:
            seen[old] = f"{counter[0]:02d}"
            counter[0] += 1
        return f"%{seen[old]}"
 
    return pattern.sub(replacer, safe_str)


def permute_safe_fragments(safe_str: str, max_perms: int = 4, rng: random.Random = None) -> list[str]:
    # Given a SAFE string, return up to max_perms fragment-permuted variants that round-trip to the same canonical SMILES, excluding the original order.
    # Returns an empty list if the molecule has only one fragment (nothing to permute) or if round-trip validation fails for all permutations
    if rng is None:
        rng = random.Random()
 
    # Get canonical SMILES of the original via safe decode
    try:
        mol = sf.decode(safe_str, as_mol=True, ignore_errors=True)
        if mol is None:
            return []
        canon_smi = Chem.MolToSmiles(mol)
    except Exception:
        return []
 
    # Split on '.' to get fragments
    # Each fragment is a SMILES token-string; the '.' is purely a separator
    fragments = safe_str.split('.')
    if len(fragments) <= 1:
        return []   # nothing to permute
 
    # Generate all permutations (or a random subset for large fragment counts)
    all_perms = list(permutations(range(len(fragments))))
    # Remove identity permutation
    identity = tuple(range(len(fragments)))
    all_perms = [p for p in all_perms if p != identity]
 
    if len(all_perms) == 0:
        return []
 
    # For large numbers of fragments, sample randomly
    if len(all_perms) > max_perms * 10:
        rng.shuffle(all_perms)
        all_perms = all_perms[: max_perms * 10]
 
    results = []
    seen_strings = {safe_str}
 
    for perm in all_perms:
        if len(results) >= max_perms:
            break
 
        permuted_frags = [fragments[i] for i in perm]
        candidate = '.'.join(permuted_frags)
 
        # Re-number ring digits so they are consistent after reordering
        candidate = _renumber_ring_digits(candidate)
 
        if candidate in seen_strings:
            continue
 
        # Round-trip validation
        try:
            mol_c = sf.decode(candidate, as_mol=True, ignore_errors=True)
            if mol_c is None:
                continue
            canon_c = Chem.MolToSmiles(mol_c)
            if canon_c != canon_smi:
                continue
        except Exception:
            continue
 
        seen_strings.add(candidate)
        results.append(candidate)
 
    return results

# Augmentation at the Reactional Level

def _molecules_from_safe_side(side: str) -> list[str]:
    # Rxn side SAFE string to individual molecule SAFE strings. 
    return side.split('~')
 
 
def _safe_side_from_molecules(molecules: list[str]) -> str:
    return '~'.join(molecules)
 
 
def augment_reaction(src_line: str, tgt_line: str,
                     max_aug: int, rng: random.Random,
                     augment_src: bool = True,
                     augment_tgt: bool = True) -> list[tuple[str, str]]:
    """
    Produce up to `max_aug` augmented (src, tgt) pairs for a single reaction.
 
    Augmentation strategy:
      - For the src (product): permute its fragments.
      - For the tgt (precursors): permute fragments of each molecule independently,
        then combine. We keep the molecule ORDER in tgt fixed (permuting molecule
        order would change reaction meaning) and only permute intra-molecule
        fragment order.
 
    Returns a list of (src, tgt) tuples (NOT including the original).
    """
    src_safe = src_line.strip()
    tgt_safe = tgt_line.strip()
 
    # Permutations of SRC 
    src_variants = [src_safe]
    if augment_src:
        src_variants += permute_safe_fragments(src_safe, max_perms=max_aug, rng=rng)
        # Keep at most max_aug+1 total (1 original + max_aug)
        src_variants = src_variants[: max_aug + 1]
 
    # Permutations of TGT (per-molecule)
    tgt_molecules = _molecules_from_safe_side(tgt_safe)
    # For each molecule, gather its permutations (or keep original)
    mol_variant_lists = []
    for mol_safe in tgt_molecules:
        mol_perms = [mol_safe] + permute_safe_fragments(
            mol_safe, max_perms=max_aug, rng=rng)
        mol_variant_lists.append(mol_perms)
 
    # Build tgt variants by picking one permutation per molecule.
    # Simple strategy: use the i-th permutation for each molecule (if available).
    tgt_variants = [tgt_safe]
    for i in range(1, max_aug + 1):
        new_mols = []
        changed = False
        for mol_variants in mol_variant_lists:
            idx = min(i, len(mol_variants) - 1)
            new_mols.append(mol_variants[idx])
            if mol_variants[idx] != mol_variants[0]:
                changed = True
        if changed:
            tgt_variants.append(_safe_side_from_molecules(new_mols))
 
    tgt_variants = tgt_variants[: max_aug + 1]
 
    # Combine: pair each src variant with each tgt variant
    augmented = []
    seen = {(src_safe, tgt_safe)}
    for s in src_variants:
        for t in tgt_variants:
            pair = (s, t)
            if pair not in seen:
                seen.add(pair)
                augmented.append(pair)
            if len(augmented) >= max_aug:
                return augmented
 
    return augmented

# Command Line Interface for Fragment Augmentation
 
def parse_args():
    p = argparse.ArgumentParser(
        description="Fragment-permutation augmentation for SAFE retrosynthesis data")
    p.add_argument("--src_train", required=True,
                   help="Path to src-train.txt (product SAFE, one per line)")
    p.add_argument("--tgt_train", required=True,
                   help="Path to tgt-train.txt (precursor SAFE, one per line)")
    p.add_argument("--out_dir", required=True,
                   help="Output directory for augmented files")
    p.add_argument("--max_aug", type=int, default=4,
                   help="Max augmented copies per training example (default: 4)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--no_augment_src", action="store_true",
                   help="Do not permute the src (product) side")
    p.add_argument("--no_augment_tgt", action="store_true",
                   help="Do not permute the tgt (precursors) side")
    return p.parse_args()
 
 
def main():
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    src_lines = Path(args.src_train).read_text().splitlines()
    tgt_lines = Path(args.tgt_train).read_text().splitlines()
 
    if len(src_lines) != len(tgt_lines):
        sys.exit(f"Line count mismatch: src={len(src_lines)}, tgt={len(tgt_lines)}")
 
    n_orig = len(src_lines)
    aug_src, aug_tgt = list(src_lines), list(tgt_lines)
 
    stats = defaultdict(int)
    stats["total_original"] = n_orig
 
    print(f"Augmenting {n_orig} training examples (max_aug={args.max_aug}) …")
 
    for i, (s, t) in enumerate(zip(src_lines, tgt_lines)):
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{n_orig} processed, {len(aug_src) - n_orig} augmented so far")
 
        new_pairs = augment_reaction(
            s, t,
            max_aug=args.max_aug,
            rng=rng,
            augment_src=not args.no_augment_src,
            augment_tgt=not args.no_augment_tgt,
        )
        for ns, nt in new_pairs:
            aug_src.append(ns)
            aug_tgt.append(nt)
            stats["total_augmented"] += 1
 
        if new_pairs:
            stats["reactions_augmented"] += 1
        else:
            stats["reactions_not_augmented"] += 1
 
    # Write outputs
    out_src = out_dir / "src-train-aug.txt"
    out_tgt = out_dir / "tgt-train-aug.txt"
    out_src.write_text('\n'.join(aug_src) + '\n')
    out_tgt.write_text('\n'.join(aug_tgt) + '\n')
 
    # Report
    report_lines = [
        "=== Fragment Augmentation Report ===",
        f"Original examples    : {stats['total_original']}",
        f"Reactions augmented  : {stats['reactions_augmented']}",
        f"Reactions unchanged  : {stats['reactions_not_augmented']}",
        f"New augmented lines  : {stats['total_augmented']}",
        f"Total output lines   : {len(aug_src)}",
        f"Expansion factor     : {len(aug_src) / max(n_orig,1):.2f}x",
        "",
        f"Output src : {out_src}",
        f"Output tgt : {out_tgt}",
    ]
    report_text = '\n'.join(report_lines)
    print('\n' + report_text)
    (out_dir / "augmentation_report.txt").write_text(report_text + '\n')
    print("\nDone.")
 
 
if __name__ == "__main__":
    main()
 