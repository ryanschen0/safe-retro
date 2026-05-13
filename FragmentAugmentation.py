#!/usr/bin/env python3
"""
Fragment-permutation augmentation for SAFE retrosynthesis training data.
Single-threaded, streaming — no multiprocessing, no RAM buildup.
"""

import argparse
import random
import re
import sys
from itertools import permutations
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import safe as sf

SMI_REGEX_PATTERN = (
    r'(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p'
    r'|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])'
)
_REGEX = re.compile(SMI_REGEX_PATTERN)

def tokenize(s):
    return ' '.join(_REGEX.findall(s))

def detokenize(s):
    return s.replace(' ', '')

def _renumber_ring_digits(safe_str):
    pattern = re.compile(r'%(\d{2})')
    seen = {}
    counter = [10]
    def replacer(m):
        old = m.group(1)
        if old not in seen:
            seen[old] = f"{counter[0]:02d}"
            counter[0] += 1
        return f"%{seen[old]}"
    return pattern.sub(replacer, safe_str)

def permute_safe_fragments(safe_str, max_perms=4, rng=None):
    if rng is None:
        rng = random.Random()
    try:
        mol = sf.decode(safe_str, as_mol=True, ignore_errors=True)
        if mol is None:
            return []
        canon_smi = Chem.MolToSmiles(mol)
    except Exception:
        return []

    fragments = safe_str.split('.')
    if len(fragments) <= 1:
        return []

    all_perms = list(permutations(range(len(fragments))))
    identity = tuple(range(len(fragments)))
    all_perms = [p for p in all_perms if p != identity]
    if not all_perms:
        return []

    if len(all_perms) > max_perms * 10:
        rng.shuffle(all_perms)
        all_perms = all_perms[:max_perms * 10]

    results = []
    seen_strings = {safe_str}
    for perm in all_perms:
        if len(results) >= max_perms:
            break
        candidate = _renumber_ring_digits('.'.join(fragments[i] for i in perm))
        if candidate in seen_strings:
            continue
        try:
            mol_c = sf.decode(candidate, as_mol=True, ignore_errors=True)
            if mol_c is None:
                continue
            if Chem.MolToSmiles(mol_c) != canon_smi:
                continue
        except Exception:
            continue
        seen_strings.add(candidate)
        results.append(candidate)
    return results

def process_reaction(tok_src, tok_tgt, max_aug, rng):
    raw_tgt = detokenize(tok_tgt)
    tgt_molecules = raw_tgt.split('~')

    mol_variant_lists = []
    for mol_safe in tgt_molecules:
        variants = [mol_safe] + permute_safe_fragments(mol_safe, max_perms=max_aug, rng=rng)
        mol_variant_lists.append(variants)

    seen_tgts = {raw_tgt}
    new_tgts = []
    for i in range(1, max_aug + 1):
        new_mols = []
        changed = False
        for mol_variants in mol_variant_lists:
            idx = min(i, len(mol_variants) - 1)
            new_mols.append(mol_variants[idx])
            if mol_variants[idx] != mol_variants[0]:
                changed = True
        if changed:
            new_tgt = '~'.join(new_mols)
            if new_tgt not in seen_tgts:
                seen_tgts.add(new_tgt)
                new_tgts.append(new_tgt)
        if len(new_tgts) >= max_aug:
            break

    pairs = [(tok_src, tok_tgt)]
    for new_tgt in new_tgts:
        pairs.append((tok_src, tokenize(new_tgt)))
    return pairs

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src_train", required=True)
    p.add_argument("--tgt_train", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_aug", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_src = out_dir / "src-train-aug.txt"
    out_tgt = out_dir / "tgt-train-aug.txt"

    print(f"Starting augmentation (max_aug={args.max_aug}, single-threaded) …")
    sys.stdout.flush()

    total_written = 0
    total_augmented = 0
    i = 0

    with open(args.src_train) as f_src, \
         open(args.tgt_train) as f_tgt, \
         open(out_src, 'w') as o_src, \
         open(out_tgt, 'w') as o_tgt:

        for s, t in zip(f_src, f_tgt):
            s, t = s.rstrip('\n'), t.rstrip('\n')
            pairs = process_reaction(s, t, args.max_aug, rng)
            for ps, pt in pairs:
                o_src.write(ps + '\n')
                o_tgt.write(pt + '\n')
                total_written += 1
            total_augmented += len(pairs) - 1
            i += 1
            if i % 5000 == 0:
                print(f"  {i} processed, {total_augmented} augmented so far")
                sys.stdout.flush()

    print(f"\n=== Augmentation Report ===")
    print(f"Total input   : {i}")
    print(f"Augmented     : {total_augmented}")
    print(f"Total output  : {total_written}")
    print(f"Expansion     : {total_written / max(i, 1):.2f}x")
    (out_dir / "augmentation_report.txt").write_text(
        f"Total input   : {i}\nAugmented     : {total_augmented}\n"
        f"Total output  : {total_written}\nExpansion     : {total_written / max(i, 1):.2f}x\n"
    )
    print("Done.")

if __name__ == "__main__":
    main()