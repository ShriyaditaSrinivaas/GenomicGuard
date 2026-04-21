#!/usr/bin/env python3
"""
Generate synthetic genomic dataset for GenomicGuard.

Usage:
    python scripts/generate_data.py [--n-samples 2000] [--seed 42]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genomicguard.config import DataConfig, DATA_DIR
from genomicguard.data.generator import GenomicDataGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic genomic data")
    parser.add_argument("--n-samples", type=int, default=2000, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    print("=" * 60)
    print("  GenomicGuard - Synthetic Data Generation")
    print("=" * 60)

    config = DataConfig(n_samples=args.n_samples, random_seed=args.seed)
    output_dir = Path(args.output_dir) if args.output_dir else DATA_DIR

    print(f"\n  Configuration:")
    print(f"    Samples:     {config.n_samples}")
    print(f"    SNPs:        {config.n_snps}")
    print(f"    Populations: {list(config.population_weights.keys())}")
    print(f"    Diseases:    {list(config.disease_prevalence.keys())}")
    print(f"    Seed:        {config.random_seed}")
    print(f"    Output:      {output_dir}")

    print(f"\n  Generating dataset...")
    generator = GenomicDataGenerator(config)
    dataset = generator.generate_full_dataset()

    print(f"\n  Saving to disk...")
    generator.save_dataset(dataset, output_dir)

    # Summary statistics
    print(f"\n  Dataset Summary:")
    for name, df in dataset.items():
        print(f"    {name}: {df.shape[0]} rows × {df.shape[1]} cols")

    # Population distribution
    pop_counts = dataset["sample_metadata"]["population"].value_counts()
    print(f"\n  Population Distribution:")
    for pop, count in pop_counts.items():
        print(f"    {pop}: {count} ({count/len(dataset['sample_metadata'])*100:.1f}%)")

    # Disease prevalence
    phenotypes = dataset["phenotypes"]
    print(f"\n  Disease Prevalence:")
    for col in phenotypes.columns:
        if col.endswith("_label"):
            disease = col.replace("_label", "").replace("_", " ").title()
            prev = phenotypes[col].mean()
            print(f"    {disease}: {prev:.1%}")

    print(f"\n✓ Done! Dataset saved to {output_dir}")


if __name__ == "__main__":
    main()
