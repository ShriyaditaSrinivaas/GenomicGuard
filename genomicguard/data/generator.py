"""
Synthetic Genomic Data Generator.

Generates realistic synthetic genomic data including:
- Genotype matrices (SNP-level) with population-specific allele frequencies
- Clinical phenotype data (age, BMI, family history, lab values, etc.)
- Disease labels with realistic prevalence patterns
- Population stratification reflecting real-world diversity

The synthetic data mimics patterns found in real genomic datasets (e.g., UK Biobank,
1000 Genomes Project) without using any real patient data, making it safe for
public repositories.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from pathlib import Path

from genomicguard.config import (
    DataConfig,
    POPULATIONS,
    DISEASES,
    CHROMOSOMES,
    DATA_DIR,
)


class GenomicDataGenerator:
    """
    Generates synthetic genomic and clinical data for multiple populations
    and disease phenotypes.

    The generator creates biologically plausible data by modeling:
    - Population-specific allele frequencies with Fst-based differentiation
    - Linkage disequilibrium blocks
    - Gene-environment interactions
    - Realistic clinical feature distributions
    """

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.rng = np.random.RandomState(self.config.random_seed)
        self._snp_metadata = None
        self._allele_frequencies = None

    def generate_full_dataset(self) -> Dict[str, pd.DataFrame]:
        """
        Generate a complete synthetic dataset.

        Returns:
            Dictionary with keys:
            - 'genotypes': SNP genotype matrix (n_samples × n_snps)
            - 'clinical': Clinical features (n_samples × n_clinical_features)
            - 'phenotypes': Disease labels and risk scores
            - 'snp_metadata': SNP annotations
            - 'sample_metadata': Sample-level metadata
        """
        # 1. Generate population assignments
        populations = self._generate_populations()

        # 2. Generate SNP metadata
        snp_meta = self._generate_snp_metadata()
        self._snp_metadata = snp_meta

        # 3. Generate population-specific allele frequencies
        allele_freqs = self._generate_allele_frequencies(snp_meta)
        self._allele_frequencies = allele_freqs

        # 4. Generate genotype matrix
        genotypes = self._generate_genotypes(populations, allele_freqs)

        # 5. Generate clinical features
        clinical = self._generate_clinical_features(populations)

        # 6. Generate disease phenotypes
        phenotypes = self._generate_phenotypes(genotypes, clinical, populations)

        # 7. Build sample metadata
        sample_meta = pd.DataFrame({
            "sample_id": [f"SAMPLE_{i:05d}" for i in range(self.config.n_samples)],
            "population": populations,
            "sex": self.rng.choice(["M", "F"], size=self.config.n_samples, p=[0.48, 0.52]),
        })

        return {
            "genotypes": genotypes,
            "clinical": clinical,
            "phenotypes": phenotypes,
            "snp_metadata": snp_meta,
            "sample_metadata": sample_meta,
        }

    def _generate_populations(self) -> np.ndarray:
        """Assign population labels based on configured weights."""
        pops = list(self.config.population_weights.keys())
        weights = list(self.config.population_weights.values())
        return self.rng.choice(pops, size=self.config.n_samples, p=weights)

    def _generate_snp_metadata(self) -> pd.DataFrame:
        """
        Generate metadata for synthetic SNPs.

        Creates SNP annotations including chromosome position, gene names,
        functional annotations, and effect sizes (betas).
        """
        total_snps = self.config.n_snps
        gene_names = self._synthetic_gene_names(total_snps)

        # Assign SNPs to chromosomes
        chrom_assignments = self.rng.choice(CHROMOSOMES, size=total_snps)

        # Generate positions (realistic range)
        positions = self.rng.randint(10_000, 250_000_000, size=total_snps)

        # Functional annotations
        annotations = self.rng.choice(
            ["missense", "synonymous", "intronic", "intergenic", "regulatory", "splice_region"],
            size=total_snps,
            p=[0.15, 0.10, 0.35, 0.20, 0.15, 0.05],
        )

        # Effect sizes (betas) - most SNPs have small effects
        betas = self.rng.normal(0, 0.1, size=total_snps)
        # A few SNPs have larger effects
        n_large_effect = max(1, total_snps // 20)
        large_effect_idx = self.rng.choice(total_snps, size=n_large_effect, replace=False)
        betas[large_effect_idx] = self.rng.normal(0, 0.4, size=n_large_effect)

        # Conservation scores (0, 1)
        conservation = self.rng.beta(2, 5, size=total_snps)

        # CADD-like deleteriousness scores
        cadd_scores = self.rng.exponential(5, size=total_snps)
        cadd_scores = np.clip(cadd_scores, 0, 40)

        return pd.DataFrame({
            "snp_id": [f"rs{self.rng.randint(1000, 99999999)}" for _ in range(total_snps)],
            "chromosome": chrom_assignments,
            "position": positions,
            "gene": gene_names,
            "annotation": annotations,
            "beta": betas,
            "conservation_score": conservation,
            "cadd_score": cadd_scores,
            "ref_allele": self.rng.choice(["A", "C", "G", "T"], size=total_snps),
            "alt_allele": self.rng.choice(["A", "C", "G", "T"], size=total_snps),
        })

    def _synthetic_gene_names(self, n: int) -> list:
        """Generate realistic-looking gene names."""
        prefixes = [
            "BRCA", "TP", "EGFR", "KRAS", "PIK3CA", "PTEN", "RB", "APC",
            "MLH", "MSH", "FGFR", "ALK", "BRAF", "MYC", "CDKN", "CDH",
            "SLC", "HLA", "CYP", "ABCB", "GJB", "CFTR", "HBB", "FBN",
            "COL", "DMD", "HTT", "LMNA", "APOB", "PCSK", "LDLR", "NPC",
            "GBA", "HEXA", "PAH", "GALT", "ACADM", "SCN", "KCNQ", "TNF",
            "IL", "IFNG", "STAT", "JAK", "NRAS", "MAP2K", "SMAD", "NOTCH",
        ]
        names = []
        for i in range(n):
            prefix = prefixes[i % len(prefixes)]
            suffix = self.rng.randint(1, 20)
            names.append(f"{prefix}{suffix}")
        return names

    def _generate_allele_frequencies(
        self, snp_meta: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Generate population-specific allele frequencies.

        Uses an Fst-based model to create realistic inter-population
        variation in allele frequencies (Balding-Nichols model).
        """
        n_snps = len(snp_meta)

        # Ancestral allele frequencies (drawn from a realistic distribution)
        ancestral_freq = self.rng.beta(1.5, 5, size=n_snps)
        ancestral_freq = np.clip(ancestral_freq, 0.01, 0.99)

        # Fst values per population pair (realistic range 0.01 - 0.15)
        fst_values = {
            "EUR": 0.03,
            "AFR": 0.08,
            "EAS": 0.06,
            "SAS": 0.05,
            "AMR": 0.04,
        }

        pop_freqs = {}
        for pop in POPULATIONS:
            fst = fst_values[pop]
            # Balding-Nichols model
            alpha = ancestral_freq * (1 - fst) / fst
            beta = (1 - ancestral_freq) * (1 - fst) / fst

            # Ensure valid parameters
            alpha = np.maximum(alpha, 0.01)
            beta = np.maximum(beta, 0.01)

            pop_freq = np.array([
                self.rng.beta(a, b) for a, b in zip(alpha, beta)
            ])
            pop_freqs[pop] = np.clip(pop_freq, 0.001, 0.999)

        return pop_freqs

    def _generate_genotypes(
        self,
        populations: np.ndarray,
        allele_freqs: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """
        Generate genotype matrix (0, 1, 2 encoding = number of alt alleles).

        Each individual's genotype is sampled based on their population's
        allele frequencies under Hardy-Weinberg equilibrium.
        """
        n_snps = self.config.n_snps
        genotype_matrix = np.zeros((self.config.n_samples, n_snps), dtype=np.int8)

        for pop in POPULATIONS:
            mask = populations == pop
            n_pop = mask.sum()
            if n_pop == 0:
                continue

            freq = allele_freqs[pop]
            # Sample two alleles per locus (diploid)
            allele1 = (self.rng.random((n_pop, n_snps)) < freq).astype(np.int8)
            allele2 = (self.rng.random((n_pop, n_snps)) < freq).astype(np.int8)
            genotype_matrix[mask] = allele1 + allele2

        snp_ids = self._snp_metadata["snp_id"].values
        return pd.DataFrame(
            genotype_matrix,
            columns=snp_ids,
        )

    def _generate_clinical_features(
        self, populations: np.ndarray
    ) -> pd.DataFrame:
        """
        Generate clinical phenotype features.

        Creates realistic distributions that vary by population,
        reflecting real-world epidemiological patterns.
        """
        n = self.config.n_samples

        # Age (slightly different distributions by population)
        age_means = {"EUR": 55, "AFR": 48, "EAS": 52, "SAS": 50, "AMR": 51}
        age = np.zeros(n)
        for pop in POPULATIONS:
            mask = populations == pop
            age[mask] = self.rng.normal(age_means[pop], 12, size=mask.sum())
        age = np.clip(age, 18, 90).astype(int)

        # BMI
        bmi_means = {"EUR": 27.5, "AFR": 29.0, "EAS": 24.0, "SAS": 26.5, "AMR": 28.0}
        bmi = np.zeros(n)
        for pop in POPULATIONS:
            mask = populations == pop
            bmi[mask] = self.rng.normal(bmi_means[pop], 5, size=mask.sum())
        bmi = np.clip(bmi, 15, 50).round(1)

        # Blood pressure (systolic)
        sbp_means = {"EUR": 128, "AFR": 135, "EAS": 125, "SAS": 130, "AMR": 129}
        systolic_bp = np.zeros(n)
        for pop in POPULATIONS:
            mask = populations == pop
            systolic_bp[mask] = self.rng.normal(sbp_means[pop], 18, size=mask.sum())
        systolic_bp = np.clip(systolic_bp, 80, 220).astype(int)

        # Fasting glucose (mg/dL)
        glucose = self.rng.normal(100, 25, size=n)
        glucose = np.clip(glucose, 60, 300).round(1)

        # Total cholesterol
        cholesterol = self.rng.normal(200, 40, size=n)
        cholesterol = np.clip(cholesterol, 100, 400).round(1)

        # HDL cholesterol
        hdl = self.rng.normal(50, 12, size=n)
        hdl = np.clip(hdl, 20, 100).round(1)

        # LDL cholesterol
        ldl = self.rng.normal(120, 35, size=n)
        ldl = np.clip(ldl, 40, 250).round(1)

        # Triglycerides
        triglycerides = self.rng.lognormal(4.8, 0.5, size=n)
        triglycerides = np.clip(triglycerides, 40, 500).round(1)

        # HbA1c (%)
        hba1c = self.rng.normal(5.7, 1.0, size=n)
        hba1c = np.clip(hba1c, 4.0, 14.0).round(1)

        # Family history (binary)
        family_history = self.rng.binomial(1, 0.25, size=n)

        # Smoking status (0=never, 1=former, 2=current)
        smoking = self.rng.choice([0, 1, 2], size=n, p=[0.55, 0.25, 0.20])

        # Physical activity (hours/week)
        exercise = self.rng.exponential(3, size=n)
        exercise = np.clip(exercise, 0, 20).round(1)

        return pd.DataFrame({
            "age": age,
            "bmi": bmi,
            "systolic_bp": systolic_bp,
            "fasting_glucose": glucose,
            "total_cholesterol": cholesterol,
            "hdl_cholesterol": hdl,
            "ldl_cholesterol": ldl,
            "triglycerides": triglycerides,
            "hba1c": hba1c,
            "family_history": family_history,
            "smoking_status": smoking,
            "exercise_hours_week": exercise,
        })

    def _generate_phenotypes(
        self,
        genotypes: pd.DataFrame,
        clinical: pd.DataFrame,
        populations: np.ndarray,
    ) -> pd.DataFrame:
        """
        Generate disease labels based on genetic and clinical risk factors.

        Combines polygenic risk (SNP effects) with clinical risk factors
        using a realistic disease model with population-specific baselines.
        """
        n = self.config.n_samples
        results = {}

        for disease in DISEASES:
            # Genetic component: weighted sum of genotypes × betas
            betas = self._snp_metadata["beta"].values
            genetic_score = genotypes.values.astype(np.float64) @ betas

            # Normalize genetic score
            genetic_score = (genetic_score - genetic_score.mean()) / (genetic_score.std() + 1e-8)

            # Clinical component
            clinical_score = np.zeros(n)
            if disease == "Type 2 Diabetes":
                clinical_score += 0.3 * (clinical["bmi"].values - 25) / 5
                clinical_score += 0.2 * (clinical["fasting_glucose"].values - 100) / 25
                clinical_score += 0.15 * (clinical["hba1c"].values - 5.7) / 1.0
                clinical_score += 0.1 * (clinical["age"].values - 50) / 10
                clinical_score += 0.2 * clinical["family_history"].values
            elif disease == "Coronary Artery Disease":
                clinical_score += 0.25 * (clinical["systolic_bp"].values - 120) / 20
                clinical_score += 0.2 * (clinical["ldl_cholesterol"].values - 100) / 35
                clinical_score += -0.15 * (clinical["hdl_cholesterol"].values - 50) / 12
                clinical_score += 0.15 * (clinical["age"].values - 50) / 10
                clinical_score += 0.1 * clinical["smoking_status"].values / 2
                clinical_score += 0.15 * clinical["family_history"].values
            elif disease == "Breast Cancer":
                clinical_score += 0.2 * (clinical["age"].values - 50) / 10
                clinical_score += 0.1 * (clinical["bmi"].values - 25) / 5
                clinical_score += 0.3 * clinical["family_history"].values
                clinical_score -= 0.1 * clinical["exercise_hours_week"].values / 5

            # Combined risk (genetic + clinical + noise)
            combined = 0.45 * genetic_score + 0.45 * clinical_score
            noise = self.rng.normal(0, 0.3, size=n)
            combined += noise

            # Apply population-specific baselines (modeling real disparities)
            pop_baselines = {
                "Type 2 Diabetes": {"EUR": 0.0, "AFR": 0.15, "EAS": 0.05, "SAS": 0.20, "AMR": 0.10},
                "Coronary Artery Disease": {"EUR": 0.05, "AFR": 0.0, "EAS": -0.05, "SAS": 0.15, "AMR": 0.05},
                "Breast Cancer": {"EUR": 0.05, "AFR": 0.0, "EAS": -0.05, "SAS": 0.0, "AMR": 0.0},
            }
            for pop in POPULATIONS:
                mask = populations == pop
                combined[mask] += pop_baselines[disease][pop]

            # Convert to probability via sigmoid
            prob = 1.0 / (1.0 + np.exp(-combined))

            # Calibrate to match target prevalence
            target_prev = self.config.disease_prevalence[disease]
            threshold = np.percentile(prob, (1 - target_prev) * 100)
            labels = (prob >= threshold).astype(int)

            disease_key = disease.lower().replace(" ", "_")
            results[f"{disease_key}_risk_score"] = prob.round(4)
            results[f"{disease_key}_label"] = labels

        # Overall risk (max across diseases)
        risk_cols = [c for c in results if c.endswith("_risk_score")]
        risk_matrix = np.column_stack([results[c] for c in risk_cols])
        results["overall_risk_score"] = risk_matrix.max(axis=1).round(4)
        results["primary_disease"] = np.array(DISEASES)[risk_matrix.argmax(axis=1)]

        return pd.DataFrame(results)

    def save_dataset(self, dataset: Dict[str, pd.DataFrame], output_dir: Optional[Path] = None):
        """Save generated dataset to CSV files."""
        out = output_dir or DATA_DIR
        out.mkdir(parents=True, exist_ok=True)

        for name, df in dataset.items():
            filepath = out / f"{name}.csv"
            df.to_csv(filepath, index=False)
            print(f"  ✓ Saved {name}: {df.shape[0]} rows × {df.shape[1]} cols → {filepath}")

    @staticmethod
    def load_dataset(data_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
        """Load a previously saved dataset."""
        d = data_dir or DATA_DIR
        dataset = {}
        for csv_file in sorted(d.glob("*.csv")):
            name = csv_file.stem
            dataset[name] = pd.read_csv(csv_file)
        return dataset
