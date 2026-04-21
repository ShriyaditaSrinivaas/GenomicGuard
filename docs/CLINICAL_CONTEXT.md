# Clinical Context

## Background

Polygenic Risk Scores (PRS) aggregate the effects of many genetic variants across the genome to estimate an individual's genetic predisposition to a disease. While PRS has shown promise in research settings, translating these scores into clinical practice requires addressing several key challenges:

### The Interpretability Gap

Clinicians need more than a number. They need to understand:
- **Which genetic factors** contribute most to the risk score
- **Which clinical factors** amplify or attenuate genetic risk
- **How confident** the model is in its prediction
- **What actions** should be taken based on the results

GenomicGuard addresses this through SHAP-based explanations that decompose each prediction into individual feature contributions.

### The Equity Gap

PRS models have a well-documented transferability problem:
- Most GWAS (Genome-Wide Association Studies) have been conducted predominantly in European-ancestry populations
- PRS accuracy drops significantly when applied to non-European populations (Martin et al., 2019)
- This creates a risk of widening existing health disparities

GenomicGuard addresses this through built-in fairness auditing that detects and quantifies performance disparities across population groups.

## Diseases Modeled

### Type 2 Diabetes (T2D)
- Global prevalence: ~10.5% (537 million adults, IDF 2021)
- Known genetic component: >400 associated loci
- Key clinical risk factors: BMI, fasting glucose, HbA1c, family history, age

### Coronary Artery Disease (CAD)
- Leading cause of death globally
- Known genetic component: >160 associated loci
- Key clinical risk factors: Blood pressure, cholesterol, smoking, age, family history

### Breast Cancer
- Most common cancer in women worldwide
- Known genetic component: BRCA1/2 and polygenic risk
- Key clinical risk factors: Age, family history, BMI, physical activity

## Clinical Workflow Integration

GenomicGuard is designed to fit into a clinical genomics workflow:

1. **Sample Collection** → Genomic sequencing and clinical data collection
2. **Risk Assessment** → GenomicGuard computes calibrated risk scores
3. **Clinical Report** → Structured report with explanations and recommendations
4. **Clinical Review** → Healthcare provider reviews report with patient
5. **Action Plan** → Evidence-based screening and intervention decisions

## References

- Martin, A.R., et al. (2019). Clinical use of current polygenic risk scores may exacerbate health disparities. *Nature Genetics*, 51(4), 584-591.
- Sirugo, G., et al. (2019). The missing diversity in human genetic studies. *Cell*, 177(1), 26-31.
- Wand, H., et al. (2021). Improving reporting standards for polygenic scores in risk prediction studies. *Nature*, 591(7849), 211-219.
