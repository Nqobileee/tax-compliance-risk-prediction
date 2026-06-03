# TaxGuard documentation

This folder holds reference papers and the project’s **technical methodology document**, written in the same scholarly style as the example credit-scoring papers.

| Document | Description |
|----------|-------------|
| [TaxGuard_Corporate_Tax_Risk_Scoring_Methodology.md](TaxGuard_Corporate_Tax_Risk_Scoring_Methodology.md) | Full methods paper: definitions, formulas, algorithms, evaluation metrics, results tables, and references |
| `Credit+Scoring+Using+Machine+Learning+Algorithims.pdf` | Example (local only, gitignored) — Nyoni & Matshisela (2018) |
| `Optimisation_of_the_Linear_Probability_M.pdf` | Example (local only, gitignored) — Nyathi et al. (2014) |

**Related repository artefacts**

- Executive report: [../TAXGUARD_RESEARCH_SUMMARY.md](../TAXGUARD_RESEARCH_SUMMARY.md)
- Figures: [../notebooks/figures/README.md](../notebooks/figures/README.md)
- Metrics JSON: [../outputs/metrics/experiment_summary.json](../outputs/metrics/experiment_summary.json)
- Feature code: [../src/taxguard_features.py](../src/taxguard_features.py)

To regenerate metrics and figures:

```bash
python scripts/generate_report_assets.py
```

### PDF export (methodology paper)

```bash
pip install pypandoc_binary playwright
playwright install chromium
python scripts/build_methodology_pdf.py
```

Output: `docs/TaxGuard_Corporate_Tax_Risk_Scoring_Methodology.pdf` (KaTeX-rendered equations).
