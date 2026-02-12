# Predicting Wildfire Size in the United States

**UC Berkeley MIDS - DATASCI 207: Applied Machine Learning (Fall 2025)**

Machine learning models for multi-class wildfire size prediction using 2.3 million U.S. wildfire incidents from the FPA FOD-Attributes dataset.

## Abstract

Wildfires pose escalating risks to U.S. communities, with climate change driving increasingly severe fire seasons. We developed machine learning models to predict wildfire size categories (small, medium, large, very large) using the FPA FOD-Attributes dataset containing 2.3 million incidents and 308 environmental, meteorological, and social features. Our primary challenge was extreme class imbalance: 97% of fires remain under 100 acres. We evaluated eight preprocessing pipelines combining imputation strategies, imbalance handling (undersampling, SMOTENC, class weights), and location representations. We compared Random Forest (baseline), XGBoost, Feed-Forward Neural Networks, Logistic Regression, and Entity Embeddings. **The FFNN with class weights achieved the best macro F1 score (0.49 validation, 0.47 test)**, demonstrating balanced performance across all fire size categories.

## Problem

| Fire Size | Acres | Samples | Percent |
|-----------|-------|---------|---------|
| Small | 0-100 | 2,241,807 | 97.36% |
| Medium | 100-4,999 | 55,930 | 2.43% |
| Large | 5,000-29,000 | 3,682 | 0.16% |
| Very Large | 29,000+ | 1,102 | 0.05% |

The extreme class imbalance (97% small fires) made this a challenging multi-class classification problem requiring careful handling of minority classes.

## Dataset

**FPA FOD-Attributes Dataset** ([Pourmohamad et al., 2023](https://doi.org/10.5194/essd-16-3045-2024))
- 2,302,521 U.S. wildfire incidents (1992-2020)
- 308 features: physical (weather, climate), biological (vegetation), social (population, vulnerability), administrative (preparedness, fire stations)
- Stratified 60/20/20 train/validation/test splits by fire year and size label

## Models & Results

| Model | Val Accuracy | Val Macro F1 | Test Macro F1 |
|-------|--------------|--------------|---------------|
| Logistic Regression + Class Weights | 49.6% | 0.33 | 0.33 |
| Random Forest + Class Weights | 52.0% | 0.52 | 0.50 |
| XGBoost + Class Weights | 52.5% | 0.53 | 0.51 |
| **FFNN + Class Weights** | **51.1%** | **0.49** | **0.47** |
| Entity Embeddings + Class Weights | 51.1% | 0.51 | 0.49 |

Class weights substantially outperformed SMOTENC oversampling for handling imbalance.

## Repository Structure

```
.
├── data/
│   ├── raw/           # Original FPA FOD CSV files
│   ├── interim/       # Cleaned parquet files
│   └── processed/     # Train/val/test splits
├── notebooks/         # Jupyter notebooks for EDA and modeling
├── reports/           # Final project report
├── src/               # Python modules
│   ├── data_loader.py           # Memory-efficient data loading with Polars
│   ├── preprocessing_pipeline.py # End-to-end preprocessing
│   ├── train_test_splitter.py   # Stratified splitting utilities
│   └── dataviz_builders.py      # Visualization functions
└── requirements.txt
```

**Note:** Each team member's individual code contributions can be found in their respective development branches (e.g., `colton-dev`, `leo-dev`, `nedim-dev`, `shanti-dev`).

## Team Contributions

- **Colton Hester** ([colton-dev](https://github.com/ColtonHester/mids-w207-section1-team1-finalproject/tree/colton-dev)): Data ingestion with Polars, undersampling pipelines, modeling experiments (RF, FFNN, Entity Embeddings). Report: Abstract, Introduction, Related Work.

- **Leo Lazzarini** ([leo-dev](https://github.com/ColtonHester/mids-w207-section1-team1-finalproject/tree/leo-dev)): FFNN with class weights (final model), model comparison framework, preprocessing pipeline, regional subgroup evaluation. Report: Results, Discussion, Methodology.

- **Nedim Hodzic** ([nedim-dev](https://github.com/ColtonHester/mids-w207-section1-team1-finalproject/tree/nedim-dev)): Binning strategy, RF/FFNN/XGBoost configurations, East/West location variable, EDA visualizations. Presentation: EDA, Conclusion, Improvements.

- **Shanti Agung** ([shanti-dev](https://github.com/ColtonHester/mids-w207-section1-team1-finalproject/tree/shanti-dev)): Feature selection, preprocessing pipelines (feature engineering, subgroup mean imputation, SMOTENC), RF/FFNN/Logistic models. Report: Preprocessing, Methodology, Experiments.

## Links

- **Medium Article**: [Predicting Wildfire Size in the United States](https://medium.com/@nedimhodzic0111/predicting-wildfire-size-in-the-united-states-699de63de24c)
- **Final Report**: [reports/DATASCI_207_Fall_2025_Project_FinalReport_Team_1.pdf](reports/DATASCI_207_Fall_2025_Project_FinalReport_Team_1.pdf)

## References

Pourmohamad, Y., et al. (2023). Physical, social, and biological attributes for improved understanding and prediction of wildfires: FPA FOD-attributes dataset. *Earth System Science Data*, 16(6), 3045-2024. https://doi.org/10.5194/essd-16-3045-2024
