# Predicting Wildfire Size in the United States

**Colton Hester's Development Branch** | UC Berkeley MIDS - DATASCI 207 (Fall 2025)

This branch contains my individual contributions to the team's wildfire size prediction project.

## My Contributions

### Data Ingestion & Pipeline Engineering

**Memory-Efficient Data Loading** ([src/data_loader.py](src/data_loader.py))
- Implemented Polars-based CSV loading that reduced RAM usage from ~20GB to ~2GB
- Selective column loading for the 48 required features (out of 308 total)
- GRIDMET climate variable detection via pattern matching

**End-to-End Preprocessing Pipeline** ([src/preprocessing_pipeline.py](src/preprocessing_pipeline.py))
- Configurable pipeline supporting multiple imbalance strategies: undersampling, SMOTE, class weights
- Stratified train/val/test splitting preserving temporal structure across 29 years
- Entity embeddings infrastructure with integer encoding for categorical features
- Feature imputation: zero-fill for infrastructure, mean imputation for climate variables

### Model Training & Experiments

**Modeling Notebook** ([notebooks/03_Model_Training_Undersampling_CH.ipynb](notebooks/03_Model_Training_Undersampling_CH.ipynb))
- Random Forest baseline implementation
- Feed-Forward Neural Network with dropout regularization
- Entity Embeddings model using Keras Embedding layers for learned categorical representations
- Comprehensive evaluation with macro-averaged metrics (precision, recall, F1)
- Class weights approach comparison against undersampling

**Data Loading Notebook** ([notebooks/01_Data_Loading_polars_CH.ipynb](notebooks/01_Data_Loading_polars_CH.ipynb))
- Initial Polars-based data exploration
- MTBS threshold filtering (West: 1000+ acres, East: 500+ acres)
- Parquet conversion for efficient downstream processing

### Supporting Modules

- **Train/Test Splitter** ([src/train_test_splitter.py](src/train_test_splitter.py)): Stratified splitting with optional shuffling
- **Data Visualization** ([src/dataviz_builders.py](src/dataviz_builders.py)): Geographic maps, regional distributions, infrastructure analysis

## Key Technical Highlights

| Component | Technique | Purpose |
|-----------|-----------|---------|
| Data Loading | Polars lazy evaluation | Memory efficiency for 2.3M records |
| Imbalance | Undersampling to minority class | Balance 97%/2%/0.2%/0.05% distribution |
| Categoricals | Entity Embeddings | Learn dense representations for fire cause, GACC level |
| Neural Network | FFNN (64→32→4) + Dropout | Capture nonlinear feature interactions |
| Evaluation | Macro F1 | Fair assessment across imbalanced classes |

## Results (Undersampling Approach)

| Model | Val Accuracy | Val Macro F1 | Test Macro F1 |
|-------|--------------|--------------|---------------|
| Random Forest | 64.5% | 0.22 | 0.22 |
| FFNN | 67.0% | 0.22 | 0.22 |
| Entity Embeddings | 65.8% | 0.22 | 0.22 |
| Entity Embed + Class Weights | 66.4% | 0.23 | 0.23 |

Note: The team's best model (FFNN + class weights on full dataset) achieved macro F1=0.47. See [main branch](https://github.com/ColtonHester/mids-w207-section1-team1-finalproject) for consolidated results.

## Project Context

**Problem**: Predict wildfire size categories (small/medium/large/very large) from 2.3M U.S. wildfire incidents with 308 features.

**Challenge**: Extreme class imbalance (97% small fires, 0.05% very large fires).

**Dataset**: FPA FOD-Attributes ([Pourmohamad et al., 2023](https://doi.org/10.5194/essd-16-3045-2024))

## Links

- [Main Branch (Full Project)](https://github.com/ColtonHester/mids-w207-section1-team1-finalproject)
- [Medium Article](https://medium.com/@nedimhodzic0111/predicting-wildfire-size-in-the-united-states-699de63de24c)
- [Final Report](reports/DATASCI_207_Fall_2025_Project_FinalReport_Team_1.pdf)
