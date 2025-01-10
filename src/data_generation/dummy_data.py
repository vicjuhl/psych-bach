import numpy as np
import pandas as pd
import pathlib as pl

from src.data_utils.fields import fields, sorted_fields, five_d_asc
from src.config import seed_iterators, n_main_feats, n_middle_feats, n_relevant_feats, n_noise_feats

def gen_dummy_set(n: int, drug_sens: float, sex_sens: float, age_sens: float, path_dir: pl.Path, test = False):
    """Make dummy data set of size n with given sensetivities.
    Demographics and labels are not affected by sensitivities.
    """
    # Set random seed for reproducibility
    np.random.seed(next(seed_iterators[0]))
    
    data = []

    feature_weights = np.random.normal(0, 1, (5, n_relevant_feats))  # weights matrix for 5 outputs

    for sub in range(1, n + 1):
        # Generate demographics
        age = np.random.randint(18, 80)
        sex = np.random.choice(['M', 'F'])

        for active_drug in [False, True]:
            # Generate features most affected by drug, sex and age
            f_base = np.random.normal(0, 1, n_main_feats)
            f_drug = np.random.normal(drug_sens * active_drug, 0.1, n_main_feats)
            f_sex = np.random.normal(1, 0.2, n_main_feats) * (1 if sex == 'M' else -1) * sex_sens
            f_age = np.random.normal(0, 3, n_main_feats) * (age - 49) / 31 * age_sens  # Normalize age effect

            main_features = f_base + f_drug + f_sex + f_age
            main_features[0] += active_drug * drug_sens * 1 # extra effect of active drug on f1
            main_features[1] += active_drug * drug_sens * -1.2 # extra (lesser) effect of active drug on f2
            main_features[2] += active_drug * drug_sens * main_features[1] # non-linear effect
            main_features[3] -= main_features[1]**2 * main_features[2] # non-linear effect
            main_features[4] = np.sign(main_features[4]) * (np.abs(main_features[4])) ** (1 + active_drug * drug_sens * 0.5) # Non-linear effect of active drug on f5

            # Generate features which are equally (and less) affected by drug, sex and age
            middle_features = np.random.normal(
                drug_sens*active_drug * 0.3 + sex_sens*(sex == 'F') * 0.1 + age_sens*(age - 49) / 31 * 0.1,
                3, n_middle_feats
            )

            # Generate remaining features (f9 to f12) which are noise
            noise_features = np.random.normal(0, 1, n_noise_feats)

            # Not actually noise! TODO
            noise_features[20] = main_features[0]
            noise_features[30] += float(active_drug)

            # Generate labels as dependent on features
            five_d_asc_base = np.random.normal(0, 2, 5)
            
            # Compute contribution from features (both initial f1-f3 and remaining f4-f8)
            relevant_features = np.concatenate([main_features, middle_features])  # first 8 features
            feature_contribution = feature_weights @ relevant_features
            # Add small demographic effects
            demographic_effect = np.random.normal(0, 0, 5) * ((sex == 'M') * 0 + (age - 49) / 31 * 0)
            
            five_d_asc_score = five_d_asc_base + feature_contribution + demographic_effect

            row = [sub, active_drug] + five_d_asc_score.tolist() + relevant_features.tolist() + noise_features.tolist()
            data.append(row)        

    # Create DataFrame
    dummy = pd.DataFrame(data, columns=sorted_fields)

    # Normalize five_d_asc columns to range 0-10
    for col in five_d_asc.keys():
        dummy[col] = (dummy[col] - dummy[col].min()) / (dummy[col].max() - dummy[col].min()) * 100

    # Ensure correct data types
    for col, dtype in fields.items():
        dummy[col] = dummy[col].astype(dtype)
    dummy.to_csv(path_dir / f"dummy{'_test' if test else ''}_n{n}_sens{drug_sens}_{sex_sens}_{age_sens}.csv")