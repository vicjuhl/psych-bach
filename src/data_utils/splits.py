import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from src.config import n_splits

def get_splits(df: pd.DataFrame, seed_iterator):
    # Initialize split columns
    split_columns = ['test_split', 'val_split', 'sel_split']
    df = df.assign(**{col: -1 for col in split_columns})
    
    # Get unique pids
    unique_pids = df['pid'].unique()
    
    # Outer split
    outer_kf = KFold(n_splits=n_splits["outer"], shuffle=True, random_state=next(seed_iterator))
    for fold, (_, test_idx) in enumerate(outer_kf.split(unique_pids)):
        test_pids = unique_pids[test_idx]
        df.loc[df['pid'].isin(test_pids), 'test_split'] = fold
    
    # Middle split - modified approach
    for outer_fold in range(n_splits["outer"]):
        # Collect PIDs for this outer fold
        outer_mask = df['test_split'] == outer_fold
        fold_pids = df[outer_mask]['pid'].unique()
        
        # Shuffle PIDs
        rng = np.random.RandomState(next(seed_iterator))
        shuffled_pids = rng.permutation(fold_pids)
        
        # Calculate sizes for even distribution
        n_total = len(shuffled_pids)
        base_size = n_total // n_splits["middle"]
        remainder = n_total % n_splits["middle"]
        
        # Rotate which fold gets the remainder based on outer_fold
        remainder_offset = outer_fold % n_splits["middle"]
        
        current_idx = 0
        for fold in range(n_splits["middle"]):
            fold_size = base_size
            # Add one extra if there's remainder (rotating which fold gets it)
            if remainder > 0 and (fold - remainder_offset) % n_splits["middle"] < remainder:
                fold_size += 1
            
            val_pids = shuffled_pids[current_idx:current_idx + fold_size]
            df.loc[df['pid'].isin(val_pids), 'val_split'] = fold
            current_idx += fold_size
    
    # Assign pids for each test/val combination
    for outer_fold in range(n_splits["outer"]):
        for middle_fold in range(n_splits["middle"]):
            mask = (df['test_split'] == outer_fold) & (df['val_split'] == middle_fold)
            fold_pids = df[mask]['pid'].unique()
            
            rng = np.random.RandomState(next(seed_iterator))
            shuffled_pids = rng.permutation(fold_pids)
            
            n_total = len(shuffled_pids)
            base_size = n_total // n_splits["inner"]
            remainder = n_total % n_splits["inner"]
            
            # Rotate which fold gets the remainder based on outer and middle fold
            remainder_offset = (outer_fold * n_splits["middle"] + middle_fold) % n_splits["inner"]
            
            current_idx = 0
            for fold in range(n_splits["inner"]):
                fold_size = base_size
                # Add one extra if there's remainder (rotating which fold gets it)
                if remainder > 0 and (fold - remainder_offset) % n_splits["inner"] < remainder:
                    fold_size += 1
                
                fold_pids = shuffled_pids[current_idx:current_idx + fold_size]
                df.loc[df['pid'].isin(fold_pids), 'sel_split'] = fold
                current_idx += fold_size
    
    return df