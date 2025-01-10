import pandas as pd
from itertools import cycle

from src.data_utils.splits import get_splits
from src.config import n_splits

def test_get_splits():
    # Load test data
    df = pd.read_csv("data/dummy_n79_sens0.7_0.2_0.1.csv")
    df = get_splits(df, cycle([4, 5, 6]))
    
    # Test that split columns exist
    split_cols = ['test_split', 'val_split', 'sel_split']
    for col in split_cols:
        assert col in df.columns, f"Split column {col} is missing"
    
    # Test that no -1 values remain (all rows assigned to folds)
    for col in split_cols:
        assert not (df[col] == -1).any(), f"Found unassigned fold (-1) in {col}"
    
    # Test that fold numbers are valid
    assert df['test_split'].isin(range(n_splits['outer'])).all(), "Invalid fold numbers in outer split"
    assert df['val_split'].isin(range(n_splits['middle'])).all(), "Invalid fold numbers in middle split"
    assert df['sel_split'].isin(range(n_splits['inner'])).all(), "Invalid fold numbers in inner split"

    # Test stratification
    total_rows = len(df)
    for test_split in range(n_splits['outer']):
        # Split test from train/val
        df_test = df[df['test_split'] == test_split]
        df_train_val = df[df['test_split'] != test_split]
        # Count split sizes
        n_test = df_test.shape[0]
        n_train_val = df_train_val.shape[0]
        # Count active/inactive
        n_active_test = df_test[df_test['active_drug'] == 1].shape[0]
        n_active_train_val = df_train_val[df_train_val['active_drug'] == 1].shape[0]
        
        # Test split sizes
        assert abs(len(df_test) - total_rows / n_splits["outer"]) <= 2, f"Test set size deviates from expected 1/{n_splits['outer']}"
        assert abs(len(df_train_val) - (n_splits['outer']-1)/n_splits['outer'] * total_rows) <= 2, f"Train/val set size deviates from expected {n_splits['outer']-1}/{n_splits['outer']}"
        
        # Test that each pid appears exactly twice in both splits
        assert all(df_test.groupby('pid').size() == 2), "Not all PIDs appear exactly twice in test set"
        assert all(df_train_val.groupby('pid').size() == 2), "Not all PIDs appear exactly twice in train/val set"
        # Test stratification
        assert abs(n_active_train_val - n_train_val / 2) <= 1, "Unequal distribution of active/inactive in train/val set"
        assert abs(n_active_test - n_test / 2) <= 1, "Unequal distribution of active/inactive in test set"
        
        # Split train from val
        for val_split in range(n_splits['middle']):
            # print(f"train_val len: {len(df_train_val)}")
            # Split val from train
            df_val = df_train_val[df_train_val['val_split'] == val_split]
            df_train = df_train_val[df_train_val['val_split'] != val_split]
            # print(f"val len: {len(df_val)}")
            # print(f"train len: {len(df_train)}")
            # Count split sizes
            n_val = df_val.shape[0]
            n_train = df_train.shape[0]
            # Count active/inactive
            n_active_val = df_val[df_val['active_drug'] == 1].shape[0]
            n_active_train = df_train[df_train['active_drug'] == 1].shape[0]

            assert abs(len(df_val) - len(df_train_val)/n_splits['middle']) <= 2, f"Validation set size deviates from expected 1/{n_splits['middle']} of train/val"
            assert abs(len(df_train) - (n_splits['middle']-1)/n_splits['middle'] * len(df_train_val)) <= 2, f"Training set size deviates from expected {n_splits['middle']-1}/{n_splits['middle']} of train/val"
            
            # Test that each pid appears exactly twice in both splits
            assert all(df_val.groupby('pid').size() == 2), "Not all PIDs appear exactly twice in validation set"
            assert all(df_train.groupby('pid').size() == 2), "Not all PIDs appear exactly twice in training set"
            # Test stratification
            assert abs(n_active_train - n_train / 2) <= 1, "Unequal distribution of active/inactive in train/val set"  
            assert abs(n_active_val - n_val / 2) <= 1, "Unequal distribution of active/inactive in val set"
            # Split train from test
            for sel_split in range(n_splits['inner']):
                df_sel = df_train[df_train['sel_split'] == sel_split]
                df_try = df_train[df_train['sel_split'] != sel_split]
                print(f"sel/try unique pids: {len(df_sel.pid.unique()), len(df_try.pid.unique())}")
                # Count split sizes
                n_sel = df_sel.shape[0]
                n_try = df_try.shape[0]
                # Count active/inactive
                n_active_sel = df_sel[df_sel['active_drug'] == 1].shape[0]
                n_active_try = df_try[df_try['active_drug'] == 1].shape[0]
                
                # Test split sizes
                assert abs(len(df_sel) - 1/n_splits['inner'] * len(df_train)) <= 2, f"Selection set size deviates from expected 1/{n_splits['inner']} of training"
                assert abs(len(df_try) - (n_splits['inner']-1)/n_splits['inner'] * len(df_train)) <= 2, f"Try set size deviates from expected {n_splits['inner']-1}/{n_splits['inner']} of training"
                
                # Test that each pid appears exactly twice in both splits
                assert all(df_sel.groupby('pid').size() == 2), "Not all PIDs appear exactly twice in sel set"
                assert all(df_try.groupby('pid').size() == 2), "Not all PIDs appear exactly twice in try set"
                # Test stratification
                assert abs(n_active_sel - n_sel / 2) == 0, "Unequal distribution of active/inactive in train/test set"
                assert abs(n_active_try - n_try / 2) == 0, "Unequal distribution of active/inactive in try/sel set"