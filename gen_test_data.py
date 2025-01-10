import pathlib as pl

from src.data_generation.dummy_data import gen_dummy_set

gen_dummy_set(80, 0.7, 0.2, 0.1, pl.Path("data/"), test=True)
