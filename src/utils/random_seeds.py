import random
from pathlib import Path
import multiprocessing
from itertools import cycle

def generate_random_seeds(dump: bool, size=100000, min_val=0, max_val=99999, seed=654):
    """
    Generate a list of unique random integers. They will always be the same if seed is fixed.
    """
    if max_val - min_val + 1 < size:
        raise ValueError("Range is smaller than requested size")
    
    # Create a separate random instance with a fixed seed
    rng = random.Random(seed)
    seeds = rng.sample(range(min_val, max_val + 1), size)

    if dump:
        output_path = Path(__file__).parent / "random_seeds.txt"
        with open(output_path, "w") as f:
            for seed in seeds:
                f.write(f"{seed}\n")
    else:
        return seeds
    
def partition_seed_iterators(seeds: list[int], n_cores: int):
    seeds_per_core = len(seeds) // n_cores
    seed_partitions = [seeds[i * seeds_per_core:(i + 1) * seeds_per_core] for i in range(n_cores)]
    iterators = [
        cycle(partition)
        for partition in seed_partitions
    ]
    return iterators