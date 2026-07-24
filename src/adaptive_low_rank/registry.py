from adaptive_low_rank.algorithms import (
    adaptive,
    batch_max,
    greedy,
    greedy_pp,
    random,
)

ALGORITHMS = {
    "adaptive": adaptive.select_rows,
    "batch_max": batch_max.select_rows,
    "greedy_pp": greedy_pp.select_rows,
    "greedy": greedy.select_rows,
    "random": random.select_rows,
}