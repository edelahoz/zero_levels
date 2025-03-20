#%%
#!/usr/bin/env python3

import os
import numpy as np

try:
    from ..src import IO

except ImportError:
    import sys
    
    sys.path.append(
        os.path.dirname(os.path.dirname(__file__))
    )
    
    import src.IO as IO

db_name = "test_io.db"

IO.create_tables(db_name=db_name)  # Ensure tables exist

# Store results for a combination of two maps
test_maps = ["A", "B"]
test_iterations = np.array([[0.5, 1.2, -0.5, 0.3, 0.6, 1.3, -0.6, 0.4],
                            [0.4, 1.1, -0.4, 0.25, 0.5, 1.2, -0.5, 0.35]])

test_total_iterations = test_iterations[0] + test_iterations[1]
# test_iterations = [
#     {"A": np.array([0.5, 1.2, -0.5, 0.3]), "B": np.array([0.6, 1.3, -0.6, 0.4])},  # Iteration 1
#     {"A": np.array([0.4, 1.1, -0.4, 0.25]), "B": np.array([0.5, 1.2, -0.5, 0.35])}  # Iteration 2
# ]
IO.store_results(db_name, test_maps, test_iterations, "Testing two maps")

# Read results
res1 = IO.read_results(db_name, test_maps, last_iteration_only=True)
res2 = IO.read_results(db_name, test_maps, selected_map="A")


assert (
    np.all(res1[0][2:] == test_total_iterations[:4]) and 
    np.all(res1[1][2:] == test_total_iterations[4:8])
), "The stored results for the last iteration are not correct."

os.remove(db_name)

# %%
