[PROJECT 3]

# Overcooked

## Project Overview

This project involves the implementation of a multi-agent RL algorithm (MAPPO) to solve the **Overcooked** challenge where two agents must coordinate in 3 kitchen layouts (cramped room, coordination ring,counter circuit o 1order) to cook and deliver at least 7 soups per episode.
## Structure of the Repository

[UPDATED] The gymnasium compliant implementations are organized as follows:

``` bash
Overcooked_MAPPO.py
```

-   `Overcooked_MAPPO.py`: Contains the implementation of the MAPPO algorithm for training and testing the trained agents on and generating the required performance graphs and tables with the results.

## How to Run

1.  Install the libraries in the `requirements.txt` provided in the folder and then follow the next steps. The python version is `Python 3.11.11`

2.  To generate the metrics and graphs with the finalized hyperparameters (or change them) and the pre-set number of episodes, run Overcooked_MAPPO.py. The progress will be printed on the screen as execution runs through the 3 kitchen layouts.

3.  Once the code has successfully executed the following files are generated:

- `combined_avg_soups_per_100_episodes.png`: Contains the graphs for the soups delivered per episode during training across the 3 layouts
- `combined_avg_onion_per_100_episodes.png`: Contains the graphs for the onion pickups per episode during training across the 3 layouts
- `combined_avg_dish_per_100_episodes.png`: Contains the graphs for the dish pickups per episode during training across the 3 layouts

- ` combine_test_soups_100_episodes.png`: Contains the graphs for the soups delivered per episode during testing across the 3 layouts
- `combine_test_onion_100_episodes.png`: Contains the graphs for the onion pickups per episode during testing across the 3 layouts
- ` combine_test_dish_100_episodes.png`: Contains the graphs for the dish pickups per episode during testing across the 3 layouts

- ` test_results_cramped_room.txt`: Contains final testing results for the metrics Avg soups per episode delivered in 100 consective epsiode, along with Avg Onion pickups and Avg Dishpickups for the cramped_room layout.
- ` test_results_coordination_ring.txt`: Contains final testing results for the metrics Avg soups per episode delivered in 100 consective epsiode, along with Avg Onion pickups and Avg Dishpickups for the coordination_ring layout.
- ` test_results_counter_circuit_o_1order.txt`: Contains final testing results for the metrics Avg soups per episode delivered in 100 consective epsiode, along with Avg Onion pickups and Avg Dishpickups for the counter_circuit_o_1order layout.
