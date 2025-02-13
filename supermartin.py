import numpy as np
import matplotlib.pyplot as plt
import time
import os
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

# ============================
# Global Simulation Parameters
# ============================
NUM_INSTANCES    = 200000          # number of independent simulations per permutation
STARTING_CAPITAL = 1_000.0         # starting capital per simulation
BATCH_SIZE       = 1000            # simulate 1000 bets per kernel launch

# ============================
# Runtime Limits (in seconds)
# ============================
TOTAL_ALLOWED_RUNTIME = 86400  #86400- 24 hours total allowed time for all permutations
MAX_SIM_TIME          = 6000   # 1800 -30 minutes maximum per simulation permutation

# ============================
# List of Parameter Permutations
# ============================
# Four different starting bets (all orders of magnitude apart, all ≤ 0.01)
initial_bets = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
# Six different target capitals (stopping values)
target_capitals = [2000.0, 5000.0, 10000.0, 50000.0, 100000.0, 300000.0, 800000.0]

# ============================
# CUDA Kernel (modified to accept target and base bet)
# ============================
@cuda.jit
def simulate_batch(batch_size, accounts, bets, active, rng_states, target_capital, base_bet):
    """
    For each simulation instance (one per thread), simulate a batch of bets.

    Parameters:
      batch_size:     Number of bets to simulate in this kernel launch.
      accounts:       1D array of account balances.
      bets:           1D array of current bet sizes.
      active:         1D array (of ints) indicating if the simulation is still active.
      rng_states:     Pre-initialized per-thread RNG states.
      target_capital: The stopping capital (simulation terminates if account >= this).
      base_bet:       The initial bet size (used to reset the bet after a win).
    """
    i = cuda.grid(1)
    if i >= accounts.size:
        return

    if active[i] == 0:
        return

    for j in range(batch_size):
        # Termination criteria: bust (not enough to cover bet) or reached/exceeded target.
        if accounts[i] < bets[i] or accounts[i] >= target_capital:
            active[i] = 0
            break

        # Simulate a fair coin toss.
        r = xoroshiro128p_uniform_float32(rng_states, i)
        if r < 0.5:
            # Loss: subtract the bet and double the bet.
            accounts[i] -= bets[i]
            bets[i] *= 2.0
        else:
            # Win: add the bet and reset bet size.
            accounts[i] += bets[i]
            bets[i] = base_bet

# ============================
# Main Simulation Routine
# ============================
def main():
    # Create folder to save graphs.
    output_folder = "simulation_results"
    os.makedirs(output_folder, exist_ok=True)

    # Dictionary to store summary results for final plotting.
    # Keys: target capital; Values: list of (base_bet, final_return_percent) tuples.
    summary_results = {}

    total_runtime_accumulated = 0.0
    permutation_index = 0  # used to vary RNG seeds

    # Loop over all parameter permutations.
    for target_capital in target_capitals:
        for base_bet in initial_bets:
            # Check remaining overall time.
            remaining_total_time = TOTAL_ALLOWED_RUNTIME - total_runtime_accumulated
            if remaining_total_time <= 0:
                print("Total allowed simulation time reached. Stopping further simulations.")
                break

            # Allowed simulation time for this permutation.
            sim_time_allowed = min(MAX_SIM_TIME, remaining_total_time)
            print(f"\nStarting simulation permutation {permutation_index+1}:")
            print(f"  Base bet: {base_bet}")
            print(f"  Target capital: {target_capital}")
            print(f"  Allowed simulation time: {sim_time_allowed:.1f} seconds")

            # Initialize host arrays for simulation state.
            accounts_host = np.full(NUM_INSTANCES, STARTING_CAPITAL, dtype=np.float64)
            bets_host     = np.full(NUM_INSTANCES, base_bet, dtype=np.float64)
            active_host   = np.ones(NUM_INSTANCES, dtype=np.int32)  # 1 means active

            # Copy state arrays to device.
            d_accounts = cuda.to_device(accounts_host)
            d_bets     = cuda.to_device(bets_host)
            d_active   = cuda.to_device(active_host)

            # Initialize per-thread RNG states (vary seed by permutation_index).
            rng_states = create_xoroshiro128p_states(NUM_INSTANCES, seed=42 + permutation_index)

            # Determine CUDA kernel launch configuration.
            threads_per_block = 128
            blocks = (NUM_INSTANCES + threads_per_block - 1) // threads_per_block

            # Lists to record simulation data (for graphing the average).
            bet_counts   = []  # cumulative bets executed (x-axis)
            avg_accounts = []  # average account value (y-axis)
            total_bets   = 0   # cumulative bets counter

            permutation_start_time = time.time()
            sim_elapsed = 0.0

            # Main simulation loop for this permutation.
            while sim_elapsed < sim_time_allowed:
                simulate_batch[blocks, threads_per_block](BATCH_SIZE, d_accounts, d_bets, d_active,
                                                            rng_states, target_capital, base_bet)
                cuda.synchronize()  # wait for kernel to complete

                total_bets += BATCH_SIZE

                # Copy current state back to host.
                accounts_host = d_accounts.copy_to_host()
                active_host   = d_active.copy_to_host()
                avg_value = np.mean(accounts_host)
                bet_counts.append(total_bets)
                avg_accounts.append(avg_value)

                # Terminate simulation early if no accounts remain active.
                if np.sum(active_host) == 0:
                    print("  All accounts terminated (busted or reached target).")
                    break

                sim_elapsed = time.time() - permutation_start_time

            permutation_total_time = time.time() - permutation_start_time
            total_runtime_accumulated += permutation_total_time
            print(f"Permutation {permutation_index+1} completed in {permutation_total_time:.2f} seconds")
            print(f"  Total bets executed per instance: {total_bets}")

            # Calculate final average return percentage.
            final_avg_account = np.mean(accounts_host)
            final_return_percent = ((final_avg_account - STARTING_CAPITAL) / STARTING_CAPITAL) * 100
            print(f"  Final average return: {final_return_percent:.2f}%")

            # Save the result for the final summary graph.
            if target_capital not in summary_results:
                summary_results[target_capital] = []
            summary_results[target_capital].append((base_bet, final_return_percent))

            # --------------------------
            # Save Graph 1: Average Account Value vs. Total Bets Executed
            # --------------------------
            plt.figure(figsize=(10, 6))
            plt.plot(bet_counts, avg_accounts, lw=2)
            plt.xlabel("Total Bets Executed (per instance)", fontsize=14)
            plt.ylabel("Average Account Value ($)", fontsize=14)
            plt.title(f"Avg Account Value vs Bets\n(Base Bet = {base_bet}, Target = {target_capital})", fontsize=16)
            plt.grid(True)
            plt.tight_layout()
            graph_filename1 = os.path.join(output_folder,
                                           f"avg_account_value_bet_{base_bet:.0e}_target_{target_capital:.0e}.png")
            plt.savefig(graph_filename1, dpi=300)
            plt.close()

            # --------------------------
            # Save Graph 2: Distribution of Account Values at End of Simulation
            # --------------------------
            plt.figure(figsize=(10, 6))
            plt.hist(accounts_host, bins=50, edgecolor='black')
            plt.xlabel("Account Value ($)", fontsize=14)
            plt.ylabel("Frequency", fontsize=14)
            plt.title(f"Account Value Distribution\n(Base Bet = {base_bet}, Target = {target_capital})", fontsize=16)
            plt.grid(True)
            plt.tight_layout()
            graph_filename2 = os.path.join(output_folder,
                                           f"account_distribution_bet_{base_bet:.0e}_target_{target_capital:.0e}.png")
            plt.savefig(graph_filename2, dpi=300)
            plt.close()

            permutation_index += 1

        # Check if overall total allowed time has been reached.
        if TOTAL_ALLOWED_RUNTIME - total_runtime_accumulated <= 0:
            print("\nOverall total allowed simulation time reached. Exiting parameter loop.")
            break

    # --------------------------
    # Final Summary Graph:
    # Plot final return percentage vs. initial bet for each target capital.
    # --------------------------
    plt.figure(figsize=(10, 6))
    for target_capital, results in summary_results.items():
        # Sort results by base_bet.
        results.sort(key=lambda x: x[0])
        bets_list = [r[0] for r in results]
        returns_list = [r[1] for r in results]
        plt.plot(bets_list, returns_list, marker='o', label=f"Target {target_capital:.0f}")
    plt.xscale("log")
    plt.xlabel("Initial Bet (log scale)", fontsize=14)
    plt.ylabel("Final Return (%)", fontsize=14)
    plt.title("Final Return Percentage vs Initial Bet for Different Targets", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    final_graph_filename = os.path.join(output_folder, "final_summary.png")
    plt.savefig(final_graph_filename, dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
