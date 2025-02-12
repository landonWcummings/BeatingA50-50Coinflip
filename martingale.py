import numpy as np
import matplotlib.pyplot as plt
import time
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

# ============================
# Simulation Parameters
# ============================
NUM_INSTANCES    = 100000          # number of independent simulations
TARGET_CAPITAL   = 800_000.0      # stop if account >= this value
STARTING_CAPITAL = 1_000.0        # starting capital per simulation
BASE_BET         = 0.00001        # initial bet
BATCH_SIZE       = 1000           # simulate 1000 bets per kernel launch
SIM_TIME         = 70000.0        # overall simulation time in seconds

# ============================
# CUDA Kernel
# ============================
@cuda.jit
def simulate_batch(batch_size, accounts, bets, active, rng_states):
    """
    For each simulation instance (one per thread), simulate a batch of bets.

    Parameters:
      batch_size: Number of bets to simulate in this kernel launch.
      accounts:   1D array of account balances.
      bets:       1D array of current bet sizes.
      active:     1D array (of ints) indicating if the simulation is still active.
      rng_states: Pre-initialized per-thread RNG states.
    """
    i = cuda.grid(1)
    if i >= accounts.size:
        return

    # Skip simulation if this instance is no longer active.
    if active[i] == 0:
        return

    for j in range(batch_size):
        # Check stopping criteria before placing a bet:
        #   1. Bust: not enough money to cover the current bet.
        #   2. Reached or exceeded target capital.
        if accounts[i] < bets[i] or accounts[i] >= TARGET_CAPITAL:
            active[i] = 0  # mark this simulation as terminated
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
            bets[i] = BASE_BET

# ============================
# Main Simulation Routine
# ============================
def main():
    # Allocate host arrays for simulation state.
    accounts_host = np.full(NUM_INSTANCES, STARTING_CAPITAL, dtype=np.float64)
    bets_host     = np.full(NUM_INSTANCES, BASE_BET, dtype=np.float64)
    active_host   = np.ones(NUM_INSTANCES, dtype=np.int32)  # 1 means "active"

    # Copy state arrays to the device.
    d_accounts = cuda.to_device(accounts_host)
    d_bets     = cuda.to_device(bets_host)
    d_active   = cuda.to_device(active_host)

    # Initialize per-thread RNG states.
    rng_states = create_xoroshiro128p_states(NUM_INSTANCES, seed=42)

    # Determine CUDA kernel launch configuration.
    threads_per_block = 128
    blocks = (NUM_INSTANCES + threads_per_block - 1) // threads_per_block

    # Lists to record simulation data for plotting the average.
    bet_counts   = []   # cumulative bets executed (x-axis)
    avg_accounts = []   # average account value (y-axis)
    total_bets   = 0    # cumulative bets counter

    # Start the simulation timer.
    start_time = time.time()
    elapsed = 0.0

    # Main simulation loop: launch batches until SIM_TIME seconds have elapsed.
    while elapsed < SIM_TIME:
        simulate_batch[blocks, threads_per_block](BATCH_SIZE, d_accounts, d_bets, d_active, rng_states)
        cuda.synchronize()  # wait for the kernel to complete

        total_bets += BATCH_SIZE

        # Copy the current account values back to host.
        accounts_host = d_accounts.copy_to_host()
        avg_value = np.mean(accounts_host)
        bet_counts.append(total_bets)
        avg_accounts.append(avg_value)

        # Update elapsed time.
        elapsed = time.time() - start_time

    print("Simulation complete.")
    print(f"Total bets executed (per instance): {total_bets}")

    # How many accounts reached or exceeded the target capital?
    max_win = np.count_nonzero(accounts_host >= TARGET_CAPITAL)
    print(f"Number of accounts at or above the target capital: {max_win}")

    # --------------------------
    # Plot 1: Average Account Value vs. Total Bets Executed
    # --------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(bet_counts, avg_accounts, lw=2)
    plt.xlabel("Total Bets Executed (per instance)", fontsize=14)
    plt.ylabel("Average Account Value ($)", fontsize=14)
    plt.title("Monte Carlo Simulation of a Martingale Betting Strategy", fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    # Save this figure
    plt.savefig("average_account_value.png", dpi=300)
    plt.show()

    # --------------------------
    # Plot 2: Distribution of Account Values
    # --------------------------
    plt.figure(figsize=(10, 6))
    plt.hist(accounts_host, bins=50, edgecolor='black')
    plt.xlabel("Account Value ($)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title("Distribution of Account Values at End of Simulation", fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    # Save this figure
    plt.savefig("account_value_distribution.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
