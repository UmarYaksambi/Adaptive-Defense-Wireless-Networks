# plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
SUMMARY_FILE = "ALL_EXPERIMENTS_summary.csv" # Assumes you have this combined summary

# Ensure plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data(file_path):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def plot_avg_payoffs_by_model(df, filename="avg_payoffs_by_model.png"):
    """Plots average attacker and defender payoffs grouped by game model."""
    if df is None or df.empty:
        print("No data to plot for average payoffs by model.")
        return

    # Group by game_model and calculate mean payoffs
    grouped = df.groupby("game_model")[["avg_attacker_payoff", "avg_defender_payoff"]].mean().reset_index()

    if grouped.empty:
        print("No grouped data to plot for average payoffs by model.")
        return

    plt.figure(figsize=(10, 6))
    grouped.set_index("game_model").plot(kind="bar", ax=plt.gca())
    plt.title("Average Payoffs by Game Model")
    plt.ylabel("Average Payoff")
    plt.xlabel("Game Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_network_health_by_model(df, filename="network_health_by_model.png"):
    """Plots final network health grouped by game model."""
    if df is None or df.empty:
        print("No data to plot for network health by model.")
        return

    grouped = df.groupby("game_model")["final_network_health"].mean().reset_index()

    if grouped.empty:
        print("No grouped data to plot for network health by model.")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(x="game_model", y="final_network_health", data=grouped, palette="viridis")
    plt.title("Average Final Network Health by Game Model")
    plt.ylabel("Average Network Health")
    plt.xlabel("Game Model")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1) # Health is between 0 and 1
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_payoffs_vs_nodes(df, game_model_filter, topology_filter, filename_prefix="payoffs_vs_nodes"):
    """
    Plots average payoffs vs. number of nodes for a specific game model and topology.
    """
    if df is None or df.empty:
        print(f"No data to plot for payoffs vs nodes ({game_model_filter}, {topology_filter}).")
        return

    filtered_df = df[(df["game_model"] == game_model_filter) & (df["topology"] == topology_filter)]

    if filtered_df.empty:
        print(f"No data after filtering for {game_model_filter} and {topology_filter}.")
        return

    grouped = filtered_df.groupby("num_nodes")[["avg_attacker_payoff", "avg_defender_payoff"]].mean().reset_index()

    if grouped.empty:
        print(f"No grouped data for payoffs vs nodes ({game_model_filter}, {topology_filter}).")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(grouped["num_nodes"], grouped["avg_attacker_payoff"], marker='o', label="Attacker Payoff")
    plt.plot(grouped["num_nodes"], grouped["avg_defender_payoff"], marker='x', label="Defender Payoff")
    plt.title(f"Average Payoffs vs. Number of Nodes\n(Model: {game_model_filter}, Topology: {topology_filter})")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Average Payoff")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"{filename_prefix}_{game_model_filter.replace(' ','_')}_{topology_filter.replace(' ','_').replace('(','').replace(')','').replace('–','')}.png"
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_strategy_frequencies(df, player_type, game_model_filter=None, filename_prefix="strategy_freq"):
    """
    Plots strategy frequencies for attacker or defender, optionally filtered by game model.
    Expects columns like 'atk_freq_strategyname' or 'def_freq_strategyname'.
    """
    if df is None or df.empty:
        print(f"No data to plot for {player_type} strategy frequencies.")
        return

    if game_model_filter:
        df_filtered = df[df["game_model"] == game_model_filter].copy() # Use .copy() to avoid SettingWithCopyWarning
        title_suffix = f" (Model: {game_model_filter})"
        filename_suffix = f"_{game_model_filter.replace(' ','_')}"
    else:
        df_filtered = df.copy()
        title_suffix = " (All Models)"
        filename_suffix = "_all_models"

    if df_filtered.empty:
        print(f"No data after filtering for {player_type} strategy frequencies{title_suffix}.")
        return

    freq_cols = [col for col in df_filtered.columns if col.startswith(f"{player_type}_freq_")]
    if not freq_cols:
        print(f"No {player_type} frequency columns found (e.g., '{player_type}_freq_strategyname').")
        return

    # Sum frequencies across all trials for the (filtered) dataframe
    # We need to average these frequencies per simulation setting (model, topo, nodes, etc.)
    # Or, if we just want an overall picture for the filtered data:
    strategy_sums = df_filtered[freq_cols].sum() # Sum of counts for each strategy
    
    # Normalize to get proportions if desired, or plot raw counts
    # total_plays = strategy_sums.sum()
    # strategy_proportions = strategy_sums / total_plays if total_plays > 0 else strategy_sums

    if strategy_sums.empty:
        print(f"No strategy sum data for {player_type}{title_suffix}.")
        return

    plt.figure(figsize=(12, 7))
    strategy_sums.rename(index=lambda x: x.replace(f"{player_type}_freq_", "")).plot(kind="bar") # Clean up labels
    plt.title(f"{player_type.capitalize()} Strategy Usage Frequencies{title_suffix}")
    plt.ylabel("Total Times Chosen (Across Trials in Filtered Set)")
    plt.xlabel("Strategy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    filename = f"{filename_prefix}_{player_type}{filename_suffix}.png"
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


# --- Main plotting logic ---
if __name__ == "__main__":
    summary_file_path = os.path.join(RESULTS_DIR, SUMMARY_FILE)
    data = load_data(summary_file_path)

    if data is not None:
        print("Generating plots...")
        plot_avg_payoffs_by_model(data)
        plot_network_health_by_model(data)

        # Example: Plot payoffs vs nodes for a specific configuration
        # You might want to loop through interesting combinations
        if not data[(data["game_model"] == "Q-Learning") & (data["topology"] == "Random (Erdős–Rényi)")].empty:
             plot_payoffs_vs_nodes(data, game_model_filter="Q-Learning", topology_filter="Random (Erdős–Rényi)")
        else:
             print("Skipping Q-Learning/Random (Erdős–Rényi) payoff vs nodes plot - no data.")

        if not data[(data["game_model"] == "Bayesian Game") & (data["topology"] == "Star")].empty:
            plot_payoffs_vs_nodes(data, game_model_filter="Bayesian Game", topology_filter="Star")
        else:
            print("Skipping Bayesian Game/Star payoff vs nodes plot - no data.")


        # Plot overall strategy frequencies
        plot_strategy_frequencies(data, player_type="atk", filename_prefix="overall_atk_strategy_freq")
        plot_strategy_frequencies(data, player_type="def", filename_prefix="overall_def_strategy_freq")

        # Plot strategy frequencies for a specific game model
        if "Q-Learning" in data["game_model"].unique():
            plot_strategy_frequencies(data, player_type="atk", game_model_filter="Q-Learning", filename_prefix="qlearning_atk_strategy_freq")
            plot_strategy_frequencies(data, player_type="def", game_model_filter="Q-Learning", filename_prefix="qlearning_def_strategy_freq")
        else:
            print("Skipping Q-Learning specific strategy frequency plots - Q-Learning model not found in data.")

        print("Plotting complete. Check the 'plots' directory.")
    else:
        print("Could not load data. No plots will be generated.") 