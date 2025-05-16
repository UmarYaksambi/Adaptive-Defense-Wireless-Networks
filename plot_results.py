# plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # For arange, std
import os
from scipy.stats import ttest_ind, f_oneway # For hypothesis testing

# --- Configuration ---
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
SUMMARY_FILE = "ALL_EXPERIMENTS_summary.csv" # Assumes you have this combined summary
DETAILED_HISTORIES_SUBDIR = "detailed_histories" # Subdirectory within RESULTS_DIR for detailed trial logs

# Ensure plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data(file_path):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        # Ensure numeric types for calculation columns
        cols_to_numeric = ['avg_attacker_payoff', 'avg_defender_payoff', 'final_network_health']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def plot_avg_payoffs_by_model(df, filename="avg_payoffs_by_model_with_error.png"):
    """Plots average attacker and defender payoffs grouped by game model with error bars (std)."""
    if df is None or df.empty:
        print("No data to plot for average payoffs by model.")
        return

    grouped = df.groupby("game_model").agg(
        avg_attacker_payoff=('avg_attacker_payoff', 'mean'),
        std_attacker_payoff=('avg_attacker_payoff', 'std'),
        avg_defender_payoff=('avg_defender_payoff', 'mean'),
        std_defender_payoff=('avg_defender_payoff', 'std')
    ).reset_index()

    # If std is NaN (e.g., only one trial per group), fill with 0 for plotting
    grouped = grouped.fillna(0)

    if grouped.empty:
        print("No grouped data to plot for average payoffs by model.")
        return

    plt.figure(figsize=(12, 7))
    n_models = len(grouped["game_model"])
    index = np.arange(n_models)
    bar_width = 0.35

    plt.bar(index - bar_width/2, grouped["avg_attacker_payoff"], bar_width,
            yerr=grouped["std_attacker_payoff"], label="Attacker Payoff", capsize=5, color='skyblue', ecolor='black')
    plt.bar(index + bar_width/2, grouped["avg_defender_payoff"], bar_width,
            yerr=grouped["std_defender_payoff"], label="Defender Payoff", capsize=5, color='lightcoral', ecolor='black')

    plt.title("Average Payoffs by Game Model (with Std Dev)")
    plt.ylabel("Average Payoff")
    plt.xlabel("Game Model")
    plt.xticks(index, grouped["game_model"], rotation=45, ha="right")
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_network_health_by_model(df, filename="network_health_by_model_with_error.png"):
    """Plots final network health grouped by game model with error bars (std)."""
    if df is None or df.empty:
        print("No data to plot for network health by model.")
        return

    grouped = df.groupby("game_model").agg(
        avg_health=('final_network_health', 'mean'),
        std_health=('final_network_health', 'std')
    ).reset_index()

    grouped = grouped.fillna(0)

    if grouped.empty:
        print("No grouped data to plot for network health by model.")
        return

    plt.figure(figsize=(10, 6))
    plt.bar(grouped["game_model"], grouped["avg_health"],
            yerr=grouped["std_health"], capsize=5, color=sns.color_palette("viridis", len(grouped)), ecolor='black')

    plt.title("Average Final Network Health by Game Model (with Std Dev)")
    plt.ylabel("Average Network Health")
    plt.xlabel("Game Model")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_payoffs_vs_nodes(df, game_model_filter, topology_filter, filename_prefix="payoffs_vs_nodes"):
    """Plots average payoffs vs. number of nodes for a specific game model and topology."""
    if df is None or df.empty:
        print(f"No data to plot for payoffs vs nodes ({game_model_filter}, {topology_filter}).")
        return

    filtered_df = df[(df["game_model"] == game_model_filter) & (df["topology"] == topology_filter)]

    if filtered_df.empty:
        print(f"No data after filtering for {game_model_filter} and {topology_filter}.")
        return

    # Group by num_nodes and calculate mean and std for error bars
    grouped = filtered_df.groupby("num_nodes").agg(
        avg_attacker_payoff=('avg_attacker_payoff', 'mean'),
        std_attacker_payoff=('avg_attacker_payoff', 'std'),
        avg_defender_payoff=('avg_defender_payoff', 'mean'),
        std_defender_payoff=('avg_defender_payoff', 'std')
    ).reset_index().fillna(0)


    if grouped.empty:
        print(f"No grouped data for payoffs vs nodes ({game_model_filter}, {topology_filter}).")
        return

    plt.figure(figsize=(10, 6))
    plt.errorbar(grouped["num_nodes"], grouped["avg_attacker_payoff"], yerr=grouped["std_attacker_payoff"], marker='o', label="Attacker Payoff", capsize=3)
    plt.errorbar(grouped["num_nodes"], grouped["avg_defender_payoff"], yerr=grouped["std_defender_payoff"], marker='x', label="Defender Payoff", capsize=3)
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
    """Plots strategy frequencies for attacker or defender, optionally filtered by game model."""
    if df is None or df.empty:
        print(f"No data to plot for {player_type} strategy frequencies.")
        return

    df_filtered = df.copy()
    title_suffix = " (All Models Combined)"
    filename_suffix = "_all_models"

    if game_model_filter:
        df_filtered = df[df["game_model"] == game_model_filter].copy()
        title_suffix = f" (Model: {game_model_filter})"
        filename_suffix = f"_{game_model_filter.replace(' ','_')}"

    if df_filtered.empty:
        print(f"No data after filtering for {player_type} strategy frequencies{title_suffix}.")
        return

    freq_cols = [col for col in df_filtered.columns if col.startswith(f"{player_type}_freq_")]
    if not freq_cols:
        print(f"No {player_type} frequency columns found (e.g., '{player_type}_freq_strategyname').")
        return

    # Calculate average frequency per strategy across the filtered set of experiments/trials
    # The columns store counts per trial. We need to sum these counts then average if we group by (model, topo, nodes).
    # For an overall plot, we can sum all counts.
    strategy_total_counts = df_filtered[freq_cols].sum() # Sum of counts for each strategy across all trials in the filter
    
    if strategy_total_counts.empty or strategy_total_counts.sum() == 0:
        print(f"No strategy count data for {player_type}{title_suffix}.")
        return

    plt.figure(figsize=(12, 7))
    # Clean up labels for plotting
    strategy_total_counts.rename(index=lambda x: x.replace(f"{player_type}_freq_", "").replace("_", " ").title()).plot(kind="bar", color=sns.color_palette("pastel"))
    plt.title(f"{player_type.capitalize()} Strategy Usage Frequencies{title_suffix}")
    plt.ylabel("Total Times Chosen (Across Trials in Filtered Set)")
    plt.xlabel("Strategy")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()

    filename = f"{filename_prefix}_{player_type}{filename_suffix}.png"
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_convergence(detailed_history_file, player_payoff_col, window_size=10, filename_suffix="convergence"):
    """Plots rolling average of payoffs over steps from a detailed history file."""
    try:
        detail_df = pd.read_csv(detailed_history_file)
    except FileNotFoundError:
        print(f"Detailed history file not found: {detailed_history_file}")
        return
    except Exception as e:
        print(f"Error loading detailed history {detailed_history_file}: {e}")
        return

    if player_payoff_col not in detail_df.columns:
        print(f"Payoff column '{player_payoff_col}' not found in {detailed_history_file}.")
        return

    if detail_df.empty or len(detail_df) < window_size:
        print(f"Not enough data in {detailed_history_file} for rolling average with window {window_size}.")
        return

    detail_df[player_payoff_col] = pd.to_numeric(detail_df[player_payoff_col], errors='coerce')
    rolling_avg = detail_df[player_payoff_col].rolling(window=window_size, min_periods=1).mean() # min_periods=1 to show early values

    plt.figure(figsize=(12, 6))
    plt.plot(detail_df["step"], detail_df[player_payoff_col], label=f"Raw {player_payoff_col.replace('_', ' ').title()}", alpha=0.3, linestyle=':')
    plt.plot(detail_df["step"], rolling_avg, label=f"Rolling Avg (w={window_size}) {player_payoff_col.replace('_', ' ').title()}", color='red')
    
    file_basename = os.path.splitext(os.path.basename(detailed_history_file))[0].replace("_details", "")
    plt.title(f"Payoff Convergence: {player_payoff_col.replace('_', ' ').title()}\n({file_basename})")
    plt.xlabel("Simulation Step")
    plt.ylabel("Payoff")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_filename = f"{file_basename}_{player_payoff_col}_{filename_suffix}.png"
    save_path = os.path.join(PLOTS_DIR, save_filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved convergence plot: {save_path}")

def perform_statistical_tests(df):
    """Performs and prints results of statistical tests."""
    if df is None or df.empty:
        print("No data for statistical tests.")
        return

    print("\n--- Performing Statistical Tests ---")

    # T-test: Defender Payoff - Q-Learning vs Static
    q_learning_def_payoffs = df[df["game_model"] == "Q-Learning"]["avg_defender_payoff"].dropna()
    static_def_payoffs = df[df["game_model"] == "Static"]["avg_defender_payoff"].dropna()

    if len(q_learning_def_payoffs) > 1 and len(static_def_payoffs) > 1:
        stat, pval = ttest_ind(q_learning_def_payoffs, static_def_payoffs, equal_var=False) # Welch's t-test
        print(f"\nT-test (Defender Payoff): Q-Learning vs Static")
        print(f"  Q-Learning Mean: {q_learning_def_payoffs.mean():.2f}, Static Mean: {static_def_payoffs.mean():.2f}")
        print(f"  Statistic={stat:.3f}, p-value={pval:.4f}")
        if pval < 0.05: print("  Result: Statistically significant difference.")
        else: print("  Result: No statistically significant difference.")
    else:
        print("\nNot enough data for T-test: Q-Learning vs Static (Defender Payoff).")

    # ANOVA: Final Network Health across specified game models
    models_for_anova_health = ["Q-Learning", "Bayesian Game", "Static", "Coalition Formation"]
    health_data_for_anova = []
    valid_models_for_anova = []

    for model_name in models_for_anova_health:
        model_health_data = df[df["game_model"] == model_name]["final_network_health"].dropna()
        if len(model_health_data) >= 2: # Each group needs at least 2 samples for ANOVA
            health_data_for_anova.append(model_health_data)
            valid_models_for_anova.append(model_name)
    
    if len(health_data_for_anova) >= 2: # Need at least 2 groups for ANOVA
        stat_anova, pval_anova = f_oneway(*health_data_for_anova)
        print(f"\nANOVA (Final Network Health) across models: {', '.join(valid_models_for_anova)}")
        for i, model_name in enumerate(valid_models_for_anova):
            print(f"  {model_name} Mean Health: {health_data_for_anova[i].mean():.2f}")
        print(f"  F-statistic={stat_anova:.3f}, p-value={pval_anova:.4f}")
        if pval_anova < 0.05:
            print("  Result: Statistically significant difference exists between at least two models.")
            # Note: Post-hoc tests (e.g., Tukey's HSD) would be needed to find specific pair differences.
        else:
            print("  Result: No statistically significant overall difference between these models.")
    else:
        print("\nNot enough groups/data for ANOVA on Final Network Health across specified models.")
            
    print("\n--- Statistical Tests Complete ---")


# --- Main plotting and analysis logic ---
if __name__ == "__main__":
    summary_file_path = os.path.join(RESULTS_DIR, SUMMARY_FILE)
    data = load_data(summary_file_path)

    if data is not None:
        print("Generating plots...")
        plot_avg_payoffs_by_model(data)
        plot_network_health_by_model(data)

        # Example: Plot payoffs vs nodes for specific configurations
        # Loop through unique combinations or define specific ones
        unique_models_topologies = data[["game_model", "topology"]].drop_duplicates()
        for index, row in unique_models_topologies.iterrows():
            gm = row["game_model"]
            tp = row["topology"]
            if not data[(data["game_model"] == gm) & (data["topology"] == tp)].empty:
                plot_payoffs_vs_nodes(data, game_model_filter=gm, topology_filter=tp)
            else:
                print(f"Skipping {gm}/{tp} payoff vs nodes plot - no data.")
        
        # Plot overall strategy frequencies
        plot_strategy_frequencies(data, player_type="atk", filename_prefix="overall_atk_strategy_freq")
        plot_strategy_frequencies(data, player_type="def", filename_prefix="overall_def_strategy_freq")

        # Plot strategy frequencies for each game model
        for model_name in data["game_model"].unique():
            plot_strategy_frequencies(data, player_type="atk", game_model_filter=model_name, filename_prefix=f"{model_name.replace(' ','_')}_atk_strategy_freq")
            plot_strategy_frequencies(data, player_type="def", game_model_filter=model_name, filename_prefix=f"{model_name.replace(' ','_')}_def_strategy_freq")

        # --- Convergence Plots (Example) ---
        # This requires detailed history files to be saved by run_experiments.py
        print("\n--- Generating Convergence Plots (Example for Q-Learning) ---")
        detailed_history_dir = os.path.join(RESULTS_DIR, DETAILED_HISTORIES_SUBDIR)
        if os.path.exists(detailed_history_dir):
            # Try to find some Q-Learning detailed files
            found_detailed_files = False
            for f_name in os.listdir(detailed_history_dir):
                if "Q-Learning" in f_name and f_name.endswith("_details.csv"): # Adjust if naming is different
                    example_detail_file = os.path.join(detailed_history_dir, f_name)
                    print(f"Processing convergence for: {example_detail_file}")
                    plot_convergence(example_detail_file, player_payoff_col='atk_payoff', window_size=10, filename_suffix="convergence")
                    plot_convergence(example_detail_file, player_payoff_col='def_payoff', window_size=10, filename_suffix="convergence")
                    found_detailed_files = True
                    break # Just plot for one example file to keep it quick, remove break for all
            if not found_detailed_files:
                print(f"No Q-Learning detailed history files found in {detailed_history_dir} matching pattern.")
        else:
            print(f"Detailed history directory not found: {detailed_history_dir}. Skipping convergence plots.")

        print("\nPlotting complete. Check the 'plots' directory.")

        # --- Perform Statistical Tests ---
        perform_statistical_tests(data)

    else:
        print("Could not load data. No plots or statistical analysis will be performed.")