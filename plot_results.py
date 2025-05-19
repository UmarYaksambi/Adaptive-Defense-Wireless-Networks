# plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker # For formatting ticks
import seaborn as sns
import numpy as np
import os
from scipy.stats import ttest_ind, f_oneway
import glob # To find result files
import re # Import regex for robust filename cleaning

# --- Configuration ---
# RESULTS_DIR is now a fixed directory where run_experiments.py saves data (assuming it doesn't use timestamp)
RESULTS_DIR = "results"
# PLOTS_DIR is now a fixed directory in the current working directory
PLOTS_DIR = "plots"
DETAILED_HISTORIES_SUBDIR = "detailed_histories_consolidated" # Matches run_experiments.py

# Ensure plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"Plots will be saved in: {PLOTS_DIR}")


# --- Helper for robust filename cleaning ---
def clean_filename(filename):
    """Removes or replaces characters that might be invalid in filenames."""
    # Replace spaces with underscores
    filename = filename.replace(" ", "_")
    # Remove characters that are often problematic in filenames (\ / : * ? " < > |)
    filename = re.sub(r'[\\/:*?"<>|]', '', filename)
    # Replace periods, except the last one before the extension
    parts = filename.rsplit('.', 1)
    if len(parts) > 1:
        parts[0] = parts[0].replace('.', '')
        filename = '.'.join(parts)
    else:
        filename = filename.replace('.', '')
    return filename


def load_data_from_multiple_interval_logs(results_dir_path, file_pattern="*_interval_log.csv"):
    """Loads and concatenates data from multiple interval log CSV files."""
    # Ensure the results directory exists
    if not os.path.exists(results_dir_path):
        print(f"Results directory not found: '{results_dir_path}'")
        print("Please ensure run_experiments.py saved data to this directory.")
        return None

    all_files = glob.glob(os.path.join(results_dir_path, file_pattern))
    if not all_files:
        print(f"No interval log files found in '{results_dir_path}' with pattern '{file_pattern}'.")
        return None

    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception as e:
            print(f"Error loading data from {f}: {e}")
            continue

    if not df_list:
        print("No data loaded from any interval log files.")
        return None

    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(combined_df)} rows from {len(all_files)} interval log files.")

    # 🐛 Debugging: Print columns to check for 'atk_q_init_val'
    # print("\nDataFrame Columns after loading:")
    # print(combined_df.columns.tolist())
    # print("-" * 20)


    # Ensure numeric types for relevant columns
    cols_to_numeric = ['interval_avg_attacker_payoff', 'interval_avg_defender_payoff',
                       'current_network_health', 'interval_detection_rate',
                       'logged_at_step', 'num_nodes',
                       'atk_alpha', 'atk_gamma', 'atk_epsilon_start', 'atk_epsilon_decay', 'atk_epsilon_min', 'atk_hybrid_static_steps', 'current_atk_epsilon', 'atk_q_init_val', # Added atk_q_init_val
                       'def_alpha', 'def_gamma', 'def_epsilon_start', 'def_epsilon_decay', 'def_epsilon_min', 'def_hybrid_static_steps', 'current_def_epsilon', 'def_q_init_val', # Added def_q_init_val
                       'topo_avg_degree', 'topo_density', 'topo_avg_clustering_coefficient', 'topo_diameter' # Include topology metrics
                       ]
    # Add strategy frequency columns dynamically
    freq_cols = [col for col in combined_df.columns if col.startswith('interval_atk_freq_') or col.startswith('interval_def_freq_')]
    cols_to_numeric.extend(freq_cols)


    for col in cols_to_numeric:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

    # Convert connectivity_param to string to handle tuples/floats consistently
    if 'connectivity_param' in combined_df.columns:
        combined_df['connectivity_param'] = combined_df['connectivity_param'].astype(str)

    return combined_df


def plot_payoff_convergence_over_steps(df, plots_dir, # Pass plots_dir
                                       attacker_model_filter=None, defender_model_filter=None,
                                       topology_filter=None, node_count_filter=None, conn_param_filter=None,
                                       attacker_param_filters=None, defender_param_filters=None,
                                       player_type='defender', window_size=5, filename_prefix="payoff_convergence"):
    """
    Plots rolling average of payoffs over 'logged_at_step' from interval logs
    for a specific attacker-defender model matchup and network configuration.
    """
    if df is None or df.empty:
        print("No data for payoff convergence plot.")
        return

    payoff_col = f'interval_avg_{player_type}_payoff'
    if payoff_col not in df.columns:
        print(f"Payoff column '{payoff_col}' not found in DataFrame.")
        return

    df_filtered = df.copy()
    title_parts = [f"{player_type.capitalize()} Payoff Convergence"]
    filename_parts = [filename_prefix, player_type]

    # Apply filters for attacker and defender models
    if attacker_model_filter:
        df_filtered = df_filtered[df_filtered['attacker_model'] == attacker_model_filter]
        title_parts.append(f"Attacker: {attacker_model_filter}")
        filename_parts.append(f"atk_{clean_filename(attacker_model_filter)}") # Clean filename part
    if defender_model_filter:
        df_filtered = df_filtered[df_filtered['defender_model'] == defender_model_filter]
        title_parts.append(f"Defender: {defender_model_filter}")
        filename_parts.append(f"def_{clean_filename(defender_model_filter)}") # Clean filename part

    # Apply network configuration filters
    if topology_filter:
        df_filtered = df_filtered[df_filtered['topology'] == topology_filter]
        title_parts.append(f"Topo: {topology_filter}")
        filename_parts.append(clean_filename(topology_filter)) # Clean filename part
    if node_count_filter is not None:
         df_filtered = df_filtered[df_filtered['num_nodes'] == node_count_filter]
         title_parts.append(f"N: {node_count_filter}")
         filename_parts.append(f"N{node_count_filter}")
    if conn_param_filter is not None:
         # Ensure connectivity_param is treated as string for comparison
         df_filtered = df_filtered[df_filtered['connectivity_param'] == str(conn_param_filter)]
         title_parts.append(f"Conn: {conn_param_filter}")
         filename_parts.append(f"C{clean_filename(str(conn_param_filter))}") # Clean filename part


    # Apply learning parameter filters for attacker (if Q-Learning)
    if attacker_param_filters and attacker_model_filter == "Q-Learning":
        for col, val in attacker_param_filters.items():
            param_col_name = f"atk_{col}" # Parameter column name in DataFrame
            if param_col_name in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[param_col_name] == val]
                title_parts.append(f"Atk {col.replace('_',' ').title()}: {val}")
                filename_parts.append(f"atk{clean_filename(col)}{str(val).replace('.','p')}") # Clean filename part
            else:
                print(f"Warning: Attacker filter column '{param_col_name}' not found in data for convergence plot.")

    # Apply learning parameter filters for defender (if Q-Learning)
    if defender_param_filters and defender_model_filter == "Q-Learning":
        for col, val in defender_param_filters.items():
            param_col_name = f"def_{col}" # Parameter column name in DataFrame
            if param_col_name in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[param_col_name] == val]
                title_parts.append(f"Def {col.replace('_',' ').title()}: {val}")
                filename_parts.append(f"def{clean_filename(col)}{str(val).replace('.','p')}") # Clean filename part
            else:
                print(f"Warning: Defender filter column '{param_col_name}' not found in data for convergence plot.")


    if df_filtered.empty:
        # print(f"No data after filtering for payoff convergence plot. Filters: Atk={attacker_model_filter}, Def={defender_model_filter}, Topo={topology_filter}, N={node_count_filter}, Conn={conn_param_filter}, AtkParams={attacker_param_filters}, DefParams={defender_param_filters}")
        return # Don't print warning, many combinations might be empty by design


    plt.figure(figsize=(14, 7))

    # Calculate mean and std dev of payoff at each 'logged_at_step' across trials for the filtered config
    # Group by 'logged_at_step' and average across trials for this specific configuration
    grouped_by_step = df_filtered.groupby('logged_at_step')

    mean_payoffs = grouped_by_step[payoff_col].mean()
    std_payoffs = grouped_by_step[payoff_col].std().fillna(0) # Fill NaN if only one trial data for a step
    rolling_avg_mean = mean_payoffs.rolling(window=window_size, min_periods=1).mean()

    if mean_payoffs.empty:
        # print(f"Mean payoffs are empty after grouping for convergence plot. Filters: Atk={attacker_model_filter}, Def={defender_model_filter}, Topo={topology_filter}, N={node_count_filter}, Conn={conn_param_filter}, AtkParams={attacker_param_filters}, DefParams={defender_param_filters}")
        return


    steps_x = mean_payoffs.index
    plt.plot(steps_x, mean_payoffs, label=f"Mean {player_type.capitalize()} Payoff (Raw)", alpha=0.4, linestyle=':')
    plt.plot(steps_x, rolling_avg_mean, label=f"Rolling Avg (w={window_size})", color='red', linewidth=2)
    plt.fill_between(steps_x, mean_payoffs - std_payoffs, mean_payoffs + std_payoffs, alpha=0.2, label="Std Dev Across Trials")

    plt.title("\n".join(title_parts))
    plt.xlabel("Simulation Step (Logged At)")
    plt.ylabel(f"Average {player_type.capitalize()} Payoff")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    short_filename_parts = [
        filename_prefix,
        player_type,
        f"atk_{clean_filename(attacker_model_filter)}" if attacker_model_filter else "",
        f"def_{clean_filename(defender_model_filter)}" if defender_model_filter else "",
        f"topo_{clean_filename(topology_filter)}" if topology_filter else "",
        f"N{node_count_filter}" if node_count_filter is not None else "",
        f"C{clean_filename(str(conn_param_filter))}" if conn_param_filter is not None else "",
    ]
    # Remove empty parts and join
    short_filename = "_".join([p for p in short_filename_parts if p]) + ".png"
    save_path = os.path.join(plots_dir, short_filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved convergence plot: {save_path}")


# 🔹 Step 3: Modify comparison plots to use attacker_model and defender_model
def plot_metric_comparison_boxplots(df, plots_dir, # Pass plots_dir
                                    metric_col, x_axis_col="attacker_model", hue_col="defender_model",
                                    filters=None, filename="metric_boxplot.png", title_suffix=""):
    """
    Plots boxplots of a metric, grouped by x_axis_col (e.g., attacker_model)
    with hue by hue_col (e.g., defender_model).
    """
    if df is None or df.empty or metric_col not in df.columns:
        print(f"No data or metric column '{metric_col}' for boxplot.")
        return

    df_filtered = df.copy()
    if filters:
        for col, val in filters.items():
            if col in df_filtered.columns:
                # Ensure connectivity_param is handled as string filter
                if col == 'connectivity_param':
                    df_filtered = df_filtered[df_filtered[col] == str(val)]
                else:
                    df_filtered = df_filtered[df_filtered[col] == val]
            else:
                print(f"Warning: Filter column '{col}' not found for boxplot.")

    if df_filtered.empty:
        # print(f"No data after filtering for boxplot of '{metric_col}'. Filters: {filters}")
        return


    # Use data from the *last* logged step for each trial for overall performance metrics
    if 'logged_at_step' in df_filtered.columns:
        # Find the max step for each unique trial (identified by seed)
        last_steps = df_filtered.groupby('seed')['logged_at_step'].transform('max')
        df_plot = df_filtered[df_filtered['logged_at_step'] == last_steps].copy() # Use .copy() to avoid SettingWithCopyWarning
    else:
        df_plot = df_filtered.copy() # Use as is if no logged_at_step


    if df_plot.empty:
        # print(f"No data for the last logged step for boxplot of '{metric_col}'.")
        return

    plt.figure(figsize=(12, 7))
    sns.boxplot(x=x_axis_col, y=metric_col, hue=hue_col, data=df_plot, palette="Set2")

    clean_metric_name = metric_col.replace("_", " ").replace("interval avg", "Final Avg").replace("current network", "Final Network").replace("interval detection", "Final Detection").title()
    clean_x_axis_name = x_axis_col.replace("_", " ").title()
    clean_hue_name = hue_col.replace("_", " ").title() if hue_col else ""

    title = f"Distribution of {clean_metric_name} by {clean_x_axis_name}"
    if hue_col: title += f" (Hue: {clean_hue_name})"
    if title_suffix: title += f"\n{title_suffix}"

    plt.title(title)
    plt.ylabel(clean_metric_name)
    plt.xlabel(clean_x_axis_name)
    plt.xticks(rotation=45, ha="right")
    if "health" in metric_col.lower(): plt.ylim(0, 1.05) # Health typically 0-1
    if "rate" in metric_col.lower(): plt.ylim(0, 1.05) # Rates typically 0-1
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Construct filename based on filters and columns
    filter_suffix = "_".join([f"{clean_filename(k)}{clean_filename(str(v)).replace('.','p')}" for k,v in filters.items()]) if filters else ""
    filename_parts = [clean_filename(metric_col), clean_filename(x_axis_col)]
    if hue_col: filename_parts.append(f"hue{clean_filename(hue_col)}")
    if filter_suffix: filename_parts.append(f"filt{filter_suffix}")

    final_filename = "_".join(filename_parts) + "_boxplot.png"

    # Save to the plots_dir passed to the function - Explicitly join
    save_path = os.path.join(plots_dir, final_filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved boxplot: {save_path}")


# 🔹 Step 3: Modify strategy frequency heatmap to group by attacker/defender models
def plot_strategy_frequency_heatmap(df, plots_dir, # Pass plots_dir
                                    player_type, filters=None,
                                    filename="strategy_heatmap.png", title_suffix=""):
    """
    Plots a heatmap of strategy frequencies (average percentage) grouped by
    attacker_model and defender_model.
    """
    if df is None or df.empty:
        print(f"No data for strategy heatmap.")
        return

    df_filtered = df.copy()
    if filters:
        for col, val in filters.items():
            if col in df_filtered.columns:
                 # Ensure connectivity_param is handled as string filter
                if col == 'connectivity_param':
                    df_filtered = df_filtered[df_filtered[col] == str(val)]
                else:
                    df_filtered = df_filtered[df_filtered[col] == val]

    if df_filtered.empty:
        # print(f"No data after filtering for strategy heatmap. Filters: {filters}")
        return

    freq_cols_prefix = f"interval_{player_type}_freq_"
    # Dynamically find strategy frequency columns
    strat_cols = [col for col in df_filtered.columns if col.startswith(freq_cols_prefix)]
    if not strat_cols:
        print(f"No strategy frequency columns found with prefix '{freq_cols_prefix}'.")
        return

    # Consider data from the *last* logged interval for final strategy distribution
    if 'logged_at_step' in df_filtered.columns:
        last_steps = df_filtered.groupby('seed')['logged_at_step'].transform('max')
        df_agg = df_filtered[df_filtered['logged_at_step'] == last_steps].copy() # Use .copy()
    else:
        df_agg = df_filtered.copy()


    if df_agg.empty:
        # print(f"No data for the last logged step for strategy heatmap.")
        return

    # Calculate total plays per row (trial) for the last interval to normalize frequencies
    # This assumes the interval_freq columns contain counts for the LAST interval
    # If they contain cumulative counts, the normalization logic needs adjustment.
    # Based on run_experiments.py, they are counts for the LAST interval.
    df_agg['total_interval_plays'] = df_agg[strat_cols].sum(axis=1)

    # Normalize each strategy count column by total_interval_plays to get percentage for that interval
    for col in strat_cols:
        df_agg[col] = (df_agg[col] / df_agg['total_interval_plays']) * 100 # Percentage
    df_agg = df_agg.fillna(0) # Fill NaN if total_interval_plays was 0

    # Group by attacker_model and defender_model and average these percentages across trials
    heatmap_data = df_agg.groupby(["attacker_model", "defender_model"])[strat_cols].mean()

    if heatmap_data.empty:
        # print("No data for heatmap after aggregation.")
        return

    # Clean up column names for the heatmap
    heatmap_data.columns = [col.replace(freq_cols_prefix, "").replace("_", " ").title() for col in heatmap_data.columns]

    # Reshape data for heatmap: index=defender_model, columns=attacker_model
    # The data is currently indexed by (attacker_model, defender_model) MultiIndex
    # We need to pivot it.
    # Let's create a separate heatmap for each metric (avg payoff, health, detection rate)
    # with attacker models on one axis and defender models on the other.
    # This function is for STRATEGY FREQUENCIES.
    # Let's pivot the strategy frequency data to have attacker models as columns and defender models as rows (or vice versa)
    # For strategy frequency, it makes more sense to show the frequency of each strategy *within* a specific matchup.
    # A single heatmap showing strategy frequencies across ALL matchups might be too busy.
    # Instead, let's generate a separate heatmap for *each* matchup, showing the frequency of strategies used *within* that matchup.
    # This requires iterating through unique (attacker_model, defender_model) pairs.

    # The current `heatmap_data` is the average frequency for each strategy *for each matchup*.
    # We need to decide how to visualize this matrix.
    # Option A: A single heatmap showing avg frequency of ONE specific strategy across matchups. (e.g., Avg freq of 'broadband' attack vs each defender). Requires multiple heatmaps (one per strategy).
    # Option B: A single heatmap showing the distribution of strategies *within* a matchup. This is what the current `heatmap_data` represents, but needs reshaping.

    # Let's stick to the original idea of showing strategy frequencies *by game model*.
    # The current `heatmap_data` is grouped by (attacker_model, defender_model).
    # We need to pivot this to get attacker models on one axis and defender models on the other,
    # with the *value* in the cell being the average frequency of a *specific* strategy.
    # This still leads to multiple heatmaps (one per strategy).

    # Alternative: Create a heatmap of AVERAGE METRICS (payoff, health, detection rate)
    # with attacker models on one axis and defender models on the other. This is the core matchup matrix visualization.
    # Let's create a separate function for that.

    # For this `plot_strategy_frequency_heatmap` function, let's generate heatmaps showing the
    # average frequency of strategies *for a specific matchup*.
    # We need to group by attacker_model and defender_model first.

    unique_matchups = df_agg[['attacker_model', 'defender_model']].drop_duplicates().values.tolist()

    for atk_m, def_m in unique_matchups:
        matchup_df = df_agg[(df_agg['attacker_model'] == atk_m) & (df_agg['defender_model'] == def_m)].copy()

        if matchup_df.empty: continue

        # Calculate average strategy frequencies for this specific matchup
        avg_freqs = matchup_df[strat_cols].mean().to_frame(name='Average Frequency (%)')
        avg_freqs.index = [col.replace(freq_cols_prefix, "").replace("_", " ").title() for col in avg_freqs.index]
        avg_freqs = avg_freqs.sort_values(by='Average Frequency (%)', ascending=False) # Sort for better visualization

        plt.figure(figsize=(8, max(5, len(avg_freqs)*0.6))) # Adjust size
        sns.heatmap(avg_freqs, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5, cbar_kws={'label': 'Avg. % Frequency'})

        title = f"{player_type.capitalize()} Strategy Usage Frequency (%)\n{atk_m} Attacker vs {def_m} Defender"
        if title_suffix: title += f"\n{title_suffix}"
        plt.title(title)
        plt.xlabel("Strategy")
        plt.ylabel("") # Strategy names are the y-axis labels
        plt.xticks(rotation=0) # Strategy names are on Y, no need to rotate X
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Construct filename for this specific matchup heatmap
        matchup_filename = f"{player_type}_freq_{clean_filename(atk_m)}_vs_{clean_filename(def_m)}"
        filter_suffix_clean = "_".join([f"{clean_filename(k)}{clean_filename(str(v)).replace('.','p')}" for k,v in filters.items() if k not in ['attacker_model', 'defender_model']]) if filters else ""
        if filter_suffix_clean: matchup_filename += f"_filt{filter_suffix_clean}"

        # Save to the plots_dir passed to the function - Explicitly join
        save_path = os.path.join(plots_dir, f"{matchup_filename}_heatmap.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved strategy frequency heatmap: {save_path}")


# 🔹 Step 3: New function to plot the core Matchup Matrix Heatmaps (Avg Payoff, Health, Detection Rate)
def plot_matchup_matrix_heatmap(df, plots_dir, # Pass plots_dir
                                metric_col, filters=None,
                                filename="matchup_matrix_heatmap.png", title_suffix=""):
    """
    Plots a heatmap of the average of a metric across all attacker_model vs defender_model matchups.
    """
    if df is None or df.empty or metric_col not in df.columns:
        print(f"No data or metric column '{metric_col}' for matchup matrix heatmap.")
        return

    df_filtered = df.copy()
    if filters:
        for col, val in filters.items():
            if col in df_filtered.columns:
                 # Ensure connectivity_param is handled as string filter
                if col == 'connectivity_param':
                    df_filtered = df_filtered[df_filtered[col] == str(val)]
                else:
                    df_filtered = df_filtered[df_filtered[col] == val]
            else:
                print(f"Warning: Filter column '{col}' not found for matchup matrix heatmap.")

    if df_filtered.empty:
        # print(f"No data after filtering for matchup matrix heatmap of '{metric_col}'. Filters: {filters}")
        return

    # Use data from the *last* logged step for each trial
    if 'logged_at_step' in df_filtered.columns:
        last_steps = df_filtered.groupby('seed')['logged_at_step'].transform('max')
        df_agg = df_filtered[df_filtered['logged_at_step'] == last_steps].copy() # Use .copy()
    else:
        df_agg = df_filtered.copy()

    if df_agg.empty:
        # print(f"No data for the last logged step for matchup matrix heatmap of '{metric_col}'.")
        return


    # Group by attacker_model and defender_model and calculate the mean metric
    matchup_avg_metric = df_agg.groupby(["attacker_model", "defender_model"])[metric_col].mean().reset_index()

    if matchup_avg_metric.empty:
        # print(f"No grouped data for matchup matrix heatmap of '{metric_col}'.")
        return

    # Pivot the data to create the heatmap matrix
    heatmap_matrix = matchup_avg_metric.pivot(index="defender_model", columns="attacker_model", values=metric_col)

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_matrix, annot=True, fmt=".2f", cmap="viridis", linewidths=.5) # Use viridis or similar for metrics

    clean_metric_name = metric_col.replace("_", " ").replace("interval avg", "Final Avg").replace("current network", "Final Network").replace("interval detection", "Final Detection").title()

    title = f"Average {clean_metric_name} Across Attacker vs Defender Matchups"
    if title_suffix: title += f"\n{title_suffix}"

    plt.title(title)
    plt.xlabel("Attacker Model")
    plt.ylabel("Defender Model")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Construct filename based on filters and metric
    filter_suffix = "_".join([f"{clean_filename(k)}{clean_filename(str(v)).replace('.','p')}" for k,v in filters.items()]) if filters else ""
    filename_parts = [clean_filename(metric_col)]
    if filter_suffix: filename_parts.append(f"filt{filter_suffix}")

    final_filename = "_".join(filename_parts) + "_matchup_matrix_heatmap.png"

    # Save to the plots_dir passed to the function - Explicitly join
    save_path = os.path.join(plots_dir, final_filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved matchup matrix heatmap: {save_path}")


def perform_statistical_tests(df, last_step_only=True):
    """
    Performs and prints results of statistical tests comparing metrics across different
    attacker-defender model matchups.
    """
    if df is None or df.empty:
        print("No data for statistical tests.")
        return

    df_test = df.copy()
    if last_step_only and 'logged_at_step' in df.columns:
        last_steps = df_test.groupby('seed')['logged_at_step'].transform('max')
        df_test = df_test[df_test['logged_at_step'] == last_steps].copy() # Use .copy()
        print(f"\n--- Performing Statistical Tests (on data from last logged step) ---")
    else:
        print("\n--- Performing Statistical Tests (on all interval data) ---")

    # Define metrics to test
    metrics_to_test = [
        'interval_avg_attacker_payoff',
        'interval_avg_defender_payoff',
        'current_network_health',
        'interval_detection_rate'
    ]

    # Define the primary grouping (matchup combinations)
    df_test['matchup'] = df_test['attacker_model'] + ' vs ' + df_test['defender_model']
    unique_matchups = df_test['matchup'].unique()

    if len(unique_matchups) < 2:
        print("Not enough unique matchups (less than 2) to perform statistical tests.")
        return

    # Perform ANOVA for each metric across all matchups
    print("\n--- ANOVA Tests Across All Matchups ---")
    for metric in metrics_to_test:
        if metric not in df_test.columns:
            print(f"Skipping ANOVA for '{metric}': Column not found.")
            continue

        data_groups = [df_test[df_test['matchup'] == m][metric].dropna() for m in unique_matchups]
        # Filter out groups with insufficient data (less than 2 samples)
        valid_data_groups = [group for group in data_groups if len(group) >= 2]
        valid_matchups = [unique_matchups[i] for i, group in enumerate(data_groups) if len(group) >= 2]

        if len(valid_data_groups) >= 2:
            try:
                stat_anova, pval_anova = f_oneway(*valid_data_groups)
                print(f"\nANOVA ({metric.replace('_',' ').title()}): Across Matchups")
                for i, matchup_desc in enumerate(valid_matchups):
                     print(f"  {matchup_desc} Mean: {valid_data_groups[i].mean():.3f} (n={len(valid_data_groups[i])})")
                print(f"  F-statistic={stat_anova:.3f}, p-value={pval_anova:.4f}")
                if pval_anova < 0.05: print("  Result: Statistically significant difference exists between at least two matchups.")
                else: print("  Result: No statistically significant overall difference across matchups.")
            except ValueError as e:
                print(f"\nCould not perform ANOVA for '{metric}': {e} (likely due to insufficient variance or data)")
        else:
            print(f"\nNot enough valid groups/data for ANOVA on '{metric}'.")


    # You could add pairwise t-tests here if ANOVA shows significance, but that gets very verbose.
    # For a comprehensive analysis, post-hoc tests (like Tukey's HSD) would be needed after ANOVA.
    # This basic function just performs the ANOVA.

    print("\n--- Statistical Tests Complete ---")


# --- Main plotting and analysis logic ---
if __name__ == "__main__":
    # Use the fixed RESULTS_DIR and PLOTS_DIR
    # Ensure plots directory exists
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"Plots will be saved in: {PLOTS_DIR}")
    print(f"Attempting to load data from: {RESULTS_DIR}")


    all_data = load_data_from_multiple_interval_logs(RESULTS_DIR)

    if all_data is not None and not all_data.empty:
        print(f"Data loaded successfully. Total rows: {len(all_data)}")

        # Define common filters for network configuration if needed for specific plots
        # Example: Filter for a specific topology and node count
        common_filters = {
            'topology': 'Random (Erdős–Rényi)', # Or 'Small-World'
            'num_nodes': 10,
            'connectivity_param': '0.4' # Or '(6, 0.1)' as string
        }
        # Ensure connectivity_param is a string in filters if it's a tuple in reality

        # 1. Plot Matchup Matrix Heatmaps for key metrics (Avg Payoff, Health, Detection Rate)
        print("\nGenerating Matchup Matrix Heatmaps...")
        plot_matchup_matrix_heatmap(all_data, PLOTS_DIR, metric_col="interval_avg_attacker_payoff",
                                    filters=common_filters, filename="atk_payoff_matchup_matrix")
        plot_matchup_matrix_heatmap(all_data, PLOTS_DIR, metric_col="interval_avg_defender_payoff",
                                    filters=common_filters, filename="def_payoff_matchup_matrix")
        plot_matchup_matrix_heatmap(all_data, PLOTS_DIR, metric_col="current_network_health",
                                    filters=common_filters, filename="net_health_matchup_matrix")
        plot_matchup_matrix_heatmap(all_data, PLOTS_DIR, metric_col="interval_detection_rate",
                                    filters=common_filters, filename="detection_rate_matchup_matrix")


        # 2. Plot Payoff Convergence for specific matchups (e.g., Q vs Bayesian, Q vs Static)
        # This requires picking specific parameter sets if Q-Learning is involved.
        # For simplicity, let's plot convergence for the first parameter set found for Q-Learning
        # You would adjust these filters to analyze specific parameter tunings.
        print("\nGenerating Convergence Plots for Specific Matchups...")
        q_params_example_atk = {'alpha': 0.1, 'gamma': 0.9, 'epsilon_start': 0.5, 'epsilon_decay': 0.998, 'epsilon_min': 0.01, 'hybrid_static_steps': 0, 'q_init_val': 0.01}
        q_params_example_def = {'alpha': 0.1, 'gamma': 0.9, 'epsilon_start': 0.5, 'epsilon_decay': 0.998, 'epsilon_min': 0.01, 'hybrid_static_steps': 0, 'q_init_val': 0.01}

        # Example convergence plots for a specific network config
        convergence_filters = {
            'topology': 'Random (Erdős–Rényi)',
            'num_nodes': 10,
            'connectivity_param': '0.4'
        }

        # Q-Learning Attacker vs Bayesian Defender
        plot_payoff_convergence_over_steps(all_data, PLOTS_DIR, attacker_model_filter="Q-Learning", defender_model_filter="Bayesian Game",
                                           topology_filter=convergence_filters['topology'], node_count_filter=convergence_filters['num_nodes'],
                                           conn_param_filter=convergence_filters['connectivity_param'],
                                           attacker_param_filters=q_params_example_atk, # Use example Q params for attacker
                                           player_type='defender', filename_prefix="conv_QL_vs_Bayes_def")
        plot_payoff_convergence_over_steps(all_data, PLOTS_DIR, attacker_model_filter="Q-Learning", defender_model_filter="Bayesian Game",
                                           topology_filter=convergence_filters['topology'], node_count_filter=convergence_filters['num_nodes'],
                                           conn_param_filter=convergence_filters['connectivity_param'],
                                           attacker_param_filters=q_params_example_atk, # Use example Q params for attacker
                                           player_type='attacker', filename_prefix="conv_QL_vs_Bayes_atk")

        # Bayesian Attacker vs Q-Learning Defender
        plot_payoff_convergence_over_steps(all_data, PLOTS_DIR, attacker_model_filter="Bayesian Game", defender_model_filter="Q-Learning",
                                           topology_filter=convergence_filters['topology'], node_count_filter=convergence_filters['num_nodes'],
                                           conn_param_filter=convergence_filters['connectivity_param'],
                                           defender_param_filters=q_params_example_def, # Use example Q params for defender
                                           player_type='defender', filename_prefix="conv_Bayes_vs_QL_def")
        plot_payoff_convergence_over_steps(all_data, PLOTS_DIR, attacker_model_filter="Bayesian Game", defender_model_filter="Q-Learning",
                                           topology_filter=convergence_filters['topology'], node_count_filter=convergence_filters['num_nodes'],
                                           conn_param_filter=convergence_filters['connectivity_param'],
                                           defender_param_filters=q_params_example_def, # Use example Q params for defender
                                           player_type='attacker', filename_prefix="conv_Bayes_vs_QL_atk")

        # Q-Learning vs Q-Learning (Self-play convergence)
        plot_payoff_convergence_over_steps(all_data, PLOTS_DIR, attacker_model_filter="Q-Learning", defender_model_filter="Q-Learning",
                                           topology_filter=convergence_filters['topology'], node_count_filter=convergence_filters['num_nodes'],
                                           conn_param_filter=convergence_filters['connectivity_param'],
                                           attacker_param_filters=q_params_example_atk, defender_param_filters=q_params_example_def,
                                           player_type='defender', filename_prefix="conv_QL_vs_QL_def")
        plot_payoff_convergence_over_steps(all_data, PLOTS_DIR, attacker_model_filter="Q-Learning", defender_model_filter="Q-Learning",
                                           topology_filter=convergence_filters['topology'], node_count_filter=convergence_filters['num_nodes'],
                                           conn_param_filter=convergence_filters['connectivity_param'],
                                           attacker_param_filters=q_params_example_atk, defender_param_filters=q_params_example_def,
                                           player_type='attacker', filename_prefix="conv_QL_vs_QL_atk")


        # 3. Box Plots for final performance comparison (grouped by attacker model, hue by defender model)
        print("\nGenerating Boxplots...")
        plot_metric_comparison_boxplots(all_data, PLOTS_DIR, metric_col="interval_avg_attacker_payoff",
                                        x_axis_col="attacker_model", hue_col="defender_model",
                                        filters=common_filters) # Apply common network filters
        plot_metric_comparison_boxplots(all_data, PLOTS_DIR, metric_col="interval_avg_defender_payoff",
                                        x_axis_col="attacker_model", hue_col="defender_model",
                                        filters=common_filters)
        plot_metric_comparison_boxplots(all_data, PLOTS_DIR, metric_col="current_network_health",
                                        x_axis_col="attacker_model", hue_col="defender_model",
                                        filters=common_filters)
        plot_metric_comparison_boxplots(all_data, PLOTS_DIR, metric_col="interval_detection_rate",
                                        x_axis_col="attacker_model", hue_col="defender_model",
                                        filters=common_filters)

        # You can also swap x and hue axes to see defender model on x-axis
        plot_metric_comparison_boxplots(all_data, PLOTS_DIR, metric_col="interval_avg_defender_payoff",
                                        x_axis_col="defender_model", hue_col="attacker_model",
                                        filters=common_filters, filename="def_payoff_boxplot_def_hue_atk.png")


        # 4. Strategy Frequency Heatmaps (per matchup)
        # This function now generates a heatmap for EACH unique matchup found in the data,
        # showing the frequency of strategies used *within* that matchup.
        print("\nGenerating Strategy Frequency Heatmaps (per matchup)...")
        plot_strategy_frequency_heatmap(all_data, PLOTS_DIR, player_type="atk", filters=common_filters)
        plot_strategy_frequency_heatmap(all_data, PLOTS_DIR, player_type="def", filters=common_filters)


        # 5. Perform Statistical Tests
        print("\nPerforming Statistical Tests...")
        perform_statistical_tests(all_data, last_step_only=True)


        print(f"\nPlotting and analysis complete. Check the '{PLOTS_DIR}' directory.")
        print(f"Results directory used: {RESULTS_DIR}")

    else:
        print("Could not load data from interval logs. No plots or analysis will be performed.")
        print(f"Attempted to load from: {RESULTS_DIR}")
