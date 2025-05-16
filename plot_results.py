# plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker # For formatting ticks
import seaborn as sns
import numpy as np
import os
from scipy.stats import ttest_ind, f_oneway
import glob # To find result files

# --- Configuration ---
RESULTS_DIR = "results"  # Matches your run_experiments.py
PLOTS_DIR = "plots" # New plots dir to avoid mixing
DETAILED_HISTORIES_SUBDIR = "detailed_histories_consolidated" # Matches run_experiments.py

# Ensure plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data_from_multiple_interval_logs(results_dir_path, file_pattern="*_interval_log.csv"):
    """Loads and concatenates data from multiple interval log CSV files."""
    all_files = glob.glob(os.path.join(results_dir_path, file_pattern))
    if not all_files:
        print(f"No interval log files found in '{results_dir_path}' with pattern '{file_pattern}'.")
        return None
    
    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            # Extract config from filename if not directly in columns (or add it during generation)
            # For now, assume columns like 'game_model', 'topology', 'alpha', etc. are in the CSV.
            df_list.append(df)
        except Exception as e:
            print(f"Error loading data from {f}: {e}")
            continue
    
    if not df_list:
        print("No data loaded from any interval log files.")
        return None
        
    combined_df = pd.concat(df_list, ignore_index=True)
    # Ensure numeric types
    cols_to_numeric = ['interval_avg_attacker_payoff', 'interval_avg_defender_payoff', 
                       'current_network_health', 'interval_detection_rate', 'current_epsilon',
                       'logged_at_step', 'alpha', 'gamma', 'epsilon_start', 'epsilon_decay', 'epsilon_min', 'hybrid_static_steps']
    for col in cols_to_numeric:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    return combined_df


def plot_payoff_convergence_over_steps(df, game_model_filter=None, topology_filter=None, param_filters=None, 
                                       player_type='defender', window_size=5, filename_prefix="payoff_convergence"):
    """
    Plots rolling average of payoffs over 'logged_at_step' from interval logs.
    Allows filtering by game_model, topology, and other learning parameters.
    player_type: 'attacker' or 'defender'
    param_filters: dict like {'alpha': 0.1, 'hybrid_static_steps': 0}
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

    if game_model_filter:
        df_filtered = df_filtered[df_filtered['game_model'] == game_model_filter]
        title_parts.append(f"Model: {game_model_filter}")
        filename_parts.append(game_model_filter.replace(" ","_"))
    if topology_filter:
        df_filtered = df_filtered[df_filtered['topology'] == topology_filter]
        title_parts.append(f"Topo: {topology_filter}")
        filename_parts.append(topology_filter.replace(" ","_").replace("(","").replace(")","").replace("–",""))
    
    if param_filters: # Apply additional parameter filters for Q-Learning, etc.
        for col, val in param_filters.items():
            if col in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[col] == val]
                title_parts.append(f"{col.replace('_',' ').title()}: {val}")
                filename_parts.append(f"{col.replace('_','')}{str(val).replace('.','p')}")
            else:
                print(f"Warning: Filter column '{col}' not found in data for convergence plot.")


    if df_filtered.empty:
        print(f"No data after filtering for payoff convergence plot. Filters: GM={game_model_filter}, Topo={topology_filter}, Params={param_filters}")
        return

    # Group by 'logged_at_step' and average across trials for the filtered config
    # Also need to ensure we group by the specific configuration (alpha, gamma, etc.) if plotting multiple lines
    
    # For simplicity, if multiple param_filter combinations result from this, this plot might average them.
    # Better to plot one line per specific config or average across trials of ONE config.
    # Let's assume param_filters define a unique configuration, or we average trials for that config.
    
    # Identify unique configurations based on learning parameters if game_model is Q-Learning
    config_cols = []
    if game_model_filter == "Q-Learning": # Only relevant for Q-Learning
        config_cols = ['alpha', 'gamma', 'epsilon_start', 'epsilon_decay', 'epsilon_min', 'hybrid_static_steps']
        config_cols = [c for c in config_cols if c in df_filtered.columns] # Keep only existing columns

    if config_cols: # Plot separate lines for different Q-learning param sets
        grouped_by_config = df_filtered.groupby(config_cols + ['logged_at_step'])
    else: # Single line (or averaged if multiple non-Q-param configs match filters)
        grouped_by_config = df_filtered.groupby(['logged_at_step'])
        
    
    plt.figure(figsize=(14, 7))
    
    # Plot for each configuration if config_cols were used
    if config_cols:
        for name, group_data in grouped_by_config:
            # 'name' will be a tuple (alpha_val, gamma_val, ..., logged_at_step_val) if config_cols are present
            # or just logged_at_step_val if not
            
            # Extract config label from 'name' tuple (excluding the last element which is 'logged_at_step')
            # This is a bit tricky due to how groupby creates 'name'.
            # A more robust way is to iterate unique configs first, then group by step for each.
            
            # Simplified: Calculate mean payoff and std dev for each step across trials
            mean_payoffs = group_data.groupby('logged_at_step')[payoff_col].mean()
            std_payoffs = group_data.groupby('logged_at_step')[payoff_col].std().fillna(0)
            
            # To create a proper legend for different param sets, we need to iterate unique param sets
            # This current loop iterates over (param_set_tuple, logged_at_step)
            # For now, let's just plot one averaged line if there are many configs after filtering
            # Or, if param_filters pinpoint one config, this is fine.
            
            # Let's assume param_filters effectively select ONE specific hyperparameter set for Q-Learning
            # or we are plotting for a non-Q-Learning model.
            # The groupby then averages over trials for that selected config.
            pass # The plotting logic below will handle the already filtered df_filtered

    # Calculate mean and std dev of payoff at each 'logged_at_step' across trials
    mean_payoffs = df_filtered.groupby('logged_at_step')[payoff_col].mean()
    std_payoffs = df_filtered.groupby('logged_at_step')[payoff_col].std().fillna(0) # Fill NaN if only one trial data for a step
    rolling_avg_mean = mean_payoffs.rolling(window=window_size, min_periods=1).mean()

    if mean_payoffs.empty:
        print(f"Mean payoffs are empty after grouping for convergence plot. Filters: GM={game_model_filter}, Topo={topology_filter}, Params={param_filters}")
        return

    steps_x = mean_payoffs.index
    plt.plot(steps_x, mean_payoffs, label=f"Mean {player_type.capitalize()} Payoff (Raw)", alpha=0.4, linestyle=':')
    plt.plot(steps_x, rolling_avg_mean, label=f"Rolling Avg (w={window_size}) of Mean Payoff", color='red', linewidth=2)
    plt.fill_between(steps_x, mean_payoffs - std_payoffs, mean_payoffs + std_payoffs, alpha=0.2, label="Std Dev Across Trials")

    plt.title("\n".join(title_parts))
    plt.xlabel("Simulation Step (Logged At)")
    plt.ylabel(f"Average {player_type.capitalize()} Payoff")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    final_filename = "_".join(filename_parts) + ".png"
    save_path = os.path.join(PLOTS_DIR, final_filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved convergence plot: {save_path}")


def plot_metric_comparison_boxplots(df, metric_col, x_axis_col="game_model", hue_col=None, 
                                    filters=None, filename="metric_boxplot.png", title_suffix=""):
    """Generic function to plot boxplots of a metric, grouped by x_axis_col, optionally with hue."""
    if df is None or df.empty or metric_col not in df.columns:
        print(f"No data or metric column '{metric_col}' for boxplot.")
        return

    df_filtered = df.copy()
    if filters:
        for col, val in filters.items():
            if col in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[col] == val]
            else:
                print(f"Warning: Filter column '{col}' not found for boxplot.")
    
    if df_filtered.empty:
        print(f"No data after filtering for boxplot of '{metric_col}'. Filters: {filters}")
        return

    # Consider only the final logged step for each trial for overall performance metrics
    # Or use all interval data if appropriate for the metric
    # For avg payoffs from interval logs, we might want the value at the *last* logged step.
    if 'logged_at_step' in df_filtered.columns:
        last_step = df_filtered['logged_at_step'].max()
        df_plot = df_filtered[df_filtered['logged_at_step'] == last_step]
    else:
        df_plot = df_filtered # If no logged_at_step, use as is (e.g., from older summary files)

    if df_plot.empty:
        print(f"No data for the last logged step for boxplot of '{metric_col}'.")
        return

    plt.figure(figsize=(12, 7))
    sns.boxplot(x=x_axis_col, y=metric_col, hue=hue_col, data=df_plot, palette="Set2")
    
    clean_metric_name = metric_col.replace("_", " ").replace("interval avg", "Final Avg").replace("current ", "Final ").title()
    clean_x_axis_name = x_axis_col.replace("_", " ").title()
    
    title = f"Distribution of {clean_metric_name} by {clean_x_axis_name}"
    if hue_col: title += f" (Hue: {hue_col.replace('_',' ').title()})"
    if title_suffix: title += f"\n{title_suffix}"

    plt.title(title)
    plt.ylabel(clean_metric_name)
    plt.xlabel(clean_x_axis_name)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved boxplot: {save_path}")


def plot_avg_metric_bar_with_error(df, metric_col, group_by_col="game_model", 
                                   filters=None, filename="avg_metric_bar.png", title_suffix=""):
    """Plots a bar chart of the average of a metric, grouped, with error bars (std dev)."""
    if df is None or df.empty or metric_col not in df.columns:
        print(f"No data or metric column '{metric_col}' for bar plot.")
        return

    df_filtered = df.copy()
    if filters:
        for col, val in filters.items():
            if col in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[col] == val]
            else:
                print(f"Warning: Filter column '{col}' not found for bar plot.")

    if df_filtered.empty:
        print(f"No data after filtering for bar plot of '{metric_col}'. Filters: {filters}")
        return
        
    # Use data from the last logged step for overall performance summary
    if 'logged_at_step' in df_filtered.columns:
        last_step_val = df_filtered['logged_at_step'].max()
        df_agg = df_filtered[df_filtered['logged_at_step'] == last_step_val]
    else:
        df_agg = df_filtered

    if df_agg.empty:
        print(f"No data for the last logged step for bar plot of '{metric_col}'.")
        return

    grouped = df_agg.groupby(group_by_col).agg(
        avg_metric=(metric_col, 'mean'),
        std_metric=(metric_col, 'std')
    ).reset_index().fillna(0)

    if grouped.empty:
        print(f"No grouped data for bar plot of '{metric_col}'.")
        return

    plt.figure(figsize=(10, 6))
    plt.bar(grouped[group_by_col], grouped["avg_metric"],
            yerr=grouped["std_metric"], capsize=5, color=sns.color_palette("pastel", len(grouped)), ecolor='black')

    clean_metric_name = metric_col.replace("_", " ").replace("interval avg", "Final Avg").replace("current ", "Final ").title()
    clean_group_by_name = group_by_col.replace("_", " ").title()
    
    title = f"Average {clean_metric_name} by {clean_group_by_name} (with Std Dev)"
    if title_suffix: title += f"\n{title_suffix}"
    
    plt.title(title)
    plt.ylabel(f"Average {clean_metric_name}")
    plt.xlabel(clean_group_by_name)
    plt.xticks(rotation=45, ha="right")
    if "health" in metric_col.lower(): plt.ylim(0, 1.05) # Health typically 0-1
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved bar plot: {save_path}")


def plot_strategy_frequency_heatmap(df, player_type, filters=None, 
                                    filename="strategy_heatmap.png", title_suffix=""):
    """Plots a heatmap of strategy frequencies (average percentage)."""
    if df is None or df.empty:
        print(f"No data for strategy heatmap.")
        return

    df_filtered = df.copy()
    if filters:
        for col, val in filters.items():
            if col in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[col] == val]

    if df_filtered.empty:
        print(f"No data after filtering for strategy heatmap. Filters: {filters}")
        return

    freq_cols_prefix = f"interval_{player_type}_freq_"
    strat_cols = [col for col in df_filtered.columns if col.startswith(freq_cols_prefix)]
    if not strat_cols:
        print(f"No strategy frequency columns found with prefix '{freq_cols_prefix}'.")
        return
    
    # Consider data from the last logged interval for final strategy distribution
    if 'logged_at_step' in df_filtered.columns:
        last_step_val = df_filtered['logged_at_step'].max()
        df_agg = df_filtered[df_filtered['logged_at_step'] == last_step_val].copy() # Use .copy()
    else:
        df_agg = df_filtered.copy()

    if df_agg.empty:
        print(f"No data for the last logged step for strategy heatmap.")
        return

    # Calculate total plays per row (trial) to normalize frequencies to percentages
    df_agg['total_plays'] = df_agg[strat_cols].sum(axis=1)
    
    # Normalize each strategy column by total_plays
    for col in strat_cols:
        df_agg[col] = (df_agg[col] / df_agg['total_plays']) * 100 # Percentage
    df_agg = df_agg.fillna(0) # Fill NaN if total_plays was 0

    # Group by game_model and average these percentages
    # Other grouping factors like topology could be added
    heatmap_data = df_agg.groupby("game_model")[strat_cols].mean()
    heatmap_data.columns = [col.replace(freq_cols_prefix, "").replace("_", " ").title() for col in heatmap_data.columns]

    if heatmap_data.empty:
        print("No data for heatmap after aggregation.")
        return

    plt.figure(figsize=(12, max(6, len(heatmap_data)*0.8))) # Adjust height based on number of models
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5, cbar_kws={'label': 'Avg. % Frequency'})
    
    title = f"{player_type.capitalize()} Strategy Usage Frequency (%) by Game Model"
    if title_suffix: title += f"\n{title_suffix}"
    plt.title(title)
    plt.xlabel("Strategy")
    plt.ylabel("Game Model")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved strategy heatmap: {save_path}")


def perform_statistical_tests(df, last_step_only=True):
    """Performs and prints results of statistical tests from interval log data."""
    if df is None or df.empty:
        print("No data for statistical tests.")
        return

    df_test = df.copy()
    if last_step_only and 'logged_at_step' in df.columns:
        last_step_val = df['logged_at_step'].max()
        df_test = df[df['logged_at_step'] == last_step_val]
        print(f"\n--- Performing Statistical Tests (on data from last logged step: {last_step_val}) ---")
    else:
        print("\n--- Performing Statistical Tests (on all interval data) ---")


    # T-test: Defender Payoff - Q-Learning vs Static (filtered by a specific Q-param set if desired)
    # Example: Pick one set of Q-Learning parameters for a cleaner comparison
    q_filters = {'alpha': 0.1, 'hybrid_static_steps': 0} # Example filter
    
    q_data_filtered = df_test[df_test["game_model"] == "Q-Learning"]
    for col, val in q_filters.items():
        if col in q_data_filtered.columns:
            q_data_filtered = q_data_filtered[q_data_filtered[col] == val]
            
    q_learning_def_payoffs = q_data_filtered["interval_avg_defender_payoff"].dropna()
    static_def_payoffs = df_test[df_test["game_model"] == "Static"]["interval_avg_defender_payoff"].dropna()

    if len(q_learning_def_payoffs) > 1 and len(static_def_payoffs) > 1:
        stat, pval = ttest_ind(q_learning_def_payoffs, static_def_payoffs, equal_var=False)
        print(f"\nT-test (Defender Payoff): Q-Learning (params:{q_filters}) vs Static")
        print(f"  Q-L Mean: {q_learning_def_payoffs.mean():.2f} (n={len(q_learning_def_payoffs)}), Static Mean: {static_def_payoffs.mean():.2f} (n={len(static_def_payoffs)})")
        print(f"  Statistic={stat:.3f}, p-value={pval:.4f}")
        if pval < 0.05: print("  Result: Statistically significant difference.")
        else: print("  Result: No statistically significant difference.")
    else:
        print(f"\nNot enough data for T-test: Q-Learning (params:{q_filters}) vs Static (Defender Payoff).")
        print(f"  Q-L count: {len(q_learning_def_payoffs)}, Static count: {len(static_def_payoffs)}")

    # ANOVA: Network Health across Q-Learning (specific params), Bayesian, Static
    models_for_anova = ["Q-Learning", "Bayesian Game", "Static"]
    health_data_anova = []
    valid_models_anova = []

    for model_name in models_for_anova:
        if model_name == "Q-Learning":
            data_series = q_data_filtered["current_network_health"].dropna()
        else:
            data_series = df_test[df_test["game_model"] == model_name]["current_network_health"].dropna()
        
        if len(data_series) >= 2:
            health_data_anova.append(data_series)
            valid_models_anova.append(model_name if model_name != "Q-Learning" else f"Q-L(params:{q_filters})")
    
    if len(health_data_anova) >= 2:
        stat_anova, pval_anova = f_oneway(*health_data_anova)
        print(f"\nANOVA (Network Health) across models: {', '.join(valid_models_anova)}")
        for i, model_desc in enumerate(valid_models_anova):
            print(f"  {model_desc} Mean Health: {health_data_anova[i].mean():.3f} (n={len(health_data_anova[i])})")
        print(f"  F-statistic={stat_anova:.3f}, p-value={pval_anova:.4f}")
        if pval_anova < 0.05: print("  Result: Statistically significant difference exists.")
        else: print("  Result: No statistically significant overall difference.")
    else:
        print("\nNot enough groups/data for ANOVA on Network Health.")
            
    print("\n--- Statistical Tests Complete ---")


# --- Main plotting and analysis logic ---
if __name__ == "__main__":
    # Load all interval log data from the results directory
    all_data = load_data_from_multiple_interval_logs(RESULTS_DIR)

    if all_data is not None and not all_data.empty:
        print(f"Successfully loaded data from {len(all_data['seed'].unique())} unique seeds/configs (approx). Total rows: {len(all_data)}")
        
        # Define specific Q-Learning parameter set for some plots to make them cleaner
        # This should match one of the configurations you ran in PARAM_GRID_Q_LEARNING
        specific_q_params = {'alpha': 0.1, 'gamma': 0.9, 'hybrid_static_steps': 0,
                             'epsilon_start': 0.5, 'epsilon_decay': 0.998, 'epsilon_min': 0.01}

        # 1. Payoff Convergence Plots (for Q-Learning with specific params and other models)
        plot_payoff_convergence_over_steps(all_data, game_model_filter="Q-Learning", param_filters=specific_q_params,
                                           player_type='defender', filename_prefix="qlearn_specific_def_conv")
        plot_payoff_convergence_over_steps(all_data, game_model_filter="Q-Learning", param_filters=specific_q_params,
                                           player_type='attacker', filename_prefix="qlearn_specific_atk_conv")
        plot_payoff_convergence_over_steps(all_data, game_model_filter="Bayesian Game", player_type='defender', 
                                           filename_prefix="bayesian_def_conv")

        # 2. Box Plots for final performance comparison
        # Defender Payoff
        plot_metric_comparison_boxplots(all_data, metric_col="interval_avg_defender_payoff", x_axis_col="game_model",
                                        filename="final_def_payoff_boxplot_by_model.png",
                                        title_suffix=f"(Data from final logged step: {all_data['logged_at_step'].max()})")
        # Network Health
        plot_metric_comparison_boxplots(all_data, metric_col="current_network_health", x_axis_col="game_model",
                                        filename="final_net_health_boxplot_by_model.png",
                                        title_suffix=f"(Data from final logged step: {all_data['logged_at_step'].max()})")
        
        # Example: Boxplot of Q-Learning defender payoff vs alpha (at last step)
        q_learning_data_last_step = all_data[(all_data['game_model'] == "Q-Learning") & (all_data['logged_at_step'] == all_data['logged_at_step'].max())]
        if not q_learning_data_last_step.empty:
            plot_metric_comparison_boxplots(q_learning_data_last_step, metric_col="interval_avg_defender_payoff", 
                                            x_axis_col="alpha", hue_col="hybrid_static_steps",
                                            filename="qlearn_final_def_payoff_vs_alpha_hue_hybrid.png",
                                            title_suffix="For Q-Learning model at final step")

        # 3. Bar charts for average metrics (using data from last logged step)
        plot_avg_metric_bar_with_error(all_data, metric_col="interval_avg_defender_payoff", group_by_col="game_model",
                                       filename="avg_final_def_payoff_bar.png", title_suffix="At Final Logged Step")
        plot_avg_metric_bar_with_error(all_data, metric_col="current_network_health", group_by_col="game_model",
                                       filename="avg_final_net_health_bar.png", title_suffix="At Final Logged Step")

        # 4. Strategy Frequency Heatmaps (using data from last logged step)
        # Overall for each game model type
        for model in all_data['game_model'].unique():
            model_filter = {'game_model': model}
            plot_strategy_frequency_heatmap(all_data, player_type="atk", filters=model_filter,
                                            filename=f"heatmap_atk_freq_{model.replace(' ','_')}.png",
                                            title_suffix=f"For {model} at Final Logged Step")
            plot_strategy_frequency_heatmap(all_data, player_type="def", filters=model_filter,
                                            filename=f"heatmap_def_freq_{model.replace(' ','_')}.png",
                                            title_suffix=f"For {model} at Final Logged Step")
        
        # 5. Perform Statistical Tests
        perform_statistical_tests(all_data, last_step_only=True)

        print(f"\nPlotting and analysis complete. Check the '{PLOTS_DIR}' directory.")
    else:
        print("Could not load data from interval logs. No plots or analysis will be performed.")