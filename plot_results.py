# plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import os
from scipy.stats import ttest_ind, f_oneway, entropy # Added entropy
import glob
import re
import time
import itertools
import sys
import unicodedata # For robust filename cleaning

# --- IEEE Publication Configuration ---
plt.rcParams.update({
    'font.size': 10, 'font.family': 'serif', 
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'axes.titlesize': 11, 'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.titlesize': 12, 'lines.linewidth': 1.2, 'axes.linewidth': 0.8,
    'grid.linewidth': 0.5, 'patch.linewidth': 0.8, 'text.usetex': False, 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.format': 'pdf', 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
})

IEEE_SINGLE_COLUMN = 3.5
IEEE_DOUBLE_COLUMN = 7.16

# --- Configuration ---
RESULTS_DIR = None # Dynamically set
PLOTS_SUBDIR_NAME = "plots_main_paper" # Specific subdir for this set of plots
DETAILED_HISTORIES_SUBDIR = "detailed_histories_consolidated"

# --- Helper Functions ---
def find_latest_results_dir(base_pattern="results_*"):
    list_of_results_dirs = glob.glob(base_pattern)
    if not list_of_results_dirs: return None
    return max(list_of_results_dirs, key=os.path.getctime)

def clean_filename(filename_str):
    """Removes or replaces characters for filename compatibility, aims for shortness."""
    if not isinstance(filename_str, str): filename_str = str(filename_str)
    
    # Normalize Unicode to ASCII equivalents or remove
    filename_str = unicodedata.normalize('NFKD', filename_str).encode('ascii', 'ignore').decode('ascii')
    
    # Replace common separators and problematic characters with a single underscore
    filename_str = re.sub(r'[ \-()/]+', '_', filename_str) # Spaces, hyphens, parentheses
    filename_str = re.sub(r'[^\w.]', '', filename_str)    # Remove non-alphanumeric (except period for extension)
    
    # Consolidate multiple underscores
    filename_str = re.sub(r'_+', '_', filename_str)
    
    # Handle periods: ensure only the last one acts as extension separator
    if '.' in filename_str:
        parts = filename_str.rsplit('.', 1)
        base_name = parts[0].replace('.', '_') # Replace other periods in base with underscore
        extension = re.sub(r'[^\w]', '', parts[1]) # Clean extension
        filename_str = f"{base_name}.{extension}"
    else:
        filename_str = filename_str.replace('.', '_') # No extension, replace all periods

    # Remove leading/trailing underscores from the base name
    if '.' in filename_str:
        parts = filename_str.rsplit('.', 1)
        base_name = parts[0].strip('_')
        extension = parts[1]
        if not base_name: base_name = "default_fn" # Handle cases like "_.pdf"
        filename_str = f"{base_name}.{extension}"
    else:
        filename_str = filename_str.strip('_')
        if not filename_str: filename_str = "default_fn"

    # Ensure not just an extension like ".pdf"
    if filename_str.startswith('.'): filename_str = "default_fn" + filename_str
    if not filename_str: filename_str = "default_fn.pdf" # Ultimate fallback

    # Max length (conservative, leave room for path)
    MAX_FILENAME_LEN = 100 
    if len(filename_str) > MAX_FILENAME_LEN:
        name_part, ext_part = os.path.splitext(filename_str)
        # Hash the long name part to make it shorter but unique
        name_hash = str(abs(hash(name_part)) % (10**8)) # 8-digit hash
        short_name_part = name_part[:MAX_FILENAME_LEN - len(ext_part) - len(name_hash) - 2] + '_' + name_hash
        filename_str = short_name_part + ext_part
        
    return filename_str


def format_title(text):
    if not isinstance(text, str): text = str(text)
    text = text.replace("_", " ").replace("-", " ")
    replacements = {
        "attacker model": "Attacker Model", "defender model": "Defender Model",
        "network health": "Network Health", "detection rate": "Detection Rate",
        "interval avg": "Average", "current network": "Network",
        "interval detection": "Detection", "logged at step": "Simulation Step",
        "payoff": "Payoff", "q learning": "Q-Learning", "ql": "Q-L", # Shorter for titles
        "bayesian game": "Bayesian Game", "bayes": "Bayes", "bg": "BG", # Shorter
        "static": "Static", "st": "ST", # Shorter
        "random erdos renyi": "Erdős-Rényi", "rer": "ER", # Shorter
        "small world": "Small-World", "sw": "SW", "star": "Star",
        "vs": "vs.", "std dev": "Std. Dev.", "avg": "Avg.", "freq": "Freq.",
        "entropy": "Entropy"
    }
    formatted_text = text.lower()
    for old, new in replacements.items():
        formatted_text = re.sub(r'\b' + re.escape(old) + r'\b', new, formatted_text, flags=re.IGNORECASE)
    words = formatted_text.split()
    capitalized_words = []
    for word in words:
        is_part_of_replacement = any(word in v for v in replacements.values())
        if not is_part_of_replacement and (not word or not word[0].isupper()):
            capitalized_words.append(word.capitalize() if word else "")
        else:
            capitalized_words.append(word)
    return " ".join(capitalized_words)

def format_scientific(value, decimals=2):
    if pd.isna(value): return "N/A"
    if abs(value) < 1e-3 and value != 0 or abs(value) >= 1e4:
        return f"{value:.{decimals}e}"
    else:
        if 0.001 <= abs(value) < 1:
            return f"{value:.{max(decimals,3)}f}".rstrip('0').rstrip('.')
        return f"{value:.{decimals}f}".rstrip('0').rstrip('.')

# --- Data Loading ---
def load_data_from_multiple_interval_logs(results_dir_path, file_pattern="*_interval_log.csv"):
    if not os.path.exists(results_dir_path):
        print(f"Results directory not found: '{results_dir_path}'")
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
        except Exception as e: print(f"Error loading data from {f}: {e}")
    if not df_list: print("No data loaded from interval logs."); return None
    
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(combined_df)} rows from {len(all_files)} interval log files.")
    # Numeric conversions (ensure this list is comprehensive)
    cols_to_numeric = [
        'interval_avg_attacker_payoff', 'interval_avg_defender_payoff',
        'current_network_health', 'interval_detection_rate',
        'logged_at_step', 'num_nodes', 'trial', 'seed',
        'atk_alpha', 'atk_gamma', 'atk_epsilon_start', 'atk_epsilon_decay', 'atk_epsilon_min', 
        'atk_hybrid_static_steps', 'current_atk_epsilon', 'atk_q_init_val',
        'def_alpha', 'def_gamma', 'def_epsilon_start', 'def_epsilon_decay', 'def_epsilon_min', 
        'def_hybrid_static_steps', 'current_def_epsilon', 'def_q_init_val',
        'topo_avg_degree', 'topo_density', 'topo_avg_clustering_coefficient', 'topo_diameter'
    ]
    freq_cols = [col for col in combined_df.columns if col.startswith('interval_atk_freq_') or col.startswith('interval_def_freq_')]
    cols_to_numeric.extend(freq_cols)
    for col in cols_to_numeric:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    if 'connectivity_param' in combined_df.columns:
        combined_df['connectivity_param'] = combined_df['connectivity_param'].astype(str)
    if 'topo_is_connected' in combined_df.columns:
         combined_df['topo_is_connected'] = combined_df['topo_is_connected'].astype(bool)
    return combined_df

def load_detailed_histories(results_dir_path, detailed_subdir_name, file_pattern="details_*.csv"):
    detailed_dir = os.path.join(results_dir_path, detailed_subdir_name)
    if not os.path.exists(detailed_dir):
        print(f"Detailed histories directory not found: '{detailed_dir}' - Some plots may be skipped.")
        return None
    all_files = glob.glob(os.path.join(detailed_dir, file_pattern))
    if not all_files:
        print(f"No detailed history files found in '{detailed_dir}' with pattern '{file_pattern}'.")
        return None
    
    df_list = []
    for f_path in all_files:
        try:
            df_detail = pd.read_csv(f_path)
            # Attempt to parse filename for context
            fname = os.path.basename(f_path)
            # Example: details_QL_vs_BG_RER_N10_C0p4_cfgHASH_trial_1_seedSEED.csv
            # This parsing is heuristic and should match `run_experiments.py` format
            match = re.search(r'trial_(\d+)_seed(\d+)', fname)
            if match:
                df_detail['trial_id'] = int(match.group(1))
                df_detail['seed_id'] = int(match.group(2))
            
            match_models = re.match(r'details_([^_]+)_vs_([^_]+)', fname)
            if match_models:
                df_detail['attacker_model_fname'] = match_models.group(1).replace('_', ' ')
                df_detail['defender_model_fname'] = match_models.group(2).replace('_', ' ')
            
            df_list.append(df_detail)
        except Exception as e: print(f"Error loading detailed history from {f_path}: {e}")
    
    if not df_list: print("No data loaded from detailed history files."); return None
    
    combined_detailed_df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(combined_detailed_df)} rows from {len(all_files)} detailed history files.")
    detailed_cols_to_numeric = ['step', 'atk_payoff', 'def_payoff', 'net_health', 'atk_cost', 'def_cost',
                                'jammed_nodes_count', 'atk_epsilon_val', 'def_epsilon_val']
    for col in detailed_cols_to_numeric:
        if col in combined_detailed_df.columns:
            combined_detailed_df[col] = pd.to_numeric(combined_detailed_df[col], errors='coerce')
    if 'detected' in combined_detailed_df.columns:
        combined_detailed_df['detected'] = combined_detailed_df['detected'].astype(bool)
    return combined_detailed_df

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
        df_test = df_test[df_test['logged_at_step'] == last_steps].copy()
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

    print("\n--- Statistical Tests Complete ---")

# --- Plotting Functions (Adapted from previous responses and enhanced) ---

def plot_matchup_matrix_heatmap(df, plots_dir,
                                metric_col, filters=None,
                                specific_filename=None, title_suffix=""):
    if df is None or df.empty or metric_col not in df.columns: 
        print(f"Skipping heatmap for {metric_col}: Data unavailable/empty or metric column missing."); return

    df_filtered = df.copy()
    if filters:
        for col, val in filters.items():
            if col in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[col] == (str(val) if col == 'connectivity_param' else val)]
    if df_filtered.empty: print(f"Skipping heatmap for {metric_col}: No data after filters."); return

    group_col = 'seed' if 'seed' in df_filtered.columns else 'trial' if 'trial' in df_filtered.columns else None
    df_agg = pd.DataFrame()
    if 'logged_at_step' in df_filtered.columns and group_col:
        # Get data for the last logged step for each trial/seed
        last_step_indices = df_filtered.groupby(group_col)['logged_at_step'].idxmax()
        df_agg = df_filtered.loc[last_step_indices].copy()
    else: # Fallback if no grouping or step info, use all data (less ideal)
        df_agg = df_filtered.copy()
    
    if df_agg.empty or df_agg[metric_col].isnull().all():
        print(f"Skipping heatmap for {metric_col}: Aggregated data is empty or all NaN."); return

    matchup_avg_metric = df_agg.groupby(["attacker_model", "defender_model"], as_index=False)[metric_col].mean()
    if matchup_avg_metric.empty: print(f"Skipping heatmap for {metric_col}: Grouped data for pivot is empty."); return

    try:
        heatmap_matrix = matchup_avg_metric.pivot(index="defender_model", columns="attacker_model", values=metric_col)
    except Exception as e:
        print(f"Error pivoting data for heatmap {metric_col}: {e}"); return
        
    heatmap_matrix.index = [format_title(idx) for idx in heatmap_matrix.index]
    heatmap_matrix.columns = [format_title(col) for col in heatmap_matrix.columns]

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COLUMN * 1.4, IEEE_SINGLE_COLUMN * 1.2)) # Adjusted size
    
    sns.heatmap(heatmap_matrix, annot=True, fmt=".2f", cmap="viridis_r", 
                linewidths=0.5, ax=ax, cbar_kws={'label': format_title(metric_col.replace("interval_avg_", ""))})

    clean_metric_name = format_title(metric_col.replace("interval_avg_", "").replace("current_", ""))
    title = f"Avg. {clean_metric_name}\nAcross Model Matchups{title_suffix}"
    
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Attacker Model", fontweight='bold')
    ax.set_ylabel("Defender Model", fontweight='bold')
    ax.tick_params(axis='x', rotation=30)  # Removed ha='right'
    for label in ax.get_xticklabels():
        label.set_ha('right')  # Set alignment here
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    
    save_path = os.path.join(plots_dir, clean_filename(specific_filename))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {specific_filename}")

def plot_payoff_convergence_over_steps(df, plots_dir, player_type='defender', 
                                       attacker_model_filter=None, defender_model_filter=None,
                                       topology_filter=None, node_count_filter=None, conn_param_filter=None,
                                       attacker_param_filters=None, defender_param_filters=None,
                                       specific_filename=None, window_size_ratio=0.1):
    if df is None or df.empty: print(f"Skipping convergence plot {specific_filename}: Data unavailable."); return
    payoff_col = f'interval_avg_{player_type}_payoff'
    if payoff_col not in df.columns: print(f"Skipping convergence {specific_filename}: Column {payoff_col} not found."); return

    df_filtered = df.copy()
    # Apply filters
    if attacker_model_filter: df_filtered = df_filtered[df_filtered['attacker_model'] == attacker_model_filter]
    if defender_model_filter: df_filtered = df_filtered[df_filtered['defender_model'] == defender_model_filter]
    if topology_filter: df_filtered = df_filtered[df_filtered['topology'] == topology_filter]
    if node_count_filter is not None: df_filtered = df_filtered[df_filtered['num_nodes'] == node_count_filter]
    if conn_param_filter is not None: df_filtered = df_filtered[df_filtered['connectivity_param'] == str(conn_param_filter)]
    # Q-param filters (exact match)
    if attacker_param_filters and attacker_model_filter == "Q-Learning":
        for p_name, p_val in attacker_param_filters.items():
            if f'atk_{p_name}' in df_filtered.columns:
                 df_filtered = df_filtered[np.isclose(df_filtered[f'atk_{p_name}'], p_val) | df_filtered[f'atk_{p_name}'].isnull()] 
    if defender_param_filters and defender_model_filter == "Q-Learning":
        for p_name, p_val in defender_param_filters.items():
            if f'def_{p_name}' in df_filtered.columns:
                df_filtered = df_filtered[np.isclose(df_filtered[f'def_{p_name}'], p_val) | df_filtered[f'def_{p_name}'].isnull()]

    if df_filtered.empty: print(f"Skipping convergence {specific_filename}: No data after filters."); return

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COLUMN, IEEE_SINGLE_COLUMN * 0.75))
    
    # Group by step and calculate mean/std payoff
    # df_grouped = df_filtered.groupby('logged_at_step', as_index=False)[payoff_col].agg(['mean', 'std']) # Throws error if index name reused
    # Correct groupby for older pandas or specific structures:
    grouped_step_data = df_filtered.groupby('logged_at_step')[payoff_col]
    mean_payoffs = grouped_step_data.mean()
    std_payoffs = grouped_step_data.std().fillna(0) # fillna for single-point std
    steps_x = mean_payoffs.index


    if mean_payoffs.empty or len(steps_x) == 0: print(f"Skipping conv. {specific_filename}: Mean payoffs empty."); plt.close(fig); return
    
    # Dynamic window size
    window_size = max(1, int(len(steps_x) * window_size_ratio))
    rolling_avg_mean = mean_payoffs.rolling(window=window_size, min_periods=1).mean()
    
    if rolling_avg_mean.empty: print(f"Skipping conv. {specific_filename}: Rolling avg empty."); plt.close(fig); return

    ax.plot(steps_x, mean_payoffs, label="Raw Mean", alpha=0.4, linestyle=':', color='gray')
    ax.plot(steps_x, rolling_avg_mean, label=f"Rolling Avg. (w={window_size})", color='black', linewidth=1.5)
    ax.fill_between(steps_x, rolling_avg_mean - std_payoffs.loc[rolling_avg_mean.index], # Align std with rolling mean index
                            rolling_avg_mean + std_payoffs.loc[rolling_avg_mean.index],
                    alpha=0.2, label=f"±1 {format_title('std dev')}", color='lightblue')

    title_player_short = format_title(player_type)
    atk_model_short = format_title(attacker_model_filter) if attacker_model_filter else "Any"
    def_model_short = format_title(defender_model_filter) if defender_model_filter else "Any"

    ax.set_title(f"{title_player_short} Payoff Conv.\n{atk_model_short} vs {def_model_short}", fontweight='bold')
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel(f"Avg. {title_player_short} Payoff")
    ax.legend(loc='best', frameon=True, fancybox=False, shadow=False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format_scientific(x)))
    plt.tight_layout()
    
    save_path = os.path.join(plots_dir, clean_filename(specific_filename))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {specific_filename}")


def plot_metric_comparison_boxplots(df, plots_dir, metric_col, 
                                    x_axis_col="attacker_model", hue_col="defender_model",
                                    filters=None, specific_filename=None, title_suffix="", 
                                    y_label_override=None, showfliers=False):
    if df is None or df.empty or metric_col not in df.columns:
        print(f"Skipping boxplot {specific_filename}: Data unavailable or metric column missing."); return

    df_filtered = df.copy()
    if filters: # Apply general filters if any
        for col, val in filters.items():
            if col in df_filtered.columns:
                 df_filtered = df_filtered[df_filtered[col] == (str(val) if col == 'connectivity_param' else val)]
    if df_filtered.empty: print(f"Skipping boxplot {specific_filename}: No data after filters."); return
    
    # Ensure we are using final step data if applicable from how `df` was passed
    # If `df` is already `df_final_step_interval` or `df_std_devs`, this is fine.

    if df_filtered[metric_col].isnull().all():
        print(f"Skipping boxplot {specific_filename}: Metric column is all NaN."); return

    fig, ax = plt.subplots(figsize=(IEEE_DOUBLE_COLUMN * 0.8, IEEE_SINGLE_COLUMN * 0.9)) # Adjusted size
    
    # Order for consistency if models are on x-axis
    model_order = sorted(df_filtered[x_axis_col].unique(), key=lambda x: ("Q-L" not in format_title(x), format_title(x))) if x_axis_col in ["attacker_model", "defender_model"] else None
    hue_order = sorted(df_filtered[hue_col].unique(), key=lambda x: ("Q-L" not in format_title(x), format_title(x))) if hue_col and hue_col in ["attacker_model", "defender_model"] else None

    sns.boxplot(x=df_filtered[x_axis_col].apply(format_title), 
                y=metric_col, 
                hue=df_filtered[hue_col].apply(format_title) if hue_col else None, 
                data=df_filtered, 
                order=model_order, hue_order=hue_order,
                palette="Set2", ax=ax, showfliers=showfliers)
    
    y_label = y_label_override if y_label_override else format_title(metric_col.replace("interval_avg_", "").replace("current_", ""))
    x_label = format_title(x_axis_col)
    hue_label = format_title(hue_col) if hue_col else ""

    title = f"{y_label} by {x_label}"
    if hue_col: title += f" (Hue: {hue_label})"
    if title_suffix: title += f"\n{title_suffix}"

    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.tick_params(axis='x', rotation=30)  # Removed ha='right'
    for label in ax.get_xticklabels():
        label.set_ha('right')  # Set alignment here
    
    if "health" in metric_col.lower() or "rate" in metric_col.lower(): ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format_scientific(x)))
    
    if hue_col:
        legend = ax.legend(title=hue_label, loc='best', frameon=True, fancybox=False, shadow=False)
        if legend: legend.get_title().set_fontweight('bold')

    plt.tight_layout()
    save_path = os.path.join(plots_dir, clean_filename(specific_filename))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {specific_filename}")

def plot_strategy_entropy_over_time(df_interval, plots_dir, player_type, 
                                    attacker_model_filter, defender_model_filter, 
                                    topology_filter, node_count_filter, conn_param_filter,
                                    specific_filename):
    if df_interval is None or df_interval.empty: print(f"Skipping entropy {specific_filename}: Interval data missing."); return

    df_filtered = df_interval.copy()
    # Apply filters (similar to convergence plot)
    if attacker_model_filter: df_filtered = df_filtered[df_filtered['attacker_model'] == attacker_model_filter]
    if defender_model_filter: df_filtered = df_filtered[df_filtered['defender_model'] == defender_model_filter]
    # Add other filters: topology, num_nodes, conn_param_filter
    if topology_filter: df_filtered = df_filtered[df_filtered['topology'] == topology_filter]
    if node_count_filter: df_filtered = df_filtered[df_filtered['num_nodes'] == node_count_filter]
    if conn_param_filter: df_filtered = df_filtered[df_filtered['connectivity_param'] == str(conn_param_filter)]
    
    if df_filtered.empty: print(f"Skipping entropy {specific_filename}: No data after filters."); return

    freq_cols_prefix = f"interval_{player_type}_freq_"
    strat_freq_cols = [col for col in df_filtered.columns if col.startswith(freq_cols_prefix)]
    if not strat_freq_cols: print(f"Skipping entropy {specific_filename}: No strategy frequency columns for {player_type}."); return

    # Calculate entropy for each row (interval log entry)
    entropies_list = []
    for idx, row in df_filtered.iterrows():
        freqs = row[strat_freq_cols].values.astype(float)
        total_plays = np.sum(freqs)
        if total_plays > 0 and len(freqs[freqs>0]) > 0 : # Ensure there are plays and valid freqs for entropy
            probs = freqs[freqs>0] / total_plays # Use only non-zero frequencies for probs
            entropies_list.append(entropy(probs, base=2))
        else:
            entropies_list.append(0) # Or np.nan if no plays / only one strategy possible
    df_filtered['strategy_entropy'] = entropies_list

    if df_filtered['strategy_entropy'].isnull().all():
        print(f"Skipping entropy {specific_filename}: All entropy values are NaN."); return

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COLUMN, IEEE_SINGLE_COLUMN * 0.75))
    
    entropy_over_steps = df_filtered.groupby('logged_at_step')['strategy_entropy'].mean()
    std_entropy_over_steps = df_filtered.groupby('logged_at_step')['strategy_entropy'].std().fillna(0)

    if entropy_over_steps.empty: print(f"Skipping entropy {specific_filename}: Mean entropy is empty."); plt.close(fig); return

    ax.plot(entropy_over_steps.index, entropy_over_steps, label="Mean Entropy", color='black')
    ax.fill_between(entropy_over_steps.index, 
                    entropy_over_steps - std_entropy_over_steps, 
                    entropy_over_steps + std_entropy_over_steps, 
                    alpha=0.2, label="±1 Std Dev", color='lightblue')

    title_player_short = format_title(player_type)
    atk_model_short = format_title(attacker_model_filter)
    def_model_short = format_title(defender_model_filter)
    ax.set_title(f"{title_player_short} Strategy Entropy\n{atk_model_short} vs {def_model_short}", fontweight='bold')
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Strategy Entropy (bits)")
    ax.legend(loc='best')
    plt.tight_layout()
    save_path = os.path.join(plots_dir, clean_filename(specific_filename))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {specific_filename}")

def plot_metric_histogram(df_final_step, plots_dir, metric_col, 
                          specific_filename, title_suffix="", bins=20):
    if df_final_step is None or df_final_step.empty or metric_col not in df_final_step.columns:
        print(f"Skipping histogram {specific_filename}: Data unavailable."); return
    if df_final_step[metric_col].isnull().all():
        print(f"Skipping histogram {specific_filename}: Metric column is all NaN."); return
    
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COLUMN, IEEE_SINGLE_COLUMN * 0.75))
    data_to_plot = df_final_step[metric_col].dropna()
    if data_to_plot.empty: print(f"Skipping histogram {specific_filename}: No non-NaN data."); plt.close(fig); return

    ax.hist(data_to_plot, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    
    clean_metric_name = format_title(metric_col.replace("interval_avg_", "").replace("current_", ""))
    ax.set_title(f"Distribution of Final {clean_metric_name}{title_suffix}", fontweight='bold')
    ax.set_xlabel(clean_metric_name)
    ax.set_ylabel("Frequency")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{int(x)}"))
    plt.tight_layout()
    save_path = os.path.join(plots_dir, clean_filename(specific_filename))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {specific_filename}")

def calculate_std_dev_final_metrics_per_config(df_interval, metrics_to_std):
    """Calculates std dev of specified final step metrics per configuration across trials."""
    if df_interval is None or df_interval.empty: return pd.DataFrame()

    # Identify unique configurations (excluding trial/seed)
    config_cols = [col for col in ['attacker_model', 'defender_model', 'topology', 'num_nodes', 'connectivity_param',
                                   'atk_alpha', 'def_alpha'] # Add more Q-params if they define a config
                   if col in df_interval.columns]
    if not config_cols: print("Cannot determine config columns for std dev calc."); return pd.DataFrame()

    # Get final step data for each trial for each configuration
    last_step_data_list = []
    group_cols_for_last_step = config_cols + ['trial', 'seed'] # Group by full trial signature
    valid_group_cols = [c for c in group_cols_for_last_step if c in df_interval.columns]

    if not valid_group_cols or 'logged_at_step' not in df_interval.columns:
        print("Required columns for last_step selection missing in std_dev_calc"); return pd.DataFrame()

    for name, group in df_interval.groupby(valid_group_cols, dropna=False):
        last_step_data_list.append(group[group['logged_at_step'] == group['logged_at_step'].max()])
    
    if not last_step_data_list: print("No last step data found for std dev calc."); return pd.DataFrame()
    df_final_steps_all_trials = pd.concat(last_step_data_list)

    # Now, for each configuration, calculate std dev of the metrics across trials
    agg_dict = {metric: 'std' for metric in metrics_to_std}
    df_std_devs = df_final_steps_all_trials.groupby(config_cols, dropna=False).agg(agg_dict).reset_index()
    
    # Rename columns to reflect they are std devs, e.g., interval_avg_defender_payoff -> std_def_payoff
    rename_map = {metric: f"std_{metric.replace('interval_avg_', '').replace('current_', '')}" for metric in metrics_to_std}
    df_std_devs = df_std_devs.rename(columns=rename_map)
    
    # Merge back original config columns that might be lost if not in agg_dict keys (though groupby keeps them)
    # No, groupby + reset_index should preserve the config_cols.

    return df_std_devs


def plot_q_values_heatmap_placeholder(plots_dir, player_type, specific_filename):
    # (Identical to previous response)
    print(f"Placeholder: {specific_filename} (Q-tables not in CSVs)")
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COLUMN, IEEE_SINGLE_COLUMN * 0.5))
    ax.text(0.5, 0.5, "Q-Value Heatmap\n(Data Not Available from Logs)", 
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.axis('off'); plt.tight_layout()
    save_path = os.path.join(plots_dir, clean_filename(specific_filename))
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close(fig)


def plot_detailed_metric_histogram(df_detailed, plots_dir, metric_col, player_type, specific_filename, bins=30):
    if df_detailed is None or df_detailed.empty or metric_col not in df_detailed.columns:
        print(f"Skipping detailed histogram {specific_filename}: Data unavailable."); return
    if df_detailed[metric_col].isnull().all():
        print(f"Skipping detailed histogram {specific_filename}: Metric column is all NaN."); return

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COLUMN, IEEE_SINGLE_COLUMN * 0.75))
    data_to_plot = df_detailed[metric_col].dropna()
    if data_to_plot.empty: print(f"Skipping detailed histogram {specific_filename}: No non-NaN data."); plt.close(fig); return

    ax.hist(data_to_plot, bins=bins, color='coral', edgecolor='black', alpha=0.7)
    
    title_player = "Defender" if player_type == "def" else "Attacker" if player_type == "atk" else ""
    ax.set_title(f"{title_player} {format_title(metric_col)} Dist.\n(All Steps, All Trials)", fontweight='bold')
    ax.set_xlabel(format_title(metric_col))
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    save_path = os.path.join(plots_dir, clean_filename(specific_filename))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {specific_filename}")

def plot_detailed_metric_over_time(df_one_trial, plots_dir, metric_col, 
                                   y_label, title_prefix, specific_filename, is_boolean_event=False):
    if df_one_trial is None or df_one_trial.empty or metric_col not in df_one_trial.columns:
        print(f"Skipping detailed metric over time {specific_filename}: Data for one trial unavailable."); return
    if df_one_trial[metric_col].isnull().all():
        print(f"Skipping detailed metric over time {specific_filename}: Metric column is all NaN."); return

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COLUMN, IEEE_SINGLE_COLUMN * 0.75))
    
    if is_boolean_event:
        event_occurrences = df_one_trial[metric_col].astype(bool)
        window_sz = max(1, len(df_one_trial)//20)
        rolling_freq = event_occurrences.rolling(window=window_sz, min_periods=1).mean()
        ax.plot(df_one_trial['step'], rolling_freq, label=f"Rolling Freq. (w={window_sz})")
    else:
        ax.plot(df_one_trial['step'], df_one_trial[metric_col], label=format_title(metric_col))

    ax.set_title(f"{title_prefix} Over Time\n(Sample Trial)", fontweight='bold')
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel(y_label)
    ax.legend(loc='best')
    plt.tight_layout()
    save_path = os.path.join(plots_dir, clean_filename(specific_filename))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {specific_filename}")

def plot_strategy_frequency_barplot_for_matchup(df_interval, plots_dir, 
                                                attacker_model, defender_model, player_type, 
                                                topology, num_nodes, connectivity,
                                                specific_filename):
    # Renamed from heatmap to barplot as it's more suitable for single row of percentages
    if df_interval is None or df_interval.empty: print(f"Skipping strat freq barplot {specific_filename}: Interval data missing."); return
    
    df_matchup = df_interval[
        (df_interval['attacker_model'] == attacker_model) &
        (df_interval['defender_model'] == defender_model) &
        (df_interval['topology'] == topology) &
        (df_interval['num_nodes'] == num_nodes) &
        (df_interval['connectivity_param'] == str(connectivity))
    ].copy()

    if df_matchup.empty: print(f"Skipping strat freq barplot {specific_filename}: No data for matchup."); return

    group_col = 'seed' if 'seed' in df_matchup.columns else 'trial' if 'trial' in df_matchup.columns else None
    df_agg = pd.DataFrame()
    if group_col and 'logged_at_step' in df_matchup.columns:
        last_steps_indices = df_matchup.groupby(group_col)['logged_at_step'].idxmax()
        df_agg = df_matchup.loc[last_steps_indices].copy()
    else: df_agg = df_matchup.copy() # Fallback

    if df_agg.empty: print(f"Skipping strat freq barplot {specific_filename}: No aggregated data for matchup."); return

    freq_cols_prefix = f"interval_{player_type}_freq_"
    strat_cols = sorted([col for col in df_agg.columns if col.startswith(freq_cols_prefix)])
    if not strat_cols: print(f"Skipping strat freq barplot {specific_filename}: No strategy columns."); return

    df_agg['total_plays'] = df_agg[strat_cols].sum(axis=1)
    for col in strat_cols:
        df_agg[col + '_pct'] = (df_agg[col] / df_agg['total_plays'].replace(0,1)) * 100 # Avoid div by zero
    
    strategy_percentages = df_agg[[col + '_pct' for col in strat_cols]].mean().fillna(0) # Mean across trials for this config
    if strategy_percentages.empty: print(f"Skipping strat freq barplot {specific_filename}: No strategy percentages."); return

    strategy_names = [format_title(s.replace(freq_cols_prefix, "").replace("_pct", "")) for s in strategy_percentages.index]
    
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COLUMN * 1.3, IEEE_SINGLE_COLUMN * 0.9))
    
    sns.barplot(x=strategy_names, y=strategy_percentages.values, ax=ax)

    title_player_short = format_title(player_type)
    atk_model_short = format_title(attacker_model)
    def_model_short = format_title(defender_model)
    ax.set_title(f"{title_player_short} Strategy Usage Freq.\n{atk_model_short} vs {def_model_short}", fontweight='bold')
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Avg. Usage Freq. (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', rotation=45)  # Removed ha='right'
    for label in ax.get_xticklabels():
        label.set_ha('right')  # Set alignment here
    plt.tight_layout()
    save_path = os.path.join(plots_dir, clean_filename(specific_filename))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {specific_filename}")


def plot_avg_metric_by_strategy_bar(df_detailed, plots_dir, metric_col, player_type_for_strategy, 
                                    specific_filename, y_label_override=None):
    if df_detailed is None or df_detailed.empty:
        print(f"Skipping metric by strategy {specific_filename}: Detailed logs not available."); return
    
    strategy_col = 'atk_strat' if player_type_for_strategy == 'atk' else 'def_strat'
    if strategy_col not in df_detailed.columns or metric_col not in df_detailed.columns:
        print(f"Skipping metric by strategy {specific_filename}: Required columns missing."); return
    if df_detailed[metric_col].isnull().all():
        print(f"Skipping metric by strategy {specific_filename}: Metric column all NaN."); return

    # Calculate mean of metric_col, grouped by chosen strategy of player_type_for_strategy
    # This uses all steps from all detailed logs.
    avg_metric_by_strat = df_detailed.groupby(strategy_col, as_index=False)[metric_col].mean()
    avg_metric_by_strat = avg_metric_by_strat.sort_values(by=metric_col, ascending=False)

    if avg_metric_by_strat.empty: print(f"Skipping metric by strategy {specific_filename}: No data after grouping."); return

    fig, ax = plt.subplots(figsize=(IEEE_DOUBLE_COLUMN * 0.7, IEEE_SINGLE_COLUMN * 0.9)) # Adjusted size
    
    sns.barplot(x=strategy_col, y=metric_col, data=avg_metric_by_strat,
                hue=strategy_col, palette="coolwarm", legend=False, ax=ax)
    # Update x-tick labels to be formatted
    ax.set_xticks(range(len(avg_metric_by_strat[strategy_col])))
    ax.set_xticklabels([format_title(strat) for strat in avg_metric_by_strat[strategy_col]], rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha('right')  # Set alignment here


    y_label = y_label_override if y_label_override else f"Avg. {format_title(metric_col)}"
    player_title = format_title(player_type_for_strategy)
    
    ax.set_title(f"{y_label}\nby {player_title} Strategy Choice (All Steps)", fontweight='bold')
    ax.set_xlabel(f"{player_title} Strategy")
    ax.set_ylabel(y_label)
    plt.tight_layout()
    save_path = os.path.join(plots_dir, clean_filename(specific_filename))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {specific_filename}")

# --- Main Execution ---
if __name__ == "__main__":
    RESULTS_DIR = find_latest_results_dir()
    if not RESULTS_DIR:
        print("No results directory found (searched for 'results_*'). Exiting."); sys.exit(1)

    PLOTS_DIR = os.path.join(RESULTS_DIR, PLOTS_SUBDIR_NAME)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print(f"--- Plot Generation for Main Paper ---")
    print(f"Using data from: {RESULTS_DIR}")
    print(f"Saving plots to: {PLOTS_DIR}")

    df_all_results = load_data_from_multiple_interval_logs(RESULTS_DIR)
    df_detailed_logs = load_detailed_histories(RESULTS_DIR, DETAILED_HISTORIES_SUBDIR)

    if (df_all_results is None or df_all_results.empty) and \
       (df_detailed_logs is None or df_detailed_logs.empty):
        print("No data loaded from interval or detailed logs. Exiting."); sys.exit(1)
    
    # Prepare df_final_step_interval for plots needing final step values from interval logs
    df_final_step_interval = pd.DataFrame()
    if df_all_results is not None and not df_all_results.empty:
        group_col_interval = []
        if 'seed' in df_all_results.columns: group_col_interval.append('seed')
        # Add other columns that define a unique trial configuration if 'seed' is not enough
        # For now, assume seed is unique per full configuration trial
        # If not, you might need to group by attacker_model, defender_model, topology, num_nodes, connectivity_param, trial, seed
        
        # A more robust way to define a unique trial run for 'last step'
        trial_defining_cols = [c for c in ['attacker_model', 'defender_model', 'topology', 'num_nodes', 'connectivity_param', 'atk_alpha', 'def_alpha', 'trial', 'seed'] if c in df_all_results.columns]
        # Fallback if some key columns are missing for grouping
        if not any(c in trial_defining_cols for c in ['trial','seed']): trial_defining_cols = None


        if trial_defining_cols and 'logged_at_step' in df_all_results.columns:
            try:
                last_steps_indices = df_all_results.groupby(trial_defining_cols, dropna=False)['logged_at_step'].idxmax()
                df_final_step_interval = df_all_results.loc[last_steps_indices].copy()
            except KeyError as e: # If grouping by many columns leads to issues with MultiIndex if some are all NaN
                print(f"Could not robustly get last_step_interval data due to grouping error: {e}. Falling back.")
                df_final_step_interval = df_all_results[df_all_results['logged_at_step'] == df_all_results['logged_at_step'].max()].copy() # Simpler fallback
        elif 'logged_at_step' in df_all_results.columns: # Simpler fallback if grouping fails
             df_final_step_interval = df_all_results[df_all_results['logged_at_step'] == df_all_results['logged_at_step'].max()].copy()
        else: # If no logged_at_step, use all data (not ideal for "final")
            df_final_step_interval = df_all_results.copy()


    # Define a representative configuration for some specific plots
    rep_config = {}
    if df_all_results is not None and not df_all_results.empty:
        # Try to find a QL vs Bayes matchup for representative plots
        ql_vs_bayes_df = df_all_results[(df_all_results['attacker_model'] == 'Q-Learning') & (df_all_results['defender_model'] == 'Bayesian Game')]
        if not ql_vs_bayes_df.empty:
            first_row = ql_vs_bayes_df.iloc[0]
        else: # Fallback to any first row
            first_row = df_all_results.iloc[0]
        
        rep_config = {
            'topology': first_row.get('topology'),
            'num_nodes': first_row.get('num_nodes'),
            'connectivity': first_row.get('connectivity_param'), # Already string
            'atk_q_params': None, 'def_q_params': None
        }
        q_param_keys = ['alpha', 'gamma', 'epsilon_start', 'epsilon_decay', 'epsilon_min', 'hybrid_static_steps', 'q_init_val']
        if first_row.get('attacker_model') == 'Q-Learning':
            rep_config['atk_q_params'] = {k: first_row.get(f'atk_{k}') for k in q_param_keys if pd.notna(first_row.get(f'atk_{k}'))}
        if first_row.get('defender_model') == 'Q-Learning':
            rep_config['def_q_params'] = {k: first_row.get(f'def_{k}') for k in q_param_keys if pd.notna(first_row.get(f'def_{k}'))}
    else: # Minimal rep_config if no interval data
        rep_config = {'topology': None, 'num_nodes': None, 'connectivity': None, 'atk_q_params': None, 'def_q_params': None}


    print("\n--- A. Main Paper Plots ---")
    if df_all_results is not None and not df_all_results.empty:
        plot_matchup_matrix_heatmap(df_all_results, PLOTS_DIR, 'interval_avg_defender_payoff', specific_filename="fig_def_payoff_heatmap.pdf")
        plot_matchup_matrix_heatmap(df_all_results, PLOTS_DIR, 'interval_avg_attacker_payoff', specific_filename="fig_atk_payoff_heatmap.pdf")
        plot_matchup_matrix_heatmap(df_all_results, PLOTS_DIR, 'interval_detection_rate', specific_filename="fig_detection_heatmap.pdf")
        plot_matchup_matrix_heatmap(df_all_results, PLOTS_DIR, 'current_network_health', specific_filename="fig_net_health_heatmap.pdf")

        # For convergence, use QL vs Bayes if available, with rep_config network settings
        main_conv_atk_model, main_conv_def_model = "Q-Learning", "Bayesian Game"
        main_conv_atk_params = rep_config['atk_q_params'] if main_conv_atk_model == "Q-Learning" else None
        main_conv_def_params = rep_config['def_q_params'] if main_conv_def_model == "Q-Learning" else None
        
        plot_payoff_convergence_over_steps(df_all_results, PLOTS_DIR, player_type='defender', 
            attacker_model_filter=main_conv_atk_model, defender_model_filter=main_conv_def_model,
            topology_filter=rep_config['topology'], node_count_filter=rep_config['num_nodes'], conn_param_filter=rep_config['connectivity'],
            attacker_param_filters=main_conv_atk_params, defender_param_filters=main_conv_def_params,
            specific_filename="fig_def_convergence.pdf")
        plot_payoff_convergence_over_steps(df_all_results, PLOTS_DIR, player_type='attacker', 
            attacker_model_filter=main_conv_atk_model, defender_model_filter=main_conv_def_model,
            topology_filter=rep_config['topology'], node_count_filter=rep_config['num_nodes'], conn_param_filter=rep_config['connectivity'],
            attacker_param_filters=main_conv_atk_params, defender_param_filters=main_conv_def_params,
            specific_filename="fig_atk_convergence.pdf")

        if not df_final_step_interval.empty:
            plot_metric_comparison_boxplots(df_final_step_interval, PLOTS_DIR, metric_col='interval_avg_defender_payoff', 
                                            x_axis_col="attacker_model", hue_col="defender_model", 
                                            specific_filename="fig_def_payoff_boxplot.pdf", title_suffix=" (Final Step Values)")
        else: print("Skipping fig_def_payoff_boxplot.pdf: No final step interval data.")


    print("\n--- B. Deep Analysis Plots (Convergence) ---")
    if df_all_results is not None and not df_all_results.empty:
        deep_matchups = [("Bayesian Game", "Q-Learning"), ("Q-Learning", "Q-Learning"), ("Bayesian Game", "Bayesian Game")]
        for atk_m, def_m in deep_matchups:
            atk_p = rep_config['atk_q_params'] if atk_m == "Q-Learning" else None
            def_p = rep_config['def_q_params'] if def_m == "Q-Learning" else None
            plot_payoff_convergence_over_steps(df_all_results, PLOTS_DIR, player_type='defender', 
                attacker_model_filter=atk_m, defender_model_filter=def_m, 
                topology_filter=rep_config['topology'], node_count_filter=rep_config['num_nodes'], conn_param_filter=rep_config['connectivity'],
                attacker_param_filters=atk_p, defender_param_filters=def_p,
                specific_filename=f"fig_def_convergence_{clean_filename(atk_m)}_vs_{clean_filename(def_m)}.pdf")
            plot_payoff_convergence_over_steps(df_all_results, PLOTS_DIR, player_type='attacker', 
                attacker_model_filter=atk_m, defender_model_filter=def_m, 
                topology_filter=rep_config['topology'], node_count_filter=rep_config['num_nodes'], conn_param_filter=rep_config['connectivity'],
                attacker_param_filters=atk_p, defender_param_filters=def_p,
                specific_filename=f"fig_atk_convergence_{clean_filename(atk_m)}_vs_{clean_filename(def_m)}.pdf")

    print("\n--- C. Aggregates / Bar Charts (Metric by Strategy) ---")
    if df_detailed_logs is not None and not df_detailed_logs.empty:
        plot_avg_metric_by_strategy_bar(df_detailed_logs, PLOTS_DIR, 'def_payoff', 'def', "fig_avg_def_payoff_by_strategy.pdf")
        plot_avg_metric_by_strategy_bar(df_detailed_logs, PLOTS_DIR, 'atk_payoff', 'atk', "fig_avg_atk_payoff_by_strategy.pdf")
        plot_avg_metric_by_strategy_bar(df_detailed_logs, PLOTS_DIR, 'detected', 'def', 
                                        "fig_avg_detection_by_strategy.pdf", y_label_override="Avg. Detection Rate")
        plot_avg_metric_by_strategy_bar(df_detailed_logs, PLOTS_DIR, 'net_health', 'def', 
                                        "fig_avg_health_by_strategy.pdf")
    else: print("Skipping C (Aggregates by Strategy): Detailed logs not available or empty.")

    print("\n--- D. Strategy Usage Heatmaps / Frequency Maps ---")
    if df_all_results is not None and not df_all_results.empty:
        strat_map_atk_m, strat_map_def_m = "Q-Learning", "Bayesian Game"
        plot_strategy_frequency_barplot_for_matchup(df_all_results, PLOTS_DIR, 
            attacker_model=strat_map_atk_m, defender_model=strat_map_def_m, player_type='def',
            topology=rep_config['topology'], num_nodes=rep_config['num_nodes'], connectivity=rep_config['connectivity'],
            specific_filename="fig_freq_def_QL_vs_Bayes.pdf")
        plot_strategy_frequency_barplot_for_matchup(df_all_results, PLOTS_DIR, 
            attacker_model=strat_map_atk_m, defender_model=strat_map_def_m, player_type='atk',
            topology=rep_config['topology'], num_nodes=rep_config['num_nodes'], connectivity=rep_config['connectivity'],
            specific_filename="fig_freq_atk_QL_vs_Bayes.pdf")

        # Entropy for QL defender (vs first opponent found for it)
        ql_def_opponents = df_all_results[df_all_results['defender_model']=='Q-Learning']['attacker_model'].unique()
        if len(ql_def_opponents) > 0:
            plot_strategy_entropy_over_time(df_all_results, PLOTS_DIR, 'def', 
                attacker_model_filter=ql_def_opponents[0], defender_model_filter='Q-Learning',
                topology_filter=rep_config['topology'], node_count_filter=rep_config['num_nodes'], conn_param_filter=rep_config['connectivity'],
                specific_filename="fig_entropy_def_Q_learning.pdf")
        
        # Entropy for QL attacker (vs first opponent found for it)
        ql_atk_opponents = df_all_results[df_all_results['attacker_model']=='Q-Learning']['defender_model'].unique()
        if len(ql_atk_opponents) > 0:
            plot_strategy_entropy_over_time(df_all_results, PLOTS_DIR, 'atk', 
                attacker_model_filter='Q-Learning', defender_model_filter=ql_atk_opponents[0],
                topology_filter=rep_config['topology'], node_count_filter=rep_config['num_nodes'], conn_param_filter=rep_config['connectivity'],
                specific_filename="fig_entropy_atk_Q_learning.pdf")

    print("\n--- E. Variance & Stability ---")
    if df_all_results is not None and not df_all_results.empty:
        metrics_for_std_calc = ['interval_avg_defender_payoff', 'interval_avg_attacker_payoff']
        df_std_devs_final_metrics = calculate_std_dev_final_metrics_per_config(df_all_results, metrics_for_std_calc)
        
        if not df_std_devs_final_metrics.empty:
            if 'std_defender_payoff' in df_std_devs_final_metrics.columns:
                plot_metric_comparison_boxplots(df_std_devs_final_metrics, PLOTS_DIR, metric_col='std_defender_payoff',
                                                x_axis_col="attacker_model", hue_col="defender_model", showfliers=True,
                                                y_label_override="Std.Dev. of Final Def. Payoff",
                                                specific_filename="fig_def_payoff_stdev_boxplot.pdf")
            if 'std_attacker_payoff' in df_std_devs_final_metrics.columns:
                plot_metric_comparison_boxplots(df_std_devs_final_metrics, PLOTS_DIR, metric_col='std_attacker_payoff',
                                                x_axis_col="attacker_model", hue_col="defender_model", showfliers=True,
                                                y_label_override="Std.Dev. of Final Atk. Payoff",
                                                specific_filename="fig_atk_payoff_stdev_boxplot.pdf")
        else: print("Skipping E (StdDev Boxplots): Could not compute std dev of final metrics.")

        if not df_final_step_interval.empty:
            plot_metric_histogram(df_final_step_interval, PLOTS_DIR, 'interval_detection_rate', "fig_detection_variance_histogram.pdf", title_suffix=" (Final Step)")
            plot_metric_histogram(df_final_step_interval, PLOTS_DIR, 'current_network_health', "fig_health_variance_histogram.pdf", title_suffix=" (Final Step)")
        else: print("Skipping E (Histograms): No final step interval data for histograms.")


    print("\n--- F. Debug / Internal Plots ---")
    plot_q_values_heatmap_placeholder(PLOTS_DIR, "defender", "fig_q_values_heatmap_defender.pdf")
    plot_q_values_heatmap_placeholder(PLOTS_DIR, "attacker", "fig_q_values_heatmap_attacker.pdf")

    if df_detailed_logs is not None and not df_detailed_logs.empty:
        plot_detailed_metric_histogram(df_detailed_logs, PLOTS_DIR, 'def_payoff', 'def', "fig_reward_distribution_def.pdf", bins=50)
        
        # Select one trial for time-series plots from detailed logs
        # Prioritize a QL vs Bayes trial if available
        one_trial_df = pd.DataFrame()
        if 'attacker_model_fname' in df_detailed_logs.columns: # If model info parsed from filename
             ql_vs_bayes_detailed = df_detailed_logs[
                (df_detailed_logs['attacker_model_fname'].str.contains("Q-Learning", case=False, na=False)) &
                (df_detailed_logs['defender_model_fname'].str.contains("Bayesian Game", case=False, na=False))
            ]
             if not ql_vs_bayes_detailed.empty:
                 first_trial_id = ql_vs_bayes_detailed['trial_id'].min()
                 first_seed_id = ql_vs_bayes_detailed[ql_vs_bayes_detailed['trial_id']==first_trial_id]['seed_id'].min()
                 one_trial_df = ql_vs_bayes_detailed[(ql_vs_bayes_detailed['trial_id']==first_trial_id) & (ql_vs_bayes_detailed['seed_id']==first_seed_id)]

        if one_trial_df.empty and 'trial_id' in df_detailed_logs.columns: # Fallback to any first trial
            first_trial_id = df_detailed_logs['trial_id'].min()
            first_seed_id = df_detailed_logs[df_detailed_logs['trial_id']==first_trial_id]['seed_id'].min()
            one_trial_df = df_detailed_logs[(df_detailed_logs['trial_id']==first_trial_id) & (df_detailed_logs['seed_id']==first_seed_id)]

        if not one_trial_df.empty:
            plot_detailed_metric_over_time(one_trial_df, PLOTS_DIR, 'jammed_nodes_count', 
                                           "Jammed Nodes", "Jammed Nodes Count", 
                                           "fig_collision_count_over_time.pdf")
            plot_detailed_metric_over_time(one_trial_df, PLOTS_DIR, 'detected', 
                                           "Detection Event Freq.", "Detection Events", 
                                           "fig_detection_events_over_time.pdf", is_boolean_event=True)
        else: print("Skipping F (detailed over time plots): Could not select a representative trial from detailed logs.")
    else: print("Skipping some F plots: Detailed logs not available or empty.")

    print(f"\n--- Plot Generation Complete ---")

    # --- Statistical Tests ---
    perform_statistical_tests(df_all_results, last_step_only=True)

    print(f"Plots saved to: {PLOTS_DIR}")