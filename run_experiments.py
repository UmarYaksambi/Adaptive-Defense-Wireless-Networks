# run_experiments.py
import pandas as pd
import random
import numpy as np
import os
import networkx as nx

from core_sim import (generate_network, select_strategies, execute_attack,
                      execute_defense, payoff, update_learning_models,
                      initialize_learning_models, assign_coalitions, get_q_learning_state,
                      ATTACKER_STRATEGIES_DEFAULT, DEFENDER_STRATEGIES_DEFAULT) # ATTACKER_STRATEGIES_DEFAULT now includes new ones

DEFAULT_DEFENDER_STRATEGIES = DEFENDER_STRATEGIES_DEFAULT
DEFAULT_FREQUENCIES = [1, 2, 3, 4, 5, 6, 7, 8]
DETAILED_HISTORIES_SUBDIR = "detailed_histories_consolidated"
LOG_METRICS_INTERVAL = 50 # 🔹 Step 6: Log metrics every X steps

def get_topology_metrics(G):
    # ... (no changes) ...
    metrics = {"avg_degree": np.nan, "density": np.nan, "avg_clustering_coefficient": np.nan, "diameter": np.nan, "is_connected": False}
    num_nodes = G.number_of_nodes()
    if num_nodes == 0: return metrics
    metrics["density"] = nx.density(G)
    degrees = [d for n, d in G.degree()]
    if degrees: metrics["avg_degree"] = sum(degrees) / num_nodes
    try: metrics["avg_clustering_coefficient"] = nx.average_clustering(G)
    except Exception: pass
    if nx.is_connected(G):
        metrics["is_connected"] = True
        if num_nodes > 1:
            try: metrics["diameter"] = nx.diameter(G)
            except nx.NetworkXError: metrics["diameter"] = np.inf
    else:
        metrics["is_connected"] = False; metrics["diameter"] = np.inf
    return metrics


def simulate(game_model, topology, num_nodes=10, 
             steps=300, # 🔹 Step 5: Long-Horizon Simulations (300-500)
             trials=5, save_path="results.csv",
             connectivity=0.5,
             frequencies_list=None,
             attacker_strategies=None, # Will now use the updated ATTACKER_STRATEGIES_DEFAULT
             defender_strategies=None,
             seed_base=42,
             learning_alpha=0.1, learning_gamma=0.9,
             epsilon_start=0.4, epsilon_decay=0.998, epsilon_min=0.01, # Adjusted epsilon for longer runs
             save_detailed_history_models=None,
             q_bias_init=None, 
             hybrid_static_steps=0,
             log_interval=LOG_METRICS_INTERVAL): # New parameter for logging interval

    if frequencies_list is None: frequencies_list = DEFAULT_FREQUENCIES
    if attacker_strategies is None: attacker_strategies = ATTACKER_STRATEGIES_DEFAULT # Uses new strategies
    if defender_strategies is None: defender_strategies = DEFAULT_DEFENDER_STRATEGIES
    if save_detailed_history_models is None: save_detailed_history_models = []

    all_interval_results = [] # Store results at each log_interval
    results_base_dir = os.path.dirname(save_path)
    detailed_history_full_path_dir = os.path.join(results_base_dir, DETAILED_HISTORIES_SUBDIR)
    if game_model in save_detailed_history_models:
        os.makedirs(detailed_history_full_path_dir, exist_ok=True)

    for t in range(trials):
        current_trial_seed = seed_base + t
        random.seed(current_trial_seed); np.random.seed(current_trial_seed)
        print(f"    Trial {t+1}/{trials} with seed {current_trial_seed}...")

        learning_models = initialize_learning_models(
            game_model, attacker_strategies, defender_strategies,
            alpha=learning_alpha, gamma=learning_gamma,
            epsilon_start=epsilon_start, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
            q_bias=q_bias_init
        )

        G = generate_network(topology, num_nodes, connectivity, frequencies_list, seed=current_trial_seed)
        topology_metrics = get_topology_metrics(G)
        if game_model == "Coalition Formation": G = assign_coalitions(G)

        # Full history for detailed saving if needed
        full_trial_history_data = { "step": [], "atk_strat": [], "def_strat": [], "atk_payoff": [], "def_payoff": [],
                               "net_health": [], "atk_cost": [], "def_cost": [], "jammed_nodes_count": [],
                               "detected": [], "epsilon_val": [] }
        
        # For interval logging
        interval_cumulative_atk_payoff = 0.0
        interval_cumulative_def_payoff = 0.0
        interval_detection_count = 0
        interval_step_count = 0

        # Initial state values for Q-Learning (s0)
        prev_step_atk_strat = "None"; prev_step_def_strat = "None"
        prev_step_jammed_count = 0; prev_step_net_health = 1.0 

        for step_num in range(steps):
            current_q_state = None
            if game_model == "Q-Learning" or (game_model == "Coalition Formation" and 'q_attacker' in learning_models):
                 current_q_state = get_q_learning_state(full_trial_history_data, # Pass the actual history dict
                                                       step_num, steps, len(frequencies_list),
                                                       prev_step_atk_strat, prev_step_def_strat,
                                                       prev_step_jammed_count, prev_step_net_health)
            
            atk_strat, def_strat = select_strategies(
                game_model, step_num, steps, G, attacker_strategies, defender_strategies,
                full_trial_history_data, learning_models, # Pass actual history
                num_nodes, len(frequencies_list), 
                hybrid_static_steps=hybrid_static_steps
            )
            
            current_epsilon_val = learning_models.get('current_epsilon', np.nan)
            full_trial_history_data["epsilon_val"].append(current_epsilon_val)

            jammed_freqs_set, _, current_detect_prob = execute_attack(atk_strat, G, def_strat, frequencies_list)
            G = execute_defense(def_strat, G, jammed_freqs_set, frequencies_list)

            # Outcome Evaluation logic... (no change here)
            jammed_nodes_list = []; protected_nodes_list = []
            for node_id in G.nodes:
                final_freq = G.nodes[node_id].get('frequency'); is_on_jammed_freq = final_freq in jammed_freqs_set
                node_can_resist = False
                if is_on_jammed_freq:
                    if def_strat == "spread_spectrum" and len(jammed_freqs_set) < len(frequencies_list) * 0.6 : node_can_resist = True
                    elif def_strat == "error_coding" and random.random() < 0.6: node_can_resist = True
                if is_on_jammed_freq and not node_can_resist: G.nodes[node_id]['status'] = 'jammed'; jammed_nodes_list.append(node_id)
                else: G.nodes[node_id]['status'] = 'resistant' if is_on_jammed_freq and node_can_resist else 'ok'; protected_nodes_list.append(node_id)
            
            current_jammed_count = len(jammed_nodes_list)
            success_count = num_nodes - current_jammed_count
            current_net_health = success_count / num_nodes if num_nodes > 0 else 0

            atk_cost_val, def_cost_val, atk_reward, def_reward, detected_this_step = payoff(
                atk_strat, def_strat, G, success_count, jammed_nodes_list, protected_nodes_list, current_detect_prob,
                current_net_health, prev_step_net_health, num_nodes, full_trial_history_data # Pass history for payoff shaping
            )

            # Accumulate for interval logging
            interval_cumulative_atk_payoff += atk_reward
            interval_cumulative_def_payoff += def_reward
            if detected_this_step: interval_detection_count += 1
            interval_step_count += 1
            
            # Store in full history
            full_trial_history_data["step"].append(step_num + 1); full_trial_history_data["atk_strat"].append(atk_strat)
            full_trial_history_data["def_strat"].append(def_strat); full_trial_history_data["atk_payoff"].append(atk_reward)
            full_trial_history_data["def_payoff"].append(def_reward); full_trial_history_data["net_health"].append(current_net_health)
            full_trial_history_data["atk_cost"].append(atk_cost_val); full_trial_history_data["def_cost"].append(def_cost_val)
            full_trial_history_data["jammed_nodes_count"].append(current_jammed_count); full_trial_history_data["detected"].append(detected_this_step)

            next_q_state = None
            if game_model == "Q-Learning" or (game_model == "Coalition Formation" and 'q_attacker' in learning_models):
                next_q_state = get_q_learning_state(full_trial_history_data, step_num + 1, steps, len(frequencies_list),
                                                   atk_strat, def_strat, 
                                                   current_jammed_count, current_net_health)

            learning_models = update_learning_models(
                game_model, current_q_state, atk_strat, def_strat, atk_reward, def_reward,
                next_q_state, learning_models, attacker_strategies, defender_strategies,
                is_terminal_step=(step_num == steps - 1)
            )
            
            prev_step_atk_strat = atk_strat; prev_step_def_strat = def_strat
            prev_step_jammed_count = current_jammed_count; prev_step_net_health = current_net_health

            # 🔹 Step 6: Log metrics every 'log_interval' steps
            if (step_num + 1) % log_interval == 0 or (step_num + 1) == steps:
                avg_interval_atk_payoff = interval_cumulative_atk_payoff / interval_step_count if interval_step_count > 0 else 0
                avg_interval_def_payoff = interval_cumulative_def_payoff / interval_step_count if interval_step_count > 0 else 0
                interval_detect_rate = interval_detection_count / interval_step_count if interval_step_count > 0 else 0
                
                # Get strategy frequencies for this interval from full_trial_history_data up to current step
                current_interval_atk_strats = full_trial_history_data["atk_strat"][-(interval_step_count):]
                current_interval_def_strats = full_trial_history_data["def_strat"][-(interval_step_count):]
                
                interval_atk_strat_counts = {s: current_interval_atk_strats.count(s) for s in attacker_strategies}
                interval_def_strat_counts = {s: current_interval_def_strats.count(s) for s in defender_strategies}

                interval_summary = {
                    "trial": t + 1, "logged_at_step": step_num + 1,
                    "game_model": game_model, "topology": topology, "num_nodes": num_nodes,
                    "connectivity_param": str(connectivity), "seed": current_trial_seed,
                    "alpha": learning_alpha, "gamma": learning_gamma, "epsilon_start": epsilon_start,
                    "epsilon_decay": epsilon_decay, "epsilon_min": epsilon_min, "current_epsilon": current_epsilon_val,
                    "hybrid_static_steps": hybrid_static_steps,
                    "interval_avg_attacker_payoff": avg_interval_atk_payoff,
                    "interval_avg_defender_payoff": avg_interval_def_payoff,
                    "current_network_health": current_net_health, # Instantaneous health at log point
                    "interval_detection_rate": interval_detect_rate,
                    **{f"topo_{k}": v for k, v in topology_metrics.items()},
                    **{f"interval_atk_freq_{s.replace(' ', '_')}": c for s, c in interval_atk_strat_counts.items()},
                    **{f"interval_def_freq_{s.replace(' ', '_')}": c for s, c in interval_def_strat_counts.items()}
                }
                all_interval_results.append(interval_summary)
                
                # Reset interval counters
                interval_cumulative_atk_payoff = 0.0; interval_cumulative_def_payoff = 0.0
                interval_detection_count = 0; interval_step_count = 0
        
        # End of trial summary (optional if interval logging is primary)
        # print(f"    Trial {t+1} completed. Final Epsilon: {current_epsilon_val:.4f}")

        if game_model in save_detailed_history_models:
            base_filename_no_ext = os.path.splitext(os.path.basename(save_path).replace("_interval_log.csv",""))[0] # Use base name
            detailed_filename = f"{base_filename_no_ext}_trial_{t+1}_fulldetails.csv"
            full_detailed_path = os.path.join(detailed_history_full_path_dir, detailed_filename)
            try: pd.DataFrame(full_trial_history_data).to_csv(full_detailed_path, index=False)
            except Exception as e: print(f"      Error saving full detailed history: {e}")

    df_interval_results = pd.DataFrame(all_interval_results)
    try:
        df_interval_results.to_csv(save_path, index=False) # save_path now points to the interval log
        print(f"  Interval results for current configuration saved to {save_path}")
    except Exception as e: print(f"  Error saving interval results to {save_path}: {e}")
    return df_interval_results


# --- Main Experiment Execution ---
if __name__ == "__main__":
    RESULTS_MAIN_DIR = f"results" # Timestamped results folder

    MODELS = ["Q-Learning", "Bayesian Game", "Static"]
    # 🔹 Step 6: Add support for new attacker strategies (already done by ATTACKER_STRATEGIES_DEFAULT update)
    # For comparisons like "Adaptive attacker vs random attacker", you'd create a specific attacker_strategies list
    # e.g., RANDOM_ATTACKER_STRATEGIES = random.sample(ATTACKER_STRATEGIES_DEFAULT, k) or just one basic strategy
    # And then pass this list to simulate() for those specific runs.

    TOPOLOGIES = ["Random (Erdős–Rényi)", "Small-World"] # Reduced set for faster testing
    NODE_COUNTS = [10, 15] 
    CONNECTIVITY_PARAMS = { "Random (Erdős–Rényi)": [0.4], "Small-World": [(6, 0.1)] }

    RUNS_PER_SETTING = 3 # Keep low for dev, increase for paper (e.g., 10-30)
    SIMULATION_STEPS = 300 # 🔹 Step 5: Increased simulation horizon

    PARAM_GRID_Q_LEARNING = { # For Q-Learning specific tuning
        "alpha": [0.1, 0.05], "gamma": [0.9, 0.95],
        "epsilon_start": [0.5, 0.3], "epsilon_decay": [0.998, 0.995], # Slower decay for more steps
        "epsilon_min": [0.01, 0.05],
        "hybrid_static_steps": [0, int(SIMULATION_STEPS * 0.1)] # e.g., 10% of steps static
    }
    # Default params for non-Q models (or if not tuning them)
    DEFAULT_LEARNING_PARAMS = { "alpha": 0.1, "gamma": 0.9, "epsilon_start": 0.1,
                                "epsilon_decay": 1.0, "epsilon_min": 0.1, "hybrid_static_steps": 0 }
    
    SAVE_DETAILED_HISTORY_FOR_MODELS = ["Q-Learning"] # For convergence plots

    os.makedirs(RESULTS_MAIN_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_MAIN_DIR, DETAILED_HISTORIES_SUBDIR), exist_ok=True)

    all_experiment_runs_dfs = [] # To collect DFs from each `simulate` call
    config_counter = 0

    print(f"--- Starting Batch Experiment Runner (Consolidated Plan) ---")
    print(f"Results will be saved in: {RESULTS_MAIN_DIR}")
    # Simplified loop for demonstration; a full grid search for Q-params can be verbose
    for model in MODELS:
        for topo in TOPOLOGIES:
            for nodes in NODE_COUNTS:
                for conn_param_key, conn_param_values in CONNECTIVITY_PARAMS.items():
                    if topo != conn_param_key: continue # Match topology with its connectivity options
                    for conn_param in conn_param_values:
                        
                        current_params_to_run = []
                        if model == "Q-Learning":
                            # Create all combinations from PARAM_GRID_Q_LEARNING
                            # This is a simplified way to iterate; itertools.product is better for full grid
                            for alpha_val in PARAM_GRID_Q_LEARNING["alpha"]:
                                for gamma_val in PARAM_GRID_Q_LEARNING["gamma"]:
                                    # ... and so on for all Q_PARAMS ... (keeping it short for example)
                                    for hybrid_val in PARAM_GRID_Q_LEARNING["hybrid_static_steps"]:
                                        current_params_to_run.append({
                                            "alpha": alpha_val, "gamma": gamma_val,
                                            "epsilon_start": PARAM_GRID_Q_LEARNING["epsilon_start"][0], # Using first for brevity
                                            "epsilon_decay": PARAM_GRID_Q_LEARNING["epsilon_decay"][0],
                                            "epsilon_min": PARAM_GRID_Q_LEARNING["epsilon_min"][0],
                                            "hybrid_static_steps": hybrid_val
                                        })
                        else: # For non-Q models, run with default params
                            current_params_to_run.append(DEFAULT_LEARNING_PARAMS)

                        for param_set in current_params_to_run:
                            config_counter += 1
                            print(f"\n[Config {config_counter}] Running:")
                            print(f"  Model: {model}, Topo: {topo}, N: {nodes}, Conn: {conn_param}")
                            if model == "Q-Learning": print(f"  Params: {param_set}")

                            file_topo_clean = topo.replace(" (Erdős–Rényi)", "_ER").replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
                            file_conn_clean = str(conn_param).replace(".", "p").replace(" ", "").replace("(", "").replace(")", "").replace(",", "_") if conn_param is not None else "def"
                            
                            param_suffix_str = ""
                            if model == "Q-Learning":
                                param_suffix_str = f"_a{param_set['alpha']}_g{param_set['gamma']}_h{param_set['hybrid_static_steps']}".replace(".","p")

                            log_file_base = f"{model.replace(' ', '_')}_{file_topo_clean}_N{nodes}_C{file_conn_clean}{param_suffix_str}"
                            interval_log_save_path = os.path.join(RESULTS_MAIN_DIR, f"{log_file_base}_interval_log.csv")

                            # 🔹 Step 6: Comparison of attacker types (example setup)
                            current_attacker_strategies = ATTACKER_STRATEGIES_DEFAULT
                            # if condition_for_random_attacker_test:
                            #    current_attacker_strategies = ["broadband"] # or some other fixed/random set

                            trial_results_df = simulate(
                                game_model=model, topology=topo, num_nodes=nodes,
                                steps=SIMULATION_STEPS, trials=RUNS_PER_SETTING,
                                save_path=interval_log_save_path, # This will now be the interval log
                                connectivity=conn_param,
                                attacker_strategies=current_attacker_strategies, # Use the full default list
                                learning_alpha=param_set["alpha"], learning_gamma=param_set["gamma"],
                                epsilon_start=param_set["epsilon_start"], epsilon_decay=param_set["epsilon_decay"],
                                epsilon_min=param_set["epsilon_min"],
                                hybrid_static_steps=param_set["hybrid_static_steps"],
                                save_detailed_history_models=SAVE_DETAILED_HISTORY_FOR_MODELS,
                                log_interval=LOG_METRICS_INTERVAL,
                                seed_base=random.randint(1, 100000) 
                            )
                            if trial_results_df is not None and not trial_results_df.empty:
                                all_experiment_runs_dfs.append(trial_results_df)

    if all_experiment_runs_dfs:
        # The primary output is now individual interval log files.
        # A combined summary might still be useful but would need to aggregate interval data.
        # For now, let's just confirm completion.
        print(f"\nAll experiment configurations complete. Interval logs saved in '{RESULTS_MAIN_DIR}'.")
        # If you want a single grand summary of all interval logs:
        # combined_df = pd.concat(all_experiment_runs_dfs, ignore_index=True)
        # combined_path = os.path.join(RESULTS_MAIN_DIR, "ALL_EXPERIMENTS_CONSOLIDATED_intervals_summary.csv")
        # try: combined_df.to_csv(combined_path, index=False)
        # except Exception as e: print(f"Error saving combined interval summary: {e}")
    else: print("\nNo results generated from experiments.")