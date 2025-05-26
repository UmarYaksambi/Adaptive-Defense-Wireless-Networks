# run_experiments.py
import pandas as pd
import random
import numpy as np
import os
import networkx as nx
import time 

from core_sim import (generate_network, select_attacker_strategy, select_defender_strategy, # Import split functions
                      execute_attack, execute_defense, payoff, update_learning_models,
                      initialize_learning_models, assign_coalitions, get_q_learning_state,
                      ATTACKER_STRATEGIES_DEFAULT, DEFENDER_STRATEGIES_DEFAULT)

DEFAULT_DEFENDER_STRATEGIES = DEFENDER_STRATEGIES_DEFAULT
DEFAULT_FREQUENCIES = [1, 2, 3, 4, 5, 6, 7, 8]
DETAILED_HISTORIES_SUBDIR = "detailed_histories_consolidated"
LOG_METRICS_INTERVAL = 50 # 🔹 Step 6: Log metrics every X steps

def get_topology_metrics(G):
    """Calculates basic topology metrics for a network graph."""
    metrics = {"avg_degree": np.nan, "density": np.nan, "avg_clustering_coefficient": np.nan, "diameter": np.nan, "is_connected": False}
    num_nodes = G.number_of_nodes()
    if num_nodes == 0: return metrics
    metrics["density"] = nx.density(G)
    degrees = [d for n, d in G.degree()]
    if degrees: metrics["avg_degree"] = sum(degrees) / num_nodes
    try: metrics["avg_clustering_coefficient"] = nx.average_clustering(G)
    except Exception: pass # Handle cases where clustering is not well-defined (e.g., single node)

    if num_nodes > 1: # Diameter and connectivity are only meaningful for > 1 node
        if nx.is_connected(G):
            metrics["is_connected"] = True
            try: metrics["diameter"] = nx.diameter(G)
            except nx.NetworkXError: metrics["diameter"] = np.inf # Graph is connected but diameter is infinite (e.g., path graph)
        else:
            metrics["is_connected"] = False; metrics["diameter"] = np.inf
    else: # Single node case
        metrics["is_connected"] = True # A single node is trivially connected
        metrics["diameter"] = 0 # Diameter of a single node graph is 0

    return metrics


# 🔹 Step 2: Modify simulate to handle separate attacker/defender models and learning
def simulate(attacker_model, defender_model, topology, num_nodes=10,
             steps=500, # 🔹 Step 5: Long-Horizon Simulations (300-500)
             trials=5, save_path="results.csv",
             connectivity=0.5,
             frequencies_list=None,
             attacker_strategies=None,
             defender_strategies=None,
             seed_base=42,
             attacker_params=None, # Separate params for attacker
             defender_params=None, # Separate params for defender
             save_detailed_history_models=None,
             log_interval=LOG_METRICS_INTERVAL):

    if frequencies_list is None: frequencies_list = DEFAULT_FREQUENCIES
    if attacker_strategies is None: attacker_strategies = ATTACKER_STRATEGIES_DEFAULT
    if defender_strategies is None: defender_strategies = DEFENDER_STRATEGIES_DEFAULT
    if save_detailed_history_models is None: save_detailed_history_models = []
    if attacker_params is None: attacker_params = {} # Ensure params dict exists
    if defender_params is None: defender_params = {} # Ensure params dict exists


    all_interval_results = [] # Store results at each log_interval
    results_base_dir = os.path.dirname(save_path)
    detailed_history_full_path_dir = os.path.join(results_base_dir, DETAILED_HISTORIES_SUBDIR)
    # Only create detailed history subdir if at least one of the models is slated for detailed saving
    if attacker_model in save_detailed_history_models or defender_model in save_detailed_history_models:
         os.makedirs(detailed_history_full_path_dir, exist_ok=True)


    for t in range(trials):
        current_trial_seed = seed_base + t
        random.seed(current_trial_seed); np.random.seed(current_trial_seed)
        print(f"    Trial {t+1}/{trials} with seed {current_trial_seed}...")

        # 🔹 Step 2: Initialize learning models separately for each agent
        agent_models = initialize_learning_models(
            attacker_model_type=attacker_model, defender_model_type=defender_model,
            attacker_strategies=attacker_strategies, defender_strategies=defender_strategies,
            attacker_params=attacker_params, defender_params=defender_params
        )

        G = generate_network(topology, num_nodes, connectivity, frequencies_list, seed=current_trial_seed)
        topology_metrics = get_topology_metrics(G)
        # Coalition formation is a model type, not a general network property applied to all.
        # If Coalition Formation is an attacker or defender model, its logic would be within select/update.
        # The assign_coalitions function might be used *by* a Coalition Formation model.
        # For now, assume coalition assignment is part of a specific model's setup if needed.
        # if game_model == "Coalition Formation": G = assign_coalitions(G) # This line is likely misplaced here

        # Full history for detailed saving if needed
        full_trial_history_data = { "step": [], "atk_strat": [], "def_strat": [], "atk_payoff": [], "def_payoff": [],
                               "net_health": [], "atk_cost": [], "def_cost": [], "jammed_nodes_count": [],
                               "detected": [],
                               "atk_epsilon_val": [], "def_epsilon_val": [] # Track epsilon for both if Q-Learning
                               }

        # For interval logging
        interval_cumulative_atk_payoff = 0.0
        interval_cumulative_def_payoff = 0.0
        interval_detection_count = 0
        interval_step_count = 0

        # Initial state values for Q-Learning (s0) - using 'None' or default values before the first action
        prev_step_atk_strat = "None"; prev_step_def_strat = "None"
        prev_step_jammed_count = 0; prev_step_net_health = 1.0


        for step_num in range(steps):
            # 🔹 Step 2: Get Q-Learning state using outcomes from the *previous* step
            # This state is used by both attacker and defender Q-Learning models to choose their *current* action.
            current_q_state = get_q_learning_state(full_trial_history_data, # Pass the actual history dict
                                                   step_num, steps, len(frequencies_list),
                                                   prev_step_atk_strat, prev_step_def_strat,
                                                   prev_step_jammed_count, prev_step_net_health)

            # 🔹 Step 1: Select strategies using the split functions and agent-specific models/params
            # Corrected call to include defender_strategies
            atk_strat = select_attacker_strategy(
                attacker_model, step_num, steps, attacker_strategies, defender_strategies, # Added defender_strategies
                full_trial_history_data, agent_models['attacker']['model_state'], agent_models['attacker']['params'],
                num_nodes, len(frequencies_list),
                prev_step_atk_strat, prev_step_def_strat, prev_step_jammed_count, prev_step_net_health
            )

            # Corrected call to include attacker_strategies
            def_strat = select_defender_strategy(
                defender_model, step_num, steps, defender_strategies, attacker_strategies, # Added attacker_strategies
                full_trial_history_data, agent_models['defender']['model_state'], agent_models['defender']['params'],
                num_nodes, len(frequencies_list),
                prev_step_atk_strat, prev_step_def_strat, prev_step_jammed_count, prev_step_net_health
            )

            # Track epsilon values if models are Q-Learning
            atk_epsilon_val = agent_models['attacker']['model_state'].get('epsilon', np.nan)
            def_epsilon_val = agent_models['defender']['model_state'].get('epsilon', np.nan)
            full_trial_history_data["atk_epsilon_val"].append(atk_epsilon_val)
            full_trial_history_data["def_epsilon_val"].append(def_epsilon_val)

            # Execute actions
            jammed_freqs_set, _, current_detect_prob = execute_attack(atk_strat, G, def_strat, frequencies_list)
            G = execute_defense(def_strat, G, jammed_freqs_set, frequencies_list)

            # Outcome Evaluation logic...
            jammed_nodes_list = []; protected_nodes_list = []
            current_num_nodes = G.number_of_nodes() # Use current node count in case nodes are removed/added (not implemented yet)
            for node_id in G.nodes:
                final_freq = G.nodes[node_id].get('frequency', frequencies_list[0] if frequencies_list else 1)
                is_on_jammed_freq = final_freq in jammed_freqs_set
                node_can_resist = False # Default no resistance
                # Check for resistance based on defender strategy
                if is_on_jammed_freq:
                    if def_strat == "spread_spectrum" and len(jammed_freqs_set) < len(frequencies_list) * 0.6 :
                        node_can_resist = True # Spread spectrum provides resistance if jamming is not broadband
                    elif def_strat == "error_coding" and random.random() < 0.6:
                         node_can_resist = True # Error coding provides probabilistic resistance

                if is_on_jammed_freq and not node_can_resist:
                    G.nodes[node_id]['status'] = 'jammed'
                    jammed_nodes_list.append(node_id)
                else:
                    G.nodes[node_id]['status'] = 'resistant' if is_on_jammed_freq and node_can_resist else 'ok'
                    protected_nodes_list.append(node_id)

            current_jammed_count = len(jammed_nodes_list)
            success_count = current_num_nodes - current_jammed_count
            current_net_health = success_count / current_num_nodes if current_num_nodes > 0 else 0 # Handle 0 nodes

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

            # 🔹 Step 2: Get next state tuple using outcomes of the *current* step
            next_q_state = get_q_learning_state(full_trial_history_data, step_num + 1, steps, len(frequencies_list),
                                               atk_strat, def_strat, # Use current step's actions as previous for next state
                                               current_jammed_count, current_net_health) # Use current step's outcomes

            # 🔹 Step 2: Update learning models separately for each agent
            agent_models = update_learning_models(
                agent_models, current_q_state, atk_strat, def_strat, atk_reward, def_reward,
                next_q_state, attacker_strategies, defender_strategies,
                is_terminal_step=(step_num == steps - 1)
            )

            # Update previous step variables for the next iteration
            prev_step_atk_strat = atk_strat; prev_step_def_strat = def_strat
            prev_step_jammed_count = current_jammed_count; prev_step_net_health = current_net_health


            # 🔹 Step 6: Log metrics every 'log_interval' steps
            if (step_num + 1) % log_interval == 0 or (step_num + 1) == steps:
                avg_interval_atk_payoff = interval_cumulative_atk_payoff / interval_step_count if interval_step_count > 0 else 0
                avg_interval_def_payoff = interval_cumulative_def_payoff / interval_step_count if interval_step_count > 0 else 0
                interval_detect_rate = interval_detection_count / interval_step_count if interval_step_count > 0 else 0

                # Get strategy frequencies for this interval from full_trial_history_data up to current step
                # Need to be careful here - if step_num + 1 is exactly log_interval, we take the last log_interval steps.
                # If it's the very first log interval (step_num + 1 == log_interval), we take the first log_interval steps.
                start_index = max(0, len(full_trial_history_data["atk_strat"]) - interval_step_count)
                current_interval_atk_strats = full_trial_history_data["atk_strat"][start_index:]
                current_interval_def_strats = full_trial_history_data["def_strat"][start_index:]

                interval_atk_strat_counts = {s: current_interval_atk_strats.count(s) for s in attacker_strategies}
                interval_def_strat_counts = {s: current_interval_def_strats.count(s) for s in defender_strategies}

                interval_summary = {
                    "trial": t + 1, "logged_at_step": step_num + 1,
                    "attacker_model": attacker_model, "defender_model": defender_model, # Log both models
                    "topology": topology, "num_nodes": num_nodes,
                    "connectivity_param": str(connectivity), "seed": current_trial_seed,
                    # Log relevant params for *each* agent's model
                    "atk_alpha": attacker_params.get("alpha", np.nan), "atk_gamma": attacker_params.get("gamma", np.nan),
                    "atk_epsilon_start": attacker_params.get("epsilon_start", np.nan),
                    "atk_epsilon_decay": attacker_params.get("epsilon_decay", np.nan),
                    "atk_epsilon_min": attacker_params.get("epsilon_min", np.nan),
                    "atk_hybrid_static_steps": attacker_params.get("hybrid_static_steps", np.nan),
                    "current_atk_epsilon": atk_epsilon_val, # Log instantaneous epsilon

                    "def_alpha": defender_params.get("alpha", np.nan), "def_gamma": defender_params.get("gamma", np.nan),
                    "def_epsilon_start": defender_params.get("epsilon_start", np.nan),
                    "def_epsilon_decay": defender_params.get("epsilon_decay", np.nan),
                    "def_epsilon_min": defender_params.get("epsilon_min", np.nan),
                     "def_hybrid_static_steps": defender_params.get("hybrid_static_steps", np.nan),
                    "current_def_epsilon": def_epsilon_val, # Log instantaneous epsilon

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


        # Save detailed history if requested for this model type
        if attacker_model in save_detailed_history_models or defender_model in save_detailed_history_models:
            # Use a unique identifier for the file based on models and config
            models_str = f"{attacker_model.replace(' ', '_')}_vs_{defender_model.replace(' ', '_')}"
            topo_str = topology.replace(" (Erdős–Rényi)", "_ER").replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
            conn_str = str(connectivity).replace(".", "p").replace(" ", "").replace("(", "").replace(")", "").replace(",", "_") if connectivity is not None else "def"
            # Create a more robust hash for parameters
            atk_params_str = "_".join([f"{k}{v}" for k,v in sorted(attacker_params.items())])
            def_params_str = "_".join([f"{k}{v}" for k,v in sorted(defender_params.items())])
            config_hash = abs(hash(f"{atk_params_str}_{def_params_str}")) % 100000 # Larger hash range

            detailed_filename = f"details_{models_str}_{topo_str}_N{num_nodes}_C{conn_str}_cfg{config_hash}_trial_{t+1}_seed{current_trial_seed}.csv"
            full_detailed_path = os.path.join(detailed_history_full_path_dir, detailed_filename)
            try:
                pd.DataFrame(full_trial_history_data).to_csv(full_detailed_path, index=False)
                # print(f"      Detailed history saved to {full_detailed_path}") # Uncomment for verbose logging
            except Exception as e: print(f"      Error saving full detailed history for trial {t+1}: {e}")


    df_interval_results = pd.DataFrame(all_interval_results)
    try:
        df_interval_results.to_csv(save_path, index=False) # save_path now points to the interval log
        # print(f"  Interval results for configuration saved to {save_path}") # Uncomment for verbose logging
    except Exception as e: print(f"  Error saving interval results to {save_path}: {e}")
    return df_interval_results


# --- Main Experiment Execution ---
if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    RESULTS_MAIN_DIR = f"results_{timestamp}" # Timestamped results folder

    # 🔹 Step 2: Define lists for Attacker and Defender models to test
    ATTACKER_MODELS = ["Q-Learning", "Bayesian Game", "Static"]
    DEFENDER_MODELS = ["Q-Learning", "Bayesian Game", "Static"]
    # Add other models here as they are implemented (e.g., "Coalition Formation", "Stackelberg")

    TOPOLOGIES = ["Random (Erdős–Rényi)", "Small-World", "Star", "Ring"] # Reduced set for faster testing
    NODE_COUNTS = [10, 20, 50] # Keep low for dev
    CONNECTIVITY_PARAMS = { "Random (Erdős–Rényi)": [0.4], "Small-World": [(6, 0.1)], "Star": [None] }

    RUNS_PER_SETTING = 5 # Keep low for dev, increase for paper (e.g., 10-30)
    SIMULATION_STEPS = 500 # 🔹 Step 5: Increased simulation horizon (500-1000 is better for learning convergence)

    # 🔹 Step 2: Define parameter grids/defaults for each model type
    # These can be tuned independently per agent type
    Q_LEARNING_PARAMS_GRID = { # For Q-Learning specific tuning
        "alpha": [0.1], # Keep simple for now
        "gamma": [0.9],
        "epsilon_start": [0.5], # Start higher for exploration
        "epsilon_decay": [0.998], # Slower decay for more steps
        "epsilon_min": [0.01],
        "hybrid_static_steps": [0], # Can test hybrid Q-learning here
        "q_init_val": [0.01] # Optimistic initial Q value
    }

    BAYESIAN_PARAMS_DEFAULT = {} # Bayesian model typically doesn't have hyperparameters like Q-learning
    STATIC_PARAMS_DEFAULT = {} # Static model has no learning parameters

    # Map model types to their parameter grids/defaults
    MODEL_PARAMS = {
        "Q-Learning": Q_LEARNING_PARAMS_GRID,
        "Bayesian Game": BAYESIAN_PARAMS_DEFAULT,
        "Static": STATIC_PARAMS_DEFAULT,
        # Add other models here
    }

    SAVE_DETAILED_HISTORY_FOR_MODELS = ["Q-Learning"] # Save detailed history for learning models

    os.makedirs(RESULTS_MAIN_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_MAIN_DIR, DETAILED_HISTORIES_SUBDIR), exist_ok=True)

    all_experiment_runs_dfs = [] # To collect DFs from each `simulate` call (optional, primary output is CSVs)
    config_counter = 0

    print(f"--- Starting Batch Experiment Runner (Cross-Model Matchups) ---")
    print(f"Results will be saved in: {RESULTS_MAIN_DIR}")

    # 🔹 Step 2: Iterate through all combinations of attacker and defender models
    for atk_model in ATTACKER_MODELS:
        for def_model in DEFENDER_MODELS:
            # Determine parameter sets based on model types
            atk_param_sets = []
            if atk_model == "Q-Learning":
                 # Generate combinations for Q-Learning attacker
                 # Using nested loops for clarity, itertools.product is more concise for many params
                 for alpha_val in Q_LEARNING_PARAMS_GRID["alpha"]:
                     for gamma_val in Q_LEARNING_PARAMS_GRID["gamma"]:
                         for eps_start_val in Q_LEARNING_PARAMS_GRID["epsilon_start"]:
                             for eps_decay_val in Q_LEARNING_PARAMS_GRID["epsilon_decay"]:
                                 for eps_min_val in Q_LEARNING_PARAMS_GRID["epsilon_min"]:
                                     for hybrid_val in Q_LEARNING_PARAMS_GRID["hybrid_static_steps"]:
                                         for q_init_val in Q_LEARNING_PARAMS_GRID["q_init_val"]:
                                             atk_param_sets.append({
                                                 "alpha": alpha_val, "gamma": gamma_val,
                                                 "epsilon_start": eps_start_val, "epsilon_decay": eps_decay_val,
                                                 "epsilon_min": eps_min_val, "hybrid_static_steps": hybrid_val,
                                                 "q_init_val": q_init_val
                                             })
            else:
                 atk_param_sets.append(MODEL_PARAMS.get(atk_model, {})) # Use default/empty params

            def_param_sets = []
            if def_model == "Q-Learning":
                 # Generate combinations for Q-Learning defender (can use same grid or a different one)
                 for alpha_val in Q_LEARNING_PARAMS_GRID["alpha"]:
                     for gamma_val in Q_LEARNING_PARAMS_GRID["gamma"]:
                         for eps_start_val in Q_LEARNING_PARAMS_GRID["epsilon_start"]:
                             for eps_decay_val in Q_LEARNING_PARAMS_GRID["epsilon_decay"]:
                                 for eps_min_val in Q_LEARNING_PARAMS_GRID["epsilon_min"]:
                                     for hybrid_val in Q_LEARNING_PARAMS_GRID["hybrid_static_steps"]:
                                         for q_init_val in Q_LEARNING_PARAMS_GRID["q_init_val"]:
                                             def_param_sets.append({
                                                 "alpha": alpha_val, "gamma": gamma_val,
                                                 "epsilon_start": eps_start_val, "epsilon_decay": eps_decay_val,
                                                 "epsilon_min": eps_min_val, "hybrid_static_steps": hybrid_val,
                                                 "q_init_val": q_init_val
                                             })
            else:
                 def_param_sets.append(MODEL_PARAMS.get(def_model, {})) # Use default/empty params


            # Iterate through all combinations of topology and network parameters
            for topo in TOPOLOGIES:
                for nodes in NODE_COUNTS:
                    for conn_param_key, conn_param_values in CONNECTIVITY_PARAMS.items():
                        if topo != conn_param_key: continue # Match topology with its connectivity options
                        for conn_param in conn_param_values:

                            # Iterate through all combinations of attacker and defender parameters
                            for atk_params in atk_param_sets:
                                for def_params in def_param_sets:

                                    config_counter += 1
                                    print(f"\n[Config {config_counter}] Running:")
                                    print(f"  Attacker Model: {atk_model}, Defender Model: {def_model}")
                                    print(f"  Topology: {topo}, Nodes: {nodes}, Connectivity: {conn_param}")
                                    if atk_model == "Q-Learning": print(f"  Attacker Params: {atk_params}")
                                    if def_model == "Q-Learning": print(f"  Defender Params: {def_params}")


                                    file_atk_model_clean = atk_model.replace(' ', '_')
                                    file_def_model_clean = def_model.replace(' ', '_')
                                    file_topo_clean = topo.replace(" (Erdős–Rényi)", "_ER").replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
                                    file_conn_clean = str(conn_param).replace(".", "p").replace(" ", "").replace("(", "").replace(")", "").replace(",", "_") if conn_param is not None else "def"

                                    # Create a unique suffix based on parameters for the filename
                                    atk_param_suffix = "_".join([f"{k.replace('_','')}{str(v).replace('.','p')}" for k,v in sorted(atk_params.items())]) if atk_params else ""
                                    def_param_suffix = "_".join([f"{k.replace('_','')}{str(v).replace('.','p')}" for k,v in sorted(def_params.items())]) if def_params else ""

                                    file_suffix = f"_{atk_param_suffix}_{def_param_suffix}".replace("__", "_").strip("_") # Combine and clean up

                                    log_file_base = f"{file_atk_model_clean}_vs_{file_def_model_clean}_{file_topo_clean}_N{nodes}_C{file_conn_clean}_{file_suffix}"
                                    interval_log_save_path = os.path.join(RESULTS_MAIN_DIR, f"{log_file_base}_interval_log.csv")

                                    trial_results_df = simulate(
                                        attacker_model=atk_model,
                                        defender_model=def_model,
                                        topology=topo,
                                        num_nodes=nodes,
                                        steps=SIMULATION_STEPS,
                                        trials=RUNS_PER_SETTING,
                                        save_path=interval_log_save_path,
                                        connectivity=conn_param,
                                        attacker_strategies=ATTACKER_STRATEGIES_DEFAULT, # Use the full default list for all models
                                        defender_strategies=DEFENDER_STRATEGIES_DEFAULT, # Use the full default list for all models
                                        attacker_params=atk_params, # Pass specific params
                                        defender_params=def_params, # Pass specific params
                                        save_detailed_history_models=SAVE_DETAILED_HISTORY_FOR_MODELS,
                                        log_interval=LOG_METRICS_INTERVAL,
                                        seed_base=random.randint(1, 100000) # Use a fresh seed base for each config
                                    )
                                    if trial_results_df is not None and not trial_results_df.empty:
                                        all_experiment_runs_dfs.append(trial_results_df)

    if all_experiment_runs_dfs:
        # The primary output is now individual interval log files per configuration.
        # The plot_results.py script will load and consolidate these.
        print(f"\nAll experiment configurations complete. Interval logs saved in '{RESULTS_MAIN_DIR}'.")
        # Optional: Save a single combined file if needed for easier loading in plot_results.py
        # This can be large, so loading individual files in plot_results might be better.
        # combined_df = pd.concat(all_experiment_runs_dfs, ignore_index=True)
        # combined_path = os.path.join(RESULTS_MAIN_DIR, "ALL_EXPERIMENTS_CONSOLIDATED_intervals.csv")
        # try: combined_df.to_csv(combined_path, index=False)
        # except Exception as e: print(f"Error saving combined interval summary: {e}")

    else: print("\nNo results generated from experiments.")
