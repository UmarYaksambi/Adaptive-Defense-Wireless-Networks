# run_experiments.py
import pandas as pd
import random
import numpy as np
import os
import networkx as nx # For topology metrics

# Import core simulation functions
from core_sim import (generate_network, select_strategies, execute_attack,
                      execute_defense, payoff, update_learning_models,
                      initialize_learning_models, assign_coalitions,
                      ATTACKER_STRATEGIES_DEFAULT, DEFENDER_STRATEGIES_DEFAULT)

# Default configurations (can be overridden by experiment parameters)
DEFAULT_ATTACKER_STRATEGIES = ATTACKER_STRATEGIES_DEFAULT
DEFAULT_DEFENDER_STRATEGIES = DEFENDER_STRATEGIES_DEFAULT
DEFAULT_FREQUENCIES = [1, 2, 3, 4, 5]
DETAILED_HISTORIES_SUBDIR = "detailed_histories" # Subdirectory for detailed trial logs


def get_topology_metrics(G):
    """Calculates and returns key topology metrics for graph G."""
    metrics = {
        "avg_degree": np.nan,
        "density": np.nan,
        "avg_clustering_coefficient": np.nan,
        "diameter": np.nan, # Can be computationally expensive for large graphs
        "is_connected": False
    }
    num_nodes = G.number_of_nodes()

    if num_nodes == 0:
        return metrics

    metrics["density"] = nx.density(G)
    
    degrees = [d for n, d in G.degree()]
    if degrees:
        metrics["avg_degree"] = sum(degrees) / num_nodes
    
    # Avg clustering coefficient can be 0 for graphs with no triangles (e.g. star graph with N > 2, or path graph)
    # It is defined as 0 for nodes with degree < 2.
    try:
        metrics["avg_clustering_coefficient"] = nx.average_clustering(G)
    except Exception: # nx.average_clustering might raise error for some specific small graphs
        pass


    if nx.is_connected(G):
        metrics["is_connected"] = True
        if num_nodes > 1: # Diameter is defined for connected graphs with at least 2 nodes
            try:
                metrics["diameter"] = nx.diameter(G)
            except nx.NetworkXError: # Should not happen if connected, but as safeguard
                metrics["diameter"] = np.inf # Or some other indicator
    else:
        metrics["is_connected"] = False
        metrics["diameter"] = np.inf # Diameter is infinite for disconnected graphs

    return metrics


def simulate(game_model, topology, num_nodes=10, steps=30, trials=5, save_path="results.csv",
             connectivity=0.5,
             frequencies_list=None,
             attacker_strategies=None,
             defender_strategies=None,
             seed_base=42,
             learning_alpha=0.1,
             learning_gamma=0.9,
             learning_epsilon=0.2,
             save_detailed_history_models=None): # List of models to save detailed history for

    if frequencies_list is None:
        frequencies_list = DEFAULT_FREQUENCIES
    if attacker_strategies is None:
        attacker_strategies = DEFAULT_ATTACKER_STRATEGIES
    if defender_strategies is None:
        defender_strategies = DEFAULT_DEFENDER_STRATEGIES
    if save_detailed_history_models is None:
        save_detailed_history_models = []


    all_trials_results = []
    results_base_dir = os.path.dirname(save_path)
    detailed_history_full_path_dir = os.path.join(results_base_dir, DETAILED_HISTORIES_SUBDIR)

    if game_model in save_detailed_history_models:
        os.makedirs(detailed_history_full_path_dir, exist_ok=True)


    for t in range(trials):
        current_trial_seed = seed_base + t
        random.seed(current_trial_seed)
        np.random.seed(current_trial_seed)
        print(f"    Trial {t+1}/{trials} with seed {current_trial_seed}...")

        learning_models = initialize_learning_models(
            game_model, attacker_strategies, defender_strategies,
            force_reset=True,
            alpha=learning_alpha, gamma=learning_gamma, epsilon=learning_epsilon
        )

        G = generate_network(topology, num_nodes, connectivity, frequencies_list, seed=current_trial_seed)
        topology_metrics = get_topology_metrics(G) # Calculate topology metrics

        if game_model == "Coalition Formation":
             G = assign_coalitions(G)

        history = {
            "step": [], "atk_strat": [], "def_strat": [],
            "atk_payoff": [], "def_payoff": [], "net_health": [],
            "atk_cost": [], "def_cost": [], "jammed_nodes_count": [], "detected": []
        }

        cumulative_attacker_payoff = 0.0
        cumulative_defender_payoff = 0.0
        detection_count = 0
        current_epsilon = learning_epsilon

        for step_num in range(steps):
            atk_strat, def_strat = select_strategies(
                game_model, step_num, G, attacker_strategies, defender_strategies,
                history, learning_models, epsilon=current_epsilon
            )

            jammed_freqs_set, _, current_detect_prob = execute_attack(atk_strat, G, def_strat, frequencies_list)
            G = execute_defense(def_strat, G, jammed_freqs_set, frequencies_list)

            jammed_nodes_list = []
            protected_nodes_list = []
            for node_id in G.nodes:
                final_freq = G.nodes[node_id]['frequency']
                is_on_jammed_freq = final_freq in jammed_freqs_set
                node_can_resist = False
                if is_on_jammed_freq:
                    if def_strat == "spread_spectrum" and len(jammed_freqs_set) < len(frequencies_list) * 0.6:
                        node_can_resist = True
                    elif def_strat == "error_coding" and random.random() < 0.6:
                        node_can_resist = True
                if is_on_jammed_freq and not node_can_resist:
                    G.nodes[node_id]['status'] = 'jammed'
                    jammed_nodes_list.append(node_id)
                else:
                    G.nodes[node_id]['status'] = 'resistant' if is_on_jammed_freq and node_can_resist else 'ok'
                    protected_nodes_list.append(node_id)

            jammed_count = len(jammed_nodes_list)
            success_count = num_nodes - jammed_count
            network_health = success_count / num_nodes if num_nodes > 0 else 0

            atk_cost_val, def_cost_val, atk_reward, def_reward, detected_this_step = payoff(
                atk_strat, def_strat, G, success_count, jammed_nodes_list, protected_nodes_list, current_detect_prob
            )

            cumulative_attacker_payoff += atk_reward
            cumulative_defender_payoff += def_reward
            if detected_this_step:
                detection_count += 1

            history["step"].append(step_num + 1)
            history["atk_strat"].append(atk_strat)
            history["def_strat"].append(def_strat)
            history["atk_payoff"].append(atk_reward)
            history["def_payoff"].append(def_reward)
            history["net_health"].append(network_health)
            history["atk_cost"].append(atk_cost_val)
            history["def_cost"].append(def_cost_val)
            history["jammed_nodes_count"].append(jammed_count)
            history["detected"].append(detected_this_step)

            learning_models = update_learning_models(
                game_model, atk_strat, def_strat, atk_reward, def_reward, learning_models,
                attacker_strategies, defender_strategies,
                alpha=learning_models.get('alpha', learning_alpha),
                gamma=learning_models.get('gamma', learning_gamma)
            )

        avg_atk_payoff = cumulative_attacker_payoff / steps if steps > 0 else 0
        avg_def_payoff = cumulative_defender_payoff / steps if steps > 0 else 0
        final_net_health = history["net_health"][-1] if history["net_health"] else 0
        overall_detection_rate = detection_count / steps if steps > 0 else 0

        atk_strat_counts = {s: history["atk_strat"].count(s) for s in attacker_strategies}
        def_strat_counts = {s: history["def_strat"].count(s) for s in defender_strategies}

        trial_summary = {
            "trial": t + 1,
            "game_model": game_model,
            "topology": topology,
            "num_nodes": num_nodes,
            "connectivity_param": str(connectivity), # Store connectivity param as string
            "steps": steps,
            "seed": current_trial_seed,
            "avg_attacker_payoff": avg_atk_payoff,
            "avg_defender_payoff": avg_def_payoff,
            "final_network_health": final_net_health,
            "detection_rate": overall_detection_rate,
            **{f"topo_{k}": v for k, v in topology_metrics.items()}, # Add topology metrics
            **{f"atk_freq_{s.replace(' ', '_')}": c for s, c in atk_strat_counts.items()},
            **{f"def_freq_{s.replace(' ', '_')}": c for s, c in def_strat_counts.items()}
        }
        all_trials_results.append(trial_summary)

        if game_model in save_detailed_history_models:
            # Construct filename for detailed history
            base_filename_no_ext = os.path.splitext(os.path.basename(save_path))[0]
            detailed_filename = f"{base_filename_no_ext}_trial_{t+1}_details.csv"
            full_detailed_path = os.path.join(detailed_history_full_path_dir, detailed_filename)
            try:
                pd.DataFrame(history).to_csv(full_detailed_path, index=False)
                print(f"      Detailed history saved to: {full_detailed_path}")
            except Exception as e:
                print(f"      Error saving detailed history: {e}")

        print(f"    Trial {t+1} completed. Avg Atk Payoff: {avg_atk_payoff:.2f}, Final Health: {final_net_health:.2f}")

    df_results = pd.DataFrame(all_trials_results)
    try:
        df_results.to_csv(save_path, index=False)
        print(f"  Results for current configuration saved to {save_path}")
    except Exception as e:
        print(f"  Error saving results to {save_path}: {e}")

    return df_results

# --- Main Experiment Execution ---
if __name__ == "__main__":
    MODELS = ["Q-Learning", "Bayesian Game", "Static", "Coalition Formation"]
    TOPOLOGIES = ["Random (Erdős–Rényi)", "Star", "Ring", "Small-World", "Fully Connected"]
    NODE_COUNTS = [10, 20] # Example: 10 and 20 nodes
    CONNECTIVITY_PARAMS = {
        "Random (Erdős–Rényi)": [0.2, 0.5],
        "Star": [None],
        "Ring": [None],
        "Small-World": [4, (6, 0.1)], # k or (k, p_rewire)
        "Fully Connected": [None]
    }

    RUNS_PER_SETTING = 5
    SIMULATION_STEPS = 50
    BASE_FREQUENCIES = list(range(1, 6))

    ALPHA = 0.1
    GAMMA = 0.9
    EPSILON = 0.2
    
    # Specify which models should have their detailed step-by-step history saved
    SAVE_DETAILED_HISTORY_FOR_MODELS = ["Q-Learning"] # Add other models if needed

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    # Also ensure the subdir for detailed histories is globally known or created within simulate
    os.makedirs(os.path.join(results_dir, DETAILED_HISTORIES_SUBDIR), exist_ok=True)


    all_experiment_results_dfs = []
    experiment_count = 0
    total_experiments_configurations = 0
    for model in MODELS:
        for topo in TOPOLOGIES:
            for nodes in NODE_COUNTS:
                for conn_param in CONNECTIVITY_PARAMS.get(topo, [0.5]):
                    total_experiments_configurations +=1

    print(f"--- Starting Batch Experiment Runner ---")
    print(f"Total experiment configurations to run: {total_experiments_configurations}")
    print(f"Runs per setting (trials): {RUNS_PER_SETTING}")
    print(f"Simulation steps per trial: {SIMULATION_STEPS}\n")

    for model in MODELS:
        for topo in TOPOLOGIES:
            for nodes in NODE_COUNTS:
                current_conn_params_options = CONNECTIVITY_PARAMS.get(topo, [0.5]) # Default to 0.5 if not specified

                for conn_param in current_conn_params_options:
                    experiment_count += 1
                    print(f"\n[{experiment_count}/{total_experiments_configurations}] Running Configuration:")
                    print(f"  Model: {model}, Topology: {topo}, Nodes: {nodes}, Connectivity: {conn_param}")

                    file_name_topo = topo.replace(" (Erdős–Rényi)", "_ER").replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
                    file_name_conn = str(conn_param).replace(".", "p").replace(" ", "").replace("(", "").replace(")", "").replace(",", "_") if conn_param is not None else "default"
                    
                    save_file_base = f"{model.replace(' ', '_')}_{file_name_topo}_N{nodes}_C{file_name_conn}"
                    summary_save_path = os.path.join(results_dir, f"{save_file_base}_summary.csv") # Specific name for summary of this config

                    trial_results_df = simulate(
                        game_model=model,
                        topology=topo,
                        num_nodes=nodes,
                        connectivity=conn_param if conn_param is not None else (0.5 if topo == "Random (Erdős–Rényi)" else 4), # Provide default connectivity for ER and SW if None
                        steps=SIMULATION_STEPS,
                        trials=RUNS_PER_SETTING,
                        save_path=summary_save_path,
                        frequencies_list=BASE_FREQUENCIES,
                        attacker_strategies=DEFAULT_ATTACKER_STRATEGIES,
                        defender_strategies=DEFAULT_DEFENDER_STRATEGIES,
                        seed_base=random.randint(1, 10000), # Random base seed for each configuration set
                        # seed_base=42 # Use a fixed seed_base for all experiments if you want the *entire set* to be reproducible identically
                        learning_alpha=ALPHA,
                        learning_gamma=GAMMA,
                        learning_epsilon=EPSILON,
                        save_detailed_history_models=SAVE_DETAILED_HISTORY_FOR_MODELS
                    )
                    if trial_results_df is not None and not trial_results_df.empty:
                        all_experiment_results_dfs.append(trial_results_df)

    if all_experiment_results_dfs:
        combined_df = pd.concat(all_experiment_results_dfs, ignore_index=True)
        combined_path = os.path.join(results_dir, "ALL_EXPERIMENTS_summary.csv")
        try:
            combined_df.to_csv(combined_path, index=False)
            print(f"\nAll experiments complete. Combined summary saved to {combined_path}")
        except Exception as e:
            print(f"Error saving combined summary: {e}")
    else:
        print("\nNo results generated from experiments.")