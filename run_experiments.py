# run_experiments.py
import pandas as pd
import random
import numpy as np
import os

# Import core simulation functions
from core_sim import (generate_network, select_strategies, execute_attack,
                      execute_defense, payoff, update_learning_models,
                      initialize_learning_models, assign_coalitions,
                      ATTACKER_STRATEGIES_DEFAULT, DEFENDER_STRATEGIES_DEFAULT)
                      # ATTACK_COST, DEFENSE_COST are now in core_sim.py

# Define these globally or pass them around if they vary per experiment
# These can be overridden by experiment parameters if needed
DEFAULT_ATTACKER_STRATEGIES = ATTACKER_STRATEGIES_DEFAULT
DEFAULT_DEFENDER_STRATEGIES = DEFENDER_STRATEGIES_DEFAULT
DEFAULT_FREQUENCIES = [1, 2, 3, 4, 5]


def simulate(game_model, topology, num_nodes=10, steps=30, trials=5, save_path="results.csv",
             connectivity=0.5, # Can be float for Erdos-Renyi, or int (k) or tuple (k,p) for Small-World
             frequencies_list=None, 
             attacker_strategies=None, 
             defender_strategies=None,
             seed_base=42, 
             reset_learning_per_trial=True,
             learning_alpha=0.1,
             learning_gamma=0.9,
             learning_epsilon=0.2):
    
    if frequencies_list is None:
        frequencies_list = DEFAULT_FREQUENCIES
    if attacker_strategies is None:
        attacker_strategies = DEFAULT_ATTACKER_STRATEGIES
    if defender_strategies is None:
        defender_strategies = DEFAULT_DEFENDER_STRATEGIES

    all_trials_results = []

    for t in range(trials):
        current_trial_seed = seed_base + t
        random.seed(current_trial_seed)
        np.random.seed(current_trial_seed)

        # Initialize or Reset learning models for each trial
        learning_models = initialize_learning_models(
            game_model, attacker_strategies, defender_strategies,
            force_reset=True, # Always reset for a new trial in this context
            alpha=learning_alpha, gamma=learning_gamma, epsilon=learning_epsilon
        )
        
        # Generate network with its own seed for reproducibility of topology
        G = generate_network(topology, num_nodes, connectivity, frequencies_list, seed=current_trial_seed)
        
        if game_model == "Coalition Formation":
             G = assign_coalitions(G) # Assign coalitions at the start of the trial

        history = {
            "step": [], "atk_strat": [], "def_strat": [],
            "atk_payoff": [], "def_payoff": [], "net_health": [],
            "atk_cost": [], "def_cost": [], "jammed_nodes_count": [], "detected": []
        }
        
        cumulative_attacker_payoff = 0.0
        cumulative_defender_payoff = 0.0
        detection_count = 0

        current_epsilon = learning_epsilon # For potential epsilon decay

        for step_num in range(steps):
            # Epsilon decay (optional, simple linear decay example)
            # current_epsilon = learning_epsilon - (learning_epsilon * 0.9 * (step_num / steps)) if steps > 0 else learning_epsilon
            # current_epsilon = max(0.01, current_epsilon) # Ensure epsilon doesn't go to zero if exploration is always desired

            atk_strat, def_strat = select_strategies(
                game_model, step_num, G, attacker_strategies, defender_strategies,
                history, learning_models, epsilon=current_epsilon
            )
            
            jammed_freqs_set, _, current_detect_prob = execute_attack(atk_strat, G, def_strat, frequencies_list)
            G = execute_defense(def_strat, G, jammed_freqs_set, frequencies_list)

            # Outcome Evaluation
            jammed_nodes_list = []
            protected_nodes_list = [] # Includes resistant nodes
            
            for node_id in G.nodes:
                final_freq = G.nodes[node_id]['frequency']
                is_on_jammed_freq = final_freq in jammed_freqs_set
                
                node_can_resist = False
                if is_on_jammed_freq:
                    if def_strat == "spread_spectrum" and len(jammed_freqs_set) < len(frequencies_list) * 0.6: # Example resistance
                        node_can_resist = True
                    elif def_strat == "error_coding" and random.random() < 0.6: # Example 60% chance
                        node_can_resist = True
                
                if is_on_jammed_freq and not node_can_resist:
                    G.nodes[node_id]['status'] = 'jammed'
                    jammed_nodes_list.append(node_id)
                else:
                    G.nodes[node_id]['status'] = 'resistant' if is_on_jammed_freq and node_can_resist else 'ok'
                    protected_nodes_list.append(node_id)
            
            jammed_count = len(jammed_nodes_list)
            # success_count is number of non-jammed nodes
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
                alpha=learning_models.get('alpha', learning_alpha), # Use alpha from models if changed, else default
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
            "connectivity": str(connectivity), # Store connectivity param as string for varied types
            "steps": steps,
            "seed": current_trial_seed,
            "avg_attacker_payoff": avg_atk_payoff,
            "avg_defender_payoff": avg_def_payoff,
            "final_network_health": final_net_health,
            "detection_rate": overall_detection_rate,
            **{f"atk_freq_{s.replace(' ', '_')}": c for s, c in atk_strat_counts.items()},
            **{f"def_freq_{s.replace(' ', '_')}": c for s, c in def_strat_counts.items()}
        }
        all_trials_results.append(trial_summary)

        # Optional: Save detailed history for each trial
        # trial_detail_filename = os.path.join(os.path.dirname(save_path), f"{os.path.splitext(os.path.basename(save_path))[0]}_trial_{t+1}_details.csv")
        # pd.DataFrame(history).to_csv(trial_detail_filename, index=False)
        
        print(f"  Trial {t+1}/{trials} completed. Avg Atk Payoff: {avg_atk_payoff:.2f}, Avg Def Payoff: {avg_def_payoff:.2f}, Final Health: {final_net_health:.2f}")

    df_results = pd.DataFrame(all_trials_results)
    try:
        df_results.to_csv(save_path, index=False)
        print(f"Results for {game_model}, {topology}, {num_nodes} nodes saved to {save_path}")
    except Exception as e:
        print(f"Error saving results to {save_path}: {e}")


    return df_results

# --- Main Experiment Execution ---
if __name__ == "__main__":
    # --- Experiment Configuration ---
    MODELS = ["Q-Learning", "Bayesian Game", "Static", "Coalition Formation"] 
    TOPOLOGIES = ["Random (Erdős–Rényi)", "Star", "Ring", "Small-World", "Fully Connected"]
    NODE_COUNTS = [10, 20] # Example: 10 and 20 nodes
    # Connectivity parameters for each topology if they differ significantly
    # For Erdos-Renyi: p (probability of edge creation)
    # For Small-World: (k, p_rewire) tuple or just k (integer neighbors)
    CONNECTIVITY_PARAMS = {
        "Random (Erdős–Rényi)": [0.2, 0.5],
        "Star": [None], # No specific connectivity param needed other than N
        "Ring": [None], # No specific connectivity param needed other than N
        "Small-World": [4, (6, 0.1)], # k=4 or (k=6, p=0.1)
        "Fully Connected": [None]
    }

    RUNS_PER_SETTING = 5  # 'trials' in simulate function
    SIMULATION_STEPS = 50
    BASE_FREQUENCIES = list(range(1, 6)) # 5 frequencies

    # Learning parameters (can also be part of the experiment matrix)
    ALPHA = 0.1 # Learning rate
    GAMMA = 0.9 # Discount factor
    EPSILON = 0.2 # Exploration rate for Q-learning

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    all_experiment_results_dfs = []

    experiment_count = 0
    total_experiments = 0
    for model in MODELS:
        for topo in TOPOLOGIES:
            for nodes in NODE_COUNTS:
                for conn_param in CONNECTIVITY_PARAMS.get(topo, [0.5]): # Use default 0.5 if topo not in dict or has generic param
                    total_experiments +=1
    
    print(f"--- Starting Batch Experiment Runner ---")
    print(f"Total experiment configurations to run: {total_experiments}")
    print(f"Runs per setting (trials): {RUNS_PER_SETTING}")
    print(f"Simulation steps per trial: {SIMULATION_STEPS}\n")


    for model in MODELS:
        for topo in TOPOLOGIES:
            for nodes in NODE_COUNTS:
                # Determine connectivity parameter for the current topology
                current_conn_params = CONNECTIVITY_PARAMS.get(topo, [0.5]) # Default to 0.5 if not specified
                
                for conn_param in current_conn_params:
                    experiment_count += 1
                    print(f"\n[{experiment_count}/{total_experiments}] Running: Model={model}, Topology={topo}, Nodes={nodes}, Connectivity={conn_param}")
                    
                    file_name_topo = topo.replace(" (Erdős–Rényi)", "_ER").replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
                    file_name_conn = str(conn_param).replace(".", "p").replace(" ", "").replace("(", "").replace(")", "").replace(",", "_") if conn_param is not None else "default"
                    
                    save_file = f"{model.replace(' ', '_')}_{file_name_topo}_N{nodes}_C{file_name_conn}.csv"
                    save_path = os.path.join(results_dir, save_file)
                    
                    trial_results_df = simulate(
                        game_model=model,
                        topology=topo,
                        num_nodes=nodes,
                        connectivity=conn_param if conn_param is not None else 0.5, # Pass appropriate connectivity
                        steps=SIMULATION_STEPS,
                        trials=RUNS_PER_SETTING,
                        save_path=save_path,
                        frequencies_list=BASE_FREQUENCIES,
                        attacker_strategies=DEFAULT_ATTACKER_STRATEGIES, # Could vary these too
                        defender_strategies=DEFAULT_DEFENDER_STRATEGIES, # Could vary these too
                        seed_base=random.randint(1, 10000), # Use a random base seed for each setting for diversity, or fixed for full reproducibility
                        # seed_base=42 # Use a fixed seed_base for all experiments if you want the *entire set* to be reproducible identically
                        learning_alpha=ALPHA,
                        learning_gamma=GAMMA,
                        learning_epsilon=EPSILON
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