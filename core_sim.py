# core_sim.py
import networkx as nx
import numpy as np
import random
import math

# --- Constants ---
# 🔹 Step 1: Optimize Attacker Strategy Logic (Add new strategies)
ATTACKER_STRATEGIES_DEFAULT = [
    "broadband", "sweep", "reactive", "targeted", "power_burst", "intelligent",
    "reactive_strong", "central_node_targeting" # New strategies
]
DEFENDER_STRATEGIES_DEFAULT = [
    "hop", "detect_and_switch", "stay", "spread_spectrum", "error_coding", "cooperative"
]

# 🔹 Step 1: Lower cost for new attackers
ATTACK_COST = {
    "broadband": 5.0, "sweep": 3.0, "reactive": 2.5, "targeted": 2.0, "power_burst": 4.0, "intelligent": 1.5,
    "reactive_strong": 2.5, # New
    "central_node_targeting": 2.0, # New
    "default": 1.0
}
DEFENSE_COST = {
    "hop": 2.0, "detect_and_switch": 1.5, "stay": 0.2, "spread_spectrum": 3.0, "error_coding": 2.5, "cooperative": 1.0,
    "default": 0.5
}

# --- Helper Functions for State Representation (🔹 Step 3: Enhance Q-Learning State) ---
def get_health_bucket(network_health):
    """Buckets network health into low, medium, or high."""
    if network_health < 0.33: return "low"
    if network_health < 0.66: return "medium"
    return "high"

def get_jammed_freqs_bucket(jammed_freq_count, total_freqs):
    """Buckets the proportion of jammed frequencies."""
    if total_freqs == 0: return "none" # Avoid division by zero
    if jammed_freq_count == 0: return "none"
    if jammed_freq_count <= total_freqs * 0.4: return "few"
    return "many"

def get_step_phase(step_num, total_steps):
    """Determines the current phase of the simulation (early, mid, late)."""
    if total_steps == 0: return "early" # Avoid division by zero
    if step_num < total_steps / 3: return "early"
    if step_num < 2 * total_steps / 3: return "mid"
    return "late"

def get_most_common_strategy(history_list, window=5):
    """Determines the most common strategy in the recent history of a player."""
    if not history_list or len(history_list) < window:
        return "None" # Not enough history or no history
    recent_strategies = history_list[-window:]
    if not recent_strategies: return "None"
    counts = {s: recent_strategies.count(s) for s in set(recent_strategies)}
    return max(counts, key=counts.get) if counts else "None"


# 🔹 Step 3: Modified get_q_learning_state for cross-model compatibility
def get_q_learning_state(history, current_step_num, total_simulation_steps, num_frequencies,
                         last_attacker_strat, last_defender_strat, # Last actions taken by both players
                         last_jammed_count, last_net_health): # Outcomes of PREVIOUS step
    """
    Enhanced Q-Learning State Representation for both Attacker and Defender.
    Includes: network_health_bucket, jammed_bucket, step_phase,
              last_attacker_strat, last_defender_strat,
              most_common_attacker_strategy (recent), most_common_defender_strategy (recent).
    """
    health_bucket = get_health_bucket(last_net_health)
    step_phase = get_step_phase(current_step_num, total_simulation_steps)
    jammed_bucket = get_jammed_freqs_bucket(last_jammed_count, num_frequencies)

    # Get recent history for both players
    most_common_atk_hist = get_most_common_strategy(history.get("atk_strat", []), window=5)
    most_common_def_hist = get_most_common_strategy(history.get("def_strat", []), window=5)

    # The state tuple includes information about both players' recent actions and the game state
    state_tuple_elements = (
        health_bucket,
        jammed_bucket,
        step_phase,
        str(last_attacker_strat),
        str(last_defender_strat),
        str(most_common_atk_hist),
        str(most_common_def_hist),
    )
    return state_tuple_elements


# --- Core Simulation Functions (Modified) ---

def generate_network(topology_type, n_nodes, connect_param, frequencies_list, seed=None):
    """Generates a network graph based on specified topology and parameters."""
    if seed is not None: random.seed(seed); np.random.seed(seed)
    G = nx.Graph(); G.add_nodes_from(range(n_nodes))
    for i in range(n_nodes):
        G.nodes[i]['frequency'] = random.choice(frequencies_list) if frequencies_list else 1
        G.nodes[i]['status'] = 'idle'; G.nodes[i]['type'] = 'normal'; G.nodes[i]['coalition'] = None

    # Generate graph based on topology
    if topology_type == "Random (Erdős–Rényi)": G = nx.erdos_renyi_graph(n_nodes, connect_param, seed=seed)
    elif topology_type == "Star": G = nx.star_graph(n_nodes -1) if n_nodes > 1 else nx.Graph(); G.add_nodes_from(range(n_nodes))
    elif topology_type == "Ring": G = nx.cycle_graph(n_nodes) if n_nodes > 1 else nx.Graph(); G.add_nodes_from(range(n_nodes))
    elif topology_type == "Small-World":
        k = int(connect_param[0]) if isinstance(connect_param, tuple) else int(connect_param)
        p_rewire = connect_param[1] if isinstance(connect_param, tuple) else 0.1
        # Ensure k is valid for the number of nodes
        k = min(k, n_nodes - 1) if n_nodes > 1 else 0
        if k > 0: G = nx.watts_strogatz_graph(n_nodes, k, p_rewire, seed=seed)
        else: G = nx.Graph(); G.add_nodes_from(range(n_nodes)) # Handle edge case of k=0 or n_nodes=1
    elif topology_type == "Fully Connected": G = nx.complete_graph(n_nodes)
    else: G = nx.erdos_renyi_graph(n_nodes, connect_param, seed=seed) # Default to Random

    # Ensure attributes exist after graph recreation (important for some nx functions)
    for i in G.nodes():
        if 'frequency' not in G.nodes[i]: G.nodes[i]['frequency'] = random.choice(frequencies_list) if frequencies_list else 1
        if 'status' not in G.nodes[i]: G.nodes[i]['status'] = 'idle'
        if 'type' not in G.nodes[i]: G.nodes[i]['type'] = 'normal'
        if 'coalition' not in G.nodes[i]: G.nodes[i]['coalition'] = None

    # Handle single node case explicitly if graph generation somehow failed
    if n_nodes == 1 and not G.nodes():
        G.add_node(0)
        G.nodes[0]['frequency']=random.choice(frequencies_list) if frequencies_list else 1
        G.nodes[0]['status']='idle'
        G.nodes[0]['type']='normal'
        G.nodes[0]['coalition']=None

    return G

# 🔹 Step 2: Modify Payoff Function
def payoff(attacker_strat, defender_strat, network, success_count, jammed_nodes_list, protected_nodes_list, detection_prob,
           current_net_health, prev_net_health, max_nodes, history):
    """Calculates payoffs for attacker and defender based on actions and outcomes."""
    atk_cost = ATTACK_COST.get(attacker_strat, ATTACK_COST["default"])
    def_cost = DEFENSE_COST.get(defender_strat, DEFENSE_COST["default"])
    num_nodes = len(network.nodes) if network else 0 # Handle empty network

    atk_base_reward = (len(jammed_nodes_list) / num_nodes) * 10 if num_nodes > 0 else 0
    def_base_reward = (len(protected_nodes_list) / num_nodes) * 10 if num_nodes > 0 else 0

    detected_this_step = random.random() < detection_prob
    detection_penalty_attacker = 5.0 if detected_this_step else 0.0 # Direct reward penalty for attacker

    # Initial payoffs before shaping
    atk_reward = atk_base_reward - atk_cost - detection_penalty_attacker
    def_reward = def_base_reward - def_cost

    # --- Defender Payoff Shaping ---
    # Penalty for repeating same defense 3+ times (makes defender predictable)
    if history and len(history.get("def_strat", [])) >= 3:
        hist_def = history["def_strat"]
        if hist_def[-1] == hist_def[-2] and hist_def[-2] == hist_def[-3]: # Check last 3 including current
             # The check `hist_def[-1] == defender_strat` is implicit if calling this after action is chosen
             # Let's check the last 3 *recorded* strategies in history
            if len(hist_def) >= 3 and hist_def[-1] == hist_def[-2] == hist_def[-3]:
                 def_reward -= 1.5  # Penalty for being static/predictable
                 atk_reward += 1.0  # Attacker bonus for exploiting predictability


    # Bonus if network health improves significantly
    if current_net_health > prev_net_health + 0.05: # Threshold for significant improvement
        def_reward += (current_net_health - prev_net_health) * 5.0 # Larger bonus for significant recovery
    # Penalty if network health drops significantly
    elif current_net_health < prev_net_health - 0.05: # Threshold for significant drop
         def_reward += (current_net_health - prev_net_health) * 3.0 # Larger penalty for significant loss (negative value)


    # Penalize defender for heavy jamming events
    jam_threshold_ratio = 0.75
    if num_nodes > 0 and (len(jammed_nodes_list) / num_nodes) > jam_threshold_ratio:
        def_reward -= 2.5 # Stronger penalty for defender if >75% nodes jammed
        atk_reward += 1.5 # Bonus for attacker achieving this

    # --- Attacker Payoff Shaping ---
    # Attacker costs are handled by ATTACK_COST dictionary.
    # Specific bonuses already added (e.g., when defender is predictable, high jamming success).
    # Could add a bonus for attacker if network health significantly drops
    # if current_net_health < prev_net_health - 0.05:
    #     atk_reward += (prev_net_health - current_net_health) * 2.0 # Bonus for causing damage

    return atk_cost, def_cost, atk_reward, def_reward, detected_this_step

# 🔹 Step 1: Update execute_attack
def execute_attack(strategy, network, defender_strategy, frequencies_list):
    """Executes the attacker's chosen strategy."""
    jammed_freqs_set = set()
    detection_prob = 0.1 # Base detection probability
    num_nodes = network.number_of_nodes()

    # Strategy specific effects
    if strategy == "broadband":
        # Jams a fixed number of random frequencies
        k_jam = min(len(frequencies_list), max(1, len(frequencies_list) // 3)) # Jam at least 1, up to 1/3rd
        jammed_freqs_set = set(random.sample(frequencies_list, k=k_jam)) if frequencies_list else set()
        detection_prob = 0.5 # High detection risk

    elif strategy == "sweep":
        # Jams a single random frequency
        if frequencies_list: jammed_freqs_set.add(random.choice(frequencies_list))
        detection_prob = 0.3 # Moderate detection risk

    elif strategy == "reactive": # Original reactive - jams the most used frequency
        if num_nodes > 0:
            # Find the most common frequency among *all* nodes
            all_freqs = [d.get('frequency', frequencies_list[0] if frequencies_list else 1) for _, d in network.nodes(data=True)]
            if all_freqs:
                freq_counts = np.unique(all_freqs, return_counts=True)
                most_common_freq = freq_counts[0][np.argmax(freq_counts[1])]
                jammed_freqs_set.add(most_common_freq)
        detection_prob = 0.2 # Lower detection risk than broadband/sweep

    elif strategy == "targeted":
        # Jams a specific predefined frequency (e.g., the first one)
        if frequencies_list: jammed_freqs_set.add(frequencies_list[0]) # Target the first frequency
        detection_prob = 0.15 # Low detection risk

    elif strategy == "power_burst":
        # Jams a single random frequency but with higher intensity (handled in outcome eval implicitly)
        if frequencies_list: jammed_freqs_set.add(random.choice(frequencies_list))
        detection_prob = 0.4 # Higher detection risk due to power/intensity

    elif strategy == "intelligent": # Can be enhanced later - e.g., target frequencies of critical nodes
         if num_nodes > 0:
            # Simple intelligent: target frequency of a random 'normal' node
            normal_nodes = [n for n, d in network.nodes(data=True) if d.get('type') == 'normal']
            if normal_nodes:
                target_node = random.choice(normal_nodes)
                freq_to_jam = network.nodes[target_node].get('frequency', frequencies_list[0] if frequencies_list else 1)
                jammed_freqs_set.add(freq_to_jam)
            elif frequencies_list: # Fallback
                 jammed_freqs_set.add(random.choice(frequencies_list))
         detection_prob = 0.25 # Moderate detection risk

    # --- New Attacker Strategies ---
    elif strategy == "reactive_strong":
        # Jams all currently active frequencies (frequencies used by non-jammed nodes)
        if num_nodes > 0:
            active_freqs = set([d.get('frequency', frequencies_list[0] if frequencies_list else 1)
                                for _, d in network.nodes(data=True) if d.get('status') != 'jammed'])
            jammed_freqs_set.update(active_freqs)
        detection_prob = 0.4 # Higher detection risk due to broad impact

    elif strategy == "central_node_targeting":
        # Identifies and jams the frequency of the node with the highest degree (most central)
        if num_nodes > 0 and network.edges(): # Ensure network has edges to calculate degrees
            try:
                degrees = list(network.degree())
                if degrees:
                    central_node_id = max(degrees, key=lambda x: x[1])[0]
                    freq_to_jam = network.nodes[central_node_id].get('frequency', frequencies_list[0] if frequencies_list else 1)
                    jammed_freqs_set.add(freq_to_jam)
                elif frequencies_list: # Fallback if no degrees (e.g., isolated nodes)
                    jammed_freqs_set.add(random.choice(frequencies_list))
            except Exception: # Handle potential errors with degree calculation on complex graphs
                 if frequencies_list: jammed_freqs_set.add(random.choice(frequencies_list))
        elif frequencies_list: # Fallback for empty or single-node networks
             jammed_freqs_set.add(random.choice(frequencies_list))
        detection_prob = 0.35 # Moderate detection risk

    else: # Default/Unknown strategy
        if frequencies_list: jammed_freqs_set.add(random.choice(frequencies_list))

    return jammed_freqs_set, len(jammed_freqs_set), detection_prob


def execute_defense(strategy, network, jammed_freqs_set, frequencies_list):
    """Executes the defender's chosen strategy."""
    available_frequencies = [f for f in frequencies_list if f not in jammed_freqs_set] if frequencies_list else []
    for node_id in network.nodes:
        current_freq = network.nodes[node_id].get('frequency', frequencies_list[0] if frequencies_list else 1)
        new_freq = current_freq # Default is to stay

        # Strategy specific reactions to jamming
        is_currently_jammed = current_freq in jammed_freqs_set

        if strategy == "hop":
            # Simple frequency hopping if jammed and other frequencies are available
            if is_currently_jammed and available_frequencies:
                new_freq = random.choice(available_frequencies)

        elif strategy == "detect_and_switch":
            # Similar to hop, but implies some detection mechanism (handled by payoff/detection_prob)
            if is_currently_jammed and available_frequencies:
                new_freq = random.choice(available_frequencies)

        elif strategy == "stay":
            # Nodes attempt to stay on their current frequency regardless of jamming
            pass # new_freq remains current_freq

        elif strategy == "spread_spectrum" or strategy == "error_coding":
            # These strategies provide resistance to jamming *without* changing frequency.
            # The effect is handled in the outcome evaluation part of the simulation loop,
            # where nodes on jammed frequencies might still be marked as 'resistant' instead of 'jammed'.
            pass # new_freq remains current_freq

        elif strategy == "cooperative":
            # If a node is jammed, it tries to switch to a frequency used by a non-jammed neighbor.
            if is_currently_jammed and available_frequencies:
                safe_neighbor_freq = None
                # Find a neighbor whose frequency is NOT jammed
                for neighbor in network.neighbors(node_id):
                    neighbor_freq = network.nodes[neighbor].get('frequency')
                    if neighbor_freq is not None and neighbor_freq not in jammed_freqs_set:
                        safe_neighbor_freq = neighbor_freq
                        break # Found a safe frequency from a neighbor

                if safe_neighbor_freq is not None:
                    new_freq = safe_neighbor_freq
                elif available_frequencies:
                    # If no safe neighbor frequency found, hop to any available frequency
                    new_freq = random.choice(available_frequencies)

        # Update the node's frequency
        network.nodes[node_id]['frequency'] = new_freq

    return network

# 🔹 Step 2: Modify initialize_learning_models to handle separate agent models
def initialize_learning_models(attacker_model_type, defender_model_type, attacker_strategies, defender_strategies,
                               attacker_params, defender_params):
    """
    Initializes learning models (Q-tables or beliefs) for both attacker and defender
    based on their respective model types and parameters.
    """
    agent_models = {
        'attacker': {'type': attacker_model_type, 'params': attacker_params, 'model_state': {}},
        'defender': {'type': defender_model_type, 'params': defender_params, 'model_state': {}}
    }

    # Initialize Attacker's model state
    atk_type = attacker_model_type
    atk_params = attacker_params
    if atk_type == "Q-Learning":
        agent_models['attacker']['model_state']['q_table'] = {} # Q[state_tuple][action_string]
        agent_models['attacker']['model_state']['epsilon'] = atk_params.get('epsilon_start', 0.1)
    elif atk_type == "Bayesian Game":
        agent_models['attacker']['model_state']['belief_on_opponent_strategy'] = {d: 1.0 / len(defender_strategies) for d in defender_strategies}
        agent_models['attacker']['model_state']['observed_opponent_plays'] = {d: 0 for d in defender_strategies}
        agent_models['attacker']['model_state']['avg_payoffs'] = {a: {d: 0.0 for d in defender_strategies} for a in attacker_strategies}
        agent_models['attacker']['model_state']['payoff_counts'] = {a: {d: 0 for d in defender_strategies} for a in attacker_strategies}
    # Add initialization for other attacker model types here (e.g., Static needs no state)

    # Initialize Defender's model state
    def_type = defender_model_type
    def_params = defender_params
    if def_type == "Q-Learning":
        agent_models['defender']['model_state']['q_table'] = {} # Q[state_tuple][action_string]
        agent_models['defender']['model_state']['epsilon'] = def_params.get('epsilon_start', 0.1)
         # Handle potential Q-bias initialization for defender if needed
        if def_params.get('q_bias_init'):
             for (state, action), value in def_params['q_bias_init'].items():
                 if state not in agent_models['defender']['model_state']['q_table']:
                     agent_models['defender']['model_state']['q_table'][state] = {d: 0.01 for d in defender_strategies} # Optimistic init
                 agent_models['defender']['model_state']['q_table'][state][action] = value

    elif def_type == "Bayesian Game":
        agent_models['defender']['model_state']['belief_on_opponent_strategy'] = {a: 1.0 / len(attacker_strategies) for a in attacker_strategies}
        agent_models['defender']['model_state']['observed_opponent_plays'] = {a: 0 for a in attacker_strategies}
        agent_models['defender']['model_state']['avg_payoffs'] = {d: {a: 0.0 for a in attacker_strategies} for d in defender_strategies}
        agent_models['defender']['model_state']['payoff_counts'] = {d: {a: 0 for a in attacker_strategies} for d in defender_strategies}
# Add initialization for other defender model types here (e.g., Static needs no state)


    return agent_models

# 🔹 Step 1: Split select_strategies
def select_attacker_strategy(attacker_model_type, step_num, total_simulation_steps,
                             attacker_strategies, defender_strategies, history, attacker_model_state, attacker_params,
                             num_nodes, num_frequencies,
                             last_attacker_strat, last_defender_strat, last_jammed_count, last_net_health):
    """Selects the attacker's strategy based on their model type."""
    atk_strat = random.choice(attacker_strategies) # Default random

    if attacker_model_type == "Static":
        atk_strat = attacker_strategies[0] # Or a specific static strategy

    elif attacker_model_type == "Q-Learning":
        q_table = attacker_model_state['q_table']
        epsilon = attacker_model_state['epsilon']
        current_state_q = get_q_learning_state(history, step_num, total_simulation_steps, num_frequencies,
                                             last_attacker_strat, last_defender_strat, last_jammed_count, last_net_health)

        if random.random() < epsilon:
            atk_strat = random.choice(attacker_strategies)
        else:
            # Epsilon-greedy: Choose action with highest Q-value
            if current_state_q not in q_table or not q_table[current_state_q]:
                # Optimistic initialization for unseen states
                q_table[current_state_q] = {a: attacker_params.get('q_init_val', 0.01) for a in attacker_strategies}

            # Ensure all possible strategies are in the Q-table for this state (important for new strategies)
            for strat in attacker_strategies:
                 if strat not in q_table[current_state_q]:
                     q_table[current_state_q][strat] = attacker_params.get('q_init_val', 0.01)

            if q_table[current_state_q]:
                 # Handle cases where all Q values are the same (e.g., initial state) - choose randomly
                 max_q = max(q_table[current_state_q].values())
                 best_strategies = [a for a, q in q_table[current_state_q].items() if q == max_q]
                 atk_strat = random.choice(best_strategies)
            else: # Fallback if Q-table is somehow empty for state
                 atk_strat = random.choice(attacker_strategies)

    elif attacker_model_type == "Bayesian Game":
        belief = attacker_model_state['belief_on_opponent_strategy']
        avg_payoffs = attacker_model_state['avg_payoffs']

        # Calculate expected payoff for each attacker strategy
        exp_payoffs_attacker = {
            atk_s: sum(belief.get(def_s, 0) * avg_payoffs[atk_s].get(def_s, 0) # Use .get with default 0 for safety
                       for def_s in defender_strategies)
            for atk_s in attacker_strategies
        }

        # Choose the strategy with the highest expected payoff
        if exp_payoffs_attacker:
            # Handle cases where all expected payoffs are the same - choose randomly
            max_exp_payoff = max(exp_payoffs_attacker.values())
            best_strategies = [a for a, ep in exp_payoffs_attacker.items() if ep == max_exp_payoff]
            atk_strat = random.choice(best_strategies)
        else: # Fallback if no expected payoffs calculated
            atk_strat = random.choice(attacker_strategies)

    # Add selection logic for other attacker model types here (e.g., Coalition Formation)

    return atk_strat

# 🔹 Step 1: Split select_strategies
def select_defender_strategy(defender_model_type, step_num, total_simulation_steps,
                              defender_strategies, attacker_strategies, history, defender_model_state, defender_params,
                              num_nodes, num_frequencies,
                              last_attacker_strat, last_defender_strat, last_jammed_count, last_net_health):
    """Selects the defender's strategy based on their model type."""
    def_strat = random.choice(defender_strategies) # Default random

    if defender_model_type == "Static":
        def_strat = defender_strategies[0] # Or a specific static strategy

    elif defender_model_type == "Q-Learning":
        q_table = defender_model_state['q_table']
        epsilon = defender_model_state['epsilon']
        current_state_q = get_q_learning_state(history, step_num, total_simulation_steps, num_frequencies,
                                             last_attacker_strat, last_defender_strat, last_jammed_count, last_net_health)

        if random.random() < epsilon:
            def_strat = random.choice(defender_strategies)
        else:
            # Epsilon-greedy: Choose action with highest Q-value
            if current_state_q not in q_table or not q_table[current_state_q]:
                # Optimistic initialization for unseen states
                 q_table[current_state_q] = {d: defender_params.get('q_init_val', 0.01) for d in defender_strategies}

            # Ensure all possible strategies are in the Q-table for this state
            for strat in defender_strategies:
                 if strat not in q_table[current_state_q]:
                     q_table[current_state_q][strat] = defender_params.get('q_init_val', 0.01)

            if q_table[current_state_q]:
                 # Handle cases where all Q values are the same - choose randomly
                 max_q = max(q_table[current_state_q].values())
                 best_strategies = [d for d, q in q_table[current_state_q].items() if q == max_q]
                 def_strat = random.choice(best_strategies)
            else: # Fallback
                 def_strat = random.choice(defender_strategies)

    elif defender_model_type == "Bayesian Game":
        belief = defender_model_state['belief_on_opponent_strategy']
        avg_payoffs = defender_model_state['avg_payoffs']

        # Calculate expected payoff for each defender strategy
        exp_payoffs_defender = {
            def_s: sum(belief.get(atk_s, 0) * avg_payoffs[def_s].get(atk_s, 0) # Use .get with default 0 for safety
                       for atk_s in attacker_strategies)
            for def_s in defender_strategies
        }

        # Choose the strategy with the highest expected payoff
        if exp_payoffs_defender:
             # Handle cases where all expected payoffs are the same - choose randomly
             max_exp_payoff = max(exp_payoffs_defender.values())
             best_strategies = [d for d, ep in exp_payoffs_defender.items() if ep == max_exp_payoff]
             def_strat = random.choice(best_strategies)
        else: # Fallback
             def_strat = random.choice(defender_strategies)

    # Add selection logic for other defender model types here (e.g., Stackelberg)

    # Apply hybrid static steps logic if enabled for this defender
    if defender_params.get('hybrid_static_steps', 0) > 0 and step_num < defender_params['hybrid_static_steps']:
         # During hybrid steps, use a static strategy (e.g., the first one)
         def_strat = defender_strategies[0] # Or make this configurable

    return def_strat


# 🔹 Step 2: Modify update_learning_models to handle separate agent models
def update_learning_models(agent_models, current_state_tuple, atk_strat, def_strat, atk_payoff, def_payoff,
                           next_state_tuple, attacker_strategies, defender_strategies,
                           is_terminal_step=False):
    """
    Updates the learning models for both attacker and defender based on the
    current step's outcome.
    """
    # --- Attacker Update ---
    atk_model = agent_models['attacker']
    atk_type = atk_model['type']
    atk_params = atk_model['params']
    atk_state = atk_model['model_state']

    if atk_type == "Q-Learning":
        q_table = atk_state['q_table']
        alpha = atk_params.get('alpha', 0.1)
        gamma = atk_params.get('gamma', 0.9)
        epsilon = atk_state['epsilon'] # Use current epsilon

        # Ensure current state and action exist in Q-table
        if current_state_tuple not in q_table: q_table[current_state_tuple] = {a: atk_params.get('q_init_val', 0.01) for a in attacker_strategies}
        if atk_strat not in q_table[current_state_tuple]: q_table[current_state_tuple][atk_strat] = atk_params.get('q_init_val', 0.01)

        old_q_atk = q_table[current_state_tuple][atk_strat]
        max_future_q_atk = 0
        if not is_terminal_step:
            if next_state_tuple not in q_table: q_table[next_state_tuple] = {a: atk_params.get('q_init_val', 0.01) for a in attacker_strategies} # Optimistic for unseen next states
            if q_table[next_state_tuple]: max_future_q_atk = max(q_table[next_state_tuple].values())

        # Q-learning update formula
        q_table[current_state_tuple][atk_strat] = old_q_atk + alpha * (atk_payoff + gamma * max_future_q_atk - old_q_atk)

        # Epsilon decay for attacker (if Q-Learning)
        epsilon_decay = atk_params.get('epsilon_decay', 1.0)
        epsilon_min = atk_params.get('epsilon_min', 0.01)
        atk_state['epsilon'] = max(epsilon_min, epsilon * epsilon_decay)


    elif atk_type == "Bayesian Game":
        belief = atk_state['belief_on_opponent_strategy']
        observed_plays = atk_state['observed_opponent_plays']
        avg_payoffs = atk_state['avg_payoffs']
        payoff_counts = atk_state['payoff_counts']

        # Update observed plays of the opponent (defender)
        observed_plays[def_strat] += 1
        total_opponent_plays = sum(observed_plays.values())

        # Update belief about opponent's strategy distribution (simple frequency count)
        if total_opponent_plays > 0:
            for d_s in defender_strategies:
                 belief[d_s] = observed_plays[d_s] / total_opponent_plays

        # Update average payoff for the chosen strategy pair (atk_strat, def_strat)
        count_atk_def = payoff_counts[atk_strat].get(def_strat, 0) # Use .get for safety
        current_avg_payoff = avg_payoffs[atk_strat].get(def_strat, 0.0) # Use .get for safety

        avg_payoffs[atk_strat][def_strat] = (current_avg_payoff * count_atk_def + atk_payoff) / (count_atk_def + 1)
        payoff_counts[atk_strat][def_strat] = count_atk_def + 1 # Update count

    # Add update logic for other attacker model types here

    # --- Defender Update ---
    def_model = agent_models['defender']
    def_type = def_model['type']
    def_params = def_model['params']
    def_state = def_model['model_state']

    if def_type == "Q-Learning":
        q_table = def_state['q_table']
        alpha = def_params.get('alpha', 0.1)
        gamma = def_params.get('gamma', 0.9)
        epsilon = def_state['epsilon'] # Use current epsilon

        # Ensure current state and action exist in Q-table
        if current_state_tuple not in q_table: q_table[current_state_tuple] = {d: def_params.get('q_init_val', 0.01) for d in defender_strategies}
        if def_strat not in q_table[current_state_tuple]: q_table[current_state_tuple][def_strat] = def_params.get('q_init_val', 0.01)

        old_q_def = q_table[current_state_tuple][def_strat]
        max_future_q_def = 0
        if not is_terminal_step:
            if next_state_tuple not in q_table: q_table[next_state_tuple] = {d: def_params.get('q_init_val', 0.01) for d in defender_strategies}
            if q_table[next_state_tuple]: max_future_q_def = max(q_table[next_state_tuple].values())

        # Q-learning update formula
        q_table[current_state_tuple][def_strat] = old_q_def + alpha * (def_payoff + gamma * max_future_q_def - old_q_def)

        # Epsilon decay for defender (if Q-Learning)
        epsilon_decay = def_params.get('epsilon_decay', 1.0)
        epsilon_min = def_params.get('epsilon_min', 0.01)
        def_state['epsilon'] = max(epsilon_min, epsilon * epsilon_decay)

    elif def_type == "Bayesian Game":
        belief = def_state['belief_on_opponent_strategy']
        observed_plays = def_state['observed_opponent_plays']
        avg_payoffs = def_state['avg_payoffs']
        payoff_counts = def_state['payoff_counts']

        # Update observed plays of the opponent (attacker)
        observed_plays[atk_strat] += 1
        total_opponent_plays = sum(observed_plays.values())

        # Update belief about opponent's strategy distribution
        if total_opponent_plays > 0:
            for a_s in attacker_strategies:
                 belief[a_s] = observed_plays[a_s] / total_opponent_plays

        # Update average payoff for the chosen strategy pair (def_strat, atk_strat)
        count_def_atk = payoff_counts[def_strat].get(atk_strat, 0) # Use .get for safety
        current_avg_payoff = avg_payoffs[def_strat].get(atk_strat, 0.0) # Use .get for safety

        avg_payoffs[def_strat][atk_strat] = (current_avg_payoff * count_def_atk + def_payoff) / (count_def_atk + 1)
        payoff_counts[def_strat][atk_strat] = count_def_atk + 1 # Update count

    # Add update logic for other defender model types here

    return agent_models # Return the updated agent models dictionary


def assign_coalitions(network):
    """Assigns nodes to one of two coalitions randomly."""
    num_nodes = len(network.nodes);
    if num_nodes == 0: return network
    nodes_list = list(network.nodes()); random.shuffle(nodes_list)
    if num_nodes > 1:
        for i, node_id in enumerate(nodes_list): network.nodes[node_id]['coalition'] = i % 2
    elif num_nodes == 1: network.nodes[nodes_list[0]]['coalition'] = 0 # Single node in coalition 0
    return network

# Note: Coalition Formation and Stackelberg models require significantly more
# implementation detail for their strategy selection and potential learning/optimization processes.
# The current structure provides the framework to plug them in.
