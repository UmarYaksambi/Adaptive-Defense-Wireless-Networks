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
    if network_health < 0.33: return "low"
    if network_health < 0.66: return "medium"
    return "high"

def get_jammed_freqs_bucket(jammed_freq_count, total_freqs):
    if total_freqs == 0: return "none" # Avoid division by zero
    if jammed_freq_count == 0: return "none"
    if jammed_freq_count <= total_freqs * 0.4: return "few"
    return "many"

def get_step_phase(step_num, total_steps):
    if total_steps == 0: return "early" # Avoid division by zero
    if step_num < total_steps / 3: return "early"
    if step_num < 2 * total_steps / 3: return "mid"
    return "late"

def get_most_common_defense(history_def_strat, window=5):
    """Determines the most common defense strategy in the recent history."""
    if not history_def_strat or len(history_def_strat) < window:
        return "None" # Not enough history or no history
    recent_defenses = history_def_strat[-window:]
    if not recent_defenses: return "None"
    counts = {d: recent_defenses.count(d) for d in set(recent_defenses)}
    return max(counts, key=counts.get) if counts else "None"


def get_q_learning_state(history, current_step_num, total_simulation_steps, num_frequencies,
                         last_atk_strat_for_state, last_def_strat_for_state, # These are from PREVIOUS step's actions
                         last_jammed_count_for_state, last_net_health_for_state): # These are outcomes of PREVIOUS step
    """
    🔹 Step 3: Enhanced Q-Learning State Representation
    Includes: last_defense_used, most_common_defense (recent), jammed_bucket, step_phase, network_health_bucket
    """
    health_bucket = get_health_bucket(last_net_health_for_state)
    step_phase = get_step_phase(current_step_num, total_simulation_steps)
    jammed_bucket = get_jammed_freqs_bucket(last_jammed_count_for_state, num_frequencies)
    
    # last_def_strat_for_state is effectively 'last_defense_used' for the state leading to current decision
    most_common_def_hist = get_most_common_defense(history.get("def_strat", []), window=5) # Use actual history dict

    state_tuple_elements = (
        health_bucket,
        str(last_def_strat_for_state), # Last defense action taken by self (for defender) or observed (for attacker)
        str(most_common_def_hist),    # Defender's recent trend
        jammed_bucket,
        step_phase,
        # Could also add last_atk_strat_for_state if useful for defender's state
        # str(last_atk_strat_for_state)
    )
    return state_tuple_elements


# --- Core Simulation Functions (Modified) ---

def generate_network(topology_type, n_nodes, connect_param, frequencies_list, seed=None):
    # ... (no changes from previous version) ...
    if seed is not None: random.seed(seed); np.random.seed(seed)
    G = nx.Graph(); G.add_nodes_from(range(n_nodes))
    for i in range(n_nodes):
        G.nodes[i]['frequency'] = random.choice(frequencies_list) if frequencies_list else 1
        G.nodes[i]['status'] = 'idle'; G.nodes[i]['type'] = 'normal'; G.nodes[i]['coalition'] = None
    if topology_type == "Random (Erdős–Rényi)": G = nx.erdos_renyi_graph(n_nodes, connect_param, seed=seed)
    elif topology_type == "Star": G = nx.star_graph(n_nodes -1);
    elif topology_type == "Ring": G = nx.cycle_graph(n_nodes)
    elif topology_type == "Small-World":
        k = int(connect_param[0]) if isinstance(connect_param, tuple) else int(connect_param)
        p_rewire = connect_param[1] if isinstance(connect_param, tuple) else 0.1
        G = nx.watts_strogatz_graph(n_nodes, k, p_rewire, seed=seed)
    elif topology_type == "Fully Connected": G = nx.complete_graph(n_nodes)
    else: G = nx.erdos_renyi_graph(n_nodes, connect_param, seed=seed)
    for i in G.nodes(): # Ensure attributes exist after graph recreation
        if 'frequency' not in G.nodes[i]: G.nodes[i]['frequency'] = random.choice(frequencies_list) if frequencies_list else 1
        if 'status' not in G.nodes[i]: G.nodes[i]['status'] = 'idle'
        if 'type' not in G.nodes[i]: G.nodes[i]['type'] = 'normal'
        if 'coalition' not in G.nodes[i]: G.nodes[i]['coalition'] = None
    if n_nodes == 1 and not G.nodes(): G.add_node(0); G.nodes[0]['frequency']=random.choice(frequencies_list) if frequencies_list else 1; G.nodes[0]['status']='idle' # Handle single node case fully
    return G

def payoff(attacker_strat, defender_strat, network, success_count, jammed_nodes_list, protected_nodes_list, detection_prob,
           current_net_health, prev_net_health, max_nodes, history): # 🔹 Step 2: Modify Payoff Function
    atk_cost = ATTACK_COST.get(attacker_strat, ATTACK_COST["default"])
    def_cost = DEFENSE_COST.get(defender_strat, DEFENSE_COST["default"])
    num_nodes = len(network.nodes) if network else 0 # Handle empty network
    
    atk_base_reward = (len(jammed_nodes_list) / num_nodes) * 10 if num_nodes > 0 else 0
    def_base_reward = (len(protected_nodes_list) / num_nodes) * 10 if num_nodes > 0 else 0
    
    detected_this_step = random.random() < detection_prob
    detection_penalty_attacker = 5.0 if detected_this_step else 0.0 # Attacker cost penalty already removed, this is direct reward penalty

    # Initial payoffs before shaping
    atk_reward = atk_base_reward - atk_cost - detection_penalty_attacker
    def_reward = def_base_reward - def_cost

    # --- Defender Payoff Shaping ---
    # Penalty for repeating same defense 3+ times
    if history and len(history.get("def_strat", [])) >= 3:
        hist_def = history["def_strat"]
        if hist_def[-1] == hist_def[-2] == hist_def[-3] and hist_def[-1] == defender_strat : # Current action makes it 3 in a row
            def_reward -= 1.5  # Penalty for being static/predictable
            atk_reward += 1.0  # Attacker bonus for exploiting predictability

    # Bonus if network health improves
    if current_net_health > prev_net_health:
        def_reward += (current_net_health - prev_net_health) * 2.0
    # Penalty if network health drops (already implicitly handled by base reward, but can emphasize)
    # elif current_net_health < prev_net_health:
    #     def_reward += (current_net_health - prev_net_health) * 1.0 # Negative value, so a penalty

    # Penalize heavy jamming events
    jam_threshold_ratio = 0.75
    if num_nodes > 0 and (len(jammed_nodes_list) / num_nodes) > jam_threshold_ratio:
        def_reward -= 2.5 # Stronger penalty for defender if >75% nodes jammed
        atk_reward += 1.5 # Bonus for attacker achieving this

    # --- Attacker Payoff Shaping ---
    # Attacker costs are generally lower now via ATTACK_COST dictionary.
    # Specific bonuses already added (e.g., when defender is predictable, high jamming success).

    return atk_cost, def_cost, atk_reward, def_reward, detected_this_step

def execute_attack(strategy, network, defender_strategy, frequencies_list): # 🔹 Step 1: Update execute_attack
    jammed_freqs_set = set()
    detection_prob = 0.1
    num_nodes = network.number_of_nodes()

    if strategy == "broadband":
        jammed_freqs_set = set(random.sample(frequencies_list, k=min(len(frequencies_list), 3))) if frequencies_list else set()
        detection_prob = 0.5
    elif strategy == "sweep":
        if frequencies_list: jammed_freqs_set.add(random.choice(frequencies_list))
        detection_prob = 0.3
    elif strategy == "reactive": # Original reactive
        if num_nodes > 0:
            freq_counts = np.unique([d.get('frequency', frequencies_list[0] if frequencies_list else 1) for _, d in network.nodes(data=True)], return_counts=True)
            if len(freq_counts[0]) > 0:
                jammed_freqs_set.add(freq_counts[0][np.argmax(freq_counts[1])])
        detection_prob = 0.2
    elif strategy == "targeted":
        if frequencies_list: jammed_freqs_set.add(frequencies_list[0])
        detection_prob = 0.15
    elif strategy == "power_burst":
        if frequencies_list: jammed_freqs_set.add(random.choice(frequencies_list))
        detection_prob = 0.4
    elif strategy == "intelligent": # Can be enhanced later
        if num_nodes > 0:
            freq_counts = np.unique([d.get('frequency', frequencies_list[0] if frequencies_list else 1) for _, d in network.nodes(data=True)], return_counts=True)
            if len(freq_counts[0]) > 0:
                jammed_freqs_set.add(freq_counts[0][np.argmax(freq_counts[1])])
        detection_prob = 0.25
    # --- New Attacker Strategies ---
    elif strategy == "reactive_strong":
        if num_nodes > 0:
            active_freqs = set([d.get('frequency', frequencies_list[0] if frequencies_list else 1) for _, d in network.nodes(data=True) if d.get('status') != 'jammed']) # Jam non-jammed nodes' freqs
            jammed_freqs_set.update(active_freqs)
        detection_prob = 0.4
    elif strategy == "central_node_targeting":
        if num_nodes > 0:
            degrees = list(network.degree())
            if degrees: # Ensure network has edges/degrees calculated
                central_node_id = max(degrees, key=lambda x: x[1])[0]
                freq_to_jam = network.nodes[central_node_id].get('frequency', frequencies_list[0] if frequencies_list else 1)
                jammed_freqs_set.add(freq_to_jam)
            elif frequencies_list: # Fallback if no clear central node (e.g., isolated nodes)
                jammed_freqs_set.add(random.choice(frequencies_list))
        detection_prob = 0.35
    else:
        if frequencies_list: jammed_freqs_set.add(random.choice(frequencies_list))
        
    return jammed_freqs_set, len(jammed_freqs_set), detection_prob


def execute_defense(strategy, network, jammed_freqs_set, frequencies_list):
    # ... (no changes from previous version) ...
    available_frequencies = [f for f in frequencies_list if f not in jammed_freqs_set] if frequencies_list else []
    for node_id in network.nodes:
        current_freq = network.nodes[node_id].get('frequency', frequencies_list[0] if frequencies_list else 1)
        new_freq = current_freq
        if strategy == "hop":
            if current_freq in jammed_freqs_set and available_frequencies: new_freq = random.choice(available_frequencies)
        elif strategy == "detect_and_switch": # Simplified
            if current_freq in jammed_freqs_set and available_frequencies: new_freq = random.choice(available_frequencies)
        elif strategy == "stay": pass
        elif strategy == "spread_spectrum" or strategy == "error_coding": pass # Resistance handled in outcome eval
        elif strategy == "cooperative":
            if current_freq in jammed_freqs_set and available_frequencies:
                safe_neighbor_freq = None
                for neighbor in network.neighbors(node_id):
                    neighbor_freq = network.nodes[neighbor].get('frequency')
                    if neighbor_freq and neighbor_freq not in jammed_freqs_set: safe_neighbor_freq = neighbor_freq; break
                if safe_neighbor_freq: new_freq = safe_neighbor_freq
                elif available_frequencies: new_freq = random.choice(available_frequencies)
        else: # Default behavior
            if current_freq in jammed_freqs_set and available_frequencies: new_freq = random.choice(available_frequencies)
        network.nodes[node_id]['frequency'] = new_freq
    return network

def initialize_learning_models(game_model, attacker_strategies, defender_strategies,
                               alpha=0.1, gamma=0.9,
                               epsilon_start=0.2, epsilon_decay=0.995, epsilon_min=0.05,
                               q_bias=None): # q_bias for potential Stackelberg init (🔹 Step 4)
    learning_models = {}
    learning_models['alpha'] = alpha; learning_models['gamma'] = gamma
    learning_models['epsilon_start'] = epsilon_start; learning_models['current_epsilon'] = epsilon_start
    learning_models['epsilon_decay'] = epsilon_decay; learning_models['epsilon_min'] = epsilon_min
    
    if game_model == "Q-Learning":
        learning_models['q_attacker'] = {} # Stateful: Q[state_tuple][action_string]
        learning_models['q_defender'] = {}
        # For Stackelberg pre-initialization (conceptual for now)
        # If q_bias is provided (e.g., derived from Stackelberg analysis for certain states/actions)
        # This would require translating Stackelberg outcomes into Q-values for specific state-action pairs.
        # Example: if q_bias = { (state1, action_A): 5.0, (state1, action_B): -2.0 }
        # if q_bias:
        #     for (state, action), value in q_bias.items():
        #         if state not in learning_models['q_defender']: learning_models['q_defender'][state] = {}
        #         learning_models['q_defender'][state][action] = value
        # This is complex to generalize without knowing the Stackelberg output format.
        # A simpler approach for bias is optimistic initialization for new states (handled in update/select).

    elif game_model == "Bayesian Game":
        learning_models['attacker_belief_on_defender_strategy'] = {d: 1.0 / len(defender_strategies) for d in defender_strategies}
        learning_models['defender_belief_on_attacker_strategy'] = {a: 1.0 / len(attacker_strategies) for a in attacker_strategies}
        learning_models['attacker_observed_defender_plays'] = {d: 0 for d in defender_strategies}
        learning_models['defender_observed_attacker_plays'] = {a: 0 for a in attacker_strategies}
        learning_models['avg_payoffs_attacker'] = {a: {d: 0.0 for d in defender_strategies} for a in attacker_strategies}
        learning_models['avg_payoffs_defender'] = {d: {a: 0.0 for a in attacker_strategies} for d in defender_strategies}
        learning_models['payoff_counts_attacker'] = {a: {d: 0 for d in defender_strategies} for a in attacker_strategies}
        learning_models['payoff_counts_defender'] = {d: {a: 0 for a in attacker_strategies} for d in defender_strategies}
    # Coalition formation and Stackelberg direct models would need their own init
    return learning_models

def select_strategies(game_model, step_num, total_simulation_steps,
                      current_G, attacker_strategies, defender_strategies, history, learning_models,
                      num_nodes, num_frequencies, # Passed for state creation
                      hybrid_static_steps=0):
    # ... (Hybrid logic, epsilon decay, and selection for Q-Learning, Bayesian, Static as in previous "enhanced" version) ...
    # (The Q-Learning selection will now use the enhanced state from get_q_learning_state)
    atk_strat = random.choice(attacker_strategies)
    def_strat = random.choice(defender_strategies)

    if step_num < hybrid_static_steps:
        atk_strat = attacker_strategies[step_num % len(attacker_strategies)]
        def_strat = defender_strategies[step_num % len(defender_strategies)]
        return atk_strat, def_strat

    current_epsilon = learning_models.get('current_epsilon', learning_models['epsilon_start'])

    if game_model == "Q-Learning":
        q_attacker_table = learning_models['q_attacker']
        q_defender_table = learning_models['q_defender']

        last_atk = history['atk_strat'][-1] if history.get('atk_strat') else "None"
        last_def = history['def_strat'][-1] if history.get('def_strat') else "None"
        last_jam_count = history['jammed_nodes_count'][-1] if history.get('jammed_nodes_count') else 0
        last_health = history['net_health'][-1] if history.get('net_health') else 1.0

        current_state_q = get_q_learning_state(history, step_num, total_simulation_steps, num_frequencies,
                                             last_atk, last_def, last_jam_count, last_health)
        
        # Attacker
        if random.random() < current_epsilon: atk_strat = random.choice(attacker_strategies)
        else:
            if current_state_q not in q_attacker_table or not q_attacker_table[current_state_q]:
                q_attacker_table[current_state_q] = {a: 0.01 for a in attacker_strategies} # Optimistic init for new states
            if q_attacker_table[current_state_q]: atk_strat = max(q_attacker_table[current_state_q], key=q_attacker_table[current_state_q].get)
            else: atk_strat = random.choice(attacker_strategies)
        # Defender
        if random.random() < current_epsilon: def_strat = random.choice(defender_strategies)
        else:
            if current_state_q not in q_defender_table or not q_defender_table[current_state_q]:
                q_defender_table[current_state_q] = {d: 0.01 for d in defender_strategies} # Optimistic init
            if q_defender_table[current_state_q]: def_strat = max(q_defender_table[current_state_q], key=q_defender_table[current_state_q].get)
            else: def_strat = random.choice(defender_strategies)
        
        new_epsilon = current_epsilon * learning_models['epsilon_decay']
        learning_models['current_epsilon'] = max(learning_models['epsilon_min'], new_epsilon)

    elif game_model == "Bayesian Game":
        exp_payoffs_attacker = {atk_s: sum(learning_models['attacker_belief_on_defender_strategy'].get(def_s, 0) * learning_models['avg_payoffs_attacker'][atk_s][def_s] for def_s in defender_strategies) for atk_s in attacker_strategies}
        if exp_payoffs_attacker: atk_strat = max(exp_payoffs_attacker, key=exp_payoffs_attacker.get)
        
        exp_payoffs_defender = {def_s: sum(learning_models['defender_belief_on_attacker_strategy'].get(atk_s, 0) * learning_models['avg_payoffs_defender'][def_s][atk_s] for atk_s in attacker_strategies) for def_s in defender_strategies}
        if exp_payoffs_defender: def_strat = max(exp_payoffs_defender, key=exp_payoffs_defender.get)

    elif game_model == "Static":
        atk_strat = attacker_strategies[0] # Could make this configurable
        def_strat = defender_strategies[0]
    
    # Coalition Formation and Stackelberg would need their own detailed selection logic
    # For Stackelberg, leader (attacker) anticipates defender's Q-learning best response.
    # This is computationally intensive per step. Simpler to use Stackelberg outcomes to bias Q-tables.

    return atk_strat, def_strat


def update_learning_models(game_model, current_state_tuple, atk_strat, def_strat, atk_payoff, def_payoff,
                           next_state_tuple, learning_models, attacker_strategies, defender_strategies,
                           is_terminal_step=False):
    # ... (Q-Learning and Bayesian update logic as in previous "enhanced" version, using the new state tuples) ...
    alpha = learning_models['alpha']; gamma = learning_models['gamma']
    if game_model == "Q-Learning":
        q_attacker = learning_models['q_attacker']; q_defender = learning_models['q_defender']
        # Ensure state-action pairs exist, init with small positive for optimism if new
        if current_state_tuple not in q_attacker: q_attacker[current_state_tuple] = {a: 0.01 for a in attacker_strategies}
        if atk_strat not in q_attacker[current_state_tuple]: q_attacker[current_state_tuple][atk_strat] = 0.01
        if current_state_tuple not in q_defender: q_defender[current_state_tuple] = {d: 0.01 for d in defender_strategies}
        if def_strat not in q_defender[current_state_tuple]: q_defender[current_state_tuple][def_strat] = 0.01
        
        # Attacker
        old_q_atk = q_attacker[current_state_tuple][atk_strat]
        max_future_q_atk = 0
        if not is_terminal_step:
            if next_state_tuple not in q_attacker: q_attacker[next_state_tuple] = {a: 0.01 for a in attacker_strategies} # Optimistic for unseen next states
            if q_attacker[next_state_tuple]: max_future_q_atk = max(q_attacker[next_state_tuple].values())
        q_attacker[current_state_tuple][atk_strat] = old_q_atk + alpha * (atk_payoff + gamma * max_future_q_atk - old_q_atk)
        # Defender
        old_q_def = q_defender[current_state_tuple][def_strat]
        max_future_q_def = 0
        if not is_terminal_step:
            if next_state_tuple not in q_defender: q_defender[next_state_tuple] = {d: 0.01 for d in defender_strategies}
            if q_defender[next_state_tuple]: max_future_q_def = max(q_defender[next_state_tuple].values())
        q_defender[current_state_tuple][def_strat] = old_q_def + alpha * (def_payoff + gamma * max_future_q_def - old_q_def)

    elif game_model == "Bayesian Game":
        learning_models['attacker_observed_defender_plays'][def_strat] += 1
        learning_models['defender_observed_attacker_plays'][atk_strat] += 1
        total_def_plays = sum(learning_models['attacker_observed_defender_plays'].values())
        if total_def_plays > 0:
            for d_s in defender_strategies: learning_models['attacker_belief_on_defender_strategy'][d_s] = learning_models['attacker_observed_defender_plays'][d_s] / total_def_plays
        total_atk_plays = sum(learning_models['defender_observed_attacker_plays'].values())
        if total_atk_plays > 0:
            for a_s in attacker_strategies: learning_models['defender_belief_on_attacker_strategy'][a_s] = learning_models['defender_observed_attacker_plays'][a_s] / total_atk_plays
        
        # Update avg payoffs
        count_atk = learning_models['payoff_counts_attacker'][atk_strat][def_strat]
        learning_models['avg_payoffs_attacker'][atk_strat][def_strat] = (learning_models['avg_payoffs_attacker'][atk_strat][def_strat] * count_atk + atk_payoff) / (count_atk + 1)
        learning_models['payoff_counts_attacker'][atk_strat][def_strat] += 1
        count_def = learning_models['payoff_counts_defender'][def_strat][atk_strat]
        learning_models['avg_payoffs_defender'][def_strat][atk_strat] = (learning_models['avg_payoffs_defender'][def_strat][atk_strat] * count_def + def_payoff) / (count_def + 1)
        learning_models['payoff_counts_defender'][def_strat][atk_strat] += 1
    return learning_models

def assign_coalitions(network):
    # ... (no changes) ...
    num_nodes = len(network.nodes);
    if num_nodes == 0: return network
    nodes_list = list(network.nodes()); random.shuffle(nodes_list)
    if num_nodes > 1:
        for i, node_id in enumerate(nodes_list): network.nodes[node_id]['coalition'] = i % 2
    elif num_nodes == 1: network.nodes[nodes_list[0]]['coalition'] = 0
    return network