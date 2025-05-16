# core_sim.py
import networkx as nx
import numpy as np
import random
import math

# --- Constants (moved from NetSim.py or defined here) ---
ATTACKER_STRATEGIES_DEFAULT = ["broadband", "sweep", "reactive", "targeted", "power_burst", "intelligent"]
DEFENDER_STRATEGIES_DEFAULT = ["hop", "detect_and_switch", "stay", "spread_spectrum", "error_coding", "cooperative"]

ATTACK_COST = {
    "broadband": 5.0, "sweep": 3.0, "reactive": 2.5, "targeted": 2.0, "power_burst": 4.0, "intelligent": 1.5,
    "default": 1.0 # Default cost for unknown strategies
}
DEFENSE_COST = {
    "hop": 2.0, "detect_and_switch": 1.5, "stay": 0.2, "spread_spectrum": 3.0, "error_coding": 2.5, "cooperative": 1.0,
    "default": 0.5 # Default cost for unknown strategies
}

# --- Core Simulation Functions ---

def generate_network(topology_type, n_nodes, connect_param, frequencies_list, seed=None):
    """
    Generates a network graph based on specified topology.
    Nodes are initialized with a frequency and 'idle' status.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    # Assign initial attributes to nodes
    for i in range(n_nodes):
        G.nodes[i]['frequency'] = random.choice(frequencies_list)
        G.nodes[i]['status'] = 'idle' # 'idle', 'ok', 'jammed', 'resistant'
        G.nodes[i]['type'] = 'normal' # For potential future use (e.g., 'sensor', 'actuator')
        G.nodes[i]['coalition'] = None # For coalition game model

    if topology_type == "Random (Erdős–Rényi)":
        G = nx.erdos_renyi_graph(n_nodes, connect_param, seed=seed)
    elif topology_type == "Star":
        G = nx.star_graph(n_nodes -1) # nx.star_graph(n) creates a graph with n+1 nodes.
        if n_nodes == 1: G = nx.Graph(); G.add_node(0) # Handle case of 1 node
    elif topology_type == "Ring":
        G = nx.cycle_graph(n_nodes)
    elif topology_type == "Small-World":
        # For small-world, connect_param can be tuple (k, p)
        # k = Each node is joined to k nearest neighbors in ring topology
        # p = The probability of rewiring each edge
        k = int(connect_param) if isinstance(connect_param, (int, float)) else 4 # Default k
        p_rewire = 0.1 # Default p_rewire
        if isinstance(connect_param, tuple) and len(connect_param) == 2:
            k = connect_param[0]
            p_rewire = connect_param[1]
        G = nx.watts_strogatz_graph(n_nodes, k, p_rewire, seed=seed)
    elif topology_type == "Fully Connected":
        G = nx.complete_graph(n_nodes)
    else:
        # Default to Erdos-Renyi if unknown
        G = nx.erdos_renyi_graph(n_nodes, connect_param, seed=seed)

    # Re-assign attributes if graph generation overwrote them (some nx functions return new graph)
    for i in G.nodes(): # Iterate over actual nodes in G
        G.nodes[i]['frequency'] = random.choice(frequencies_list)
        G.nodes[i]['status'] = 'idle'
        G.nodes[i]['type'] = 'normal'
        G.nodes[i]['coalition'] = None

    return G

def payoff(attacker_strat, defender_strat, network, success_count, jammed_nodes_list, protected_nodes_list, detection_prob):
    """
    Calculates payoffs for attacker and defender.
    Returns: atk_cost, def_cost, atk_reward, def_reward, detected_this_step
    """
    atk_cost = ATTACK_COST.get(attacker_strat, ATTACK_COST["default"])
    def_cost = DEFENSE_COST.get(defender_strat, DEFENSE_COST["default"])

    num_nodes = len(network.nodes)
    
    # Basic reward structure:
    # Attacker: Reward for jammed nodes, penalized for cost and detection
    # Defender: Reward for protected nodes (non-jammed), penalized for cost
    
    # Example: Attacker reward proportional to % of jammed nodes
    atk_base_reward = (len(jammed_nodes_list) / num_nodes) * 10 if num_nodes > 0 else 0
    
    # Example: Defender reward proportional to % of protected nodes
    def_base_reward = (len(protected_nodes_list) / num_nodes) * 10 if num_nodes > 0 else 0

    detected_this_step = random.random() < detection_prob
    
    detection_penalty_attacker = 5.0 if detected_this_step else 0.0 # Penalty if detected

    # Modify rewards based on specific strategies or game dynamics
    if attacker_strat == "intelligent" and defender_strat == "cooperative":
        # Example: Intelligent attacker might get higher reward if successful against cooperative defense
        # Or cooperative defense might mitigate intelligent attack better
        pass # Add specific logic if needed

    atk_reward = atk_base_reward - atk_cost - detection_penalty_attacker
    def_reward = def_base_reward - def_cost

    return atk_cost, def_cost, atk_reward, def_reward, detected_this_step

def execute_attack(strategy, network, defender_strategy, frequencies_list):
    """
    Simulates an attack, returns set of jammed frequencies, (placeholder for count), and detection probability.
    """
    jammed_freqs_set = set()
    # Placeholder: detection probability can depend on attacker/defender strategy match
    detection_prob = 0.1 # Base detection prob

    if strategy == "broadband":
        jammed_freqs_set = set(random.sample(frequencies_list, k=min(len(frequencies_list), 3))) # Jam 3 random freqs or all if less than 3
        detection_prob = 0.5
    elif strategy == "sweep":
        # Sweeps through frequencies, might jam one or two at a time
        jammed_freqs_set.add(random.choice(frequencies_list))
        detection_prob = 0.3
    elif strategy == "reactive":
        # Tries to identify active frequencies and jam them. For simplicity, jam a common one.
        if network.nodes:
            freq_counts = np.unique([d['frequency'] for _, d in network.nodes(data=True)], return_counts=True)
            if len(freq_counts[0]) > 0:
                most_common_freq = freq_counts[0][np.argmax(freq_counts[1])]
                jammed_freqs_set.add(most_common_freq)
        detection_prob = 0.2
    elif strategy == "targeted":
        # Targets a specific, known frequency (e.g., most critical one)
        if frequencies_list:
            jammed_freqs_set.add(frequencies_list[0]) # Example: target the first frequency
        detection_prob = 0.15
    elif strategy == "power_burst":
        # Jams a single frequency with high power, harder to overcome
        if frequencies_list:
            jammed_freqs_set.add(random.choice(frequencies_list))
        detection_prob = 0.4 # Higher power might be more detectable
    elif strategy == "intelligent":
        # More sophisticated, adapts. For now, similar to reactive but perhaps more effective.
        if network.nodes:
            freq_counts = np.unique([d['frequency'] for _, d in network.nodes(data=True)], return_counts=True)
            if len(freq_counts[0]) > 0:
                most_common_freq = freq_counts[0][np.argmax(freq_counts[1])]
                jammed_freqs_set.add(most_common_freq)
        detection_prob = 0.25 # Intelligent might be stealthier or more obvious
    else: # Default fallback
        if frequencies_list:
            jammed_freqs_set.add(random.choice(frequencies_list))

    # Placeholder for the second return value (e.g., number of attack signals sent)
    _placeholder_value = len(jammed_freqs_set) 
    
    return jammed_freqs_set, _placeholder_value, detection_prob

def execute_defense(strategy, network, jammed_freqs_set, frequencies_list):
    """
    Simulates defense actions, updates node frequencies/status in the network.
    Returns the modified network.
    """
    available_frequencies = [f for f in frequencies_list if f not in jammed_freqs_set]
    
    for node_id in network.nodes:
        current_freq = network.nodes[node_id]['frequency']
        
        if strategy == "hop":
            if current_freq in jammed_freqs_set and available_frequencies:
                network.nodes[node_id]['frequency'] = random.choice(available_frequencies)
        elif strategy == "detect_and_switch":
            # Assumes detection happens (simplified here)
            if current_freq in jammed_freqs_set and available_frequencies:
                network.nodes[node_id]['frequency'] = random.choice(available_frequencies)
            # Could add a small chance of staying if switch is costly/unavailable
        elif strategy == "stay":
            pass # No change in frequency
        elif strategy == "spread_spectrum":
            # Nodes using spread spectrum are harder to jam unless all sub-channels are hit.
            # This logic is partly handled in the outcome evaluation in simulate()
            pass
        elif strategy == "error_coding":
            # Error coding helps tolerate some level of interference.
            # This logic is partly handled in the outcome evaluation in simulate()
            pass
        elif strategy == "cooperative":
            # Nodes might share info about clear channels or help relay.
            # Simplified: if jammed, try to move to a frequency a neighbor is successfully using.
            if current_freq in jammed_freqs_set and available_frequencies:
                # Find a safe frequency from a neighbor (if any)
                safe_neighbor_freq = None
                for neighbor in network.neighbors(node_id):
                    neighbor_freq = network.nodes[neighbor]['frequency']
                    if neighbor_freq not in jammed_freqs_set:
                        safe_neighbor_freq = neighbor_freq
                        break
                if safe_neighbor_freq:
                    network.nodes[node_id]['frequency'] = safe_neighbor_freq
                elif available_frequencies: # Else, pick a random available one
                    network.nodes[node_id]['frequency'] = random.choice(available_frequencies)
        else: # Default fallback
            if current_freq in jammed_freqs_set and available_frequencies:
                 network.nodes[node_id]['frequency'] = random.choice(available_frequencies)

    return network

def initialize_learning_models(game_model, attacker_strategies, defender_strategies, force_reset=False, alpha=0.1, gamma=0.9, epsilon=0.2):
    """
    Initializes and returns learning models (Q-tables, beliefs, etc.).
    No longer uses st.session_state.
    """
    learning_models = {}
    if game_model == "Q-Learning":
        learning_models['q_attacker'] = {a: {d: 0.0 for d in defender_strategies} for a in attacker_strategies}
        learning_models['q_defender'] = {d: {a: 0.0 for a in attacker_strategies} for d in defender_strategies}
        # print("Q-Tables Initialized")

    elif game_model == "Bayesian Game":
        learning_models['attacker_belief_on_defender'] = {d: 1.0 / len(defender_strategies) for d in defender_strategies}
        learning_models['defender_belief_on_attacker'] = {a: 1.0 / len(attacker_strategies) for a in attacker_strategies}
        
        # History of opponent's plays and resulting payoffs for Bayesian updates
        learning_models['attacker_observed_defender_payoffs'] = {a: {d: [] for d in defender_strategies} for a in attacker_strategies}
        learning_models['defender_observed_attacker_payoffs'] = {d: {a: [] for a in attacker_strategies} for d in defender_strategies}
        # print("Bayesian Beliefs Initialized")
        
    elif game_model == "Static": # Example: Static game might imply fixed strategies or no learning
        # No specific learning models needed, or could pre-define strategies
        # For now, this will make select_strategies pick randomly if not handled explicitly
        pass
        
    elif game_model == "Coalition Formation":
        # May need specific models for coalition utility, stability, etc.
        # For now, let basic Q-learning or Bayesian operate per agent/coalition if applicable
        # This part would need significant expansion for true coalition logic.
        # Let's assume for now it uses Q-learning for individual agent decisions within a coalition context.
        learning_models['q_attacker'] = {a: {d: 0.0 for d in defender_strategies} for a in attacker_strategies}
        learning_models['q_defender'] = {d: {a: 0.0 for a in attacker_strategies} for d in defender_strategies}
        # print("Coalition Game (using Q-learning base) Initialized")

    # Store learning parameters
    learning_models['alpha'] = alpha
    learning_models['gamma'] = gamma
    learning_models['epsilon'] = epsilon
    
    return learning_models

def select_strategies(game_type, step, current_network, attacker_strategies, defender_strategies, history, learning_models, epsilon=0.2):
    """
    Selects strategies for attacker and defender based on the game model and learning.
    Accepts learning_models and history as parameters.
    Returns (atk_strat, def_strat)
    """
    atk_strat = random.choice(attacker_strategies)
    def_strat = random.choice(defender_strategies)

    if game_type == "Q-Learning":
        q_attacker = learning_models.get('q_attacker', {})
        q_defender = learning_models.get('q_defender', {})

        # Attacker selection (epsilon-greedy)
        if random.random() < epsilon or not q_attacker:
            atk_strat = random.choice(attacker_strategies)
        else:
            # Find best action for attacker based on its Q-table (assuming defender plays best response or based on attacker's view)
            # This is simplified; a true Q-learner estimates value of (s,a) pair.
            # Here, we assume the "state" is implicitly the game itself.
            # Attacker needs to pick an action. Let's assume it picks based on max Q value against any defender strategy.
            best_val = -float('inf')
            best_a = atk_strat
            for a_s in attacker_strategies:
                # Expected Q value for attacker's action a_s, averaging over defender's possible responses
                # weighted by defender's Q-values for those responses (complex) or just max Q(a_s, d_s)
                current_max_q_for_a_s = max(q_attacker.get(a_s, {}).values()) if q_attacker.get(a_s) else -float('inf')
                if current_max_q_for_a_s > best_val:
                    best_val = current_max_q_for_a_s
                    best_a = a_s
            atk_strat = best_a
            
        # Defender selection (epsilon-greedy)
        if random.random() < epsilon or not q_defender:
            def_strat = random.choice(defender_strategies)
        else:
            best_val = -float('inf')
            best_d = def_strat
            for d_s in defender_strategies:
                current_max_q_for_d_s = max(q_defender.get(d_s, {}).values()) if q_defender.get(d_s) else -float('inf')
                if current_max_q_for_d_s > best_val:
                    best_val = current_max_q_for_d_s
                    best_d = d_s
            def_strat = best_d

    elif game_type == "Bayesian Game":
        # Attacker chooses strategy to maximize expected payoff given belief about defender
        attacker_belief_on_defender = learning_models.get('attacker_belief_on_defender', {})
        exp_payoffs_attacker = {}
        for atk_s in attacker_strategies:
            exp_payoffs_attacker[atk_s] = 0
            for def_s in defender_strategies:
                # Need expected payoff of (atk_s, def_s) - estimate from history or a model
                # For simplicity, use average payoff from history if available
                payoffs = learning_models.get('attacker_observed_defender_payoffs', {}).get(atk_s, {}).get(def_s, [])
                avg_payoff = np.mean(payoffs) if payoffs else 0 # Default to 0 if no history
                exp_payoffs_attacker[atk_s] += attacker_belief_on_defender.get(def_s, 0) * avg_payoff
        if exp_payoffs_attacker:
             atk_strat = max(exp_payoffs_attacker, key=exp_payoffs_attacker.get)
        else:
             atk_strat = random.choice(attacker_strategies)


        # Defender chooses strategy to maximize expected payoff given belief about attacker
        defender_belief_on_attacker = learning_models.get('defender_belief_on_attacker', {})
        exp_payoffs_defender = {}
        for def_s in defender_strategies:
            exp_payoffs_defender[def_s] = 0
            for atk_s in attacker_strategies:
                payoffs = learning_models.get('defender_observed_attacker_payoffs', {}).get(def_s, {}).get(atk_s, [])
                avg_payoff = np.mean(payoffs) if payoffs else 0
                exp_payoffs_defender[def_s] += defender_belief_on_attacker.get(atk_s, 0) * avg_payoff
        if exp_payoffs_defender:
            def_strat = max(exp_payoffs_defender, key=exp_payoffs_defender.get)
        else:
            def_strat = random.choice(defender_strategies)
            
    elif game_type == "Static":
        # Example: Predefined or random strategies if "Static" implies no learning
        # For this placeholder, let's make attacker always choose "broadband" and defender "hop"
        # if those strategies exist, otherwise random.
        atk_strat = "broadband" if "broadband" in attacker_strategies else random.choice(attacker_strategies)
        def_strat = "hop" if "hop" in defender_strategies else random.choice(defender_strategies)

    elif game_type == "Coalition Formation":
        # This is complex. If agents are Q-learners, it's similar to Q-learning.
        # For now, let's assume it falls back to a Q-learning like selection or random.
        # The assign_coalitions function would set up the G.nodes[id]['coalition'].
        # Strategy selection could then be per-coalition or by representative agents.
        # Simplified: use Q-learning logic if models are Q-type.
        if 'q_attacker' in learning_models: # Check if Q-learning models are present
             # Attacker selection (epsilon-greedy)
            q_attacker = learning_models.get('q_attacker', {})
            if random.random() < epsilon or not q_attacker:
                atk_strat = random.choice(attacker_strategies)
            else:
                best_val = -float('inf')
                best_a = random.choice(attacker_strategies)
                for a_s in attacker_strategies:
                    current_max_q_for_a_s = max(q_attacker.get(a_s, {}).values()) if q_attacker.get(a_s) else -float('inf')
                    if current_max_q_for_a_s > best_val:
                        best_val = current_max_q_for_a_s
                        best_a = a_s
                atk_strat = best_a
                
            # Defender selection (epsilon-greedy)
            q_defender = learning_models.get('q_defender', {})
            if random.random() < epsilon or not q_defender:
                def_strat = random.choice(defender_strategies)
            else:
                best_val = -float('inf')
                best_d = random.choice(defender_strategies)
                for d_s in defender_strategies:
                    current_max_q_for_d_s = max(q_defender.get(d_s, {}).values()) if q_defender.get(d_s) else -float('inf')
                    if current_max_q_for_d_s > best_val:
                        best_val = current_max_q_for_d_s
                        best_d = d_s
                def_strat = best_d
        else: # Fallback for Coalition if no Q-models found
            atk_strat = random.choice(attacker_strategies)
            def_strat = random.choice(defender_strategies)


    return atk_strat, def_strat

def update_learning_models(game_type, atk_strat, def_strat, atk_payoff, def_payoff, learning_models, attacker_strategies, defender_strategies, alpha=0.1, gamma=0.9):
    """
    Updates learning models based on the outcome of a step.
    Operates on and returns the learning_models dictionary.
    """
    if game_type == "Q-Learning":
        q_attacker = learning_models.get('q_attacker')
        q_defender = learning_models.get('q_defender')

        if q_attacker is not None:
            # Simplified Q-update: Q(s,a) = Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
            # Here, "state" is implicit. We update Q(atk_strat, def_strat).
            # "Next state max Q" is tricky without explicit states. For simplicity, assume next state value is 0 for terminal or use current estimate.
            # Or, if opponent's strategy is the "state" for the Q-value:
            # Q_A(a, d) = Q_A(a, d) + alpha * (R_A + gamma * max_{a'} Q_A(a', d') - Q_A(a, d)) -> this is not quite right for 2-player
            # For a zero-sum or general-sum game, each player updates their own Q-table.
            # Q_attacker(chosen_atk_strat, chosen_def_strat)
            old_q_atk = q_attacker.get(atk_strat, {}).get(def_strat, 0.0)
            # Max future reward for attacker: max over defender's next possible moves from attacker's POV
            # This can be simplified to max_a' Q(a', def_strat) if def_strat is fixed, or more complex.
            # Simplified: no future state consideration for this basic update:
            # Q(a,d) = (1-alpha)*Q(a,d) + alpha * (R_a)  -- this is simpler, like for multi-armed bandit
            # Let's use the standard Q-learning update for (state, action) where state is opponent's action.
            # Attacker updates Q_A(d_strat, atk_strat) - value of playing atk_strat when defender plays def_strat
            
            # Update Attacker's Q-table: Q(attacker_action, defender_action)
            # The "state" for the attacker could be considered the defender's action.
            # Q_A(s=def_strat, a=atk_strat)
            # To calculate max_a' Q_A(s'=def_strat_next, a'), we'd need a model of defender's next strategy or assume it's fixed.
            # For simplicity, let's use a common simplified update for two-player games:
            # Q_A(atk_strat, def_strat) += alpha * (atk_payoff - Q_A(atk_strat, def_strat))
            # This doesn't include gamma or max_future_q.
            # A more standard approach: Q(s,a) += alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
            # Let s be an empty state (the game itself), a be the joint action (atk_strat, def_strat) - not quite right.

            # Let's update Q_attacker[atk_strat][def_strat]
            # The 'next state' is effectively the game re-starting, so what's max_q_next?
            # max_q_next_atk = max(q_attacker[atk_strat].values()) # Max Q if attacker sticks with atk_strat, defender varies
            max_q_next_atk = 0
            if q_attacker.get(atk_strat):
                 max_q_next_atk = max(q_attacker[atk_strat].values())

            q_attacker[atk_strat][def_strat] = old_q_atk + alpha * (atk_payoff + gamma * max_q_next_atk - old_q_atk)

        if q_defender is not None:
            old_q_def = q_defender.get(def_strat, {}).get(atk_strat, 0.0)
            # max_q_next_def = max(q_defender[def_strat].values()) # Max Q if defender sticks with def_strat, attacker varies
            max_q_next_def = 0
            if q_defender.get(def_strat):
                max_q_next_def = max(q_defender[def_strat].values())
            q_defender[def_strat][atk_strat] = old_q_def + alpha * (def_payoff + gamma * max_q_next_def - old_q_def)

    elif game_type == "Bayesian Game":
        # Update observed payoffs
        learning_models.get('attacker_observed_defender_payoffs', {}).setdefault(atk_strat, {}).setdefault(def_strat, []).append(atk_payoff)
        learning_models.get('defender_observed_attacker_payoffs', {}).setdefault(def_strat, {}).setdefault(atk_strat, []).append(def_payoff)

        # Update beliefs (simplified: based on frequency of opponent's plays, or could be proper Bayesian update)
        # For a proper Bayesian update, we'd use the likelihood of observing the opponent's play given our strategy and their types.
        # Simplified: increase probability of observed opponent strategy
        # This is a very naive belief update. A real Bayesian update is more complex.
        attacker_belief_on_defender = learning_models.get('attacker_belief_on_defender', {})
        if attacker_belief_on_defender:
            # Simple history-based update: count plays and normalize
            # This requires history to be accessible or pass counts
            # For now, let's assume the select_strategies uses the current beliefs,
            # and beliefs are updated based on a more sophisticated model if needed.
            # A simple update could be: slightly increase belief in the chosen def_strat
            # and renormalize. This is not strictly Bayesian.
            # For now, let's assume beliefs are updated by observing payoffs and opponent actions.
            # Example: if def_strat was played, increase its belief score.
            # This is a placeholder for a more robust Bayesian belief update mechanism.
            # One simple way: Dirichlet distribution update based on counts.
            # Let's just record history for now, actual belief update in select_strategies.
            pass # The selection logic already uses observed payoffs to guide choice.

    elif game_type == "Static":
        # No learning updates for static strategies
        pass
        
    elif game_type == "Coalition Formation":
        # If using Q-learning base for coalitions, update Q-tables similar to "Q-Learning"
        if 'q_attacker' in learning_models and 'q_defender' in learning_models:
            q_attacker = learning_models['q_attacker']
            q_defender = learning_models['q_defender']
            
            old_q_atk = q_attacker.get(atk_strat, {}).get(def_strat, 0.0)
            max_q_next_atk = 0
            if q_attacker.get(atk_strat):
                 max_q_next_atk = max(q_attacker[atk_strat].values())
            q_attacker[atk_strat][def_strat] = old_q_atk + alpha * (atk_payoff + gamma * max_q_next_atk - old_q_atk)

            old_q_def = q_defender.get(def_strat, {}).get(atk_strat, 0.0)
            max_q_next_def = 0
            if q_defender.get(def_strat):
                max_q_next_def = max(q_defender[def_strat].values())
            q_defender[def_strat][atk_strat] = old_q_def + alpha * (def_payoff + gamma * max_q_next_def - old_q_def)


    return learning_models # Return the updated models

def assign_coalitions(network):
    """
    Assigns nodes to coalitions. Example: random assignment or based on proximity.
    Modifies and returns the network.
    """
    num_nodes = len(network.nodes)
    if num_nodes == 0:
        return network
        
    # Example: Simple random assignment to one of two coalitions
    # More complex logic would go here (e.g., based on node type, connectivity, etc.)
    # Coalition IDs could be anything, e.g., 0 and 1 for two coalitions.
    # Or, could relate to attacker/defender roles within the coalition.
    
    # For simplicity, let's say there are two main coalitions: "AttackerSide" and "DefenderSide"
    # This is a very high-level assignment. Real coalition formation is dynamic.
    
    # Example: if nodes > 0, assign half to coalition 0, half to coalition 1 (approx)
    nodes_list = list(network.nodes())
    random.shuffle(nodes_list) # Shuffle for random assignment
    
    # For simulation purposes, attacker might be one entity, defenders are nodes.
    # Or, multiple attackers form a coalition, multiple defenders form another.
    # Let's assume nodes form defensive coalitions.
    
    # Example: create 2 defender coalitions if num_nodes > 1
    if num_nodes > 1:
        num_coalitions = 2 
        for i, node_id in enumerate(nodes_list):
            network.nodes[node_id]['coalition'] = i % num_coalitions
    elif num_nodes == 1:
         network.nodes[nodes_list[0]]['coalition'] = 0 # Single node in its own coalition
         
    # print(f"Assigned coalitions: {nx.get_node_attributes(network, 'coalition')}")
    return network