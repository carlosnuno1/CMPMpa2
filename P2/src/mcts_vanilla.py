from mcts_node import MCTSNode
from random import choice
from math import sqrt, log

num_nodes = 1000    # at 1000 for testing against rollout
nodes = input("how many nodes: ")
num_nodes = int(nodes)
explore_faction = sqrt(2)   # changed from 2

def traverse_nodes(node, board, state, identity):
    """
    Selection phase: traverse the tree from root to leaf using UCB values.
    
    Args:
        node: Current node in traversal
        board: The game board
        state: Current game state
        identity: Bot's player number
    Returns:
        (leaf_node, leaf_state): The selected leaf node and its game state
    """
    # If reach terminal state or node with unexplored actions, stop
    if board.is_ended(state) or len(node.untried_actions) > 0:
        return node, state

    # Find child with highest UCB value
    best_child = None
    best_ucb = float('-inf')

    for action, child in node.child_nodes.items():
        child_ucb = ucb(child, node, identity)
        if child_ucb > best_ucb:
            best_ucb = child_ucb
            best_child = child

    # Move to best child and continue traversal
    next_state = board.next_state(state, best_child.parent_action)
    return traverse_nodes(best_child, board, next_state, identity)

def expand_leaf(node, board, state):
    """
    Expansion phase: add a new child node to our tree.
    
    Args:
        node: Node to expand from
        board: The game board
        state: Current game state
    Returns:
        (new_node, new_state): The newly created node and its game state
    """
    if board.is_ended(state) or not node.untried_actions:
        return node, state

    # Select random untried action
    action = choice(node.untried_actions)
    node.untried_actions.remove(action)
    
    # Create new child node
    next_state = board.next_state(state, action)
    child_node = MCTSNode(
        parent=node,
        parent_action=action,
        action_list=board.legal_actions(next_state)
    )
    
    node.child_nodes[action] = child_node
    return child_node, next_state

def rollout(board, state):
    """
    Simulation phase: play out the game from current state using random moves,
    but with basic winning move detection.
    
    Args:
        board: The game board
        state: Starting state for rollout
    Returns:
        Final game state after rollout
    """
    rollout_state = state
    
    while not board.is_ended(rollout_state):
        legal_moves = board.legal_actions(rollout_state)
        
        # Check for winning moves
        for move in legal_moves:
            next_state = board.next_state(rollout_state, move)
            if board.is_ended(next_state):
                points = board.points_values(next_state)
                if points[board.current_player(rollout_state)] == 1:
                    return next_state
        
        # If no winning move, play random
        action = choice(legal_moves)
        rollout_state = board.next_state(rollout_state, action)
    
    return rollout_state

def backpropagate(node, won):
    """
    Backpropagation phase: update statistics for all nodes in path from leaf to root.
    
    Args:
        node: Starting (leaf) node
        won: 1 for win, 0 for loss, 0.5 for draw
    """
    while node is not None:
        node.visits += 1
        node.wins += won
        node = node.parent

def ucb(node, parent, identity):
    """
    Calculate Upper Confidence Bound for trees (UCT) value for a node.
    This balances exploitation (winning moves) with exploration (unexplored moves).
    
    Args:
        node: The child node to calculate UCB for
        parent: The parent of the child node
        identity: The bot's player number (1 or 2)
    Returns:
        The UCB value for this node
    """
    if node.visits == 0:
        return float('inf')  # unvisited nodes potential set to inf
    
    # Calculate win rate
    win_rate = node.wins / node.visits
    # If its opponents turn, invert win rate 
    if parent.visits % 2 != identity - 1:
        win_rate = 1 - win_rate
    
    # UCB = win_rate + exploration_term
    return win_rate + explore_faction * sqrt(2 * log(parent.visits) / node.visits)

def is_win(board, state, identity):
    """
    Check if the given player has won the game.
    
    Args:
        board: The game board
        state: Current game state
        identity: Player number to check for win
    Returns:
        True if player has won, False otherwise
    """
    points = board.points_values(state)
    if points is None:
        return False
    return points[identity] == 1

def get_best_action(node):
    """
    Select the best action from the root node based on both visit count and win rate.
    
    Args:
        node: The root node of our MCTS tree
    Returns:
        The action with the best combined score
    """
    best_score = float('-inf')
    best_action = None
    
    for action, child in node.child_nodes.items():
        if child.visits == 0:
            continue
        
        # Score combines win rate , small bonus for highly visited nodes
        score = (child.wins / child.visits) + 0.1 * sqrt(child.visits)
        if score > best_score:
            best_score = score
            best_action = action
    
    # If no good move go back to most visited node
    if best_action is None:
        best_action = max(node.child_nodes.items(), 
                         key=lambda x: x[1].visits)[0]
    
    return best_action

def think(board, state):
    """
    Main MCTS function: run simulations to build game tree and select best move.
    
    Args:
        board: The game board
        state: Current game state
    Returns:
        The selected best action
    """
    identity_of_bot = board.current_player(state)
    
    # Create root node
    root_node = MCTSNode(
        parent=None,
        parent_action=None,
        action_list=board.legal_actions(state)
    )

    # Run MCTS simulations
    for _ in range(num_nodes):
        current_state = state
        node = root_node
        
        #  Find a leaf node
        node, current_state = traverse_nodes(node, board, current_state, identity_of_bot)
        
        # Add a new node if not terminal
        if not board.is_ended(current_state):
            node, current_state = expand_leaf(node, board, current_state)
        
        # Play out the game
        final_state = rollout(board, current_state)
        
        # Update node stats
        points = board.points_values(final_state)
        if points[identity_of_bot] == 1:
            backpropagate(node, 1)  # Win
        elif points[identity_of_bot] == -1:
            backpropagate(node, 0)  # Loss
        else:
            backpropagate(node, 0.5)  # Draw

    # Select and return best action
    return get_best_action(root_node)
