from mcts_node import MCTSNode
from random import choice
from math import sqrt, log

num_nodes = 1000  # Default value; can be modified via user input
nodes = input("How many nodes: ")
num_nodes = int(nodes)
explore_factor = 0.9  # Exploration factor (sqrt(2)) for UCB

def traverse_nodes(node, board, state, identity):
    """
    Selection phase: traverse the tree from root to leaf using UCB values.
    """
    while not board.is_ended(state) and not node.untried_actions:
        best_child = max(
            node.child_nodes.values(),
            key=lambda child: ucb(child, node, identity)
        )
        state = board.next_state(state, best_child.parent_action)
        node = best_child
    return node, state

def expand_leaf(node, board, state):
    """
    Expansion phase: add a new child node to our tree.
    """
    if board.is_ended(state) or not node.untried_actions:
        return node, state

    action = choice(node.untried_actions)
    node.untried_actions.remove(action)
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
    prioritizing winning moves if available.
    """
    while not board.is_ended(state):
        legal_moves = board.legal_actions(state)
        winning_move = next(
            (move for move in legal_moves if board.is_ended(board.next_state(state, move))),
            None
        )
        action = winning_move if winning_move else choice(legal_moves)
        state = board.next_state(state, action)
    return state

def backpropagate(node, won):
    """
    Backpropagation phase: update statistics for all nodes in path from leaf to root.
    """
    while node is not None:
        node.visits += 1
        node.wins += won
        node = node.parent

def ucb(node, parent, identity):
    """
    Calculate Upper Confidence Bound for trees (UCT) value for a node.
    """
    if node.visits == 0:
        return float('inf')

    win_rate = node.wins / node.visits
    if identity != parent.parent_action:  # Adjust for opponent's turn
        win_rate = 1 - win_rate
    exploration_term = explore_factor * sqrt(log(parent.visits) / node.visits)
    return win_rate + exploration_term

def get_best_action(node):
    """
    Select the best action based on visit count.
    """
    best_action = max(node.child_nodes.items(), key=lambda item: item[1].visits)[0]
    return best_action

def think(board, state):
    """
    Main MCTS function: run simulations to build game tree and select the best move.
    """
    identity_of_bot = board.current_player(state)
    root_node = MCTSNode(
        parent=None,
        parent_action=None,
        action_list=board.legal_actions(state)
    )

    for _ in range(num_nodes):
        node = root_node
        current_state = state

        # Selection
        node, current_state = traverse_nodes(node, board, current_state, identity_of_bot)

        # Expansion
        if not board.is_ended(current_state):
            node, current_state = expand_leaf(node, board, current_state)

        # Simulation
        final_state = rollout(board, current_state)

        # Backpropagation
        points = board.points_values(final_state)
        bot_score = points.get(identity_of_bot, 0)
        backpropagate(node, bot_score)

    return get_best_action(root_node)
