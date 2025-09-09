#!/usr/bin/env python3
import random
import math
import time

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()
        self.max_board_width = 20
        self.max_board_height = 20
        self.default_search_depth = 10
        self.time_per_move_seconds = 0.06
        self.transposition_table = {}
        self.history_heuristic = {}
        self.killer_moves = {}
        self.random_generator = random.Random(42)
        self.last_best_move = 0
        self.node_evaluations = 0

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move, elapsed = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": elapsed})

    

    def minimax_loop(self):

        return

    def alpha_beta_prune(self,):

        return

    def search_best_next_move(self, initial_tree_node):
        """
        Use minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """
        # Iterative deepening alpha-beta with transposition table, killer moves,
        # history heuristic, and repetition checks.
        start_time = time.time()
        # Honor the framework time threshold if available
        framework_budget = getattr(self.settings, 'time_threshold', 0.075)
        time_budget = min(self.time_per_move_seconds, max(0.01, framework_budget * 0.9))
        self.node_evaluations = 0
        self.transposition_table.clear()
        best_move_int = self.last_best_move
        available_depth = len(initial_tree_node.observations)
        max_depth = min(self.default_search_depth, available_depth)
        root_children = initial_tree_node.compute_and_get_children()
        ordered_children = self.order_children(initial_tree_node, root_children, None)
        pv = []
        for depth in range(1, max_depth + 1):
            if time.time() - start_time > time_budget:
                break
            alpha = -math.inf # The best score found so far for the maximizing player (-inf)
            beta = math.inf # The best score found so far for the minimizing player (inf)
            current_best_move = best_move_int # The best move found so far for the current node
            current_best_score = -math.inf # The best score found so far for the current node
            current_pv = []
            for child in ordered_children:
                move_int = child.move
                score, line = self.alpha_beta(
                    child,
                    depth - 1,
                    alpha,
                    beta,
                    start_time,
                    time_budget,
                    set([self.compute_state_key(initial_tree_node)])
                )
                if score > current_best_score or (score == current_best_score and move_int == self.last_best_move):
                    current_best_score = score
                    current_best_move = move_int
                    current_pv = [move_int] + line
                alpha = max(alpha, current_best_score)
                if time.time() - start_time > time_budget:
                    break
            best_move_int = current_best_move
            pv = current_pv
            # Move ordering improvement for next iteration
            ordered_children = self.order_children(initial_tree_node, root_children, best_move_int)
        self.last_best_move = best_move_int
        return ACTION_TO_STR[best_move_int], time.time() - start_time

    def alpha_beta(self, node, depth, alpha, beta, start_time, time_budget, repetition_keys):
        # Time cutoff
        if time.time() - start_time > time_budget:
            return self.evaluate_state(node.state), []
        # Terminal or depth limit
        if depth == 0 or len(node.observations) == node.depth:
            return self.evaluate_state(node.state), []
        # Repetition detection
        state_key = self.compute_state_key(node)
        if state_key in repetition_keys:
            return self.score_difference(node.state) * 0.5, []
        # Transposition table lookup
        tt_entry = self.transposition_table.get((state_key, depth))
        if tt_entry is not None:
            stored_type, stored_value, stored_move, stored_pv = tt_entry
            if stored_type == "EXACT":
                return stored_value, stored_pv
            elif stored_type == "LOWER" and stored_value > alpha:
                alpha = stored_value
            elif stored_type == "UPPER" and stored_value < beta:
                beta = stored_value
            if alpha >= beta:
                return stored_value, stored_pv
        # Expand
        children = node.compute_and_get_children()
        if not children:
            return self.evaluate_state(node.state), []
        ordered = self.order_children(node, children, tt_entry[2] if tt_entry else None)
        best_move = None
        best_pv = []
        if node.state.get_player() == 0:
            # Maximizing player
            value = -math.inf
            for child in ordered:
                repetition_keys.add(state_key)
                score, pv = self.alpha_beta(child, depth - 1, alpha, beta, start_time, time_budget, repetition_keys)
                repetition_keys.discard(state_key)
                if score > value:
                    value = score
                    best_move = child.move
                    best_pv = [child.move] + pv
                alpha = max(alpha, value)
                if alpha >= beta:
                    self.record_killer(node.depth, child.move)
                    break
            # Store in TT
            entry_type = "EXACT"
            if value <= alpha:
                entry_type = "UPPER"
            elif value >= beta:
                entry_type = "LOWER"
            self.transposition_table[(state_key, depth)] = (entry_type, value, best_move, best_pv)
            self.bump_history(node.depth, best_move, depth)
            return value, best_pv
        else:
            # Minimizing player
            value = math.inf
            for child in ordered:
                repetition_keys.add(state_key)
                score, pv = self.alpha_beta(child, depth - 1, alpha, beta, start_time, time_budget, repetition_keys)
                repetition_keys.discard(state_key)
                if score < value:
                    value = score
                    best_move = child.move
                    best_pv = [child.move] + pv
                beta = min(beta, value)
                if alpha >= beta:
                    self.record_killer(node.depth, child.move)
                    break
            entry_type = "EXACT"
            if value <= alpha:
                entry_type = "UPPER"
            elif value >= beta:
                entry_type = "LOWER"
            self.transposition_table[(state_key, depth)] = (entry_type, value, best_move, best_pv)
            self.bump_history(node.depth, best_move, depth)
            return value, best_pv

    def order_children(self, node, children, preferred_move):
        # Order: PV/TT move first, then killer moves, then by static move score heuristic, finally history heuristic
        scored = []
        killer_set = set(self.killer_moves.get(node.depth, []))
        for child in children:
            bonus = 0
            if preferred_move is not None and child.move == preferred_move:
                bonus += 1000000
            if child.move in killer_set:
                bonus += 500000
            static_score = self.static_move_score(node, child)
            hist = self.history_heuristic.get((node.depth, child.move), 0)
            scored.append((-(bonus + static_score + hist), self.random_generator.random(), child))
        scored.sort()
        return [c for _, __, c in scored]

    def static_move_score(self, node, child):
        # Prefer moves that increase our immediate score lead or catch high-value fish quickly
        parent_diff = self.score_difference(node.state)
        child_diff = self.score_difference(child.state)
        delta = child_diff - parent_diff
        caught = node.state.get_caught()
        encourage_up = 2 if caught[node.state.get_player()] is not None and child.move == 1 else 0
        return 1000 * delta + encourage_up

    def record_killer(self, depth, move):
        if depth not in self.killer_moves:
            self.killer_moves[depth] = []
        moves = self.killer_moves[depth]
        if move in moves:
            return
        moves.insert(0, move)
        if len(moves) > 2:
            moves.pop()

    def bump_history(self, depth, move, ply_depth):
        key = (depth, move)
        self.history_heuristic[key] = self.history_heuristic.get(key, 0) + ply_depth * ply_depth

    def compute_state_key(self, node):
        s = node.state
        p = s.get_player()
        scores = s.get_player_scores()
        caught = s.get_caught()
        hooks_dict = s.get_hook_positions()
        hooks = (hooks_dict[0], hooks_dict[1])
        fish_pos = tuple(sorted(s.get_fish_positions().items()))
        return (p, scores, caught, hooks, fish_pos)

    def score_difference(self, state):
        s0, s1 = state.get_player_scores()
        return s0 - s1

    def _toroidal_dx(self, x0, x1):
        return min((x1 - x0) % self.max_board_width, (x0 - x1) % self.max_board_width)

    def manhattan_distance(self, p, q):
        """
        Manhattan distance adapted to toroidal x-axis.
        p, q are (x, y)
        """
        dx = self._toroidal_dx(p[0], q[0])
        dy = abs(p[1] - q[1])
        return dx + dy

    def euclidean_distance(self, p, q):
        """
        Euclidean distance adapted to toroidal x-axis.
        p, q are (x, y)
        """
        dx = self._toroidal_dx(p[0], q[0])
        dy = abs(p[1] - q[1])
        return math.hypot(dx, dy)

    def evaluate_state(self, state):
        # Heuristic: current score diff + potential of remaining fish considering distances
        self.node_evaluations += 1
        diff = self.score_difference(state) * 1000
        hooks = state.get_hook_positions()
        fish_positions = state.get_fish_positions()
        fish_scores = state.get_fish_scores()
        caught = state.get_caught()
        # Reward having a fish on rod
        if caught[0] is not None and caught[0] in fish_positions:
            fish_y = fish_positions[caught[0]][1]
            depth_to_surface = (self.max_board_height - 1) - fish_y
            diff += 500 + 50 * max(0, 9 - depth_to_surface)
        if caught[1] is not None and caught[1] in fish_positions:
            fish_y = fish_positions[caught[1]][1]
            depth_to_surface = (self.max_board_height - 1) - fish_y
            diff -= 500 + 50 * max(0, 9 - depth_to_surface)
        # Potential of free fish
        for fish_num, (fx, fy) in fish_positions.items():
            value = fish_scores[fish_num]
            if value == 0:
                continue
            mpos = hooks[0]
            opos = hooks[1]
            #Manhattan distance for number-of-moves estimate
            dist_m = self.manhattan_distance(mpos, (fx, fy))
            dist_o = self.manhattan_distance(opos, (fx, fy))
            if dist_m == 0:
                dist_m = 0.5
            if dist_o == 0:
                dist_o = 0.5
            potential = 100 * value * (1.0 / (1.0 + dist_m))
            opp_potential = 100 * value * (1.0 / (1.0 + dist_o))
            diff += potential - 0.7 * opp_potential
        return diff
