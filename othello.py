import numpy


class OthelloBoard:
    # list all 8 possible moves' directions
    __directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

    def __init__(self, board=None, pieces=None):
        """
        :param board: if it's constructed based on another OthelloBoard, that OthelloBoard board  else None
        :param pieces:if it's constructed based on another OthelloBoard, that OthelloBoard pieces else None
        """
        self.board = numpy.zeros((8, 8), dtype=int)
        self.pieces = dict()

        if board is None:  # the initial board properties
            # white initial pieces
            self.board[3][3] = -1
            self.board[4][4] = -1
            # black initial pieces
            self.board[3][4] = 1
            self.board[4][3] = 1
            # position of current pieces on the board 1: black -1: white
            # (y,x) means row y and column x
            self.pieces = {1: [(3, 4), (4, 3)],
                           -1: [(3, 3), (4, 4)]}
        else:  # make a copy of the given boards properties
            # make a copy of current board
            self.board = numpy.array(board, copy=True)
            # make a copy of current positions of black and white pieces
            self.pieces = {key: [piece for piece in pieces[key]] for key in pieces.keys()}

    def show(self):
        """
        :return: return a string represent the current state of the board
        """
        cell_show = {0: '  *',
                     1: '  B',
                     -1: '  W'}
        board_state = '     0  1  2  3  4  5  6  7\n'
        for r_index, row in enumerate(self.board):
            board_state += '  {}'.format(r_index)
            for cell in row:
                board_state += cell_show[cell]
            board_state += '\n'
        return board_state

    def valid_moves(self, color):
        """

        :param color: =1 if we wants black's valid moves, =-1 if we wants white's valid moves
        :return: moves dict() which includes possible moves for all black pieces based on the current board
        (more detail about it's structure has been given in the code )
        """
        # moves dictionary structure
        # {current_pos:
        #       {possible_move_location: flip(list of positions where the opponents pieces have to flip by this move )}}
        moves = dict()

        for cell in self.pieces[color]:  # for each cell in which there are a piece of given color
            moves.update({cell: dict()})
            for direction in self.__directions:
                # for each 8 direction check if there is a valid move for this piece in that direction or not
                move = self.discover_move(cell, direction, color)
                if move != 0:
                    moves[cell].update({list(move.keys())[0]: move[list(move.keys())[0]]})

        return moves

    def discover_move(self, origin, direction, color):
        """

        :param origin: the piece location that it's moves are going to be discovered
        :param direction: the direction that the piece in the given location is going to move in
        :param color: which color is the piece in the given location (-1: white 1: black)
        :return: 0: if there is no possible move for the piece with the given color and location
                {(y,x): flip} which (y,x) is the distance location and
                flip is a list of pieces that have to flip
                (change their color to the given color) if this move is executed
        """
        y, x = origin
        y_dir, x_dir = direction
        flip = list()
        x += x_dir
        y += y_dir
        has_seen_opponent_piece = False
        while -1 < y < 8 and -1 < x < 8:
            if self.board[y][x] == color:
                return 0
            elif self.board[y][x] == 0 and has_seen_opponent_piece:
                return {(y, x): flip}
            elif self.board[y][x] == -color:
                has_seen_opponent_piece = True
                flip.append((y, x))
            x += x_dir
            y += y_dir
        return 0

    def execute_move(self, move, color):
        """

        :param move: a dict {destination: flip_list} (structured like execute_move function's output )
        :param color: -1: white is moving 1: black is moving
        :return:
        do: change the self.board based on this move (change the positions in flip list to the given color)
            update self.pieces for white(-1) and black(1) based on this move
        """
        for destination in move.keys():
            y, x = destination
            self.board[y][x] = color
            self.pieces[color].append((y, x))
            for p in move[destination]:
                y, x = p
                self.pieces[-color].remove((y, x))
                self.pieces[color].append((y, x))
                self.board[y][x] = color


class OthelloGame:

    def __init__(self):
        """
        create an initial othello board and initial other properties

        """
        # count constructed boards (and use it as a key (assigned index to each board)
        # to add them to self.game_boards dict())
        self._n = 0
        # a dict() like {board_index: OthelloBoard() } that contains all the boards (nodes in the min-max tree) with
        # the given index to each
        self.game_boards = {self._n: OthelloBoard()}
        self._n += 1
        # to keep each node in the min-max tree has been discverd with which node
        # {child: parent,..}
        self.path = dict()
        # two different type of weight for evaluation based on two different paper
        """
            using "The heuristic player’s strategy represented by WPC(Weighted Piece Counter)" table mentioned in the
            "Learning Board Evaluation Function for OthelloBoard by Hybridizing Coevolution with Temporal Difference Learning"
            by  Marcin Szubert, Wojciech Ja´skowski, and Krzysztof Krawiec
            on June 7, 2013
            paper
          """
        # self.weights = numpy.array([[1.00, -0.25, 0.10, 0.05, 0.05, 0.10, -0.25, 1.00],
        #                             [-0.25, -0.25, 0.01, 0.01, 0.01, 0.01, -0.25, -0.25],
        #                             [0.10, 0.01, 0.05, 0.02, 0.02, 0.05, 0.01, 0.10],
        #                             [0.05, 0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.05],
        #                             [0.05, 0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.05],
        #                             [0.10, 0.01, 0.05, 0.02, 0.02, 0.05, 0.01, 0.10],
        #                             [-0.25, -0.25, 0.01, 0.01, 0.01, 0.01, -0.25, -0.25],
        #                             [1.00, -0.25, 0.10, 0.05, 0.05, 0.10, -0.25, 1.00]]
        #                            )

        """
                    using "Reward of different squires" table mentioned in the
                    "Searching Algorithms in Playing Othello"
                    Zhifei Zhang and Yuechuan Chen
                    School of Mechanical, Industrial and Manufacturing Engineering
                    Oregon State University, Corvallis, OR 97331-6001 USA.
                    paper
                  """
        self.weights = numpy.array([
            [120, -20, 20, 5, 5, 20, -20, 120],
            [-20, -40, -5, -5, -5, -5, -40, -20],
            [20, -5, 15, 3, 3, 15, -5, 20],
            [5, -5, 3, 3, 3, 3, -5, 5],
            [5, -5, 3, 3, 3, 3, -5, 5],
            [20, -5, 15, 3, 3, 15, -5, 20],
            [-20, -40, -5, -5, -5, -5, -40, -20],
            [120, -20, 20, 5, 5, 20, -20, 120]
        ])
        # a dict() in which each key is the evaluation func result for a bord,
        # and the value the index of that board
        self.leaves = dict()
        # a dict() in which each key is the index of a board and the value is it's branching factor
        # its for when we use alpha beta
        self.branching_factor_alpha_beta = dict()
        # a dict() in which each key is the index of a board and the value is it's branching factor
        # its for when we use only min-max
        self.branching_factor_min_max = dict()

    def evaluation(self, n):
        """
        :return: f(b)= sum(bij*wij) for (i,j) in cell in the board

                for each board state b compute the evaluation function for it
                bij =1  if ints black piece in (i,j) cell
                bij =-1 if ints white piece in (i,j) cell
                bij =0  if ints nothing piece in (i,j) cell

                wij =  (i,j) cell of the weight matrix
        """
        return sum(
            [sum([bij * wij for bij, wij in zip(b_cln_i, w_cln_i)]) for b_cln_i, w_cln_i in
             zip(self.game_boards[n].board, self.weights)])

    def alpha_beta_search(self, board_num, depth):
        """
        :param board_num: given index to the initial board (which is 0)
        :param depth: depth that we wont consider deeper node and cut the tree from that depth
        :return: calculated predicted utility for the root using evluation func and alpha_beta_search
        """
        optimal_utility = self.black_player(board_num=board_num, color=1, beta=float('+inf'), depth=depth)
        return optimal_utility

    def black_player(self, board_num, color, beta, depth):
        """
        :param board_num: index of the current board
        :param color: which color is playing (1: black , -1: white)
        :param beta: beta in the alpha-beta pruning alg.
        :param depth:
        :return: u (predicted utility for this board state)

        it also update the branching factor for this node for two different algorithm (alpha-beta and min-max)
        for further usage in considering alpha-beta pruning performance
        """
        if depth == 0:  # if it gets to the depth limit, return the evaluation function value for this board
            # print(f'{board_num} black: {self.evaluation(board_num)} ')

            # save leaves and their predicted utility for find the chosen path
            pred_utility = self.evaluation(board_num)

            self.leaves.update({pred_utility: board_num})
            self.branching_factor_alpha_beta.update({board_num: 0})
            self.branching_factor_min_max.update({board_num: 0})
            return pred_utility

        cur_alpha = float('-inf')
        u = float('-inf')
        moves = self.game_boards[board_num].valid_moves(color)
        # calculate current node branches (for alpha-beta alg.)
        branch = 0
        # calculate branching factor for min_max algorithm
        self.branching_factor_min_max.update({board_num: sum([len(list(moves[init].keys())) for init in moves.keys()])})
        for init in moves.keys():
            for distance in moves[init].keys():
                branch += 1
                self.path.update({self._n: board_num})
                # create a copy of the current node to execute the possible move on it
                # (which would be the child node of the current node after executing the move)
                self.game_boards.update(
                    {self._n: OthelloBoard(self.game_boards[board_num].board, self.game_boards[board_num].pieces)})

                self.game_boards[self._n].execute_move({distance: moves[init][distance]}, color)

                self._n += 1

                u = max(u, self.white_player(self._n - 1, -color, cur_alpha, depth - 1))

                # print(f'{board_num} black {nn}: {u} ')
                if u >= beta:
                    # before going back, save this nodes branch factor
                    self.branching_factor_alpha_beta.update({board_num: branch})
                    return u
                cur_alpha = max(cur_alpha, u)

        # before going back, save this nodes branch factor
        self.branching_factor_alpha_beta.update({board_num: branch})
        return u

    def white_player(self, board_num, color, alpha, depth):
        """
                :param board_num: index of the current board
                :param color: which color is playing (1: black , -1: white)
                :param alpha: alpha in the alpha-beta pruning alg.
                :param depth:
                :return: u (predicted utility for this board state)

                it also update the branching factor for this node for two different algorithm (alpha-beta and min-max)
                for further usage in considering alpha-beta pruning performance
                """
        if depth == 0:  # if it gets to the depth limit, return the evaluation function value for this board

            # print(f'{board_num} white: {self.evaluation(board_num)} ')

            pred_utility = self.evaluation(board_num)

            # save leaves and their predicted utility for find the chosen path
            self.leaves.update({pred_utility: board_num})
            # before going back, save this nodes branch factor
            self.branching_factor_alpha_beta.update({board_num: 0})

            return pred_utility

        cur_beta = float('+inf')
        u = float('+inf')

        moves = self.game_boards[board_num].valid_moves(color)

        # calculate current node branches (for alpha-beta alg.)
        branch = 0
        # calculate branching factor for min_max algorithm
        self.branching_factor_min_max.update({board_num: sum([len(list(moves[init].keys())) for init in moves.keys()])})

        for init in moves.keys():
            for distance in moves[init].keys():
                branch += 1

                self.path.update({self._n: board_num})

                # create a copy of the current node to execute the possible move on it
                # (which would be the child node of the current node after executing the move)
                self.game_boards.update(
                    {self._n: OthelloBoard(self.game_boards[board_num].board, self.game_boards[board_num].pieces)})
                self.game_boards[self._n].execute_move({distance: moves[init][distance]}, color)

                self._n += 1
                u = min(u, self.black_player(self._n - 1, -color, cur_beta, depth - 1))
                # print(f'{board_num} black {nn}: {u} ')
                if u <= alpha:
                    # before going back, save this nodes branch factor
                    self.branching_factor_alpha_beta.update({board_num: branch})
                    return u
                cur_beta = min(cur_beta, u)
        # before going back, save this nodes branch factor
        self.branching_factor_alpha_beta.update({board_num: branch})
        return u

    def path_extraction(self, u):
        """

        :param u: the predicted utility assigned to the root by applying alpha-beta search
        :return: index of the positions in the chosen path

        using u and self.leaves to find the goal state and find the path from root to it using sel.path
        """
        # list of states number
        p = list()
        node = self.leaves[u]
        while node != 0:
            p.insert(0, node)
            node = self.path[node]
        p.insert(0, 0)
        return p

    def print_total_boards(self, final_evaluation):
        """

        :param final_evaluation: the predicted utility assigned to the root by applying alpha-beta search
        :return:

        use the path_extraction function and print each board from the root tothe goal state
        """
        for move, state in enumerate(self.path_extraction(final_evaluation)):
            if move % 2 == 0:
                print(f'     Black turn  move {move}    ')
            else:
                print(f'    White turn  move {move}    ')

            print(self.game_boards[state].show())


def check_evaluation_in_different_depth(depth):
    """

    :param depth: depth for the alpha-beta-search
    :return: evaluation_per_depth which is a dict which in each key is the used depth and
    it's value is the given predicted utility using alpha-beta-search for this depth and use it to see
    the effect of depth in the given predicted utility
    """
    # a dictionary in which each key is a depth number that  we run alpa-beta on it
    # and it's value is the returned evaluation
    evaluation_per_depth = dict()
    for i in range(depth):
        game = OthelloGame()
        evaluation_per_depth.update({i: game.alpha_beta_search(0, i)})
    return evaluation_per_depth


def check_alpha_beta_puring_performance(depth):
    """

    :param depth: depth for the alpha-beta-search
    :return: a dictionary which it's value is a string show the
    branch factir for each node in min-max and alpha-beta search (for see how effective alpha-beta pruning is)

    """
    game = OthelloGame()
    game.alpha_beta_search(0, depth)
    result = {'compare_each_node': ''
              }
    for i in game.branching_factor_alpha_beta.keys():
        result['compare_each_node'] += 'node: {}  min-max: {}     alpha-beta: {} \n'.format(i,
                                                                                            game.branching_factor_min_max[
                                                                                                i],
                                                                                            game.branching_factor_alpha_beta[
                                                                                                i])
    return result


if __name__ == '__main__':
    othello = OthelloGame()
    final_eval = othello.alpha_beta_search(board_num=0, depth=5)
    othello.print_total_boards(final_evaluation=final_eval)
    print(check_alpha_beta_puring_performance(depth=4)['compare_each_node'])
    print(check_evaluation_in_different_depth(depth=6))
