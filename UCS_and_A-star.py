import heapq
import time


# the problem's environment
class Environment:
    def __init__(self, environment_file_dir):
        self.graph_array = list()
        self.graph_rep_dir = environment_file_dir
        self.construct_graph()

    def construct_graph(self):
        with open(self.graph_rep_dir, 'r') as graph_rep_file:
            for line in graph_rep_file:
                tmp_column = list()
                for each in line.split(','):
                    tmp_column.append(int(float(each)))
                self.graph_array.append(tmp_column)


class PriorityQueue:
    def __init__(self):
        self.DONE = -100000
        self.heap = []
        self.priorities = {}  # Map from state to priority

    # Insert |state| into the heap with priority |newProject| if
    # |state| isn't the heap or |newPriority| is smaller than the existing one
    # Return whether the priority queue was updated.

    def update(self, state, new_cost):
        old_cost = self.priorities.get(state)
        if old_cost is None or new_cost < old_cost:
            self.priorities[state] = new_cost
            heapq.heappush(self.heap, (new_cost, state))
            return True
        return False

    # Returns (state with minimum priority,priority)
    # or (None,None) if the priority queue is empty.
    def remove_min(self):
        while len(self.heap) > 0:
            priority, state = heapq.heappop(self.heap)
            if self.priorities[state] == self.DONE:
                continue  # Outdated priority, skip
            self.priorities[state] = self.DONE
            return state, priority
        return None, None  # Nothing left


def ucs(env, start, end):
    result = {'cost': float('+inf'),
              'path': []}

    path_solution = {}

    frontier = PriorityQueue()

    # given cost for each move
    cost_rule = {'left': 1,
                 'right': 1,
                 'up': 2,
                 'down': 3}
    # store the current position added to explored dictionary
    current = {'pos': None,
               'cost': 0}

    # check is start point and end point aren't a block and then add it to explored
    if env.graph_array[start[1]][start[0]] == 1:
        print('The given START POINT is a BLOCK,'
              ' therefore there is no path_solution from there to any other places.')
        return False

    elif env.graph_array[end[1]][end[0]] == 1:
        print('The given END POINT is a BLOCK,'
              ' therefore there is no path_solution from there to any other places.')
        return False
    else:
        frontier.update(start, 0)
        path_solution.update({start: None})
    current['pos'], current['cost'] = start, 0

    # the current 4 directions neighborhood, check if they are possible to visit and by visiting them from
    # the current position the cost would be lower(if they've been visited previously) then update frontier
    def update_frontier(current_explored, current_cost, graph_arr):
        # left
        # check if the left position exist in the environment or not
        if current_explored[0] - 1 >= 0:
            left_pos = (current_explored[0] - 1, current_explored[1])
            # as the given text file is modeled based on the environment in the HW file, ones represent the blocks
            # check if the left position of the current_explored place can be visited or not
            if graph_arr[left_pos[1]][left_pos[0]] == 0:
                is_updated = frontier.update(left_pos, current_cost + cost_rule['left'])
                if is_updated:
                    if left_pos in path_solution.keys():
                        path_solution[left_pos] = current_explored
                    else:
                        path_solution.update({left_pos: current_explored})
        # do the same for the three other possible movement
        # right
        if current_explored[0] + 1 < len(graph_arr[current_explored[1]]):
            right_pos = (current_explored[0] + 1, current_explored[1])
            if graph_arr[right_pos[1]][right_pos[0]] == 0:
                is_updated = frontier.update(right_pos, current_cost + cost_rule['right'])
                if is_updated:
                    if right_pos in path_solution.keys():
                        path_solution[right_pos] = current_explored
                    else:
                        path_solution.update({right_pos: current_explored})
        # up
        if current_explored[1] - 1 >= 0:
            up_pos = (current_explored[0], current_explored[1] - 1)
            if graph_arr[up_pos[1]][up_pos[0]] == 0:
                is_updated = frontier.update(up_pos, current_cost + cost_rule['up'])
                if is_updated:
                    if up_pos in path_solution.keys():
                        path_solution[up_pos] = current_explored
                    else:
                        path_solution.update({up_pos: current_explored})
        # down
        if current_explored[1] + 1 < len(graph_arr):
            down_pos = (current_explored[0], current_explored[1] + 1)
            if graph_arr[down_pos[1]][down_pos[0]] == 0:
                is_updated = frontier.update(down_pos, current_cost + cost_rule['down'])
                if is_updated:
                    if down_pos in path_solution.keys():
                        path_solution[down_pos] = current_explored
                    else:
                        path_solution.update({down_pos: current_explored})

    def path():
        current_pose = end
        result['cost'] = current['cost']
        while path_solution[current_pose] is not None:
            result['path'].insert(0, current_pose)
            current_pose = path_solution[current_pose]
        result['path'].insert(0, current_pose)

    while True:
        current['pos'], current['cost'] = frontier.remove_min()
        if (current['pos'], current['cost']) == (None, None) or end == current['pos']:
            break
        update_frontier(current['pos'], current['cost'], env.graph_array)

    if end in path_solution.keys():
        path()
    else:
        print(path_solution.keys())
        return False

    return result


def a_star(env, start, end):
    result = {'cost': float('+inf'),
              'path': []}

    path_solution = {}

    frontier = PriorityQueue()

    # given cost for each move
    cost_rule = {'left': 1,
                 'right': 1,
                 'up': 2,
                 'down': 3}
    # store the current position added to explored dictionary
    current = {'pos': None,
               'cost': 0}

    # check is start point and end point aren't a block and then add it to explored
    if env.graph_array[start[1]][start[0]] == 1:
        print('The given START POINT is a BLOCK,'
              ' therefore there is no path_solution from there to any other places.')
        return False

    elif env.graph_array[end[1]][end[0]] == 1:
        print('The given END POINT is a BLOCK,'
              ' therefore there is no path_solution from there to any other places.')
        return False
    else:
        frontier.update(start, 0)
        path_solution.update({start: None})
    current['pos'], current['cost'] = start, 0

    # Use Manhattan Distance between the given position and the end position without considering the blocks
    def heuristic(position):
        return abs(position[0] - end[0]) + abs(position[1] - end[1])

    # the current 4 directions neighborhood, check if they are possible to visit and by visiting them from
    # the current position the cost would be lower(if they've been visited previously) then update frontier
    def update_frontier(current_explored, current_cost, graph_arr):
        # left
        # check if the left position exist in the environment or not
        if current_explored[0] - 1 >= 0:
            left_pos = (current_explored[0] - 1, current_explored[1])
            # as the given text file is modeled based on the environment in the HW file, ones represent the blocks
            # check if the left position of the current_explored place can be visited or not
            if graph_arr[left_pos[1]][left_pos[0]] == 0:
                is_updated = frontier.update(left_pos,
                                             current_cost + cost_rule['left'] + heuristic(left_pos) - heuristic(
                                                 current_explored))
                if is_updated:
                    if left_pos in path_solution.keys():
                        path_solution[left_pos] = current_explored
                    else:
                        path_solution.update({left_pos: current_explored})
        # do the same for the three other possible movement
        # right
        if current_explored[0] + 1 < len(graph_arr[current_explored[1]]):
            right_pos = (current_explored[0] + 1, current_explored[1])
            if graph_arr[right_pos[1]][right_pos[0]] == 0:
                is_updated = frontier.update(right_pos,
                                             current_cost + cost_rule['right'] + heuristic(right_pos) - heuristic(
                                                 current_explored))
                if is_updated:
                    if right_pos in path_solution.keys():
                        path_solution[right_pos] = current_explored
                    else:
                        path_solution.update({right_pos: current_explored})
        # up
        if current_explored[1] - 1 >= 0:
            up_pos = (current_explored[0], current_explored[1] - 1)
            if graph_arr[up_pos[1]][up_pos[0]] == 0:
                is_updated = frontier.update(up_pos, current_cost + cost_rule['up'] + heuristic(up_pos) - heuristic(
                    current_explored))
                if is_updated:
                    if up_pos in path_solution.keys():
                        path_solution[up_pos] = current_explored
                    else:
                        path_solution.update({up_pos: current_explored})
        # down
        if current_explored[1] + 1 < len(graph_arr):
            down_pos = (current_explored[0], current_explored[1] + 1)
            if graph_arr[down_pos[1]][down_pos[0]] == 0:
                is_updated = frontier.update(down_pos,
                                             current_cost + cost_rule['down'] + heuristic(down_pos) - heuristic(
                                                 current_explored))
                if is_updated:
                    if down_pos in path_solution.keys():
                        path_solution[down_pos] = current_explored
                    else:
                        path_solution.update({down_pos: current_explored})

    def path():
        current_pose = end
        result['cost'] = current['cost']
        while path_solution[current_pose] is not None:
            result['path'].insert(0, current_pose)
            current_pose = path_solution[current_pose]
        result['path'].insert(0, current_pose)

    while True:
        current['pos'], current['cost'] = frontier.remove_min()
        if (current['pos'], current['cost']) == (None, None) or end == current['pos']:
            break
        update_frontier(current['pos'], current['cost'], env.graph_array)

    if end in path_solution.keys():
        path()
    else:
        return False

    return result


def compare_runtime(start, end):
    start_time = time.time()
    ucs(p, start, end)
    ucs_time = time.time() - start_time

    start_time = time.time()
    a_star(p, start, end)
    a_star_time = time.time() - start_time

    if a_star_time > ucs_time:
        return 'UCS is faster on this example'
    else:
        return 'A* is faster on this example'


if __name__ == '__main__':
    text_file_dir = 'Environment.txt'

    p = Environment(text_file_dir)

    start_point_1 = (0, 0)
    end_point_1 = (23, 24)

    result_1 = ucs(p, start_point_1, end_point_1)

    if not result_1:
        print('there is no path from {} to {} in the given environment!!\n'.format(start_point_1, end_point_1))

    else:
        print('found a pat from {} to {} with cost {} using UCS algorithm. '
              'You can see the path in the below line:'.format(start_point_1, end_point_1, result_1['cost']))
        print(result_1['path'], '\n')

    start_point_2 = (17, 1)
    end_point_2 = (17, 29)

    result_2 = ucs(p, start_point_2, end_point_2)

    if not result_2:
        print('there is no path from {} to {} in the given environment!!\n'.format(start_point_2, end_point_2))
    else:
        print(
            'found a pat from {} to {} with cost {} using UCS algorithm.'
            ' You can see the path in the below line:'.format(start_point_2, end_point_2, result_2['cost']))
        print(result_2['path'], '\n')
