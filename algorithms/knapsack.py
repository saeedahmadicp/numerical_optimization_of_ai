import time
import itertools

__all__ = ["Item", "Knapsack"]

class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value

    def __repr__(self):
        return f"({self.weight}, {self.value})"


class Knapsack:
    def __init__(self, items, max_weight):
        self.items = items
        self.max_weight = max_weight
        self.solve_method = ""

    def solve(self, method, time_it=False):
        """
        Solves the knapsack problem using the specified method.
        The solution to a knapsack problem is a combination of items with
        the highest value that can fit into the knapsack.
        :param method: the method to use when solving the knapsack problem
        :param time_it: whether or not to time the execution of the method
        :return: the combination of items with the highest value that can fit into the knapsack
        """

        method = method.lower().strip().replace(" ", "_")

        if time_it:
            start_time = time.time()
            solution = getattr(self, method)(self.items, self.max_weight)
            end_time = time.time()
            execution_time = end_time - start_time
            return solution, execution_time
        else:
            solution = getattr(self, method)(self.items, self.max_weight)
            return solution


    def __repr__(self) -> str:
        sol = ", ".join(str(item) for item in self.items)
        total_weight = sum(item.weight for item in self.items)
        total_value = sum(item.value for item in self.items)
        return f"Method: {self.solve_method}\nKnapsack Solution: {sol}\nTotal Weight: {total_weight}\nTotal Value: {total_value}"


    @classmethod
    def brute_force(cls, items, max_weight):
        # create a list of all possible combinations of items
        combinations = []
        for i in range(1, len(items) + 1):
            combinations += list(itertools.combinations(items, i))

        # create a list of all possible combinations of items that are under the max weight
        valid_combinations = []
        for combination in combinations:
            total_weight = 0
            for item in combination:
                total_weight += item.weight
            if total_weight <= max_weight:
                valid_combinations.append(combination)

        # find the combination with the highest value
        max_value = 0
        best_combination = []
        for combination in valid_combinations:
            total_value = 0
            for item in combination:
                total_value += item.value
            if total_value > max_value:
                max_value = total_value
                best_combination = combination

        knapsack = cls(best_combination, max_weight)
        knapsack.solve_method = "Brute Force"
        return knapsack
    

    @classmethod
    def greedy(cls, items, max_weight):
        # sort the items by value
        items.sort(key=lambda x: x.value, reverse=True)

        # add items to the knapsack until it is full
        knapsack = []
        total_weight = 0
        for item in items:
            if total_weight + item.weight <= max_weight:
                knapsack.append(item)
                total_weight += item.weight

        knapsack = cls(knapsack, max_weight)
        knapsack.solve_method = "Greedy"
        return knapsack
    

    @classmethod
    def dynamic_programming(cls, items, max_weight):
        # create a 2D array to store the max value for each item and weight
        max_values = [[0 for x in range(max_weight + 1)] for x in range(len(items) + 1)]

        # fill the array
        for i in range(1, len(items) + 1):
            for j in range(1, max_weight + 1):
                if items[i - 1].weight <= j:
                    max_values[i][j] = max(items[i - 1].value + max_values[i - 1][j - items[i - 1].weight],
                                           max_values[i - 1][j])
                else:
                    max_values[i][j] = max_values[i - 1][j]

        # find the items in the knapsack
        knapsack = []
        i = len(items)
        j = max_weight
        while i > 0 and j > 0:
            if max_values[i][j] != max_values[i - 1][j]:
                knapsack.append(items[i - 1])
                j -= items[i - 1].weight
            i -= 1

        knapsack = cls(knapsack, max_weight)
        knapsack.solve_method = "Dynamic Programming"
        return knapsack