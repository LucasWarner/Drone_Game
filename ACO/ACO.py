import numpy as np
from Ant import Ant

class ACO:
    def __init__(self, env, num_ants=1, pheromone_decay=0.1, pheromone_intensity=1.0, alpha=1.0, beta=2.0):
        self.env = env
        self.pheromone_map = np.zeros((200, 20, 200))
        self.ants = [Ant(self.env) for _ in range(num_ants)]
        self.alpha = alpha
        self.beta = beta
        self.pheromone_decay = pheromone_decay
        self.pheromone_intensity = pheromone_intensity
        self.heuristic_map = np.ones((200, 20, 200))
        self.i = 0# Example initialization
        # Use PPO to determine the first action

    def update_pheromones(self):
        # Evaporate pheromones
        self.pheromone_map *= (1 - self.pheromone_decay)
        # Add new pheromones based on ant paths
        for ant in self.ants:
            for position in ant.path:
                # Convert position to integer indices
                index = tuple(map(lambda x: int(np.floor(x)), position))
                # Ensure indices are within bounds
                if (0 <= index[0] < self.pheromone_map.shape[0] and
                        0 <= index[1] < self.pheromone_map.shape[1] and
                        0 <= index[2] < self.pheromone_map.shape[2]):
                    self.pheromone_map[index] += self.pheromone_intensity / ant.total_cost

    def run(self, model):
        if self.i == 0:
            for ant in self.ants:
                obs = self.env.get_observation(ant.position)
                action, _states = model.predict(obs)
                ant.move(action)
                self.i =1
                if not self.env.is_goal_reached:
                    # Implement ACO logic to choose next action
                    next_action = self.choose_action(ant)
                    ant.move(next_action)
        else:
            for ant in self.ants:
            # Let ACO handle the rest of the path
                if not self.env.is_goal_reached:
                    # Implement ACO logic to choose next action
                    next_action = self.choose_action(ant)
                    ant.move(next_action)

        self.update_pheromones()

    def choose_action(self, ant):
        """
        Choose the next action for the ant based on pheromone levels and other factors.

        Parameters:
        ant (Ant): The ant for which to choose the next action.

        Returns:
        action: The chosen action for the ant.
        """
        # Retrieve the current position of the ant
        current_position = ant.position

        # Convert each component of the position to an integer index
        current_index = tuple(map(lambda x: int(np.floor(x)), current_position))

        # Get available actions from the current position
        available_actions = self.get_available_actions(current_index)

        # Initialize a list to store probabilities for each action
        probabilities = []

        # Total pheromone for normalization
        total_pheromone = 0

        # Calculate the probability for each action based on pheromone levels
        for action in available_actions:
            pheromone_level = self.pheromone_map[current_index[0] + 100, current_index[1], current_index[2] + 100]
            heuristic_value = self.heuristic_map[current_index[0] + 100, current_index[1], current_index[2] + 100]

            # Calculate the desirability of the action
            desirability = (pheromone_level ** self.alpha) * (heuristic_value ** self.beta)
            probabilities.append(desirability)
            total_pheromone += desirability

        # Check for zero total pheromone and handle it
        if total_pheromone == 0:
            # Assign equal probabilities if total pheromone is zero
            probabilities = [1.0 / len(available_actions)] * len(available_actions)
        else:
            # Normalize probabilities
            probabilities = [prob / total_pheromone for prob in probabilities]

        # Choose an action based on the calculated probabilities
        chosen_action = np.random.choice(available_actions, p=probabilities)

        return chosen_action
    @staticmethod
    def get_available_actions(position):
        # Placeholder method to return available actions from a given position
        # This should be implemented based on your environment's logic
        return range(9)  #9 possible actions