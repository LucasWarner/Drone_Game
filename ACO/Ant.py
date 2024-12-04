class Ant:
    def __init__(self, env):
        self.env = env
        self.position = self.env.drone.position
        self.path = [self.position]

        self.total_cost = 0

    def move(self, action):
        # Use the movement logic from drone_env
        new_position, cost = self.env.calculate_new_position(action)

        # Update the ant's position and path
        self.position = new_position
        self.path.append(new_position)

        # Update the total cost
        self.total_cost += cost
        self.position = new_position