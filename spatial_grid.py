class SpatialGrid:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.grid = {}

    def _get_cell_key(self, position):
        return (int(position[0] // self.cell_size),
                int(position[1] // self.cell_size),
                int(position[2] // self.cell_size))

    def add_obstacle(self, obstacle):
        key = self._get_cell_key(obstacle.position)
        if key not in self.grid:
            self.grid[key] = []
        self.grid[key].append(obstacle)

    def detect_nearby_obstacles(self, drone, size, radius):
        nearby_obstacles = []
        x, y, z = drone.position
        for i in range(int(x - radius), int(x + radius) + 1):
            for j in range(int(y - radius), int(y + radius)+ 1):
                for k in range(int(z - radius), int(z + radius) + 1):
                    key = (i // self.cell_size, j // self.cell_size, k // self.cell_size)
                    if key in self.grid:
                        nearby_obstacles.extend(self.grid[key])
        return nearby_obstacles

    def nearby_obstacles(self, drone, size, radius):
        nearby_obstacles = []
        x, y, z = drone.position
        for i in range(int(x - radius), int(x + radius) + 1):
            for j in range(int(y - radius), int(y + radius) + 1):
                for k in range(int(z - radius), int(z + radius) + 1):
                    key = (i // self.cell_size, j // self.cell_size, k // self.cell_size)
                    if key in self.grid:
                        nearby_obstacles.extend(self.grid[key])
        if nearby_obstacles != []:
            return True
        return False
    def get_cell(self, position):
        key = self._get_cell_key(position)
        return self.grid.get(key, [])
