# ======================================
# Parallel Cellular Algorithm (PCA)
# Traffic Light Optimization Simulation
# ======================================

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Problem Setup
# -------------------------------

grid_size = (20, 20)        # 20x20 city grid
iterations = 100            # number of simulation iterations
alpha = 0.3                 # diffusion rate
mutation_rate = 0.05        # random variation for exploration

# Each cell represents a traffic light's green duration (0-1)
cells = np.random.rand(*grid_size)
fitness = np.zeros(grid_size)

# Helper to get neighbors (Moore neighborhood)
def get_neighbors(i, j, grid):
    rows, cols = grid.shape
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = (i + di) % rows, (j + dj) % cols
            neighbors.append(grid[ni, nj])
    return np.array(neighbors)

# -------------------------------
# 2. Fitness Function (Traffic Model)
# -------------------------------
# Synthetic model:
#   - Penalize large differences with neighbors (local congestion imbalance)
#   - Penalize extreme green times (too short or too long)
def evaluate_fitness(cells):
    fitness = np.zeros_like(cells)
    for i in range(cells.shape[0]):
        for j in range(cells.shape[1]):
            green_time = cells[i, j]
            neighbors = get_neighbors(i, j, cells)
            # Congestion cost = imbalance + inefficiency
            imbalance = np.mean(np.abs(neighbors - green_time))
            inefficiency = (green_time - 0.5)**2
            fitness[i, j] = -(imbalance + inefficiency)  # maximize fitness
    return fitness

# -------------------------------
# 3. Parallel Cellular Update
# -------------------------------
def update_cells(cells, fitness):
    new_cells = np.copy(cells)
    for i in range(cells.shape[0]):
        for j in range(cells.shape[1]):
            neighbors = get_neighbors(i, j, cells)
            # Diffusion rule: move towards the average of neighbors
            avg_neighbor = np.mean(neighbors)
            x_new = cells[i, j] + alpha * (avg_neighbor - cells[i, j])
            # Small mutation for exploration
            x_new += np.random.uniform(-mutation_rate, mutation_rate)
            # Keep in [0, 1] range
            new_cells[i, j] = np.clip(x_new, 0, 1)
    return new_cells

# -------------------------------
# 4. Main Iteration Loop
# -------------------------------
best_fitness_over_time = []

for t in range(iterations):
    fitness = evaluate_fitness(cells)
    best_fitness = np.max(fitness)
    best_fitness_over_time.append(best_fitness)
    
    if t % 10 == 0:
        print(f"Iteration {t}: Best fitness = {best_fitness:.4f}")
    
    cells = update_cells(cells, fitness)

# -------------------------------
# 5. Results Visualization
# -------------------------------

plt.figure(figsize=(12, 5))

# Final grid (optimized traffic timings)
plt.subplot(1, 2, 1)
plt.imshow(cells, cmap='viridis')
plt.title("Optimized Traffic Light Durations (Green Ratio)")
plt.colorbar(label='Green Light Duration (0–1)')

# Fitness progress
plt.subplot(1, 2, 2)
plt.plot(best_fitness_over_time, 'r-', linewidth=2)
plt.title("Fitness Convergence")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")

plt.tight_layout()
plt.show()
