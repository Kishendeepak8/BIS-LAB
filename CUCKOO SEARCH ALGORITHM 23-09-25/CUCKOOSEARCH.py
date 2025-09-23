import numpy as np
import math


def traffic_delay(green_times, arrival_rates, cycle_time=120):
    if np.sum(green_times) > cycle_time:
        return 1e9

    total_delay = 0
    for i in range(len(arrival_rates)):
        demand = arrival_rates[i] * cycle_time
        capacity = green_times[i] / cycle_time * demand * 2
        delay = max(0, demand - capacity)
        total_delay += delay

    return total_delay


def levy_flight(Lambda, size):
    sigma = (math.gamma(1 + Lambda) * math.sin(np.pi * Lambda / 2) /
             (math.gamma((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2)))**(1 / Lambda)
    u = np.random.randn(size) * sigma
    v = np.random.randn(size)
    step = u / (np.abs(v)**(1 / Lambda))
    return step


def cuckoo_search(n=20, max_iter=200, dim=4, lb=10, ub=60, pa=0.25, cycle_time=120):
    nests = np.random.uniform(lb, ub, (n, dim))
    nests = (nests.T / np.sum(nests, axis=1) * cycle_time).T

    arrival_rates = np.array([0.3, 0.4, 0.2, 0.35])

    fitness = np.array([traffic_delay(x, arrival_rates, cycle_time) for x in nests])
    best_idx = np.argmin(fitness)
    best = nests[best_idx].copy()
    best_fitness = fitness[best_idx]

    print("Initial Best Solution:", best.round(2))
    print("Initial Best Fitness:", best_fitness)

    for iteration in range(max_iter):
        for i in range(n):
            step_size = levy_flight(1.5, dim)
            new_solution = nests[i] + step_size * (nests[i] - best)
            new_solution = np.clip(new_solution, lb, ub)
            new_solution = new_solution / np.sum(new_solution) * cycle_time
            f_new = traffic_delay(new_solution, arrival_rates, cycle_time)

            if f_new < fitness[i]:
                nests[i] = new_solution
                fitness[i] = f_new

            if f_new < best_fitness:
                best = new_solution.copy()
                best_fitness = f_new

        K = np.random.rand(n, dim) > pa
        steps = np.random.rand() * (nests[np.random.permutation(n)] - nests[np.random.permutation(n)])
        new_nests = nests + steps * K
        new_nests = np.clip(new_nests, lb, ub)
        new_nests = (new_nests.T / np.sum(new_nests, axis=1) * cycle_time).T


        f_new = np.array([traffic_delay(x, arrival_rates, cycle_time) for x in new_nests])
        for i in range(n):
            if f_new[i] < fitness[i]:
                nests[i] = new_nests[i]
                fitness[i] = f_new[i]

                if f_new[i] < best_fitness:
                    best = new_nests[i].copy()
                    best_fitness = f_new[i]

        print(f"Iteration {iteration+1}: Best Solution = {best.round(2)}, Best Fitness = {best_fitness}")


    return best, best_fitness


if __name__ == "__main__":
    best_solution, best_value = cuckoo_search()
    print("\n🚦 Optimized Green Times (N, E, S, W):", best_solution.round(2))
    print("⏳ Minimum Total Delay (vehicles waiting):", best_value)