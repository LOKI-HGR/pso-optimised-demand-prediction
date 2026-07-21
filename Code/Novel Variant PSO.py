import random
import numpy as np
from DemandPrediction import DemandPrediction
import csv
import os


# Particle class which can represent one potential solution
class particle:

    def __init__(self, problem):
        self.problem = problem
        self.bounds = problem.bounds()
        self.position = [random.uniform(low, high) for (low,high) in self.bounds]
        self.velocity = [random.uniform(-1, 1) for _ in range(len(self.position))]
        self.cost = self.problem.evaluate(self.position)
        self.best_position = list(self.position)
        self.best_cost = self.cost
        self.stagnation_counter = 0 # new line of code to monitor stagnation per particle

    # In my variation update velocity is now using adaptive inertia and velocity clamping
    def update_velocity(self, global_best_position, w, c1=1.4, c2=1.4):
        new_velocity = []
        # New Line of code to define a max velocity
        v_max = [0.2 * (high - low) for (low, high) in self.bounds]

        for i in range(len(self.position)):
            r1 = random.random()
            r2 = random.random()

            inertia = w * self.velocity[i]
            cognitive = c1 * r1 * (self.best_position[i] - self.position[i])
            social = c2 * r2 * (global_best_position[i] - self.position[i])

            v_new = inertia + cognitive + social

            # New Conditional Statement for velocity clamping
            if v_new > v_max[i]:
                v_new = v_max[i]
            elif v_new < -v_max[i]:
                v_new = -v_max[i]

            new_velocity.append(v_new)

        self.velocity = new_velocity

    # In my variation this class still updates and evaluates new cost but also tracks stagnation
    def update_position(self):
        new_position = [self.position[i] + self.velocity[i] for i in range(len(self.position))]

        for i, (low, high) in enumerate(self.bounds):
            if new_position[i] < low:
                new_position[i] = low
            elif new_position[i] > high:
                new_position[i] = high

        if self.problem.is_valid(new_position):
            self.position = new_position

        # New line of code to evaluate and track improvement
        old_cost = self.cost
        self.cost = self.problem.evaluate(self.position)

        # This conditional statement has been updated to track stagnation
        if self.cost < self.best_cost:
            self.best_cost = self.cost
            self.best_position = list(self.position)
            self.stagnation_counter = 0 # New line to reset if improved
        else:
            self.stagnation_counter += 1 # New line to increment if no improvement

# Swarm class: manages all particles and tracks global best
class Swarm:
    def __init__(self, problem, n_particles = 30):
        self.problem = problem
        self.particles =[particle(problem) for _ in range(n_particles)]

        # Find initial global best (lowest MAE)
        best_particle = min(self.particles, key=lambda p: p.best_cost)
        self.global_best_position = list(best_particle.best_position)
        self.global_best_cost = best_particle.best_cost

    # One PSO iteration: This iternation now has adaptive inertia, updates and reset mechanism
    def iterate(self, current_iter, total_iter,  w_max = 0.9, w_min = 0.4, c1 = 1.4, c2 = 1.4, stagnation_limit = 20):

        # New line of code for adaptive inertia: decreases each iteration
        w = w_max - ((w_max - w_min) * (current_iter / total_iter))

        for particle in self.particles:
            particle.update_velocity(self.global_best_position, w, c1, c2)
            particle.update_position()

            # Update global best if any particle improved it
            if particle.best_cost < self.global_best_cost:
                self.global_best_cost = particle.best_cost
                self.global_best_position = list(particle.best_position)

        # New flow statement for random reset for stagnant particles
        for p in self.particles:
            if p.stagnation_counter > stagnation_limit:
                p.velocity = [random.uniform(-1, 1) for _ in range(len(p.velocity))] # Randomise velocity to keep diversity
                p.stagnation_counter = 0

# Run PSO optimisation on the given problem
def run_pso(problem, n_particles=30, n_iterations=200):
    swarm = Swarm(problem, n_particles)

    print("\nRunning Novel Variant PSO optimisation...\n")
    for iteration in range(n_iterations):
        swarm.iterate(current_iter=iteration, total_iter=n_iterations)

        # Display progress every 10 iterations
        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"Iteration {iteration+1:3d} | Best MAE so far: {swarm.global_best_cost:.5f}")

    print("\nFinal best solution found:")
    print(f"Parameters: {np.round(swarm.global_best_position, 3)}")
    print(f"Training MAE: {swarm.global_best_cost:.5f}")

    return swarm.global_best_position, swarm.global_best_cost

# Main experiment: train on training data, test on test data
if __name__ == "__main__":
    # Load training and test datasets
    train_problem = DemandPrediction("train")
    test_problem = DemandPrediction("test")

    # Run Novel Variant PSO on training set
    best_params, train_mae = run_pso(train_problem, n_particles=50, n_iterations=1000)

    # Evaluate best parameters on test set
    test_mae = test_problem.evaluate(best_params)
    print(f"\nTest MAE using best PSO parameters: {test_mae:.5f}")

    # Novel Variant PSO summary
    print("\n============================================")
    print("Novel Variant PSO Results Summary")
    print("============================================")
    print(f"Train MAE: {train_mae:.5f}")
    print(f"Test MAE:  {test_mae:.5f}")

    csv_file = "novel_results.csv"

    # Make sure the file exists and has headers
    if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Train_MAE", "Test_MAE"])

    # Append just the two values
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([train_mae, test_mae])

    print(f"Added Train_MAE={train_mae} Test_MAE={test_mae} to {csv_file}")