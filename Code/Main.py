from DemandPrediction import DemandPrediction
import random
random.seed(40)

# A simple example of how the DemandPrediction class could be used as part
# of a random search. It is not expected that you use this code as part of
# your solution - it is just a demonstration of how the class's methods can be
# called and how we can use two versions of the problem (here train and
# test) to, respectively, obtain a promising set of parameters (using
# train) and then to measure their performance (using test).

def random_parameters():
    b = DemandPrediction.bounds()  
    return [low + random.random()*(high-low) for [high, low] in b]

training_problem = DemandPrediction("train")

# Generate N_TRIES random parameters and measure their MAE on the test
# problem, saving the best parameters.
N_TRIES = 100
best_parameters = random_parameters()
best_training_error = training_problem.evaluate(best_parameters)
for _ in range(N_TRIES - 1):
    parameters = random_parameters()
    # Note: due to the way parameters are generated, they are guaranteed to be
    # valid. The next line is just included as an example of how you would test
    # parameters for validity.
    if(DemandPrediction.is_valid(parameters)):
        training_error = training_problem.evaluate(parameters)
    else:
      training_error = None
    if(training_error < best_training_error):
        best_training_error = training_error
        best_parameters = parameters

print("Best training error after {} iterations: {}".format(
  N_TRIES, best_training_error))

# Check the MAE of the best parameters on the test problem.
test_problem  = DemandPrediction("test")
test_error = test_problem.evaluate(best_parameters)
print("Test error of best solution found while training: {}".format(
  test_error))