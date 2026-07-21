import pandas as pd

# Car price prediction problem
class DemandPrediction:
    N_DEMAND_INDICATORS = 13;
    # Parameters consist of a bias (intercept) for the sum and one weight for
    # each demand indicator.
    N_PARAMETERS = N_DEMAND_INDICATORS + 1;
    

    # Construct a car price prediction problem instance.
    # The parameter "dataset_name" specifies which dataset to use. Valid values
    # are "train" or "test".
    def __init__(self,dataset_name):    
      #Load the specified dataset
      if(dataset_name == "train"):
        self.__X, self.__y = DemandPrediction.__load_dataset("data/train.csv");
      elif(dataset_name == "test"):
        self.__X, self.__y = DemandPrediction.__load_dataset("data/test.csv");
      else:
        raise Exception("Only permitted arguments for " +
          "CarPricePrediction::__init__ are train and test.")
    
    # Rectangular bounds on the search space.
    # Returns a 2D array b such that b[i][0] is the minimum permissible value
    # of the ith solution component and b[i][1] is the maximum.    
    def bounds(self):
        return [[-100,100] for i in range(DemandPrediction.N_PARAMETERS)]

    # Check whether the function parameters (weights) lie within the
    # problem's feasible region.
    # There should be the correct number of weights for the predictor function.
    # Each weight should lie within the range specified by the bounds.
    def is_valid(self, parameters):
        if(len(parameters) != DemandPrediction.N_PARAMETERS):
          return False
        #All weights lie within the bounds.
        b = self.bounds();
        for i in range(len(b)):
          if(parameters[i] < b[i][0] or parameters[i] > b[i][1] ):
            return False
        return True;

    # Evaluate a set of parameters on the dataset used by the class instance
    # (train/test).
    # @param parameters An containing the bias and weights to be used to
    #   predict demand.
    # @return The mean absolute error of the predictions on the selected
    # dataset.
    def evaluate(self, parameters):
        abs_error = 0.0;
        for (x, y) in zip(self.__X,self.__y):
            #print(list(self.__X.values))
            #print(x)
            #print(y)
            #print('--')
            y_pred = DemandPrediction.__predict(x,parameters);
            abs_error += abs(y-y_pred);
        abs_error /= len(self.__X);
        return abs_error;

    def __load_dataset(filename):
        df = pd.read_csv(filename,header=None)
        y = df.iloc[:,0].values
        X = df.iloc[:,1:].values
        return X, y
        
    # Predicts demand based on a weighted sum of demand indicators. You may
    # replace this with something more complex, but will likely have to change
    # the form of the parameters array as well.
    def __predict(demand_indicators, parameters):
        prediction = parameters[0];

        for i in range(len(demand_indicators)):
            prediction += demand_indicators[1] * parameters[i+1];

        return prediction;