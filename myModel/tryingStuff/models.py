import numpy as np
import random
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.datasets import load_diabetes

class GeneticAlgorithm(BaseEstimator, MetaEstimatorMixin):
    def __init__(self, modelToTune, param_space, 
                 population_size=100, generations=1000, 
                 crossover_prob=0.8, mutation_prob=0.2,
                 tournament_size=15, cv=5, scoring='accuracy',
                 n_jobs=None, verbose=0, random_state=None):
        self.modelToTune = modelToTune #actual model
        self.param_space = param_space #parameters
        self.population_size = population_size #initial population size
        self.generations = generations #num of generations
        self.crossover_prob = crossover_prob #probability (0,1) of crossover
        self.mutation_prob = mutation_prob #probability (0,1) of mutation
        self.tournament_size = tournament_size #basically how large of
        self.cv = cv #cross val fold for fitness
        self.scoring = scoring #what method to get fitness from
        self.n_jobs = n_jobs #for parallelism 
        self.verbose = verbose 
        self.random_state = random_state
        self.best_params_ = [] #list of best parameters
        self.best_score_ = None #best score so far
        self.history_ = []
        
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
    
    def _initialize_individual(self):
        individual = {}
        for param, values in self.param_space.items(): #selecting from discreet options
            if isinstance(values, (list, tuple, range)): 
                individual[param] = random.choice(values)
            elif callable(values):
                individual[param] = values()
            else:
                raise ValueError(f"Invalid parameter range for {param}") #something went wrong
        return individual
    
    def _initialize_population(self):
        return [self._initialize_individual() for _ in range(self.population_size)]
    
    def _evaluate_individual(self, individual, X, y):
        model = self.modelToTune.set_params(**individual)
        scores = cross_val_score(model, X, y, cv=self.cv, 
                                scoring=self.scoring, n_jobs=self.n_jobs)
        return np.mean(scores)
    
    def _tournament_selection(self, population, fitness_scores):
        #I set up a tournament approach to selecing parents
        #makes models from a sample undergo a tournament, where we use the best one as the parent
        #CAN CHANGE TO SORTING BASED ON FITNESS, AND THEN JUST SELECTING BEST FROM BEST X POPULATION!!!

        selected = random.sample(range(len(population)), self.tournament_size)
        selected_fitness = [fitness_scores[i] for i in selected]
        winner = selected[np.argmax(selected_fitness)]
        return population[winner]
    
    def _crossover(self, parent1, parent2):
        #actual crossover
        child = {}
        for param in parent1.keys():
            if random.random() < 0.5:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]
        return child
    
    def _mutate(self, individual):
        #copy the OG
        mutated = individual.copy()
        #randomly select a parameter to mutate
        param_to_mutate = random.choice(list(self.param_space.keys()))
        
        #if we have a range to choose from, we select a random value from the range
        if isinstance(self.param_space[param_to_mutate], (list, tuple, range)):
            current_value = mutated[param_to_mutate]
            possible_values = [v for v in self.param_space[param_to_mutate] if v != current_value]
            if possible_values:
                mutated[param_to_mutate] = random.choice(possible_values)
                        #otherwise we just select one of out options

        elif callable(self.param_space[param_to_mutate]):
            mutated[param_to_mutate] = self.param_space[param_to_mutate]()
        
        return mutated
    
    def run(self, X, y):
        population = self._initialize_population()
        
        
        for gen in range(self.generations):
            #get fitness for all models in gen
            fitness_scores = [self._evaluate_individual(ind, X, y) for ind in population]
            #get the best one by id
            best_idx = np.argmax(fitness_scores)
            current_best_score = fitness_scores[best_idx]
            current_best_params = population[best_idx]
            
            #set the new best score if it wins
            if self.best_score_ is None or current_best_score > self.best_score_:
                self.best_score_ = current_best_score
                self.best_params_ = current_best_params.copy()
            
            self.history_.append({
                'generation': gen,
                'best_score': self.best_score_,
                'avg_score': np.mean(fitness_scores),
                'best_params': self.best_params_
            })
            

            if self.verbose > 0:
                print(f"Generation {gen+1}/{self.generations} - Best: {self.best_score_:.4f} - Avg: {np.mean(fitness_scores):.4f}")
            
            #make a new population
            new_population = []
            #add the best parameters
            new_population.append(self.best_params_)
            

            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                #if crossover selected, our child is made up of both parent
                if random.random() < self.crossover_prob:
                    child = self._crossover(parent1, parent2)
                else:
                #othterwise we just carry one of the two parents into the next gen
                    child = random.choice([parent1, parent2])
                
                #and we can mutate
                if random.random() < self.mutation_prob:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        self.bestModel = self.modelToTune.set_params(**self.best_params_)
        self.bestModel.fit(X, y)
        
        return self


#################################3
#DECISION TREE MODEL#
#################################

# Load and prepare data
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

# Convert to binary classification
y_binary = np.where(y > np.median(y), 1, 0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.3, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)





# Define parameter space
param_space = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(1, 20),
    'min_samples_split': range(2, 20),
    'min_samples_leaf': range(1, 20),
    'max_features': [None, 'sqrt', 'log2', 0.5, 0.7]
}


# Create and run genetic algorithm
dt = DecisionTreeClassifier(random_state=42)




ga = GeneticAlgorithm(
    modelToTune=dt,
    param_space=param_space,
    population_size=300,
    generations=30,
    crossover_prob=0.85,
    mutation_prob=0.4,
    tournament_size=10,
    cv=10,
    scoring='accuracy',
    verbose=1,
    random_state=42
)

ga.run(X_train, y_train)

# Results
print("\nBest parameters found:")
print(ga.best_params_)
print(f"\nBest cross-validation score: {ga.best_score_:.4f}")

# Evaluate on test set
test_score = ga.bestModel.score(X_test, y_test)
print(f"\nTest set accuracy: {test_score:.4f}")

# Feature importance analysis
if hasattr(ga.bestModel, 'feature_importances_'):
    print("\nFeature importances:")
    for name, importance in zip(feature_names, ga.bestModel.feature_importances_):
        print(f"{name}: {importance:.4f}")

###########################
#LIN REG MODEL
#########################



import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import warnings


# Load data
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

y_train_binary = np.where(y_train > np.median(y_train), 1, 0)
y_test_binary = np.where(y_test > np.median(y_test), 1, 0)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="l1_ratio parameter is only used when penalty is 'elasticnet'")

#Had to split this up weirdly because L1 and L2 kept causing incompatability errors
logreg_param_space = {
    'C': list(np.logspace(-4, 4, 20)),
    'penalty': ['l1', 'l2'],
    'solver': {  
        'l1': ['liblinear', 'saga'],
        'l2': ['lbfgs', 'newton-cg', 'sag', 'saga', 'liblinear']
    },
    'max_iter': list(range(100, 2001, 100)),
    'tol': [1e-4, 1e-5, 1e-6],
    'class_weight': [None, 'balanced'],
    'fit_intercept': [True, False]
}

#configuring the ga to work with log reg. We could look into using an OR tree for mutation/crossover if we have time
class logGA(GeneticAlgorithm):
    def _initialize_individual(self):
        individual = {}
        
        # 1. First we choose a penalty
        individual['penalty'] = random.choice(self.param_space['penalty'])
        
        # 2. then we choose  solver
        individual['solver'] = random.choice(self.param_space['solver'][individual['penalty']])
        
        # 3. Remaining params do not depend on penalty so can go with any
        for param in ['C', 'max_iter', 'tol', 'class_weight', 'fit_intercept']:
            individual[param] = random.choice(self.param_space[param])
            
        return individual
    
    def _crossover(self, parent1, parent2):
        child = {}
        
        # get penalty
        child['penalty'] = parent1['penalty'] if random.random() < 0.5 else parent2['penalty']
        
        # get solver that works with penalty
        if child['penalty'] == parent1['penalty']:
            child['solver'] = parent1['solver']
        else:
            child['solver'] = parent2['solver']
        
        # Allow solver mutation during crossover. Just added this because we needed a bit more diversity. But can remove
        if random.random() < 0.2: 
            child['solver'] = random.choice(
                [s for s in self.param_space['solver'][child['penalty']] if s != child['solver']]
            )
        
        # Crossover other parameters
        for param in ['C', 'max_iter', 'tol', 'class_weight', 'fit_intercept']:
            child[param] = parent1[param] if random.random() < 0.5 else parent2[param]
            
        return child
    
    def _mutate(self, individual):
        mutated = individual.copy()
        
        # Decide which parameter to mutate
        param_to_mutate = random.choice(list(self.param_space.keys()))
        
        # dealing with penalty and solver 
        if param_to_mutate == 'penalty':
            new_penalty = random.choice([p for p in self.param_space['penalty'] if p != mutated['penalty']])
            mutated['penalty'] = new_penalty
            mutated['solver'] = random.choice(self.param_space['solver'][new_penalty])
            
        elif param_to_mutate == 'solver':
            # Only mutate to other compatible solvers
            compatible_solvers = [s for s in self.param_space['solver'][mutated['penalty']] 
                                if s != mutated['solver']]
            if compatible_solvers:
                mutated['solver'] = random.choice(compatible_solvers)
                
        else:  # everything else is like normal
            current_val = mutated[param_to_mutate]
            options = [v for v in self.param_space[param_to_mutate] if v != current_val]
            if options:
                mutated[param_to_mutate] = random.choice(options)
        
        return mutated
    
    #can remove this for time if we want
    def _evaluate_individual(self, individual, X, y):
        # make sure compatible
        if individual['solver'] not in self.param_space['solver'][individual['penalty']]:
            return -np.inf  # Invalid stuff is bad scored
            
        try:
            model = LogisticRegression(**individual, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring)
            return np.mean(scores)
        except:
            return -np.inf

# Initialize
ga_logreg = logGA(
    modelToTune=LogisticRegression(random_state=42),
    param_space=logreg_param_space,
    population_size=300,
    generations=30,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1,
    random_state=42
)

print("Running Penalty-Aware Genetic Algorithm...")
ga_logreg.run(X_train, y_train_binary)


# Results
print("\n=== OPTIMIZATION RESULTS ===")
print(f"Best Parameters: {ga_logreg.best_params_}")
print(f"Best CV Accuracy: {ga_logreg.best_score_:.4f}")
print(f"Test Accuracy: {ga_logreg.bestModel.score(X_test, y_test_binary):.4f}")

# Feature coefficients
print("\nFeature Coefficients:")
for name, coef in zip(diabetes.feature_names, ga_logreg.bestModel.coef_[0]):
    print(f"{name:>8}: {coef:>10.4f}")




################
#NEURAL NET
###################


import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load and prepare data
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

y_binary = np.where(y > np.median(y), 1, 0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.3, random_state=42
)

# Scale features 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#nn PARAMS

nn_param_space = {
    'hidden_layer_sizes': {
        'values': [(50,), (100,), (50, 50), (100, 50), (50, 30, 10)],
        'type': 'tuple'
    },
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    'alpha': list(np.logspace(-6, -1, 20)),
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': list(np.logspace(-4, -1, 10)),
    'max_iter': [200, 500, 1000],
    'batch_size': [32, 64, 128],
    'early_stopping': [True, False]
}

#DOING GA

class NNGABase(GeneticAlgorithm):
    def _initialize_individual(self):
        individual = {}
        
        # Deal with special params
        for param, config in self.param_space.items():
            if param == 'hidden_layer_sizes':
                individual[param] = random.choice(config['values'])
            else:
                individual[param] = random.choice(config) if isinstance(config, list) else config
        
        return individual
    

    
    def _mutate(self, individual):
        """Mutation with special handling for NN parameters"""
        mutated = individual.copy()
        param_to_mutate = random.choice(list(self.param_space.keys()))
        
        if param_to_mutate == 'hidden_layer_sizes':
            current = mutated[param_to_mutate]
            options = [v for v in self.param_space[param_to_mutate]['values'] if v != current]
            if options:
                mutated[param_to_mutate] = random.choice(options)
        else:
            current = mutated[param_to_mutate]
            options = [v for v in self.param_space[param_to_mutate] if v != current]
            if options:
                mutated[param_to_mutate] = random.choice(options)
        
        return mutated

# Initialize and run GA
ga_nn = NNGABase(
    modelToTune=MLPClassifier(random_state=42),
    param_space=nn_param_space,
    population_size=30,
    generations=20,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1,
    random_state=42
)

print("Running Neural Network GA...")
ga_nn.run(X_train, y_train)

#grid search
param_grid = {
    'hidden_layer_sizes': nn_param_space['hidden_layer_sizes']['values'],
    'activation': nn_param_space['activation'],
    'solver': nn_param_space['solver'],
    'alpha': nn_param_space['alpha'],
    'learning_rate': nn_param_space['learning_rate'],
    'learning_rate_init': nn_param_space['learning_rate_init'],
    'max_iter': nn_param_space['max_iter'],
    'batch_size': nn_param_space['batch_size'],
    'early_stopping': nn_param_space['early_stopping']
}

grid_search = GridSearchCV(
    estimator=MLPClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

print("\nRunning Grid Search...")
grid_search.fit(X_train, y_train)

# RESULTS

print("\n=== FINAL COMPARISON ===")

# GA Results
print("\nGenetic Algorithm Results:")
print(f"Best Parameters: {ga_nn.best_params_}")
print(f"Best CV Accuracy: {ga_nn.best_score_:.4f}")
print(f"Test Accuracy: {ga_nn.bestModel.score(X_test, y_test):.4f}")

# Grid Search Results
print("\nGrid Search Results:")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")
print(f"Test Accuracy: {grid_search.score(X_test, y_test):.4f}")

# Timing Comparison
print("\nPerformance Comparison:")
print(f"GA Time: {ga_nn.total_time_elapsed_:.2f} seconds")
print(f"Grid Search Time: {grid_search.refit_time_:.2f} seconds")
