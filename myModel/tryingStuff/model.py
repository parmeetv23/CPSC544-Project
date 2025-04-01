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
    population_size=50,
    generations=10,
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