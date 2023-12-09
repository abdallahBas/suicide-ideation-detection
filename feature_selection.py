from feature_extraction import *
import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


def avg(l):
    """
    Returns the average between list elements
    """
    return (sum(l)/float(len(l)))
def getFitness(individual, X, y):
    """
    Feature subset fitness function
    """

    if(individual.count(0) != len(individual)):
        # get index with value 0
        cols = [index for index in range(
            len(individual)) if individual[index] == 0]
        # get features subset
        X_parsed = X.drop(X.columns[cols], axis=1)
        X_subset = pd.get_dummies(X_parsed)
        # apply classification algorithm
        clf = LogisticRegression(max_iter = 300)
        return (avg(cross_val_score(clf, X_subset, y, cv=5)),)
    else:
        return(0,)
    
def geneticAlgorithm(X, y, n_population, n_generation):
    """
    Deap global variables
    Initialize variables to use eaSimple
    """
    # create individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # create toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_bool, len(X.columns))
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)
    toolbox.register("evaluate", getFitness, X=X, y=y)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # initialize parameters
    pop = toolbox.population(n=n_population)
    hof = tools.HallOfFame(n_population * n_generation)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # genetic algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                                   ngen=n_generation, stats=stats, halloffame=hof,
                                   verbose=True)

    # return hall of fame
    return hof

def bestIndividual(hof, X, y):
    """
    Get the best individual
    """
    maxAccurcy = 0.0
    for individual in hof:

        if(individual.fitness.values[0]> maxAccurcy):
            maxAccurcy = individual.fitness.values
            _individual = individual

    _individualHeader = [list(X)[i] for i in range(
        len(_individual)) if _individual[i] == 1]
    return _individual.fitness.values, _individual, _individualHeader





# # Step 2: Apply preprocessing
# data = normalize_corpus(df_data)

# # Step 3: Feature Extraction
# df_basic = statistics(df_data)
# df_tfidf = tfidf(df_data)
# tags_all = get_all_tags(df_data)
# df_pos = pos(df_data, tags_all)
# num_topics = 10
# df_topic = topics(df_data, num_topics)
# df_features = pd.concat([df_basic, df_tfidf, df_pos, df_topic], axis=1)
# df_all = pd.concat([df_features, df_data['y']], axis=1)

