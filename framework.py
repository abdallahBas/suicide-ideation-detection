from contractions import CONTRACTION_MAP
from text_normalizer import normalize_corpus
from feature_extraction import *
from feature_selection import *
from classification import *

# Phase 1: Read the dataset with csv extension
df_data = pd.read_csv('dataset.csv')

# Phase 2: Apply preprocessing
data = normalize_corpus(df_data)

# Phase 3: Feature Extraction
df_basic = statistics(df_data)
df_tfidf = tfidf(df_data)
tags_all = get_all_tags(df_data)
df_pos = pos(df_data, tags_all)
df_topic = topics(df_data, 10)
df_ngram = ngram(data,3,100)

# combining features in two features sets
df_original_features = pd.concat([df_basic,  df_pos, df_topic, df_tfidf, df_ngram], axis=1)
df_original_all = pd.concat([df_original_features, df_data['y']], axis=1)

df_linguistic_features = pd.concat([ df_topic, df_tfidf, df_ngram], axis=1)
df_linguistic_all = pd.concat([df_linguistic_features, df_data['y']], axis=1)

'''Nowwe will use linguistic features but you can test agin using original features'''

# Phase 4: Feature Selection
n_pop, n_gen = 200 , 100
le = LabelEncoder()
le.fit(df_linguistic_all.iloc[:, -1])
y = le.transform(df_linguistic_all.iloc[:, -1])
X = df_linguistic_all.iloc[:, :-1]
# get accuracy with all features
individual = [1 for i in range(len(X.columns))]
print("Accuracy with all features: \t" +
    str(getFitness(individual, X, y)) + "\n")

# apply genetic algorithm
hof = geneticAlgorithm(X, y, n_pop, n_gen)

# select the best individual
accuracy, individual, header = bestIndividual(hof, X, y)
print('Best Accuracy: \t' + str(accuracy))
print('Number of Features in Subset: \t' + str(individual.count(1)))
print('Individual: \t\t' + str(individual))
print('Feature Subset\t: ' + str(header))

print('\n\ncreating a new classifier with the result')

# read dataframe from csv one more time

df = pd.read_csv('dataset.csv', sep=',')

# with feature subset
X = df[header]

clf = LogisticRegression()
scores = cross_val_score(clf, X, y, cv=5)
print("Accuracy with Feature Subset: \t" + str(avg(scores)) + "\n")

# Step 5: Classification Phase
test_ratio = 0.3
X_train,  X_test,  y_train,  y_test  = train_test_split(X, y, test_size=test_ratio, random_state=42 , shuffle=True)

# 1- Random Forest Classifier
random_forest_classifier_train(X_train ,  y_train)
random_forest_classifier_test(X_test , y_test)

# 2- KNN classifier
KNN_classifier_train(X_train, y_train)
KNN_classifier_test(X_test, y_test)

# 3- GBDT classifier
GBDT_classifier_train(X_train, y_train )
GBDT_classifier_test(X_test, y_test)

# 4- XGBoost classifier
Xgboost_classifier_train(X_train, y_train )
Xgboost_classifier_test(X_test , y_test)
