

# In[8]:


import sys
from time import time
sys.path.append("C:\\Users\\Malek\\Downloads\\Email-Classification-master\\")
from email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score
### features_train et features_test sont les fonctionnalités de l'apprentissage
### et tester des ensembles de données, respectivement
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# définir le classificateur
clf = tree.DecisionTreeClassifier()

print("\nLength of Features Train", len(features_train[0]))

#prédire le temps d'apprentissage et des tests
t0 = time()
clf.fit(features_train, labels_train)
print("\nTraining time:", round(time()-t0, 3), "s\n")
t1 = time()
pred = clf.predict(features_test)
print("Temps de prédiction :", round(time()-t1, 3), "s\n")

#calculer et afficher la précision de l'algorithme
print("Précision de l'algorithme Arbre de décision : ", accuracy_score(pred,labels_test))


# In[ ]:
##pour l'algorithme Decision tree le temps d'apprentissage est : 237.012 s
#et le temps de prediction est :  0.062 s 
# et la précision est égale à 0.9908987485779295 # la précision la plus élévée par rapport SVM et Naive Bayes



