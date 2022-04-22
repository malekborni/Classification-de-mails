

# In[1]:


import sys
from time import time
sys.path.append("C:\\Users\\Malek\\Downloads\\Email-Classification-master\\")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


### features_train et features_test sont les fonctionnalités de la formation
### et tester des ensembles de données, respectivement
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#définir le classificateur
clf = SVC(kernel = 'linear', C=1)

#prédire le temps d'apprentissage et des tests
t0 = time()
clf.fit(features_train, labels_train)
print("\nTraining time:", round(time()-t0, 3), "s\n")
t1 = time()
pred = clf.predict(features_test)
print("Temps de prédiction :", round(time()-t1, 3), "s\n")
#calculer et afficher la précision de l'algorithme
print("Précision de l'algorithme SVM : ", clf.score(features_test, labels_test))

# In[ ]:
##pour l'algorithme Naive bauyes le temps d'apprentissage est : 176.277 s #L'algorithme SVM a eu le temps d'apprentissage le plus long
#et le temps de prediction est :  18.632 s
# et la précision est égale à 0.9840728100113766



