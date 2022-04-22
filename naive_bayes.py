
# In[2]:


import sys
from time import time
sys.path.append("C:\\Users\\Malek\\Downloads\\Email-Classification-master\\")
from email_preprocess import preprocess
import numpy as np

#en utilisant l'algorithme Gaussian Bayes pour la classification des e-mails.
#l'algorithme est importé de la bibliothèque sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#iinitialiser les fonctionnalités et les Labels de test et d'entraînement
#la fonction preprocess est importée de email_preprocess.py 
features_train, features_test, labels_train, labels_test = preprocess()

#définir le classificateur
clf = GaussianNB()

#prédire le temps d' apprentissage et des tests
t0 = time()
clf.fit(features_train, labels_train)
print("\nTraining time:", round(time()-t0, 3), "s\n")
t1 = time()
pred = clf.predict(features_test)
print("Temps de prédiction :", round(time()-t1, 3), "s\n")

#calculer et afficher la précision
print(" Précision de l'algorithme Naive Bayes: ", accuracy_score(pred,labels_test))


# In[ ]:

##pour l'algorithme Naive bauyes le temps d'apprentissage est : 9.556 s
#et le temps de prediction est :  0.465 s #L'algorithme Naive Bayes a eu le temps de prédiction le plus rapide
# et la précision est égale à 0.9732650739476678


