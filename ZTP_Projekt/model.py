from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import numpy as np

#klasa z modelem QDA
class model(object):
    def __init__(self,data_frame,class_name,target_name,without=None,test_size=0.3):
        self.model_data_frame=data_frame
        if without != None:
            self.X = self.model_data_frame.drop(columns=[class_name,target_name,without])
        else:
            self.X = self.model_data_frame.drop(columns=[class_name,target_name])
        self.y = self.model_data_frame[class_name]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size)
        self.model = QuadraticDiscriminantAnalysis() 
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.y_propa = self.model.predict_proba(self.X_test)

    #metoda zwracająca procent prawidłowo sklasyfikowanych obiektów
    def display_score(self):
        score = accuracy_score(self.y_test, self.y_pred)
        return score

    #metoda wyświetlająca wykresu macierzy błędów.
    def display_confusion_matrix(self):
        sns.heatmap(confusion_matrix(self.y_test, self.y_pred), annot=True)
        plt.title('Macierz błędów (Confusion Matrix)')
        plt.show()

    #metoda wyświetlająca wykres prawdopodobieństwa choroby serca z wykorzystaniem QDA
    def diplay_propa(self):
        fig, ax = plt.subplots()
        prop = self.model.fit(self.X,self.y).predict_proba(self.X)
        x = [i[0] for i in prop[:]]
        y = [i[1] for i in prop[:]]
        colors=[]
        for i in self.y:
            if i == 1.0:
                colors.append('green')
            else:
                colors.append('red')
        j=0
        for i in prop:
            ax.scatter(i[0], i[1], c=colors[j], label=colors[j])
            j+=1
        lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]), 
    np.max([ax.get_xlim(), ax.get_ylim()]),]
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.legend(['x=y','presence','absence'])
        plt.xlabel('P(absence)')
        plt.ylabel('P(presence)')
        plt.title('Prawdopodbieństwo choroby serca w oparciu o kwadratową analizę dyskryminacyjną (QDA)')
        plt.show()

    #metoda podająca prawdopodobieństwo choroby serca dla podanych przez użytkownika parametrów
    def predict_new(self,X):
        try:
            print('prawdopodobieństwo choroby serca wynosi: ', self.model.predict_proba([X])[0][1])
        except ValueError as error:
            print('Wprowadzono nieprawidłowe dane')