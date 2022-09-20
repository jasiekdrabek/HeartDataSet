import pandas
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

#klasa w której importujemy dataset przygotowujemy go naszego modelu oraz wizualizujemy niektóre dane
class dataset(object):

    def __init__(self,file):
        #import danych z pliku csv
        self.heart = pandas.read_csv(file)
        #dodanie kolumny 'target' z nazwami clasy do datasetu
        file = open('heart.csv','r')
        lines = file.readlines()
        target=[]
        target_names=['','absence','presence']
        for line in lines:
            line = line.replace('\n','')
            *_,last = line
            target.append(last)
        _,*target = target
        target = [int(i) for i in target]
        self.heart['target'] = pandas.Categorical.from_codes(target, target_names)
        encoder = OneHotEncoder(sparse=False)
        self.heart_copy = self.heart.copy()

        #przygotowanie danych kategorycznych
        transformed_heart_data = pandas.DataFrame(encoder.fit_transform(
                self.heart[['sex', 'chest', 'resting_electrocardiographic_results', 'slope', 'thal']]))
        transformed_heart_data.columns = encoder.get_feature_names_out(
                ['sex', 'chest', 'resting_electrocardiographic_results', 'slope', 'thal'])
        self.heart.drop(['sex', 'chest', 'resting_electrocardiographic_results', 'slope', 'thal'], axis=1,
                                       inplace=True)
        transformed_heart_data = transformed_heart_data.rename(columns={'sex_0.0' : 'sex: female', 'sex_1.0' : 'sex: male',
                                                                       'chest_1.0':'chest: typical angina', 'chest_2.0':'chest: atypical angina', 'chest_3.0':'chest: nonanginal pain', 'chest_4.0':'chest: asymptomatic',
                                                                      'resting_electrocardiographic_results_0.0':'resting_electrocardiographic_results: normal', 'resting_electrocardiographic_results_1.0':'resting_electrocardiographic_results: having ST-T wave abnormality', 'resting_electrocardiographic_results_2.0':'resting_electrocardiographic_results: left ventricular hypertrophy',
                                                                      'slope_1.0':'slope: upsloping','slope_2.0':'slope: flat', 'slope_3.0':'slope: downsloping',
                                                                      'thal_3.0':'thal: normal', 'thal_6.0':'thal: fixed defect', 'thal_7.0':'thal: reversible defect'})
        self.heart = pandas.concat([transformed_heart_data,self.heart], axis=1)
        # tasowanie rekordów
        self.heart = self.heart.sample(frac=1)
        self.heart_copy = self.heart_copy.sample(frac=1)

    #metoda służąca do zwrócenia stworzonego obiektu DataFrame (po zmianie atrybutów kategorycznych w numeryczne).
    def heart_data_df(self):
        return self.heart

    #metoda służąca do zwrócenia stworzonego obiektu DataFrame (bez zmiany atrybutów kategorycznych).
    def heart_data_df_not_changed(self):
        return self.heart_copy

    #metoda odpowiedzialna za wyświetlanie x pierwszych wierszy.
    def display_first_rows(self,number_of_rows_to_display=10,afterChange=True):
        pandas.options.display.max_columns = None
        try:
            print('Wyświetlenie pierwszych ', number_of_rows_to_display, ' wierszy:')
            if(afterChange==True):
                print('Zmieniono dane kategoryczne')
                print(self.heart.head(number_of_rows_to_display))
            else:
                print('Dane oryginalne, nie zmienione')
                print(self.heart_copy.head(number_of_rows_to_display))
        except ValueError as error:
            print('Wprowadzono nieprawidłową liczbę')

    #metoda sprawdzająca, czy są atrybuty, które zawierają wartość null.
    def check_if_data_have_nulls(self):
        print('Ilość wartości null dla poszczególnych kolumn:')
        print(self.heart.isnull().sum())

    #Metoda odpowiedzialna za wyświetlanie informacji o ilosci rozpatrywanych przypadków,osobach chorych, osobach zdrowych
    def display_statistics(self):
        print('Ilość diagnozowanych przypadków:',len(self.heart))
        print('Ilość osób chorych:',len(self.heart[self.heart['class']==2]))
        print('Ilość osób zdrowych:',len(self.heart[self.heart['class']==1]))

    #metoda wyświetlająca dane atrybutu, który występuje w postaci numerycznej.
    def display_numerical_attribute(self,attributeName,xlabel,title):
        sns.displot(
            data=self.heart_copy[self.heart_copy['class'] == 2],
            x=attributeName)
        plt.xlabel(xlabel)
        plt.ylabel('Liczba przypadków')
        plt.title(title)
        plt.show()

    #metoda wyświetlająca dane atrybutu, który występuje w postaci kategorycznej.
    def display_categorical_attribute(self,attributeName,xlabel,title):
        dataframe2 = self.heart_copy.copy()
        if attributeName == 'sex':
            dataframe2[attributeName] = dataframe2[attributeName].apply(self.change_sex)
        elif attributeName == 'slope':
            dataframe2[attributeName] = dataframe2[attributeName].apply(self.change_slope)
        dataframe2['class'] = dataframe2['class'].apply(self.change_class)
        sns.countplot(data=dataframe2, x=attributeName, hue='class',palette=['#0FCE25','#CE310F'])
        plt.xlabel(xlabel)
        plt.ylabel('Liczba przypadków')
        plt.title(title)
        plt.show()

#metody umieszczone poniżej zmieniają wartości danego atrybutu w polskie odpowiedniki.


    def change_class(self,className):
        if className == 1:
            return 'brak choroby serca'
        else:
            return 'choroba serca'

    def change_sex(self,sex):
        if sex == 1.0:
            return 'Mężczyzna'
        else:
            return 'Kobieta'

    def change_slope(self, slope):
        if slope == 1.0:
            return 'Wznoszące'
        elif slope == 2.0:
            return 'Płaskie'
        elif slope == 3.0:
            return 'Spadkowe'