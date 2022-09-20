import model
import dataset

#Inicjacja danych oraz wyświetlanie kiklu pierwszych wierszy, informacji o ilości obiektów należących do danych klas oraz sprawdzenie czy w naszych danych nie ma nulli.
data=dataset.dataset('heart.csv')
data.display_first_rows(10,True)
data.display_first_rows(10,False)
data.display_statistics()
data.check_if_data_have_nulls()

#Wizualizacja wybranych  atrybutów zawierajacych dane numeryczne
data.display_numerical_attribute('age','Wiek','Wiek pacjentów, u których wykryto chorobę układu sercowego.')
data.display_numerical_attribute('serum_cholestoral','Cholesterol','Cholesterol pacjentów, u których wykryto chorobę układu sercowego.\n')

#wizualizacja wybranych atrybutów zawierających dane kategoryczne
data.display_categorical_attribute('sex','Płeć','Płeć a choroba')
data.display_categorical_attribute('slope','Nachylenie szczytowego odcinka ST wysiłkowego','Nachylenie szczytowego odcinka ST wysiłkowego a choroba\n')

#ocena modelu
heart_data_frame=data.heart_data_df_not_changed()
heart_model=model.model(heart_data_frame,'class','target',None,0.3)
heart_model.diplay_propa()
s=0.0
for i in range(100):
    heart_model=model.model(heart_data_frame,'class','target',None,0.3)
    s +=heart_model.display_score()
s = s/100
print(s)
heart_model.display_confusion_matrix()

#dodawanie  nowego obiektu przez użytkownika i podanie prawdopodobienstwa choroby serca dla tych parametrów
while True:
    if input('1-dodaj nowy obiekt. inny input-wyjście z aplikacji: ') != '1':
        break
    print('Podawaj wszystkie wartości jako liczby!!!')
    try:
        age = float(input('podaj wiek: '))
        sex = int(input('podaj płeć (0-kobieta, 1-mężczyzna): '))
        chest = int(input('podaj typ bólu klatki piersiowej (1-typowaangina, 2-nietypowa anigina, 3-nieaginalny bół, 4-bezobjawowy): '))
        blood_presure = float(input('podaj ciśnienie krwi w spoczynku(94-200): '))
        cholesterol = float(input('podaj cholesterol w surowicy(126-564): '))
        blood_sugar = float(input('podaj cuker we krwi na czczo: '))
        if blood_sugar >= 120:
            blood_sugar = 1
        else:
            blood_sugar = 0
        electrocardiographic = int(input('podaj spoczynkowe wyniki elektrokardiograficzne (0-normalne,1-nieprawidłowości załamka ST-T,2-LVH): '))
        heart_rate = float(input('podaj maxymalne tętno(71-200): '))
        angine = int(input('czy masz angine wywołaną wysiłkem fizycznym?(0-nie,1-tak): '))
        old_peak = float(input('obniżenie odcinka ST wywołane wysiłkiem fizycznym w stosunku do odpoczynku(0-6.2) :'))
        slope = int(input('podaj nachylenie szczytowego odcinka ST podczas ćwiczenia(1-wznoszące,2-płaskie,3-spadkowe): '))
        major_vassels = int(input('podaj liczbę zabarwionych naczyń głównych(0,1,2,3): '))
        thal = int(input('podaj wynik badania thala (3-prawidłowy,6-wada naprawiona,7-wada możliwa do naprawy): '))
        heart_model.predict_new([age,sex,chest,blood_presure,cholesterol,blood_sugar,electrocardiographic,heart_rate,angine,old_peak,slope,major_vassels,thal])
    except ValueError as error:
            print('nie wprowadziłeś liczby!')