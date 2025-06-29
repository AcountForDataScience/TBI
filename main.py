import telebot
from telebot import types
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from scipy import stats
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import io
import re

import heapq
plt.rcParams['figure.figsize'] = [10, 7]

import csv

from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index as index

## Variables ##
Access_dic = {
    'aramasht@gmail.com': '6719',
    'test@test.com': 'test'
}
Access_dic_0 = str(list(Access_dic.keys())[0])
Access_dic_1 = str(list(Access_dic.keys())[1])

#Відкривання очей (E - Eye opening):
#4 - спонтанне відкривання (spontaneous opening)
#3 - відкривання на мовлення (opening during a conversation)
#2 - відкривання на больовий стимул(opening to painful stimulus)
#1 - немає реакції (no reaction)
Eye_Opening_Digits_Dir = {'spontaneous_opening': 4, 'opening_during_a_conversation': 3, 'opening_to_painful_stimulus': 2, 'no_reaction': 1}
Eye_Opening_Digits_Dir_0 = str(list(Eye_Opening_Digits_Dir.keys())[0])
Eye_Opening_Digits_Dir_1 = str(list(Eye_Opening_Digits_Dir.keys())[1])
Eye_Opening_Digits_Dir_2 = str(list(Eye_Opening_Digits_Dir.keys())[2])
Eye_Opening_Digits_Dir_3 = str(list(Eye_Opening_Digits_Dir.keys())[3])

Eye_Opening_Translation_Dir = {'spontaneous_opening': 'спонтанне відкривання', 'opening_during_a_conversation': 'відкривання на мовлення', 'opening_to_painful_stimulus': 'відкривання на больовий стимул', 'no_reaction': 'немає реакції'}
Eye_Opening_Translation_Dir_0 = str(list(Eye_Opening_Translation_Dir.keys())[0])
Eye_Opening_Translation_Dir_1 = str(list(Eye_Opening_Translation_Dir.keys())[1])
Eye_Opening_Translation_Dir_2 = str(list(Eye_Opening_Translation_Dir.keys())[2])
Eye_Opening_Translation_Dir_3 = str(list(Eye_Opening_Translation_Dir.keys())[3])

#Вербальна реакція (V - Verbal response):

#5 - орієнтована (oriented)
#4 - дезорієнтована (disoriented)
#3 - недоречні слова (inappropriate words)
#2 - незрозумілі звуки
#1 - немає реакції (no reaction)
Verbal_Response_Digits_Dir = {'oriented': 5, 'disoriented': 4, 'inappropriate_words': 3, 'unintelligible_sounds': 2, 'no_reaction': 1}

Verbal_Response_Digits_Dir_0 = str(list(Verbal_Response_Digits_Dir.keys())[0])
Verbal_Response_Digits_Dir_1 = str(list(Verbal_Response_Digits_Dir.keys())[1])
Verbal_Response_Digits_Dir_2 = str(list(Verbal_Response_Digits_Dir.keys())[2])
Verbal_Response_Digits_Dir_3 = str(list(Verbal_Response_Digits_Dir.keys())[3])
Verbal_Response_Digits_Dir_4 = str(list(Verbal_Response_Digits_Dir.keys())[4])

Verbal_Response_Translation_Dir = {'oriented': 'орієнтована', 'disoriented': 'дезорієнтована', 'inappropriate_words': 'недоречні слова', 'unintelligible_sounds': 'незрозумілі звуки', 'no_reaction': 'немає реакції'}
Verbal_Response_Translation_Dir_0 = str(list(Verbal_Response_Translation_Dir.keys())[0])
Verbal_Response_Translation_Dir_1 = str(list(Verbal_Response_Translation_Dir.keys())[1])
Verbal_Response_Translation_Dir_2 = str(list(Verbal_Response_Translation_Dir.keys())[2])
Verbal_Response_Translation_Dir_3 = str(list(Verbal_Response_Translation_Dir.keys())[3])
Verbal_Response_Translation_Dir_4 = str(list(Verbal_Response_Translation_Dir.keys())[4])

#Рухова реакція (M - Motor response):

#6 - виконує команди (executes commands)
#5 - локалізує біль (localizes pain)
#4 - відсмикує на біль (recoils from pain)
#3 - патологічне згинання (декортикація) (pathological bending)
#2 - патологічне розгинання (децеребрація) (pathological extension)
#1 - немає реакції

Motor_Response_Digits_Dir = {'executes commands': 6, 'localizes pain': 5, 'recoils from pain': 4, 'pathological bending': 3, 'pathological extension': 2, 'no reaction': 1}
Motor_Response_Digits_Dir_0 = str(list(Motor_Response_Digits_Dir.keys())[0])
Motor_Response_Digits_Dir_1 = str(list(Motor_Response_Digits_Dir.keys())[1])
Motor_Response_Digits_Dir_2 = str(list(Motor_Response_Digits_Dir.keys())[2])
Motor_Response_Digits_Dir_3 = str(list(Motor_Response_Digits_Dir.keys())[3])
Motor_Response_Digits_Dir_4 = str(list(Motor_Response_Digits_Dir.keys())[4])
Motor_Response_Digits_Dir_5 = str(list(Motor_Response_Digits_Dir.keys())[5])

Motor_Response_Translation_Dir = {'executes commands': 'виконує команди', 'localizes pain': 'локалізує біль', 'recoils from pain': 'відсмикує на біль', 'pathological bending': 'патологічне згинання', 'pathological extension': 'патологічне розгинання', 'no reaction': 'немає реакції'}
Motor_Response_Translation_Dir_0 = str(list(Motor_Response_Translation_Dir.keys())[0])
Motor_Response_Translation_Dir_1 = str(list(Motor_Response_Translation_Dir.keys())[1])
Motor_Response_Translation_Dir_2 = str(list(Motor_Response_Translation_Dir.keys())[2])
Motor_Response_Translation_Dir_3 = str(list(Motor_Response_Translation_Dir.keys())[3])
Motor_Response_Translation_Dir_4 = str(list(Motor_Response_Translation_Dir.keys())[4])
Motor_Response_Translation_Dir_5 = str(list(Motor_Response_Translation_Dir.keys())[5])

YesNo_dict = {
    'No': 0,
    'Yes': 1
}
YesNo_dict_0 = str(list(YesNo_dict.keys())[0])
YesNo_dict_1 = str(list(YesNo_dict.keys())[1])
#///////////////////////////////////////////////////////////////////
Eye_Opening_Dic = {
'Spontaneous - Opens eyes spontaneously': 4,
'To Speech - Opens eyes in response to verbal command': 3,
'To Pain - Opens eyes in response to pain': 2,
'No Response - No eye opening': 1
}
Eye_Opening_Dic_0 = str(list(Eye_Opening_Dic.keys())[0])
Eye_Opening_Dic_1 = str(list(Eye_Opening_Dic.keys())[1])
Eye_Opening_Dic_2 = str(list(Eye_Opening_Dic.keys())[2])
Eye_Opening_Dic_3 = str(list(Eye_Opening_Dic.keys())[3])

Eye_Opening = None

Verbal_Response_Dic = {
'Oriented - Oriented to time, place, and person': 5,
'Confused - Confused conversation, but able to answer questions': 4,
'Inappropriate Words - Incoherent or random words': 3,
'Incomprehensible Sounds - Moaning, groaning (but no words)': 2,
'No Response - No verbal response': 1
                    }
Verbal_Response_Dic_0 = str(list(Verbal_Response_Dic.keys())[0])
Verbal_Response_Dic_1 = str(list(Verbal_Response_Dic.keys())[1])
Verbal_Response_Dic_2 = str(list(Verbal_Response_Dic.keys())[2])
Verbal_Response_Dic_3 = str(list(Verbal_Response_Dic.keys())[3])
Verbal_Response_Dic_4 = str(list(Verbal_Response_Dic.keys())[4])

Verbal_Response = None

Motor_Response_Dic = {
'Obeys Commands - Obeys simple commands': 6,
'Localizes to Pain - Purposeful movement towards a painful stimulus': 5,
'Withdraws from Pain - Withdraws part of body from pain': 4,
'Flexion (Abnormal) - Abnormal flexion (decorticate posturing)': 3,
'Extension (Abnormal) - Abnormal extension (decerebrate posturing)': 2,
'No Response - No motor response' : 1
}
Motor_Response_Dic_0 = str(list(Motor_Response_Dic.keys())[0])
Motor_Response_Dic_1 = str(list(Motor_Response_Dic.keys())[1])
Motor_Response_Dic_2 = str(list(Motor_Response_Dic.keys())[2])
Motor_Response_Dic_3 = str(list(Motor_Response_Dic.keys())[3])
Motor_Response_Dic_4 = str(list(Motor_Response_Dic.keys())[4])
Motor_Response_Dic_5 = str(list(Motor_Response_Dic.keys())[5])

Motor_Response = None
CGS = None

Neurological_Outcome_Scale_Dic = {
'Good Recovery':5,
'Moderate Disability':4,
'Severe Disability':3,
'Vegetative State':2,
'Death':1
}
GOS_Dic_0 = str(list(Neurological_Outcome_Scale_Dic.keys())[0])
GOS_Dic_1 = str(list(Neurological_Outcome_Scale_Dic.keys())[1])
GOS_Dic_2 = str(list(Neurological_Outcome_Scale_Dic.keys())[2])
GOS_Dic_3 = str(list(Neurological_Outcome_Scale_Dic.keys())[3])
GOS_Dic_4 = str(list(Neurological_Outcome_Scale_Dic.keys())[4])

ID = None
Initial_GCS = None
Age = None
Hematoma_size_ml = None
Midline_shift_mm = None
#Midline Shift (MLS) in mm refers to the degree to which the brain's structures have been pushed away from their normal central position. It's a critical measurement, usually taken from a CT scan of the head.
#Зсув(Сдвиг), Зміщення середньої лінії (ЗСЛ) у мм вказує на ступінь зміщення структур мозку від їхнього нормального центрального положення. Це критично важливий показник, який зазвичай визначається за допомогою комп'ютерної томографії голови.
Time_to_surgery_h = None
ICP_max_to_surgery = None
Concomitant_traumas = None
Survival = None
Neurological_outcome_scale = None
Complications = None
NewPatient = None
ComplicationsProbability = None
RandomForestComplicationsProbability = None

## Functions ##
# Password#
def Check_Password(password):
  for value in Access_dic.values():
    if value == password:
      return True
result = Check_Password('6719')
print(result)

# Pattern is_valid_number(value) #
def is_valid_number(value):
    pattern = r'^\d+(\.\d+)?$'  # Matches integers like '4' and floats like '4.5'
    return bool(re.match(pattern, str(value)))

#RandomForestComplications#
def RandomForestComplicationsProbabilityFunc(x1, x2, x3, x4, x5, x6, x7):
  # Завантаження даних у Pandas DataFrame
  #df = pd.read_csv(io.StringIO(csv_data))

  df = pd.read_csv('BrainInjuryComplication.csv')

  # Визначення ознак (X) та цільової змінної (y)
  # Ми прогнозуємо стовпець 'Ускладнення'
  X = df.drop(['ID', 'Survival','Complications', 'Neurological_outcome_scale'], axis=1)
  y = df['Complications']

  # Розділення даних на тренувальний та тестовий набори
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  # Ініціалізація та навчання моделі (використовуємо Random Forest для класифікації)
  model = RandomForestClassifier(random_state=42)
  model.fit(X_train, y_train)

  # Прогнозування на тестовому наборі
  y_pred = model.predict(X_test)

  # Тепер ви можете використовувати навчену модель для прогнозування ускладнень для нових пацієнтів
  # Наприклад, для нового пацієнта з такими характеристиками:
  NewPatient = pd.DataFrame({
   'Initial_GCS': [x1],
   'Age': [x2],
   'Hematoma_size_ml': [x3],
   'Midline_shift_mm': [x4],
   'Time_to_surgery_h': [x5],
   'ICP_max_to_surgery': [x6],
   'Concomitant_traumas': [x7]
  })

  ComplicationsProbability = model.predict(NewPatient)
  if ComplicationsProbability < 1:
    ComplicationsProbabilityAnswer = 'not expected'
  else:
    ComplicationsProbabilityAnswer = 'is expected'
  ComplicationsProbabilityPercent = model.predict_proba(NewPatient)
  ComplicationsProbabilityPercent = ComplicationsProbabilityPercent[-1][1]
  ComplicationsProbabilityPercent = ComplicationsProbabilityPercent*100
  return ComplicationsProbabilityAnswer, ComplicationsProbabilityPercent

# LogisticRegressionComplications #
def LogisticRegressionComplicationsProbabilityFunc(x1, x2, x3, x4, x5, x6, x7):
#LogisticRegression для прогнозування Ускладнення
  # Завантаження даних у Pandas DataFrame
  #df = pd.read_csv(io.StringIO(csv_data))

  df = pd.read_csv('BrainInjuryComplication.csv')

  # Визначення ознак (X) та цільової змінної (y)
  # Ми прогнозуємо стовпець 'Ускладнення'
  X = df.drop(['ID', 'Survival','Complications', 'Neurological_outcome_scale'], axis=1)
  y = df['Complications']

  # Розділення даних на тренувальний та тестовий набори
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


  # Ініціалізація та навчання моделі (використовуємо Random Forest для класифікації)
  model = LogisticRegression(random_state=42)
  model.fit(X_train, y_train)

  # Прогнозування на тестовому наборі
  y_pred = model.predict(X_test)

  # Тепер ви можете використовувати навчену модель для прогнозування ускладнень для нових пацієнтів
  # Наприклад, для нового пацієнта з такими характеристиками:
  NewPatient = pd.DataFrame({
   'Initial_GCS': [x1],
   'Age': [x2],
   'Hematoma_size_ml': [x3],
   'Midline_shift_mm': [x4],
   'Time_to_surgery_h': [x5],
   'ICP_max_to_surgery': [x6],
   'Concomitant_traumas': [x7]
  })

  #Resume_predicted_proba_lr_Cognitive_Disorders = Resume_predicted_proba_lr_Cognitive_Disorders[-1][1]

  LogComplicationsProbability = model.predict(NewPatient)
  if LogComplicationsProbability < 1:
    LogComplicationsProbabilityAnswer = 'not expected'
  else:
    LogComplicationsProbabilityAnswer = 'is expected'
  LogComplicationsProbabilityPercent = model.predict_proba(NewPatient)
  LogComplicationsProbabilityPercent = LogComplicationsProbabilityPercent[-1][1]
  LogComplicationsProbabilityPercent = LogComplicationsProbabilityPercent*100

  return LogComplicationsProbabilityAnswer, LogComplicationsProbabilityPercent

# Survival #
def RandomForestSurvivalProbabilityFunc(x1, x2, x3, x4, x5, x6, x7):
#RandomForestClassifier для прогнозування Ускладнення
  # Завантаження даних у Pandas DataFrame
  #df = pd.read_csv(io.StringIO(csv_data))

  df = pd.read_csv('BrainInjuryComplication.csv')

  # Визначення ознак (X) та цільової змінної (y)
  # Ми прогнозуємо стовпець 'Ускладнення'
  X = df.drop(['ID', 'Survival','Complications', 'Neurological_outcome_scale'], axis=1)
  y = df['Survival']

  # Розділення даних на тренувальний та тестовий набори
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


  # Ініціалізація та навчання моделі (використовуємо Random Forest для класифікації)
  model = RandomForestClassifier(random_state=42)
  model.fit(X_train, y_train)

  # Прогнозування на тестовому наборі
  y_pred = model.predict(X_test)

   # Тепер ви можете використовувати навчену модель для прогнозування ускладнень для нових пацієнтів
  # Наприклад, для нового пацієнта з такими характеристиками:
  NewPatient = pd.DataFrame({
   'Initial_GCS': [x1],
   'Age': [x2],
   'Hematoma_size_ml': [x3],
   'Midline_shift_mm': [x4],
   'Time_to_surgery_h': [x5],
   'ICP_max_to_surgery': [x6],
   'Concomitant_traumas': [x7],
  })

  SurvivalProbability = model.predict(NewPatient)
  if SurvivalProbability < 1:
    SurvivalProbabilityAnswer = 'not expected'
  else:
    SurvivalProbabilityAnswer = 'is expected'
  SurvivalProbabilityPercent = model.predict_proba(NewPatient)
  SurvivalProbabilityPercent = SurvivalProbabilityPercent[-1][1]
  SurvivalProbabilityPercent = SurvivalProbabilityPercent*100
  return SurvivalProbabilityAnswer, SurvivalProbabilityPercent

# Neurological Outcome #
def NeurologicalOutcomeFunc(x1, x2, x3, x4, x5, x6, x7):
  df = pd.read_csv('BrainInjuryComplication.csv')

  # Наприклад: 1–3 — поганий вихід, 4–5 — добрий, бінарна класифікацію (наприклад, "інвалідність" vs. "відновлення"):
  df['Neurological_binary'] = df['Neurological_outcome_scale'].apply(lambda x: 1 if x >= 4 else 0)

  # Ознаки
  features = ['Initial_GCS', 'Age', 'Hematoma_size_ml', 'Midline_shift_mm',
            'Time_to_surgery_h', 'ICP_max_to_surgery', 'Concomitant_traumas']
  X = df[features]
  y = df['Neurological_binary']

  # Розділення на train/test
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Навчання Random Forest
  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)

  # Оцінка моделі
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Точність моделі: {accuracy:.2f}")

  print("\nЗвіт про класифікацію:")
  print(classification_report(y_test, y_pred))

  print("\nМатриця плутанини:")
  print(confusion_matrix(y_test, y_pred))

  # Тепер ви можете використовувати навчену модель для прогнозування ускладнень для нових пацієнтів
  # Наприклад, для нового пацієнта з такими характеристиками:
  NewPatient = pd.DataFrame({
   'Initial_GCS': [x1],
   'Age': [x2],
   'Hematoma_size_ml': [x3],
   'Midline_shift_mm': [x4],
   'Time_to_surgery_h': [x5],
   'ICP_max_to_surgery': [x6],
   'Concomitant_traumas': [x7]
  })

  #Resume_predicted_proba_lr_Cognitive_Disorders = Resume_predicted_proba_lr_Cognitive_Disorders[-1][1]

  NeurologicalOutcomeProbability = model.predict(NewPatient)
  if NeurologicalOutcomeProbability < 1:
    NeurologicalOutcomeProbabilityAnswer = ': invalidity'
  else:
    NeurologicalOutcomeProbabilityAnswer = ': significant recovery'
  NeurologicalOutcomeProbabilityPercent = model.predict_proba(NewPatient)
  NeurologicalOutcomeProbabilityPercent = NeurologicalOutcomeProbabilityPercent[-1][1]
  NeurologicalOutcomeProbabilityPercent = NeurologicalOutcomeProbabilityPercent*100

  features = ['Initial_GCS',
   'Age',
   'Hematoma_size_ml',
   'Midline_shift_mm',
   'Time_to_surgery_h',
   'ICP_max_to_surgery',
   'Concomitant_traumas']
  # Важливість ознак
  importances = model.feature_importances_
  feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
  feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
  feature_importance_dict = feature_importance_df.set_index('Feature')['Importance'].to_dict()

  return feature_importance_dict, NeurologicalOutcomeProbabilityAnswer, NeurologicalOutcomeProbabilityPercent

# Вибор між Трепанація з дренуванням, Краніотомія, Малоінвазивні втручання #    
# Дані
Treatment_type_Dic = {
    1: 'Craniotomy',
    2: 'Trepanation_with_drainage',
    3: 'Minimally_invasive_interventions'
}
df = pd.read_csv('BrainInjuryTreatmentTypes.csv')

# Створимо колонку ефективності (1 = вижив + без ускладнень)
df['Effective'] = np.where((df['Survival'] == 1) & (df['Complication'] == 0), 1, 0)

# Особливості, які будемо використовувати
features = ['Initial_GCS', 'Age', 'Hematoma_size_ml', 'Midline_shift_mm',
            'Time_to_surgery_h', 'ICP_max_to_surgery', 'Concomitant_traumas']

# Тренуємо окрему модель для кожного типу лікування
models = {}
effectiveness_scores = {}

for treatment_id, treatment_name in Treatment_type_Dic.items():
    treatment_data = df[df['Treatment'] == treatment_id]

    X = treatment_data[features]
    y = treatment_data['Effective']

    if len(y.unique()) < 2:
        print(f"⚠️ Недостатньо варіації для: {treatment_name}")
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    models[treatment_id] = model

#========

def recommend_best_treatment(patient_data: dict):
    effectiveness_results = {}

    for treatment_id, model in models.items():
        input_df = pd.DataFrame([patient_data])
        predicted_proba = model.predict_proba(input_df)[0][1]  # Імовірність ефективності
        treatment_name = Treatment_type_Dic[treatment_id]
        effectiveness_results[treatment_name] = predicted_proba

    # Вибір найефективнішого
    best_treatment = max(effectiveness_results, key=effectiveness_results.get)

    #print("📊 Прогнозована ефективність по кожному типу лікування:")
    for t_name, score in effectiveness_results.items():
        print(f"   - {t_name}: {score:.2%}")

    #Craniotomy_Result = np.float64(effectiveness_results['Craniotomy'])
    effectiveness_results_str_dic = {key: str(value) for key, value in effectiveness_results.items()}

    #return f"\n✅ Рекомендоване лікування: {best_treatment} (найвища ефективність)"
    return best_treatment, effectiveness_results_str_dic

# CGS #
def Calculate_CGS(x1, x2, x3):
  CGS = x1 + x2 + x3
  return CGS

## Bot ##

bot = telebot.TeleBot('8044522836:AAGsgb6d3r4CEGKhQMjD9Lk1wPN7pU-bNGk')

@bot.message_handler(commands=['help', 'start'])

def send_welcome(message):
    msg = bot.send_message(message.chat.id, "\n\nHello, I'm the bot \"Ai Medical Assistant\" for the treatment of Craniotomy(Traumatic brain injury)!")
    chat_id = message.chat.id
    markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
    markup.add('Next')
    msg = bot.reply_to(message, 'Please enter your password', reply_markup=markup)
    bot.register_next_step_handler(msg, process_Password_step)

def process_Password_step(message):
  try:
    chat_id = message.chat.id
    Password_message = message.text
    result = Check_Password(Password_message)
    if result == True:
      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add('Next')
      msg = bot.reply_to(message, 'You are welcome. Please press Next to continue', reply_markup=markup)
      bot.register_next_step_handler(msg, process_Eye_Opening_step)
    else:
      print(Password_message)
      msg = bot.reply_to(message, '❌ Incorrect password. Please try again.')
      bot.register_next_step_handler(msg, process_Password_step)
  except Exception as e:
    bot.reply_to(message, 'oooops process_Password_step')

def process_Eye_Opening_step(message):
    try:
        chat_id = message.chat.id
        Next = message.text
        if (Next == 'Next'):
          markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
          markup.add(Eye_Opening_Dic_0, Eye_Opening_Dic_1, Eye_Opening_Dic_2, Eye_Opening_Dic_3)
          msg = bot.reply_to(message, 'To assess the level of consciousness (Glasgow Neurological Coma Scale), please enter the eye opening value.', reply_markup=markup)
          bot.register_next_step_handler(msg, process_Verbal_Response_step)
        else:
          raise Exception("Eye_Opening ")
    except Exception as e:
        bot.reply_to(message, 'oooops Eye_Opening_step')

def process_Verbal_Response_step(message):
    try:
        chat_id = message.chat.id
        Eye_Opening_message = message.text
        global Eye_Opening
        Eye_Opening = Eye_Opening_Dic[Eye_Opening_message]
        if (Eye_Opening_message == Eye_Opening_Dic_0) or (Eye_Opening_message == Eye_Opening_Dic_1) or (Eye_Opening_message == Eye_Opening_Dic_2) or (Eye_Opening_message == Eye_Opening_Dic_3):
          markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
          markup.add(Verbal_Response_Dic_0, Verbal_Response_Dic_1, Verbal_Response_Dic_2, Verbal_Response_Dic_3, Verbal_Response_Dic_4)
          msg = bot.reply_to(message, 'To assess the level of consciousness (Glasgow Neurological Coma Scale), please enter the verbal response value.', reply_markup=markup)
          bot.register_next_step_handler(msg, process_Motor_Response_step)
        else:
          raise Exception("Verbal_Response_step ")
    except Exception as e:
        bot.reply_to(message, 'oooops process_Verbal_Response_step')

def process_Motor_Response_step(message):
    try:
        chat_id = message.chat.id
        Verbal_Response_message = message.text
        global Verbal_Response
        Verbal_Response = Verbal_Response_Dic[Verbal_Response_message]
        if (Verbal_Response_message == Verbal_Response_Dic_0) or (Verbal_Response_message == Verbal_Response_Dic_1) or (Verbal_Response_message == Verbal_Response_Dic_2) or (Verbal_Response_message == Verbal_Response_Dic_3) or (Verbal_Response_message == Verbal_Response_Dic_4):
          markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
          markup.add(Motor_Response_Dic_0, Motor_Response_Dic_1, Motor_Response_Dic_2, Motor_Response_Dic_3, Motor_Response_Dic_4, Motor_Response_Dic_5)
          msg = bot.reply_to(message, 'To assess the level of consciousness (Glasgow Neurological Coma Scale), please enter the Motor response value.', reply_markup=markup)
          bot.register_next_step_handler(msg, process_Initial_GCS_step)
        else:
          raise Exception("process_Motor_Response_step ")
    except Exception as e:
        bot.reply_to(message, 'oooops process_Motor_Response_step')

def process_Initial_GCS_step(message):
    try:
        chat_id = message.chat.id
        Motor_Response_message = message.text
        global Motor_Response
        Motor_Response = Motor_Response_Dic[Motor_Response_message]
        if (Motor_Response_message == Motor_Response_Dic_0) or (Motor_Response_message == Motor_Response_Dic_1) or (Motor_Response_message == Motor_Response_Dic_2) or (Motor_Response_message == Motor_Response_Dic_3) or (Motor_Response_message == Motor_Response_Dic_4) or (Motor_Response_message == Motor_Response_Dic_5):
          markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
          markup.add('Next')
          msg = bot.reply_to(message, 'To calculate the level of consciousness (Glasgow Neurological Coma Scale) please press Next.', reply_markup=markup)
          bot.register_next_step_handler(msg, process_Initial_GCS_calculate_step)
        else:
          raise Exception("process_Initial_GCS_step ")
    except Exception as e:
        bot.reply_to(message, 'oooops process_Initial_GCS_step')

def process_Initial_GCS_calculate_step(message):
    try:
        chat_id = message.chat.id
        Initial_GCS_calculate_message = message.text
        if (Initial_GCS_calculate_message == 'Next'):
          Glasgow_Neurological_Coma_Scale = Calculate_CGS(Eye_Opening, Verbal_Response, Motor_Response)
          bot.send_message(chat_id,
          '\n - The level of consciousness (Glasgow Neurological Coma Scale) is: ' + str(Glasgow_Neurological_Coma_Scale)
          )
          global Initial_GCS
          Initial_GCS = Glasgow_Neurological_Coma_Scale
          msg = bot.reply_to(message, 'Please enter Age')
          bot.register_next_step_handler(msg, process_Age_step)
        else:
          raise Exception("process_Initial_GCS_calculate_step")
    except Exception as e:
        bot.reply_to(message, 'oooops process_Initial_GCS_calculate_step')

def process_Age_step(message):
  try:
    chat_id = message.chat.id
    global Age
    Age = message.text
    if not Age.isdigit():
      msg = bot.reply_to(message, 'Age must be a number. Please enter an age.')
      bot.register_next_step_handler(msg, process_Age_step)
    else:
      msg = bot.reply_to(message, 'Please enter hematoma size')
      bot.register_next_step_handler(msg, process_Hematoma_Size_step)
  except Exception as e:
    bot.reply_to(message, 'oooops process_Age_step')

def process_Hematoma_Size_step(message):
  try:
    chat_id = message.chat.id
    global Hematoma_size_ml
    Hematoma_size_ml = message.text
    if not Hematoma_size_ml.isdigit():
      msg = bot.reply_to(message, 'The size of the hematoma should be a number.')
      bot.register_next_step_handler(msg, process_Hematoma_Size_step)
    else:
      msg = bot.reply_to(message, 'Please enter the value of midline shift(the degree to which the brains structures are pushed away from their normal central position)')
      bot.register_next_step_handler(msg, process_Midline_shift_step)
  except Exception as e:
    bot.reply_to(message, 'oooops process_Hematoma_Size_step')

def process_Midline_shift_step(message):
  try:
    chat_id = message.chat.id
    global Midline_shift_mm
    Midline_shift_mm = message.text
    if not Midline_shift_mm.isdigit():
      msg = bot.reply_to(message, 'The midline offset must be a number.')
      bot.register_next_step_handler(msg, process_Midline_shift_step)
    else:
      msg = bot.reply_to(message, 'Please enter the time to surgery')
      bot.register_next_step_handler(msg, process_Time_to_surgery_step)
  except Exception as e:
    bot.reply_to(message, 'oooops process_Midline_shift_step')

def process_Time_to_surgery_step(message):
  try:
    chat_id = message.chat.id
    global Time_to_surgery_h
    Time_to_surgery_h = message.text
    if not Time_to_surgery_h.isdigit():
      msg = bot.reply_to(message, 'The time to surgery should be a number.')
      bot.register_next_step_handler(msg, process_Time_to_surgery_step)
    else:
      msg = bot.reply_to(message, 'Please enter the value of maximum intracranial pressure (ICP) before surgery')
      bot.register_next_step_handler(msg, process_ICP_max_to_surgery_step)
  except Exception as e:
    bot.reply_to(message, 'oooops process_Time_to_surgery_step')

def process_ICP_max_to_surgery_step(message):
  try:
    chat_id = message.chat.id
    global ICP_max_to_surgery
    ICP_max_to_surgery = message.text
    if not ICP_max_to_surgery.isdigit():
      msg = bot.reply_to(message, 'The maximum intracranial pressure (ICP) before surgery should be a number.')
      bot.register_next_step_handler(msg, process_ICP_max_to_surgery_step)
    else:
      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add(YesNo_dict_1)
      msg = bot.reply_to(message, 'Presence of concomitant injuries (other injuries that a patient has in addition to the primary condition)', reply_markup=markup)
      bot.register_next_step_handler(msg, process_Concomitant_traumas_step)
  except Exception as e:
    bot.reply_to(message, 'oooops process_ICP_max_to_surgery_step')

def process_Concomitant_traumas_step(message):
  try:
    chat_id = message.chat.id
    global Concomitant_traumas
    Concomitant_traumas_message = message.text
    if (Concomitant_traumas_message == YesNo_dict_0) or (Concomitant_traumas_message == YesNo_dict_1):
      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add(YesNo_dict_1)
      msg = bot.reply_to(message, 'Predict the consequences, survival and Neurological_outcome?', reply_markup=markup)
      global Concomitant_traumas
      Concomitant_traumas = YesNo_dict[Concomitant_traumas_message]
      bot.register_next_step_handler(msg, predict_craniotomy_complication_step)
  except Exception as e:
    bot.reply_to(message, 'oooops process_Concomitant_traumas_step')

#def process_Survival_step(message):
#  try:
#    chat_id = message.chat.id   
#    Survival_message = message.text
#    if Survival_message == YesNo_dict_1:
#      global Survival
#      Survival = 
#      msg = bot.reply_to(message, 'Neurilogical')
#      bot.register_next_step_handler(msg, process_Neurological_outcome_scale_step)
#  except Exception as e:
#   bot.reply_to(message, 'oooops process_Survival_step')

#def process_Neurological_outcome_scale_step(message):
#  try:
#    chat_id = message.chat.id
#    global Neurological_outcome_scale
#    Neurological_outcome_scale_message = message.text
#    Neurological_outcome_scale = Neurological_Outcome_Scale_Dic[Neurological_outcome_scale_message]
#    if (Neurological_outcome_scale_message == GOS_Dic_0) or (Neurological_outcome_scale_message == GOS_Dic_1) or (Neurological_outcome_scale_message == GOS_Dic_2) or (Neurological_outcome_scale_message == GOS_Dic_3) or (Neurological_outcome_scale_message == GOS_Dic_4):
#      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
#      markup.add(YesNo_dict_0, YesNo_dict_1)
#      msg = bot.reply_to(message, 'Predict the consequences and survival?', reply_markup=markup)
#      bot.register_next_step_handler(msg, predict_craniotomy_complication_step)
#  except Exception as e:
#    bot.reply_to(message, 'oooops process_Survival_step')

def predict_craniotomy_complication_step(message):
  try:
    chat_id = message.chat.id
    Predict_Complication = message.text
    if (Predict_Complication == YesNo_dict_0) or (Predict_Complication == YesNo_dict_1):

      ComplicationsProbabilityAnswer, ComplicationsProbabilityPercent= RandomForestComplicationsProbabilityFunc(Initial_GCS, Age, Hematoma_size_ml, Midline_shift_mm, Time_to_surgery_h, ICP_max_to_surgery, Concomitant_traumas)
      LogComplicationsProbabilityAnswer, LogComplicationsProbabilityPercent = LogisticRegressionComplicationsProbabilityFunc(Initial_GCS, Age, Hematoma_size_ml, Midline_shift_mm, Time_to_surgery_h, ICP_max_to_surgery, Concomitant_traumas)
      SurvivalProbabilityAnswer, SurvivalProbabilityPercent = RandomForestSurvivalProbabilityFunc(Initial_GCS, Age, Hematoma_size_ml, Midline_shift_mm, Time_to_surgery_h, ICP_max_to_surgery, Concomitant_traumas)
      feature_importance_dict, NeurologicalOutcomeProbabilityAnswer, NeurologicalOutcomeProbabilityPercent  = NeurologicalOutcomeFunc(Initial_GCS, Age, Hematoma_size_ml, Midline_shift_mm, Time_to_surgery_h, ICP_max_to_surgery, Concomitant_traumas)

      bot.send_message(chat_id,

      '\n - Complication ' + str(ComplicationsProbabilityAnswer)+
      '\n - Probability of complication in percent: ' + str(ComplicationsProbabilityPercent) + ' %' +
      '\n'+ '(RandomForest)' +

      '\n\n - Complication ' + str(LogComplicationsProbabilityAnswer)+
      '\n - Probability of complication in percent: ' + str(LogComplicationsProbabilityPercent) + ' %'  +
      '\n' + '(LogisticRegression)' +
      '\n______________________________________' +

      '\n\n - Survival ' + str(SurvivalProbabilityAnswer)+
      '\n - Survival probability in percent: ' + str(SurvivalProbabilityPercent) + ' %'
      '\n' +
      '______________________________________' +

      '\n\n - Probability of neurological outcome (significant recovery vs. disability) ' + str(NeurologicalOutcomeProbabilityAnswer)+
      '\n - Probability of neurological outcome (significant recovery vs. disability) in percent ' + str(NeurologicalOutcomeProbabilityPercent) + ' %'
      '\n' +

      '______________________________________' +
      '\n\n - Importance of factors\n' +
      str(feature_importance_dict)
      )

      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add(YesNo_dict_0, YesNo_dict_1)
      msg = bot.reply_to(message, 'Recommend the most effective treatment?', reply_markup=markup)
      bot.register_next_step_handler(msg, Recommendations_step)

  except Exception as e:
    bot.reply_to(message, 'oooops predict_craniotomy_complication_step')



def Recommendations_step(message):
  try:
    chat_id = message.chat.id

    Recomendations = message.text
    if (Recomendations == YesNo_dict_0) or (Recomendations == YesNo_dict_1):
      new_patient = {'Initial_GCS': Initial_GCS, 'Age': Age, 'Hematoma_size_ml': Hematoma_size_ml, 'Midline_shift_mm': Midline_shift_mm, 'Time_to_surgery_h': Time_to_surgery_h, 'ICP_max_to_surgery': ICP_max_to_surgery, 'Concomitant_traumas': Concomitant_traumas}
      best_treatment, effectiveness_results_str_dic = recommend_best_treatment(new_patient)
      CraniotomyRes = effectiveness_results_str_dic['Craniotomy']
      Trepanation_with_drainageRes = effectiveness_results_str_dic['Trepanation_with_drainage']
      Minimally_invasive_interventionsRes = effectiveness_results_str_dic['Minimally_invasive_interventions']
      bot.send_message(chat_id,
      '\n\n - Recommended treatment: \n' + str(best_treatment) + ' (highest efficiency)' +
      '\n\nPredicted effectiveness for each type of treatment: ' +
      '\nCraniotomy:  '  + str(CraniotomyRes) + ' %'+
      '\nTrepanation_with_drainage:  '  + str(Trepanation_with_drainageRes) + ' %'+
      '\nMinimally_invasive_interventions:  '  + str(Minimally_invasive_interventionsRes) + ' %'
      )
      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add('Далі')
      msg = bot.reply_to(message, 'Спробувати знову.', reply_markup=markup)
      bot.register_next_step_handler(msg, send_welcome)
  except Exception as e:
    bot.reply_to(message, 'oooops Recommendations_step')

#The end
bot.infinity_polling()
