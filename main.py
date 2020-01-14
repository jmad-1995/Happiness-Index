"""
Import Neccessary Libraries
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import seaborn as sns
from mpl_toolkits import mplot3d


"""
Scaler Object to scale feature space.
"""
scaler = MinMaxScaler() 

"""
Separating Features and Output for 2015 dataset.
Normalisiation of the feature space is done.
"""
df_2015 = pd.read_csv("2015.csv")
y_2015 = df_2015.iloc[:,3].values
features_2015 = scaler.fit_transform(df_2015.iloc[:,5:])

"""
We use the standard Linear Regression Model for 2015 dataset.
"""
reg = LinearRegression().fit(features_2015, y_2015)

"""
We use a MLP with 1 hidden layer, 40 hidden nodes.
"""
MLPreg = MLPRegressor(hidden_layer_sizes=(40,), random_state=5, max_iter=1000).fit(features_2015, y_2015)

print("~~~~~~~~2015~~~~~~~~")
print(reg.score(features_2015, y_2015))
print(MLPreg.score(features_2015, y_2015))

"""
Prediction is done for Linear Regression and MLP.
"""
y_pred_LR = reg.predict(features_2015)
y_pred_MLPR = MLPreg.predict(features_2015)


print("Mean squared error LR: %.2f"
      % sqrt(mean_squared_error(y_2015, y_pred_LR)))


print("Mean squared error MLP: %.2f"
      % sqrt(mean_squared_error(y_2015, y_pred_MLPR)))

print("Features are : Economy (GDP per Capita), Family, Health (Life Expectancy), Freedom, Trust (Government Corruption), Generosity, Dystopia Residual")
print("LR Weights : ", reg.coef_)
print("MLP Params : ", MLPreg.get_params)

"""
Separating Features and Output for 2016 dataset.
Normalisiation of the feature space is done.
"""
df_2016 = pd.read_csv("2016.csv")
y_2016 = df_2016.iloc[:,3].values
features_2016 = scaler.fit_transform(df_2016.iloc[:,6:])

"""
We use the standard Linear Regression Model for 2016 dataset.
"""
reg = LinearRegression().fit(features_2016, y_2016)

"""
We use a MLP with 1 hidden layer, 50 hidden nodes.
"""
MLPreg = MLPRegressor(hidden_layer_sizes=(50,), random_state=5, max_iter=1000).fit(features_2016, y_2016)
print("~~~~~~~~2016~~~~~~~~")

print(reg.score(features_2016, y_2016))
print(MLPreg.score(features_2016, y_2016))

y_pred_LR = reg.predict(features_2016)
y_pred_MLPR = MLPreg.predict(features_2016)


print("Mean squared error LR: %.2f"
      % sqrt(mean_squared_error(y_2016, y_pred_LR)))


print("Mean squared error MLP: %.2f"
      % sqrt(mean_squared_error(y_2016, y_pred_MLPR)))
print("Features are : Economy (GDP per Capita), Family, Health (Life Expectancy), Freedom, Trust (Government Corruption), Generosity, Dystopia Residual")

print("LR Weights : ", reg.coef_)
print("MLP Params : ", MLPreg.get_params)



df_2017 = pd.read_csv("2017.csv")
y_2017 = df_2017.iloc[:,3].values
features_2017 = scaler.fit_transform(df_2017.iloc[:,5:])

"""
We use the standard Linear Regression Model for 2017 dataset.
"""
reg = LinearRegression().fit(features_2017, y_2017)

"""
We use a MLP with 1 hidden layer, 50 hidden nodes.
"""
MLPreg = MLPRegressor(hidden_layer_sizes=(50,), random_state=0, max_iter=1000).fit(features_2017, y_2017)
print("~~~~~~~~2017~~~~~~~~")

print(reg.score(features_2017, y_2017))
print(MLPreg.score(features_2017, y_2017))

y_pred_LR = reg.predict(features_2017)
y_pred_MLPR = MLPreg.predict(features_2017)


print("Mean squared error LR: %.2f"
      % sqrt(mean_squared_error(y_2017, y_pred_LR)))


print("Mean squared error MLP: %.2f"
      % sqrt(mean_squared_error(y_2017, y_pred_MLPR)))

print("Features are : Economy (GDP per Capita), Family, Health (Life Expectancy), Freedom, Trust (Government Corruption), Generosity, Dystopia Residual")

print("LR Weights : ", reg.coef_)
print("MLP Params : ", MLPreg.get_params)


"""
We combine dataset of all 3 years and perform similar analysis.
"""

features_combined = np.concatenate((features_2015,features_2016,features_2017), axis=0)
y_combined = np.concatenate((y_2015, y_2016, y_2017), axis=0)

"""
We use the standard Linear Regression Model for combined dataset.
"""
reg = LinearRegression().fit(features_combined, y_combined)

"""
We use a MLP with 1 hidden layer, 100 hidden nodes.
"""
MLPreg = MLPRegressor(hidden_layer_sizes=(100,), random_state=5, max_iter=1000).fit(features_combined, y_combined)
print("~~~~~~~~Combined~~~~~~~~")

print(reg.score(features_combined, y_combined))
print(MLPreg.score(features_combined, y_combined))

y_pred_LR = reg.predict(features_combined)
y_pred_MLPR = MLPreg.predict(features_combined)


print("Mean squared error LR: %.2f"
      % sqrt(mean_squared_error(y_combined, y_pred_LR)))


print("Mean squared error MLP: %.2f"
      % sqrt(mean_squared_error(y_combined, y_pred_MLPR)))

print("Features are : Economy (GDP per Capita), Family, Health (Life Expectancy), Freedom, Trust (Government Corruption), Generosity, Dystopia Residual")

print("LR Weights : ", reg.coef_)
print("MLP Params : ", MLPreg.get_params)



"""
We drop non-common feature columns for 2015 and 2016 dataset.
"""

df_2015 = df_2015.drop(columns=['Standard Error','Region','Happiness Rank'])
df_2016 = df_2016.drop(columns=['Happiness Rank','Lower Confidence Interval','Upper Confidence Interval','Region'])

"""
We drop non-common feature columns.
2017 dataset contains Column Names which are named different.
We rename those columns as per 2015 and 2016 dataset.
"""
df_2017 = df_2017.drop(columns=['Happiness.Rank','Whisker.high','Whisker.low'])
df_2017.rename(columns={'Economy..GDP.per.Capita.':'Economy (GDP per Capita)', 
                          'Health..Life.Expectancy.':'Health (Life Expectancy)',
                          'Trust..Government.Corruption.':'Trust (Government Corruption)',
                          'Dystopia.Residual':'Dystopia Residual',
                          'Happiness.Score':'Happiness Score',
                          'Happiness.Rank':'Happiness Rank'}, inplace=True)

"""
For Plotting Correlation Matrix
"""
sns.set(style="whitegrid", palette="muted")
current_palette = sns.color_palette()

df_combined = pd.concat([df_2015, df_2016, df_2017], sort=True)

corrmat = df_combined.corr()
sns.heatmap(corrmat, vmax=.9, square=True)


"""
Plotting 2 features v/s Happiness score.
"""
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(221, projection='3d')
f1 = df_2015['Economy (GDP per Capita)'].values
f2 = df_2015['Family'].values
y = df_2015['Happiness Score'].values
ax.scatter(f1, f2, y)
ax.set_xlabel('Economy (GDP per Capita)')
ax.set_ylabel('Family')
ax.set_zlabel('Happiness Score')

ax = fig.add_subplot(222, projection='3d')
f1 = df_2015['Trust (Government Corruption)'].values
f2 = df_2015['Freedom'].values
y = df_2015['Happiness Score'].values
ax.scatter(f1, f2, y)
ax.set_xlabel('Trust (Government Corruption)')
ax.set_ylabel('Freedom')
ax.set_zlabel('Happiness Score')


ax = fig.add_subplot(223, projection='3d')
f1 = df_2015['Health (Life Expectancy)'].values
f2 = df_2015['Family'].values
y = df_2015['Happiness Score'].values
ax.scatter(f1, f2, y)
ax.set_xlabel('Health (Life Expectancy)')
ax.set_ylabel('Family')
ax.set_zlabel('Happiness Score')


ax = fig.add_subplot(224, projection='3d')
f1 = df_2015['Generosity'].values
f2 = df_2015['Dystopia Residual'].values
y = df_2015['Happiness Score'].values
ax.scatter(f1, f2, y)
ax.set_xlabel('Generosity')
ax.set_ylabel('Dystopia Residual')
ax.set_zlabel('Happiness Score')
