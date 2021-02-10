import pandas as pd
import glob
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


path = r'H:\Documents\HoustonData\On-road Driving Study\Quantitative Data'
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

onRoad_data = pd.concat(li, axis=0, ignore_index=True)

print(onRoad_data.columns)
onRoad_data['pp'].fillna(onRoad_data['pp'].mean)


#Target object of ML is Accelerator and Breaking Pressure
y = onRoad_data.Accelerator
y1 = onRoad_data.Brake

#Features to train on
features = ['HR', 'BR', 'SkinTemp', 'Posture', 'Steering']
X = onRoad_data[features]

#Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
train_X1, val_X1, train_y1, val_y1 = train_test_split(X, y1, random_state=1)

#Specify Model
accelerator_model = RandomForestRegressor(random_state=1)
brake_model = RandomForestRegressor(random_state=1)

#Fit Model
accelerator_model.fit(train_X, train_y)
brake_model.fit(train_X1, train_y1)

#Make validation prediction and calculate mean absolute error
val_predictions = accelerator_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for Accelerator Random Forest Model: {}".format(val_mae))

val_predictionsB = accelerator_model.predict(val_X1)
val_maeB = mean_absolute_error(val_predictionsB, val_y1)
print("Validation MAE for Braking Random Forest Model: {}".format(val_maeB))

#Display Accelerator from Tung ML Data
acceldataT001 = pd.read_csv('H:\Documents\HoustonData\Test Track Study 1\Quantitative Data\T001.csv')

firstLapAcceldataT001 = acceldataT001.iloc[1:3500]
fig = px.line(firstLapAcceldataT001, x = 'Time', y = 'Acceleration', title = 'Acceleration Curve with Tung ML')
fig.show()



