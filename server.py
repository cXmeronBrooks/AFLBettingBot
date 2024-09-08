from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

teamcodes ={"Adelaide": 1, "Port Adelaide": 2, "Carlton": 3, "Essendon": 4, "Richmond": 5, "Collingwood": 6,
        "Fremantle": 7, "West Coast": 8, "Greater Western Sydney": 9, "Brisbane": 10, "Sydney": 11,
        "St Kilda": 12, "Melbourne": 13, "Western Bulldogs": 14, "Geelong": 15, "Gold Coast": 16,
        "Hawthorn": 17, "North Melbourne": 18}

app = Flask(__name__)

df = pd.read_csv('afldata.csv')
df = df.drop(columns=['homescore', 'awayscore'])
y = df['winningmargin']
X = df.drop(columns=['winningmargin'])
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

@app.route("/")
def home():
    return render_template("Betting app.html")

@app.route("/predict", methods=['POST'])
def predict():
    data = request.json
    hometeam = teamcodes[data['hometeam']]
    awayteam = teamcodes[data['awayteam']]
    attendance = data['attendance']
    if attendance == "Small":
        attendance = np.random.randint(20000)
    elif attendance == "Medium":
        attendance = np.random.randint(20001, 60000)
    else:
        attendance = np.random.randint(60001, 100000)
    input_data = pd.DataFrame([[hometeam, awayteam, attendance]], columns=['hometeam', 'awayteam', 'attendance'])
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, port=8000, host='0.0.0.0')
