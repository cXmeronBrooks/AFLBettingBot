# AFLBettingBot
A starter project which I made to get familiar with machine learning, this is my first ever ML project
A basic logistic model which takes in hometeam and awayteam and returns a 1 if the model thinks the home team will win, or a 0 if the model thinks the away team will win.
each team has a code 1-18 which is in the "teams" dict.
Only has data since 2020

*When entering your home and away team, the team name has to match the "teams" dictionary*

teams ={"Adelaide": 1, "Port Adelaide": 2, "Carlton": 3, "Essendon": 4, "Richmond": 5, "Collingwood": 6,
        "Fremantle": 7, "West Coast": 8, "Greater Western Sydney": 9, "Brisbane Lions": 10, "Sydney": 11,
        "St Kilda": 12, "Melbourne": 13, "Western Bulldogs": 14, "Geelong": 15, "Gold Coast": 16,
        "Hawthorn": 17, "North Melbourne": 18}

VERSION 2.0 Update

- Complete overhall
- New model type RandomForestRegressor, predicts winning margin instead of just a win or loss
- negative values mean the away team is predicted to win, positive values mean the home team is predicted to win
- Now with a HTML interface to designed to be more user friendly
- Is now able to be hosted by a server using the Python Flask backend
