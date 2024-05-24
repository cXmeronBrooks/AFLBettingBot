import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

teams ={"Adelaide": 1, "Port Adelaide": 2, "Carlton": 3, "Essendon": 4, "Richmond": 5, "Collingwood": 6,
        "Fremantle": 7, "West Coast": 8, "Greater Western Sydney": 9, "Brisbane Lions": 10, "Sydney": 11,
        "St Kilda": 12, "Melbourne": 13, "Western Bulldogs": 14, "Geelong": 15, "Gold Coast": 16,
        "Hawthorn": 17, "North Melbourne": 18}

everymatch_2020 = ['Richmond', 'Carlton', 'Western Bulldogs', 'Collingwood', 'Essendon', 'Fremantle', 'Adelaide', 'Sydney', 'Gold Coast',
                   'Port Adelaide', 'Greater Western Sydney', 'Geelong', 'North Melbourne', 'St Kilda', 'Hawthorn', 'Brisbane Lions', 'West Coast',
                   'Melbourne',
                   'Collingwood', 'Richmond', 'Geelong', 'Hawthorn', 'Brisbane Lions', 'Fremantle', 'Carlton', 'Melbourne', 'Port Adelaide',
                   'Adelaide', 'Gold Coast', 'West Coast', 'Greater Western Sydney', 'North Melbourne', 'Sydney', 'Essendon', 'St Kilda', 'Western Bulldogs',

                   'Richmond', 'Hawthorn', 'Western Bulldogs', 'Greater Western Sydney', 'North Melbourne', 'Sydney', 'Collingwood', 'St Kilda', 'Geelong',
                   'Carlton', 'Brisbane Lions', 'West Coast', 'Gold Coast', 'Adelaide', 'Fremantle', 'Port Adelaide',

                   'Sydney',
                   'Western Bulldogs', 'Greater Western Sydney', 'Collingwood', 'Port Adelaide', 'West Coast', 'St Kilda', 'Richmond', 'Essendon', 'Carlton',
                   'Gold Coast', 'Fremantle', 'Brisbane Lions', 'Adelaide', 'Melbourne', 'Geelong', 'Hawthorn', 'North Melbourne',

                   'Carlton', 'St Kilda',
                   'Collingwood', 'Essendon', 'West Coast', 'Sydney', 'Geelong', 'Gold Coast', 'Western Bulldogs', 'North Melbourne', 'Brisbane Lions',
                   'Port Adelaide', 'Adelaide', 'Fremantle', 'Melbourne', 'Richmond', 'Greater Western Sydney', 'Hawthorn',

                   'Geelong', 'Brisbane Lions',
                   'Collingwood', 'Hawthorn', 'Fremantle', 'St Kilda', 'West Coast', 'Adelaide', 'Melbourne', 'Gold Coast', 'Essendon', 'North Melbourne',
                   'Port Adelaide', 'Greater Western Sydney', 'Richmond', 'Sydney', 'Carlton', 'Western Bulldogs',

                   'Geelong', 'Collingwood', 'Essendon',
                   'Western Bulldogs', 'Greater Western Sydney', 'Brisbane Lions', 'Sydney', 'Gold Coast', 'Richmond', 'North Melbourne', 'Carlton', 'Port Adelaide',
                   'Hawthorn', 'Melbourne', 'Fremantle', 'West Coast', 'Adelaide', 'St Kilda',

                   'Gold Coast', 'Western Bulldogs', 'Greater Western Sydney',
                   'Richmond', 'North Melbourne', 'Carlton', 'Sydney', 'Hawthorn', 'Port Adelaide', 'St Kilda', 'Adelaide', 'Essendon', 'West Coast',
                   'Collingwood', 'Melbourne', 'Brisbane Lions', 'Fremantle', 'Geelong',

                   'Western Bulldogs', 'Richmond', 'Melbourne', 'Port Adelaide',
                   'Carlton', 'Hawthorn', 'Essendon', 'Brisbane Lions', 'North Melbourne', 'Adelaide', 'St Kilda', 'Sydney', 'West Coast', 'Geelong',
                   'Gold Coast', 'Greater Western Sydney', 'Fremantle', 'Collingwood',

                   'Port Adelaide', 'Western Bulldogs', 'Richmond', 'Brisbane Lions',
                   'Geelong', 'North Melbourne', 'Adelaide', 'Melbourne', 'Collingwood', 'Sydney', 'Gold Coast', 'St Kilda', 'Essendon', 'Greater Western Sydney',


                   'Port Adelaide', 'Richmond', 'Brisbane Lions', 'Western Bulldogs', 'West Coast', 'Carlton',
                   'Melbourne', 'North Melbourne', 'St Kilda', 'Geelong', 'Fremantle', 'Hawthorn', 'Adelaide', 'Collingwood', 'Gold Coast', 'Essendon',

                   'Sydney', 'Greater Western Sydney', 'Geelong', 'Port Adelaide', 'North Melbourne', 'Brisbane Lions', 'Melbourne',
                   'Collingwood', 'Fremantle', 'Carlton', 'Western Bulldogs', 'Adelaide', 'St Kilda', 'Essendon', 'West Coast', 'Hawthorn', 'Richmond', 'Gold Coast',

                   'Gold Coast', 'Carlton', 'Western Bulldogs', 'Melbourne', 'Port Adelaide', 'Hawthorn', 'Essendon', 'Richmond', 'Fremantle', 'Sydney', 'Adelaide',
                   'Geelong', 'Brisbane Lions', 'St Kilda', 'West Coast', 'Greater Western Sydney', 'Collingwood', 'North Melbourne',

                   'Hawthorn', 'Essendon', 'Richmond',
                   'West Coast', 'Western Bulldogs', 'Geelong', 'Port Adelaide', 'Sydney', 'Fremantle', 'Greater Western Sydney', 'Melbourne', 'St Kilda', 'Carlton',
                   'Collingwood', 'Gold Coast', 'North Melbourne',

                   'Hawthorn', 'Adelaide', 'West Coast', 'Essendon', 'Richmond', 'Fremantle',
                   'Sydney', 'Melbourne', 'Greater Western Sydney', 'Carlton', 'Brisbane Lions', 'Collingwood',


                   'North Melbourne', 'Port Adelaide', 'St Kilda', 'Hawthorn', 'Geelong', 'Essendon', 'Western Bulldogs',
                   'West Coast', 'Melbourne', 'Fremantle', 'Adelaide', 'Greater Western Sydney', 'Carlton', 'Sydney', 'Brisbane Lions', 'Gold Coast',

                    'St Kilda', 'West Coast', 'Geelong', 'Richmond', 'North Melbourne', 'Fremantle', 'Port Adelaide', 'Essendon', 'Greater Western Sydney',
                   'Melbourne', 'Carlton', 'Adelaide', 'Hawthorn', 'Western Bulldogs', 'Sydney', 'Brisbane Lions', 'Collingwood', 'Gold Coast',

                   'North Melbourne',
                   'West Coast', 'St Kilda', 'Greater Western Sydney', 'Essendon', 'Melbourne', 'Adelaide', 'Richmond', 'Brisbane Lions', 'Carlton', 'Hawthorn',
                   'Gold Coast', 'Sydney', 'Geelong', 'Fremantle', 'Western Bulldogs', 'Collingwood', 'Port Adelaide',

                   'Port Adelaide', 'Geelong', 'Brisbane Lions', 'Richmond', 'St Kilda', 'Western Bulldogs',
                   'West Coast', 'Collingwood', 'Richmond', 'St Kilda', 'Geelong', 'Collingwood', 'Port Adelaide', 'Richmond', 'Brisbane Lions', 'Geelong', 'Richmond', 'Geelong']

home2020 = []
away2020 = []

for team in range(0, len(everymatch_2020), 2):
  home2020.append(everymatch_2020[team])

for tea in range(1, len(everymatch_2020), 2):
  away2020.append(everymatch_2020[tea])

print(home2020)

winners2020 = ['Richmond', 'Collingwood', 'Essendon', 'Sydney', 'Port Adelaide', 'Greater Western Sydney', 'North Melbourne', 'Hawthorn', 'West Coast',

'Collingwood', 'Geelong', 'Brisbane Lions', 'Melbourne', 'Port Adelaide', 'Gold Coast', 'North Melbourne', 'Essendon', 'St Kilda',

'Hawthorn', 'Western Bulldogs', 'Sydney', 'Collingwood', 'Carlton', 'Brisbane Lions', 'Gold Coast', 'Port Adelaide',

 'Western Bulldogs', 'Greater Western Sydney', 'Port Adelaide', 'St Kilda', 'Carlton', 'Gold Coast', 'Brisbane Lions', 'Geelong', 'Hawthorn',

 'St Kilda', 'Essendon', 'West Coast', 'Geelong', 'Western Bulldogs', 'Brisbane Lions', 'Fremantle', 'Richmond', 'Greater Western Sydney',

 'Geelong', 'Collingwood', 'Fremantle', 'West Coast', 'Melbourne', 'Essendon', 'Port Adelaide', 'Richmond', 'Carlton',

 'Collingwood', 'Western Bulldogs', 'Brisbane Lions', 'Gold Coast', 'Richmond', 'Port Adelaide', 'Melbourne', 'West Coast', 'St Kilda',

 'Western Bulldogs', 'Greater Western Sydney', 'Carlton', 'Sydney', 'St Kilda', 'Essendon', 'West Coast', 'Brisbane Lions', 'Geelong',

 'Richmond', 'Port Adelaide', 'Hawthorn', 'Brisbane Lions', 'North Melbourne', 'St Kilda', 'West Coast', 'Greater Western Sydney', 'Fremantle',

 'Port Adelaide', 'Richmond', 'Geelong', 'Melbourne', 'Collingwood', 'St Kilda', 'Greater Western Sydney',

 'Port Adelaide', 'Brisbane Lions', 'West Coast', 'Melbourne', 'Geelong', 'Fremantle', 'Collingwood', 'Gold Coast',

 'Sydney', 'Geelong', 'Brisbane Lions', 'Melbourne', 'Carlton', 'Western Bulldogs', 'St Kilda', 'West Coast', 'Richmond',

 'Carlton', 'Western Bulldogs', 'Port Adelaide', 'Richmond', 'Fremantle', 'Geelong', 'Brisbane Lions', 'West Coast', 'Collingwood',

 'Essendon', 'Richmond', 'Geelong', 'Port Adelaide', 'Greater Western Sydney', 'Melbourne', 'Collingwood', 'Gold Coast',

 'Adelaide', 'West Coast', 'Richmond', 'Sydney', 'Greater Western Sydney', 'Brisbane Lions',

 'Port Adelaide', 'St Kilda', 'Geelong', 'Western Bulldogs', 'Fremantle', 'Adelaide', 'Carlton', 'Brisbane Lions',

 'West Coast', 'Richmond', 'Fremantle', 'Port Adelaide', 'Melbourne', 'Adelaide', 'Western Bulldogs', 'Brisbane Lions', 'Collingwood',

 'West Coast', 'St Kilda', 'Melbourne', 'Richmond', 'Brisbane Lions', 'Hawthorn', 'Geelong', 'Western Bulldogs', 'Port Adelaide',

 'Port Adelaide', 'Brisbane Lions', 'St Kilda', 'Collingwood', 'Richmond', 'Geelong', 'Richmond', 'Geelong', 'Richmond']

winnerInt2020 = []

for team in range(0, len(winners2020)):
  if winners2020[team] == home2020[team]:
    winnerInt2020.append(1)
  else:
    winnerInt2020.append(0)

print(winnerInt2020)
print(len(winnerInt2020))



winners2021 = ['Richmond', 'Western Bulldogs', 'Melbourne', 'Adelaide', 'Hawthorn', 'Sydney', 'Port Adelaide', 'St Kilda', 'West Coast',

 'Collingwood', 'Geelong', 'Sydney', 'Port Adelaide', 'Melbourne', 'Gold Coast', 'Richmond', 'Western Bulldogs', 'Fremantle',

 'Brisbane Lions', 'Western Bulldogs', 'Adelaide', 'Sydney', 'Essendon', 'West Coast', 'Carlton', 'Melbourne', 'Geelong',

 'Sydney', 'Port Adelaide', 'Western Bulldogs', 'St Kilda', 'Carlton', 'Greater Western Sydney', 'Adelaide', 'Melbourne', 'Fremantle',

 'Richmond', 'West Coast', 'Western Bulldogs', 'Greater Western Sydney', 'Port Adelaide', 'Brisbane Lions', 'Fremantle', 'Melbourne', 'Geelong',

 'Western Bulldogs', 'Geelong', 'Gold Coast', 'Brisbane Lions', 'Melbourne', 'Fremantle', 'Hawthorn', 'Essendon', 'Port Adelaide',

 'Richmond', 'Gold Coast', 'Greater Western Sydney', 'St Kilda', 'Sydney', 'Brisbane Lions', 'Melbourne', 'Carlton', 'West Coast',

 'Geelong', 'St Kilda', 'Greater Western Sydney', 'Collingwood', 'Melbourne', 'Port Adelaide', 'West Coast', 'Western Bulldogs', 'Brisbane Lions',

 'Geelong', 'Sydney', 'North Melbourne', 'Brisbane Lions', 'Richmond', 'Western Bulldogs', 'Essendon', 'Melbourne', 'West Coast',

 'Brisbane Lions', 'Carlton', 'Geelong', 'Adelaide', 'Western Bulldogs', 'Fremantle', 'Greater Western Sydney', 'Port Adelaide', 'Essendon',

 'Melbourne', 'Geelong', 'Brisbane Lions', 'St Kilda', 'Essendon', 'Gold Coast', 'Richmond', 'Sydney', 'Port Adelaide',

 'Melbourne', 'Sydney', 'Collingwood', 'Richmond', 'West Coast', 'Western Bulldogs',


 'Geelong', 'Hawthorn', 'Fremantle', 'Adelaide', 'North Melbourne', 'West Coast', 'Collingwood',

'Geelong', 'Port Adelaide', 'Brisbane Lions', 'Greater Western Sydney', 'Essendon',

 'Brisbane Lions', 'St Kilda', 'Fremantle', 'North Melbourne', 'Port Adelaide', 'Melbourne', 'Hawthorn', 'Western Bulldogs', 'Carlton',

 'Gold Coast', 'Geelong', 'Greater Western Sydney', 'Brisbane Lions', 'Carlton', 'Port Adelaide', 'Sydney', 'St Kilda', 'Western Bulldogs',

 'Melbourne', 'Essendon', 'Fremantle', 'Geelong', 'St Kilda', 'Gold Coast', 'Sydney', 'Collingwood', 'North Melbourne',

 'Geelong', 'Richmond', 'Port Adelaide', 'Western Bulldogs', 'Essendon', 'Carlton', 'West Coast', 'Sydney',

 'Port Adelaide', 'North Melbourne', 'Brisbane Lions', 'West Coast', 'Western Bulldogs', 'Adelaide', 'Sydney', 'Geelong', 'Greater Western Sydney',

 'Carlton', 'Western Bulldogs', 'Geelong', 'Collingwood', 'Melbourne', 'Hawthorn', 'Sydney', 'Fremantle', 'Port Adelaide',

 'Greater Western Sydney', 'Gold Coast', 'Richmond', 'Port Adelaide', 'St Kilda', 'Hawthorn', 'Essendon', 'Brisbane Lions', 'Melbourne',

 'Greater Western Sydney', 'Hawthorn', 'Port Adelaide', 'Geelong', 'Brisbane Lions', 'Sydney', 'Melbourne', 'Essendon', 'Fremantle',

 'Port Adelaide', 'Sydney', 'Brisbane Lions', 'Melbourne', 'Greater Western Sydney', 'St Kilda', 'Essendon', 'Adelaide',

 'Port Adelaide', 'Greater Western Sydney', 'Melbourne', 'Western Bulldogs', 'Geelong', 'Western Bulldogs', 'Melbourne', 'Western Bulldogs', 'Melbourne']


print(len(winners2021))

#every away team in the 2022 season

away_teams_22 = ['Western Bulldogs', 'Richmond', 'Collingwood', 'Essendon', 'Sydney',
                 'Port Adelaide', 'North Melbourne', 'Fremantle', 'Gold Coast',

                 'Carlton',
                 'Geelong', 'Adelaide', 'Brisbane Lions', 'Hawthorn', 'Melbourne', 'West Coast',
                 'Greater Western Sydney', 'St Kilda',

                 'Sydney', 'Essendon', 'Port Adelaide',
                 'Gold Coast', 'Geelong', 'North Melbourne', 'Hawthorn', 'Richmond', 'Fremantle',

                 'Melbourne', 'Brisbane Lions', 'North Melbourne', 'West Coast', 'Greater Western Sydney',
                 'Western Bulldogs', 'Adelaide', 'St Kilda', 'Carlton',

                 'Collingwood', 'Western Bulldogs',
                 'Sydney', 'Gold Coast', 'Richmond', 'Greater Western Sydney', 'Port Adelaide', 'Fremantle',
                 'Geelong',

                 'St Kilda', 'Adelaide', 'West Coast', 'Carlton', 'Geelong', 'Brisbane Lions',
                 'Melbourne', 'Sydney', 'Collingwood',

                 'Richmond', 'Fremantle', 'Greater Western Sydney',
                 'Hawthorn', 'North Melbourne', 'Port Adelaide', 'Gold Coast', 'Essendon', 'Brisbane Lions',

                 'Western Bulldogs', 'North Melbourne', 'Collingwood', 'Gold Coast', 'Geelong', 'Hawthorn',
                 'West Coast', 'St Kilda', 'Adelaide',

                 'Western Bulldogs', 'Richmond', 'Port Adelaide',
                 'Geelong', 'Essendon', 'Brisbane Lions', 'Fremantle', 'Carlton', 'Melbourne',

                 'Sydney',
                 'Gold Coast', 'Port Adelaide', 'Melbourne', 'St Kilda', 'Essendon', 'West Coast',
                 'Brisbane Lions', 'Collingwood',

                 'Richmond', 'Adelaide', 'Greater Western Sydney',
                 'Fremantle', 'Western Bulldogs', 'Hawthorn', 'North Melbourne', 'Carlton', 'Essendon',

                 'Geelong', 'West Coast', 'North Melbourne', 'Sydney', 'Collingwood', 'Brisbane Lions',

                  'Port Adelaide', 'Carlton', 'Hawthorn',
                 'St Kilda', 'Greater Western Sydney', 'Melbourne',

                 'Carlton', 'Essendon', 'Sydney', 'Geelong', 'Western Bulldogs', 'Adelaide',

                 'Brisbane Lions', 'Hawthorn', 'Essendon', 'Fremantle', 'Richmond',
                 'St Kilda', 'Adelaide', 'Greater Western Sydney', 'Gold Coast',

                 'Western Bulldogs', 'St Kilda',
                 'Sydney', 'Melbourne', 'Collingwood', 'North Melbourne', 'West Coast', 'Hawthorn', 'Port Adelaide',


                 'Melbourne', 'Western Bulldogs', 'North Melbourne', 'Richmond', 'Fremantle', 'Greater Western Sydney',
                 'Essendon', 'Adelaide', 'Carlton',

                 'St Kilda', 'Collingwood', 'Brisbane Lions', 'Richmond', 'Geelong',
                 'Sydney', 'West Coast', 'Port Adelaide', 'Gold Coast',

                 'Fremantle', 'Adelaide', 'Hawthorn', 'Geelong',
                 'Gold Coast', 'Melbourne', 'Greater Western Sydney', 'Essendon', 'St Kilda',

                 'Melbourne', 'Port Adelaide',
                 'Greater Western Sydney', 'Hawthorn', 'Western Bulldogs', 'Carlton', 'West Coast', 'Brisbane Lions',
                 'North Melbourne',

                 'Collingwood', 'Gold Coast', 'Essendon', 'Fremantle', 'St Kilda', 'Richmond',
                 'Sydney', 'Carlton', 'Adelaide',

                 'Brisbane Lions', 'Greater Western Sydney', 'North Melbourne',
                 'Geelong', 'Carlton', 'West Coast', 'Hawthorn', 'Collingwood', 'Port Adelaide',

                 'Melbourne',
                 'Fremantle', 'Gold Coast', 'West Coast', 'Richmond', 'Adelaide', 'Western Bulldogs', 'Collingwood',
                 'Sydney',

                 'Richmond', 'Sydney', 'Collingwood', 'Western Bulldogs', 'Brisbane Lions', 'Fremantle',
                 'Brisbane Lions', 'Collingwood', 'Sydney']

away_teams22_int = []

for x in away_teams_22:
  away_teams22_int.append(teams[x])

print(len(away_teams22_int))

#every home team in the 2022 season

home_teams_22 = ['Melbourne', 'Carlton', 'St Kilda', 'Geelong', 'Greater Western Sydney',
                 'Brisbane Lions', 'Hawthorn', 'Adelaide', 'West Coast',

                 'Western Bulldogs',
                 'Sydney', 'Collingwood', 'Essendon', 'Port Adelaide', 'Gold Coast', 'North Melbourne',
                 'Richmond', 'Fremantle',

                 'Western Bulldogs', 'Melbourne', 'Adelaide', 'Greater Western Sydney',
                 'Collingwood', 'Brisbane Lions', 'Carlton', 'St Kilda', 'West Coast',

                 'Port Adelaide',
                 'Geelong', 'Sydney', 'Collingwood', 'Fremantle', 'Richmond', 'Essendon', 'Hawthorn',
                 'Gold Coast',

                 'Brisbane Lions', 'North Melbourne', 'West Coast', 'St Kilda', 'Adelaide',
                 'Melbourne', 'Carlton', 'Essendon', 'Hawthorn',

                 'Greater Western Sydney',
                 'Western Bulldogs', 'Port Adelaide', 'Fremantle', 'North Melbourne', 'Gold Coast',
                 'Richmond', 'Hawthorn', 'Essendon',

                 'West Coast', 'Geelong', 'Adelaide', 'Melbourne',
                 'Carlton', 'St Kilda', 'Collingwood', 'Western Bulldogs', 'Sydney',

                 'Port Adelaide',
                 'Fremantle', 'Richmond', 'Sydney', 'Greater Western Sydney', 'Essendon', 'Brisbane Lions',
                 'Melbourne', 'Carlton',

                 'Collingwood', 'Hawthorn', 'North Melbourne', 'St Kilda', 'Sydney',
                 'Adelaide', 'Gold Coast', 'Greater Western Sydney', 'West Coast',

                 'Carlton', 'Western Bulldogs',
                 'Geelong', 'North Melbourne', 'Adelaide', 'Richmond', 'Greater Western Sydney', 'Hawthorn',
                 'Fremantle',

                 'Sydney', 'Geelong', 'Brisbane Lions', 'Melbourne', 'West Coast', 'Gold Coast',
                 'St Kilda', 'Collingwood', 'Port Adelaide',

                 'Western Bulldogs', 'Adelaide', 'Gold Coast',
                 'Melbourne', 'Hawthorn', 'Fremantle',

                 'Richmond',
                 'Essendon', 'Fremantle', 'Brisbane Lions', 'North Melbourne', 'Collingwood',

                 'Richmond', 'St Kilda', 'Port Adelaide', 'West Coast', 'Greater Western Sydney',
                 'Gold Coast',

                 'Melbourne', 'Western Bulldogs',
                 'West Coast', 'Carlton', 'Geelong', 'Sydney', 'North Melbourne', 'Collingwood', 'Port Adelaide',
                 'Brisbane Lions', 'Carlton', 'Essendon', 'Adelaide', 'Gold Coast', 'Geelong', 'Richmond',
                 'Greater Western Sydney', 'Fremantle', 'Geelong', 'Sydney', 'Collingwood', 'Gold Coast',
                 'St Kilda', 'Port Adelaide', 'Brisbane Lions', 'Hawthorn', 'West Coast', 'Western Bulldogs',
                 'Adelaide', 'Greater Western Sydney', 'North Melbourne', 'Carlton', 'Fremantle', 'Hawthorn',
                 'Melbourne', 'Essendon', 'Richmond', 'Sydney', 'North Melbourne', 'Port Adelaide', 'Brisbane Lions',
                 'Western Bulldogs', 'Carlton', 'Collingwood', 'West Coast', 'Fremantle', 'Collingwood',
                 'Sydney', 'St Kilda', 'Geelong', 'Adelaide', 'Gold Coast', 'Richmond', 'Essendon', 'Melbourne',
                 'Hawthorn', 'Greater Western Sydney', 'Western Bulldogs', 'Geelong', 'Port Adelaide',
                 'North Melbourne', 'Brisbane Lions', 'West Coast', 'St Kilda', 'Western Bulldogs', 'Adelaide',
                 'Gold Coast', 'Melbourne', 'Fremantle', 'Richmond', 'Sydney', 'Essendon', 'Brisbane Lions',
                 'Greater Western Sydney', 'North Melbourne', 'Geelong', 'Essendon', 'Port Adelaide', 'Hawthorn',
                 'Carlton', 'St Kilda', 'Brisbane Lions', 'Melbourne', 'Geelong', 'Fremantle', 'Melbourne',
                 'Collingwood', 'Geelong', 'Sydney', 'Geelong']

home_teams22_int = []

for y in home_teams_22:
  home_teams22_int.append(teams[y])

print(len(home_teams22_int))

# The winners of all (207) 2022 afl games

winners22 = ['Melbourne', 'Carlton', 'Collingwood', 'Geelong', 'Sydney', 'Brisbane Lions', 'Hawthorn',
             'Fremantle', 'Gold Coast',
             'Carlton', 'Sydney', 'Collingwood', 'Brisbane Lions', 'Hawthorn',
             'Melbourne', 'North Melbourne', 'Richmond', 'St Kilda',

             'Western Bulldogs', 'Melbourne',
             'Adelaide', 'Greater Western Sydney', 'Geelong', 'Brisbane Lions', 'Carlton', 'St Kilda',
             'Fremantle',

             'Melbourne', 'Geelong', 'Sydney', 'West Coast', 'Fremantle', 'Richmond', 'Essendon',
             'St Kilda', 'Gold Coast',

             'Brisbane Lions', 'Western Bulldogs', 'Sydney', 'St Kilda', 'Adelaide',
             'Melbourne', 'Carlton', 'Fremantle', 'Hawthorn',

             'St Kilda', 'Adelaide', 'Port Adelaide', 'Fremantle',
             'Geelong', 'Brisbane Lions', 'Melbourne', 'Sydney', 'Collingwood',

             'Richmond', 'Fremantle',
             'Greater Western Sydney', 'Melbourne', 'Carlton', 'Port Adelaide', 'Collingwood', 'Western Bulldogs',
             'Brisbane Lions',

             'Port Adelaide', 'Fremantle', 'Richmond', 'Gold Coast', 'Geelong', 'Essendon',
             'Brisbane Lions', 'Melbourne', 'Carlton',

             'Western Bulldogs', 'Richmond', 'Port Adelaide',
             'St Kilda', 'Sydney', 'Brisbane Lions', 'Gold Coast', 'Carlton', 'Melbourne',

             'Carlton',
             'Western Bulldogs', 'Geelong', 'Melbourne', 'St Kilda', 'Richmond', 'Greater Western Sydney',
             'Hawthorn', 'Collingwood',

             'Sydney', 'Geelong', 'Brisbane Lions', 'Fremantle', 'Western Bulldogs',
             'Gold Coast', 'St Kilda', 'Collingwood', 'Port Adelaide',

             'Geelong', 'Adelaide', 'Gold Coast',
             'Sydney', 'Collingwood', 'Fremantle',

             'Richmond', 'Carlton', 'Fremantle', 'Brisbane Lions',
             'Greater Western Sydney', 'Collingwood',

             'Richmond', 'Essendon', 'Port Adelaide', 'Geelong',
             'Western Bulldogs', 'Gold Coast',

             'Melbourne', 'Western Bulldogs', 'West Coast', 'Carlton',
             'Geelong', 'Sydney', 'Adelaide', 'Collingwood', 'Port Adelaide',

             'Brisbane Lions', 'St Kilda',
             'Essendon', 'Melbourne', 'Collingwood', 'Geelong', 'Richmond', 'Greater Western Sydney',
             'Fremantle',

             'Geelong', 'Sydney', 'Collingwood', 'Gold Coast', 'Fremantle', 'Port Adelaide',
             'Essendon', 'Hawthorn', 'Carlton',

             'Western Bulldogs', 'Collingwood', 'Brisbane Lions',
             'North Melbourne', 'Geelong', 'Sydney', 'Hawthorn', 'Melbourne', 'Essendon',

             'Richmond',
             'Sydney', 'Hawthorn', 'Geelong', 'Brisbane Lions', 'Western Bulldogs', 'Carlton', 'Collingwood',
             'St Kilda',

             'Melbourne', 'Collingwood', 'Sydney', 'St Kilda', 'Geelong', 'Adelaide', 'Gold Coast',
             'Richmond', 'Essendon',

             'Collingwood', 'Hawthorn', 'Greater Western Sydney', 'Fremantle', 'Geelong',
             'Richmond', 'Sydney', 'Brisbane Lions', 'Adelaide',

             'Brisbane Lions', 'Western Bulldogs', 'Adelaide',
             'Geelong', 'Melbourne', 'Fremantle', 'Richmond', 'Sydney', 'Port Adelaide',

             'Melbourne', 'Fremantle',
             'Gold Coast', 'Geelong', 'Richmond', 'Port Adelaide', 'Western Bulldogs', 'Collingwood', 'Sydney',

             'Brisbane Lions', 'Sydney', 'Geelong', 'Fremantle', 'Brisbane Lions', 'Collingwood', 'Geelong', 'Sydney', 'Geelong']



print(len(winners22))

#just and array of (home_team, away_team, homewin/homeloss) teams according to above dict

data23 = [(5, 3, 1), (15, 6, 0), (18, 8, 1), (2, 10, 1), (13, 14, 1), (16, 11, 0), (9, 1, 1), (17, 4, 0), (12, 7, 1),
        (3, 15, 1), (10, 13, 1), (6, 2, 1), (1, 5, 0), (14, 12, 0), (7, 18, 0), (11, 17, 1), (4, 16, 1), (8, 9, 1),
        (14, 10, 1), (6, 5, 1), (17, 18, 1), (9, 3, 0), (12, 4, 1), (2, 1, 0), (16, 15, 1), (13, 11, 1), (7, 8, 1),
        (10, 6, 1), (18, 3, 0), (1, 7, 1), (5, 14, 0), (11, 2, 0), (12, 16, 1), (4, 9, 1), (8, 13, 0), (15, 17, 1),
        (1, 3, 1), (7, 16, 1), (5, 11, 1), (10, 18, 1), (4, 13, 1), (2, 14, 1), (15, 8, 1), (9, 17, 1), (6, 12, 1),
        (7, 14, 0), (2, 8, 1), (9, 10, 0), (15, 11, 1), (17, 1, 0), (3, 12, 0), (16, 18, 1), (13, 5, 1), (6, 4, 1),
        (12, 2, 0), (10, 7, 1), (11, 9, 0), (14, 17, 1), (13, 18, 1), (8, 3, 0), (4, 15, 0), (1, 6, 0), (5, 16, 0),
        (3, 10, 0), (5, 8, 1), (15, 1, 1), (16, 13, 0), (7, 17, 1), (9, 14, 0), (2, 4, 1), (6, 11, 1), (18, 12, 0),
        (5, 15, 1), (8, 16, 0), (11, 7, 0), (18, 2, 0), (17, 13, 0), (10, 4, 1), (3, 14, 0), (1, 12, 1), (6, 9, 1),
        (2, 13, 1), (18, 11, 0), (14, 1, 1), (7, 15, 1), (10, 16, 1), (4, 5, 1), (17, 8, 1), (3, 6, 0), (9, 12, 0),
        (11, 3, 1), (12, 17, 0), (13, 7, 0), (15, 9, 0), (16, 14, 1), (8, 4, 0), (5, 2, 0), (6,18, 1), (1, 10, 1),
        (13, 3, 1), (2, 17, 1), (8, 6, 0), (14, 15, 0), (16, 1, 1), (9, 5, 0), (4, 18, 1),
        (11, 12, 0), (14, 2, 0), (17, 10, 1), (1, 8, 1), (7, 5, 0), (18, 9, 0), (3, 4, 0), (13, 6, 1),
        (2, 15, 1), (10, 11, 1), (9, 7, 1), (5, 12, 1), (3, 16, 1), (18, 14, 0),
        (15, 13, 1), (12, 10, 0), (11, 8, 1), (7, 4, 1), (3, 1, 1), (16, 17, 1),
        (10, 5, 1), (11, 15, 1), (1, 18, 1), (14, 7, 1), (16, 6, 0), (4, 2, 0), (17, 3, 0), (13, 9, 0), (8, 12, 0),
        (5, 11, 1), (14, 6, 0), (10, 8, 1), (9, 17, 1), (12, 13, 0), (2, 16, 1), (15, 18, 1), (4, 1, 1), (7, 3, 0),
        (11, 14, 1), (13, 10, 1), (6, 7, 1), (16, 12, 1), (3, 2, 1), (15, 4, 1), (1, 9, 0), (18, 17, 0), (8, 5, 0),
        (4,14, 0), (5, 17, 1), (3, 8, 1), (10, 15, 1), (7, 11, 0), (2, 6, 0), (9, 16, 1), (13, 1, 1), (12, 18, 1),
        (6, 3, 0), (14, 9, 0), (15, 7, 0), (16, 10, 1), (4, 11, 0), (1, 2, 1), (17, 12, 0), (5, 13, 0), (8, 18, 1),
        (14, 5, 1), (4, 8, 1), (1, 16, 1), (17, 6, 1), (15, 2, 1), (9, 11, 0), (18, 13, 0), (12, 3, 0), (7, 10, 0),
        (6, 15, 1), (18, 4, 0), (11, 16, 1), (10, 1, 1), (3, 13, 1), (8, 7, 0), (17, 14, 1), (12, 5, 1), (2, 9, 1),
        (6, 10, 0), (5, 18, 1), (16, 3, 0), (9, 4, 1), (12, 15, 1), (1, 11, 0), (14, 8, 0), (13, 17, 1), (7, 2, 0),
        (4, 6, 0), (17, 7, 0), (18, 16, 1), (10, 12, 1), (15, 14, 0), (8, 1, 0), (2, 5, 1), (11, 13, 0), (3, 9, 0)]

print(len(data23))

# it goes homescore, away score, homescore, awayscore, ..... for every game in 2023. Count in twos index 0 and 1 are a single match
# index 2 and 3 are a game and so forth

scores_23 = ['58', '58', '103', '125', '87', '82', '126', '72', '115', '65', '61', '110', '106',
              '90', '65', '124', '67', '52', '90', '82', '93', '82', '135', '64', '76', '108',
              '41', '92', '72', '73', '118', '37', '108', '80', '100', '81', '67', '53', '63',
              '49', '80', '61', '64', '74', '92', '74', '86', '117', '73', '54', '134', '84',
              '108', '67', '116', '83', '84', '107', '111', '72', '84', '89', '64', '66', '113',
              '60', '88', '75', '63', '126', '127', '45', '118', '62', '100', '90', '78', '122',
              '152', '77', '104', '77', '70', '56', '136', '89', '77', '75', '70', '64', '69',
              '118', '109', '69', '87', '108', '130', '37', '76', '79', '60', '82', '97', '54',
              '96', '78', '90', '77', '76', '83', '115', '67', '106', '107', '94', '65', '139',
              '49', '44', '152', '104', '132', '48', '72', '58', '59', '74', '100', '104', '58',
              '98', '72', '85', '90', '117', '48', '71', '86', '92', '87', '77', '48', '34', '64',
              '102', '78', '43', '113', '86', '103', '65', '135', '49', '103', '87', '45', '59',
              '79', '121', '69', '120', '55', '80', '76', '90', '93', '85', '40', '106', '77', '107',
              '64', '71', '70', '142', '26', '57', '85', '80', '92', '77', '51', '78', '88',
              '72', '79', '74', '81', '84', '77', '46', '96', '67', '77', '105', '70', '95', '78',
              '61', '44', '151', '96', '57', '120', '75', '97', '112', '87', '104', '110', '105',
              '99', '66', '80', '85', '107', '98', '73', '174', '52', '70', '85', '75', '103', '52',
              '86', '66', '62', '110', '72', '97', '81', '106', '36', '90', '70', '120', '61', '84',
              '105', '78', '63', '56', '84', '205', '34', '93', '61', '82', '80', '101', '34', '134',
              '53', '54', '54', '138', '72', '102', '73', '42', '120', '74', '78', '52', '112', '45',
              '47', '77', '85', '88', '75', '77', '89', '116', '35', '85', '72', '58', '79', '106',
              '73', '125', '63', '115', '97', '45', '98', '78', '76', '105', '104', '113', '67', '77',
              '51', '122', '72', '122', '45', '57', '71', '40', '88', '60', '98', '49', '90', '96',
              '95', '140', '69', '64', '53', '76', '105', '83', '85', '103', '63', '97', '93', '69',
              '61', '76', '93', '73', '78', '64', '71', '96', '55', '99', '101', '112', '65', '93',
              '122', '98', '130', '72', '67', '126', '71', '73', '72', '89', '61', '105', '73', '97',
              '85', '85', '96', '71', '103', '54', '73', '74', '77', '109', '101', '77', '86', '114',
              '90', '99', '93', '60', '56', '33', '134', '67', '64', '93', '57', '136', '85', '100',
              '124', '101', '72', '87', '91', '162', '36', '88', '55', '73', '74', '85', '92', '87',
              '60', '58', '74', '31', '101', '56', '93', '132', '97', '72', '60', '79', '104', '78',
              '123', '94', '63', '56', '77', '73', '105']

#getting the difference in scores for every game
diff = []
for x in range(0, len(scores_23), 2):
  home_score = int(scores_23[x])
  away_score = int(scores_23[x+1])
  difference = home_score - away_score
  diff.append(difference)

#excluding finals to match above data

print(diff, len(diff))
print(len(scores_23))

#function to extract the even (home team) scores from an array also away teams

home_scores_23 = []
away_scores_23 = []
def get_home_scores(data, arr):
  for x in range(0, len(data), 2):
    arr.append(int(data[x]))
  return arr

def get_away_scores(data, arr):
  for i in range(1, len(data), 2):
    arr.append(int(data[i]))
  return arr

get_home_scores(scores_23, home_scores_23)
get_away_scores(scores_23, away_scores_23)

print(home_scores_23, len(home_scores_23))
print(len(away_scores_23))


home_teams = []
away_teams = []
wins = []

for gh in home2020:
  home_teams.append(teams[gh])

for ahhh in away2020:
  away_teams.append(teams[ahhh])

for x in home_teams_22:
  home_teams.append(teams[x])

for y in away_teams_22:
  away_teams.append(teams[y])

#data to add to wins below

for t in winnerInt2020:
  wins.append(t)

for z in range(0, len(winners22)):
  if home_teams_22[z] == winners22[z]:
    wins.append(1)
  else:
    wins.append(0)

for x in data23:
  home_teams.append(x[0])
  away_teams.append(x[1])
  wins.append(x[2])

print("wins:",len(wins))
print("away teams:", len(away_teams))
print("Home teams:", len(home_teams))

X = np.column_stack((home_teams, away_teams))
y = wins
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: ", accuracy)

def num_to_team(teams, number):
    # Invert the dictionary
    number_to_team = {value: key for key, value in teams.items()}
    return number_to_team.get(number, "No team found for this number")


def predictor(team1, team2):
  prediction = model.predict([[team1,team2]])
  if prediction == [1]:
    x = "will win"
  else:
    x = "will loose"
  print(num_to_team(teams, team1),x)

adelaide_wins = 0
port_wins = 0
carlton_wins = 0
essendon_wins = 0
richmond_wins = 0
collingwood_wins = 0
fremantle_wins = 0
westcoast_wins = 0
gws_wins = 0
brisbane_wins = 0
sydney_wins = 0
stkilda_wins = 0
melbourne_wins = 0
bulldogs_wins = 0
geelong_wins = 0
goldcoast_wins = 0
hawthorn_wins = 0
northmelbourne_wins = 0

adelaide_wins1 = 0
port_wins1 = 0
carlton_wins1 = 0
essendon_wins1 = 0
richmond_wins1 = 0
collingwood_wins1 = 0
fremantle_wins1 = 0
westcoast_wins1 = 0
gws_wins1 = 0
brisbane_wins1 = 0
sydney_wins1 = 0
stkilda_wins1 = 0
melbourne_wins1 = 0
bulldogs_wins1 = 0
geelong_wins1 = 0
goldcoast_wins1 = 0
hawthorn_wins1 = 0
northmelbourne_wins1 = 0

graph_arr = []

for team in winners2020:
  graph_arr.append(team)

for team in winners22:
  graph_arr.append(team)

#for wins in winners2020

for wins in winners2020:
  if wins == "Adelaide":
    adelaide_wins1 += 1
  elif wins == "Port Adelaide":
    port_wins1 += 1
  elif wins == "Carlton":
    carlton_wins1 += 1
  elif wins == "Essendon":
    essendon_wins1 += 1
  elif wins == "Richmond":
    richmond_wins1 += 1
  elif wins == "Collingwood":
    collingwood_wins1 += 1
  elif wins == "Fremantle":
    fremantle_wins1 += 1
  elif wins == "West Coast":
    westcoast_wins1 += 1
  elif wins == "Greater Western Sydney":
    gws_wins1 += 1
  elif wins == "Brisbane Lions":
    brisbane_wins1 += 1
  elif wins == "Sydney":
    sydney_wins1 += 1
  elif wins == "St Kilda":
    stkilda_wins1 += 1
  elif wins == "Melbourne":
    melbourne_wins1 += 1
  elif wins == "Western Bulldogs":
    bulldogs_wins1 += 1
  elif wins == "Geelong":
    geelong_wins1 += 1
  elif wins == "Gold Coast":
    goldcoast_wins1 += 1
  elif wins == "Hawthorn":
    hawthorn_wins1 += 1
  elif wins == "North Melbourne":
    northmelbourne_wins1 += 1


for wins in winners22:
  if wins == "Adelaide":
    adelaide_wins += 1
  elif wins == "Port Adelaide":
    port_wins += 1
  elif wins == "Carlton":
    carlton_wins += 1
  elif wins == "Essendon":
    essendon_wins += 1
  elif wins == "Richmond":
    richmond_wins += 1
  elif wins == "Collingwood":
    collingwood_wins += 1
  elif wins == "Fremantle":
    fremantle_wins += 1
  elif wins == "West Coast":
    westcoast_wins += 1
  elif wins == "Greater Western Sydney":
    gws_wins += 1
  elif wins == "Brisbane Lions":
    brisbane_wins += 1
  elif wins == "Sydney":
    sydney_wins += 1
  elif wins == "St Kilda":
    stkilda_wins += 1
  elif wins == "Melbourne":
    melbourne_wins += 1
  elif wins == "Western Bulldogs":
    bulldogs_wins += 1
  elif wins == "Geelong":
    geelong_wins += 1
  elif wins == "Gold Coast":
    goldcoast_wins += 1
  elif wins == "Hawthorn":
    hawthorn_wins += 1
  elif wins == "North Melbourne":
    northmelbourne_wins += 1

teamwinnings20 = [adelaide_wins1, port_wins1, carlton_wins1, essendon_wins1, richmond_wins1, collingwood_wins1, fremantle_wins1,
                  westcoast_wins1, gws_wins1, brisbane_wins1, sydney_wins1, stkilda_wins1, melbourne_wins1, bulldogs_wins1, geelong_wins1,
                  goldcoast_wins1, hawthorn_wins1, northmelbourne_wins1]

teamwinnings22 = [adelaide_wins, port_wins, carlton_wins, essendon_wins, richmond_wins, collingwood_wins, fremantle_wins,
                  westcoast_wins, gws_wins, brisbane_wins, sydney_wins, stkilda_wins, melbourne_wins, bulldogs_wins, geelong_wins,
                  goldcoast_wins, hawthorn_wins, northmelbourne_wins]

allteamnames = ["Adelaide", "Port Adelaide", "Carlton", "Essendon", "Richmond", "Collingwood", "Fremantle", "West Coast",
                "GWS", "Brisbane Lions", "Sydney", "St Kilda", "Melbourne", "Western Bulldogs", "Geelong",
                "Gold Coast", "Hawthorn", "North Melbourne"]
plt.bar(allteamnames, teamwinnings20)
plt.title("Wins per team in 2020")
plt.xlabel("Teams")
plt.ylabel("Number of Wins")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


plt.bar(allteamnames, teamwinnings22, color = "green")
plt.title("Wins per team in 2022")

predictor(teams["ENTER HOME TEAM"], teams["ENTER AWAY TEAM"])


plt.xlabel("Teams")
plt.ylabel("Number of wins")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()



