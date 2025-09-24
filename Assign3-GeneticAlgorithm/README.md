# PySimbot

PySimbot is a software for robot simulation

## Requirements

- Require Python3 and pip [Python](https://www.python.org/downloads/)
- For Windows, install dependencies run `pip install -r requirements_GA_windows.txt`
- For MacOS, install dependencies run `pip3 install -r requirements_GA_macos.txt`

# setting 
Settings:
Population: 100
Step per gen: 250
Selection: Roulette wheel
Crossover: Single point crossover
Mutation: 2% mutation chance
Fitness: 
Factors are based on the distance between the bot and food and also bot and its spawn point encourage them to get out from starting point. bothe and also I deduct the fitness by number of collision times 3 and to ensure that robot that able to achieve its objective I give bonus 300 point to them 
