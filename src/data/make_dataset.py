from ucimlrepo import fetch_ucirepo 
import pandas as pd
  
# fetch dataset 
mushroom = fetch_ucirepo(id=73) 
  
# data (as pandas dataframes) 
X = mushroom.data.features 
y = mushroom.data.targets 
  
# metadata 
print(mushroom.metadata) 
  
# variable information 
print(mushroom.variables) 


df = pd.concat([X, y], axis = 1)
df.to_pickle("../../data/interim/mushrooms.pkl")