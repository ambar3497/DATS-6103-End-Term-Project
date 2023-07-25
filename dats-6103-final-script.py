#%%
##### Import Packages and Data #####

import pandas as pd
import numpy as np

data = pd.read_csv('DC-AirBnB-Listings.csv')
print(data.columns)
print(f'The file has {data.shape[0]} rows and {data.shape[1]} features.')
data.head()

# %%
##### Subset the DataFrame to get the Features #####
df = data[['host_since', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost', 'neighbourhood_cleansed', 'neighbourhood', 'neighbourhood_group_cleansed', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'price', 'instant_bookable', 'review_scores_rating', 'latitude', 'longitude', 'amenities']]
# %%
##### Check for NAs #####
df.isna().sum()

#%%
##### Check for Duplicates #####
print(df.duplicated().sum())
df[df.duplicated()]

