#%%
##### Import Packages and Data #####

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import folium

data = pd.read_csv('DC-AirBnB-Listings.csv')
print(data.columns)
print(f'The file has {data.shape[0]} rows and {data.shape[1]} features.')
print(data.dtypes)
data.head()

# %%
##### Subset the DataFrame to get the Features #####
df = data[['host_since', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost', 'neighbourhood_cleansed', 'neighbourhood', 'neighbourhood_group_cleansed', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'price', 'instant_bookable', 'review_scores_rating', 'latitude', 'longitude', 'amenities']]

#%%
##### Check for Duplicates #####
print(df.duplicated().sum())
df[df.duplicated()]

df.drop_duplicates(keep='last', inplace=True)

# %%
##### Check for NAs #####
print(df.isna().sum())
df.drop(columns=['bathrooms', 'neighbourhood_group_cleansed', 'neighbourhood'], inplace=True) # Drop columns with all NA values
df = df[~df['review_scores_rating'].isna()] # Ensure there are no NAs for target variable

#%% 
##### Check for Unique Values and Data Types for Columns with NAs #####
nas = ['host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost', 'bedrooms', 'beds']
    
def get_value_counts(df, column):
    print(f"Column Name: {column}")
    print(f"Column Data Type: {df[column].dtype}")
    print(f"Number of Unique Values: {df[column].nunique()}")
    print(df[column].value_counts())
    
for column in nas:
    get_value_counts(df, column)

#%%
##### Part 1: Imputation for Columns with NAs #####
# First we will convert the columns to numeric variables 
# Host Response Time: Categorical
# Host Response Rate: Percentage as a String
# Host Acceptance Rate: Percentage as a String
# Host is Superhost: Boolean as a String
# Bedrooms: Float
# Beds: Float

df['host_response_time'].replace({'within an hour':1, 'within a few hours': 2, 'within a day': 3, 'a few days or more':4}, inplace=True)
df['host_is_superhost'].replace({'t':1, 'f': 0}, inplace=True)

acceptance = []
for idx in df.index:
    if type(df['host_acceptance_rate'][idx]) == str:
        acceptance.append(int(df['host_acceptance_rate'][idx][:-1])/100)
    else:
        acceptance.append(df['host_acceptance_rate'][idx])
        
df['host_acceptance_rate'] = acceptance
        
response = []
for idx in df.index:
    if type(df['host_response_rate'][idx]) == str:
        response.append(int(df['host_response_rate'][idx][:-1])/100)
    else:
        response.append(df['host_response_rate'][idx])
        
df['host_response_rate'] = response
        
#%%
##### Part 2: Imputation for Columns with NAs #####
# Now we will replace the NA values with the column means

def impute_mean(df, column):
    df[column].fillna(df[~df[column].isna()][column].mean(), inplace=True)

impute_mean(df, 'host_response_time')
impute_mean(df, 'host_response_rate')
impute_mean(df, 'host_acceptance_rate')
impute_mean(df, 'host_is_superhost')
impute_mean(df, 'bedrooms')
impute_mean(df, 'beds')
#%% 
##### Part 1: Feature Engineering for Amenities Variable #####
# First we will create a list of all amenities and get the number of unique amenities

df['amenities'] = df['amenities'].apply(lambda x: ast.literal_eval(x))
all_amenities = []

for idx in df.index:
    for item in df['amenities'][idx]:
        if item not in all_amenities:
            all_amenities.append(item)

all_amenities_arr = np.array(all_amenities)
print(f'There are {len(np.unique(all_amenities_arr))} unique amenities listed.')
      
#%%
##### Part 2: Feature Engineering for Amenities Variable #####
# Next we will generate amenity groups given there are over 2,000 amenities

all_amenities = [x.lower() for x in all_amenities]

streaming = ['netflix', 'amazon prime video', 'hbo max', 'hulu', 'disney+', 'apple tv', 'chromecast', 'fire tv', 'premium cable']
outdoor = ['outdoor pool', 'grill', 'bbq grill', 'hot tub', 'fire pit', 'outdoor kitchen', 'kayak', 'outdoor dining area', 'patio or balcony', 'lake access', 'resort access', 'outdoor shower', 'waterfront', 'hammock', 'beach access']
view = ['canal view','vineyard view','river view','ocean view','beach view','valley view','mountain view','harbor view',
 'pool view','bay view','resort view','sea view','golf course view','lake view','desert view','marina view', 'garden view','city skyline view','courtyard view', 'park view']
entertainment = ['game console', 'gaming console', 'ps3', 'ps4', 'ps5', 'xbox', 'hdtv', 'dvd player', 'nintendo wii', 'nintendo switch']
kitchen_appl = ['rice maker', 'coffee maker', 'hot water kettle', 'keurig', 'dishwasher', 'trash compactor', 'toaster', 'toaster oven', 'air fryer']
other_appl = ['free washer', 'free dryer', 'iron']
tech = ['fast wifi', 'google home speaker', 'google home', 'amazon alexa', 'sound bar', 'sound system']
luxury = ['sauna', 'indoor fireplace']
security = ['safe', 'smart lock', 'security cameras', 'carbon monoxide alarm', 'smoke alarm', 'window guards', 'lockbox', 'fire extinguisher', 'first aid kit', 'smart lock']
parking = ['free parking', 'free driveway parking', 'free carport', 'free residential garage']
gym = ['exercise equipment', 'bikes', 'gym']

#%%
##### Part 3: Feature Engineering for Amenities Variable #####
# Finally we will create the variables for amenity groups and add them to the dataframe 

am_streaming = []
am_outdoor = []
am_view = []
am_entertainment = []
am_kitchenappl = []
am_otherappl = []
am_tech = []
am_luxury = []
am_security = []
am_parking = []
am_gym = []

for idx in df.index:
    if any(x.lower() in streaming for x in df['amenities'][idx]):
        am_streaming.append(1)
    else:
        am_streaming.append(0)
        
    if any(x.lower() in outdoor for x in df['amenities'][idx]):
        am_outdoor.append(1)
    else:
        am_outdoor.append(0)
        
    if any(x.lower() in view for x in df['amenities'][idx]):
        am_view.append(1)
    else:
        am_view.append(0)
        
    if any(x.lower() in entertainment for x in df['amenities'][idx]):
        am_entertainment.append(1)
    else:
        am_entertainment.append(0)
        
    if any(x.lower() in kitchen_appl for x in df['amenities'][idx]):
        am_kitchenappl.append(1)
    else:
        am_kitchenappl.append(0)
        
    if any(x.lower() in other_appl for x in df['amenities'][idx]):
        am_otherappl.append(1)
    else:
        am_otherappl.append(0)
        
    if any(x.lower() in tech for x in df['amenities'][idx]):
        am_tech.append(1)
    else:
        am_tech.append(0)
    
    if any(x.lower() in luxury for x in df['amenities'][idx]):
        am_luxury.append(1)
    else:
        am_luxury.append(0)
        
    if any(x.lower() in security for x in df['amenities'][idx]):
        am_security.append(1)
    else:
        am_security.append(0)
        
    if any(x.lower() in parking for x in df['amenities'][idx]):
        am_parking.append(1)
    else:
        am_parking.append(0)
        
    if any(x.lower() in gym for x in df['amenities'][idx]):
        am_gym.append(1)
    else:
        am_gym.append(0)
    

df['Clothing'] = am_otherappl
df['Entertainment'] = am_entertainment
df['Exercise'] = am_gym
df['Kitchen'] = am_kitchenappl
df['Luxury'] = am_luxury
df['Outdoor'] = am_outdoor
df['Parking'] = am_parking
df['Security'] = am_security
df['Streaming'] = am_streaming
df['Technology'] = am_tech
df['Views'] = am_view

#%% 
##### Part 1: Feature Engineering for Neighbourhood Variable #####
# First we will create a list of all neighbourhoods and get the number of unique neighbourhoods

all_neighborhoods = []

for idx in df.index:
    neighborhood_list = [x.strip() for x in df['neighbourhood_cleansed'][idx].split(',')] 
    for item in neighborhood_list:
        if item not in all_neighborhoods:
            all_neighborhoods.append(item)

all_neighborhoods_arr = np.array(all_neighborhoods)
print(f'There are {len(np.unique(all_neighborhoods_arr))} unique neighborhoods listed.')

#%%
##### Part 2: Feature Engineering for Neighbourhood Variable #####
# Next we will generate neighborhood groups given there are over 120 neighborhoods
# Reference: https://publicsafety.fandom.com/wiki/List_of_neighborhoods_of_the_District_of_Columbia_by_ward#Ward_7

all_neighborhoods = [x.lower() for x in all_neighborhoods]

ward1 = ['adams morgan', 'columbia heights', 'howard university', 'le droit park', 'lanier heights', 'mt. pleasant', 'park view', 'pleasant plains', 'shaw']

ward2 = ['burleith/hillandale', 'chinatown', 'downtown', 'dupont circle', 'foggy bottom', 'georgetown', 'gwu','georgetown reservoir', 'kalorama heights', 'logan circle', 'mount vernon square', 'penn quarters', 'southwest employment area', 'west end']

ward3 = ['american university park', 'cathedral heights', 'chevy chase', 'cleveland park', 'colonial village', 'forest hills', 'foxhall crescent', 'foxhall village', 'friendship heights', 'glover park', 'massachusetts avenue heights', 'mclean gardens', 'north cleveland park', 'palisades', 'spring valley', 'tenleytown', 'wesley heights', 'woodland-normanstone terrace', 'woodland/fort stanton','woodley park']

ward4 = ['barnaby woods', 'brightwood','brightwood park','colonial village', 'crestwood', 'fort totten', 'hawthorne', 'manor park', 'petworth', 'lamont riggs', 'shepherd park', 'takoma']

ward5 = ['arboretum', 'bloomingdale', 'brentwood', 'brookland', 'carver langston', 'eckington', 'edgewood', 'fort lincoln', 'fort totten', 'gateway', 'ivy city', 'langdon', 'michigan park', 'north michigan park', 'pleasant hill', 'queens chapel', 'trinidad', 'truxton circle', 'woodridge']

ward6 = ['capitol hill', 'lincoln park', 'kingman park', 'southwest/waterfront']

ward7 = ['benning', 'benning heights', 'burrville', 'capitol view', 'deanwood', 'dupont park', 'eastland gardens', 'fairfax village', 'fort davis park', 'fort dupont', 'greenway', 'hillbrook', 'hillcrest', 'kenilworth', 'lincoln heights', 'marshall heights', 'mayfair', 'naylor gardens', 'penn branch', 'randle highlands', 'river terrace', 'twining']

ward8 = ['barry farm', 'bellevue', 'buena vista', 'congress heights', 'douglas', 'fairlawn', 'garfield heights', 'knox hill', 'navy yard', 'shipley terrace', 'washington highlands']

all_wards = ward1 + ward2 + ward3 + ward4 + ward5 + ward6 + ward7 + ward8 

# Get neighborhoods not currently assigned to any ward
list(set(all_neighborhoods) - set(all_wards)) 

ward1.append('cardozo/shaw')
ward2 = ward2 + ['connecticut avenue/k street', 'sheridan']
ward3.append('van ness')
ward4.append('north portal estates')
ward5 = ward5 + ['university heights', 'north capitol street']
ward6 = ward6 + ['union station', 'stanton park', 'buzzard point', 'fort mcnair']
ward7 = ward7 + ['mahaning heights', 'summit park', 'grant park', 'fairmount heights']
ward8 = ward8 + ['near southeast', 'historic anacostia', 'fort stanton']

#%%
##### Part 3: Feature Engineering for Neighbourhood Variable #####
# Finally we will create the variable for wards and add them to the dataframe 

ward = []

for idx in df.index:
    neighborhood_list = [x.strip() for x in df['neighbourhood_cleansed'][idx].split(',')] 
    
    if len(neighborhood_list) > 1: 
        if any(x.lower() in ward1 for x in neighborhood_list):
            ward.append(1)
        elif any(x.lower() in ward2 for x in neighborhood_list):
            ward.append(2)
        elif any(x.lower() in ward3 for x in neighborhood_list):
            ward.append(3)
        elif any(x.lower() in ward4 for x in neighborhood_list):
            ward.append(4)
        elif any(x.lower() in ward5 for x in neighborhood_list):
            ward.append(5)
        elif any(x.lower() in ward6 for x in neighborhood_list):
            ward.append(6)
        elif any(x.lower() in ward7 for x in neighborhood_list):
            ward.append(7)
        elif any(x.lower() in ward8 for x in neighborhood_list):
            ward.append(8)
    elif len(neighborhood_list) < 2:
        
        holder = ''
        neighborhood_list = holder.join(neighborhood_list).lower()
        if neighborhood_list in ward1:
            ward.append(1)
        elif neighborhood_list in ward2:
            ward.append(2)
        elif neighborhood_list in ward3:
            ward.append(3)
        elif neighborhood_list in ward4:
            ward.append(4)
        elif neighborhood_list in ward5:
            ward.append(5)
        elif neighborhood_list in ward6:
            ward.append(6)
        elif neighborhood_list in ward7:
            ward.append(7)
        elif neighborhood_list in ward8:
            ward.append(8)
            
df['Ward'] = ward

#%% 
##### Convert Room Type to Ordinal Variable #####

df['Privacy'] = df['room_type'].replace({'Entire home/apt':1, 'Private room':2, 'Shared room': 3, 'Hotel room':4})

##### Convert Instant Bookable to Boolean Integer #####
df['instant_bookable'] = df['instant_bookable'].replace({'t':1, 'f':0})

#%% 
##### Convert Price to Float #####

prices = []
for price in df['price']:
    if "," in price:
        prices.append(float(''.join(price[1:].split(','))))
    else:
        prices.append(float(price[1:]))
        
df['price'] = prices

#%%
##### Finalizing DataFrames #####
# The df dataframe can be used for EDA
# The listings dataframe is processed for modeling 

listings = df[['host_response_time', 'host_response_rate',
       'host_acceptance_rate', 'host_is_superhost', 'accommodates', 'bedrooms', 'beds', 'instant_bookable', 'price', 'Clothing', 'Entertainment', 'Exercise',
       'Kitchen', 'Luxury', 'Outdoor', 'Parking', 'Security', 'Streaming',
       'Technology', 'Views', 'Ward', 'Privacy', 'review_scores_rating']]


# %%
#Exploratory Data Analysis

#barplot for hosts with more than 1 property


host_property_count = data['host_id'].value_counts()
host_property_count_df = pd.DataFrame({'host_id': host_property_count.index, 'property_count': host_property_count.values})
host_property_count_df = host_property_count_df.sort_values(by='property_count', ascending=False)

#range and rnage labels
ranges = [1,2,5,10,15,20,25,50,100,200,float('inf')] 
range_labels = [' Upto 2 Properties',' Upto 5 Properties',' Upto 10 Properties',' Upto 15 Properties', 'Upto 20 Properties', 'Upto 25 Properties','Upto 50 Properties', 'Upto 100 Properties', 'Upto 200 Properties', 'More than 200 Properties']
host_property_count_df['property_count_range'] = pd.cut(host_property_count_df['property_count'], bins=ranges, labels=range_labels, right=False)
# Group the data by property count range and count the number of hosts in each range
grouped_counts = host_property_count_df.groupby('property_count_range')['host_id'].count()


plt.figure(figsize=(10, 6))
grouped_counts.plot(kind='bar')
plt.xlabel('Host Groups')
plt.ylabel('Number of Properties')
plt.title('Number of Properties Listed by Host')
plt.xticks(rotation=40)  # To avoid overlap of host IDs on x-axis
plt.tight_layout()
#plt.ylim(top = 100)
plt.show()


plt.figure(figsize=(10, 6))
grouped_counts.plot(kind='bar')
plt.xlabel('Host Groups')
plt.ylabel('Number of Properties')
plt.title('Number of Properties Listed by Host')
plt.xticks(rotation=40)  # To avoid overlap of host IDs on x-axis
plt.tight_layout()
plt.ylim(top = 100)
plt.show()



base_keywords=['Camper/RV','Casa Particular', 'Entire','Floor','Room','Shared room','Private room','Tiny home','Tower']

keyword_counts = {keyword: 0 for keyword in base_keywords}

for keyword in base_keywords:
    propcount = data[data['property_type'].str.contains(keyword,case=False,na=False)].shape[0]
    keyword_counts[keyword] = propcount

plt.figure(figsize=(10, 6))
plt.bar(keyword_counts.keys(),keyword_counts.values())
plt.xlabel('Property Type')
plt.ylabel('Count of Properties')
plt.title('Count of Properties by Keyword')
plt.xticks(rotation=40)  # To avoid overlap of host IDs on x-axis
plt.tight_layout()
plt.show()


plt.figure(figsize=(16,8))
ax= plt.axes()
correlation_mat = listings.corr()
sns.heatmap(correlation_mat,annot=True,cmap='coolwarm',ax=ax, fmt = '0.2f', annot_kws = {'size' : 10},vmin=-1,center=0,vmax=1)
plt.title('Correlation Matrix', fontsize = 16)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment = 'right')
cbar = ax.collections[0].colorbar
# cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
# cbar.set_ticklabels(['-1', '-0.5', '0', '0.5', '1'])
plt.tight_layout()
plt.show()




# integer_cols = data.select_dtypes(include = ['int64'])
# (integer_cols.head())
# intcols = integer_cols.columns
# intcols21 = intcols.drop(["Reached.on.Time_Y.N"])
# plt.figure(figsize = (16, 20))
# sns.set_theme(style="ticks", palette="pastel")
# nplot = 1
# for i in range(len(intcols21)):
#     if nplot <= len(intcols21):
#         ax = plt.subplot(4, 2, nplot)
#         sns.boxplot(x = intcols21[i], data = data, ax = ax)
#         plt.title("Boxplots for On-time delivery by "f"{intcols21[i]}" , fontsize = 13)
#         nplot += 1
# plt.show()


plt.figure(figsize=(16, 8))
sns.scatterplot(x='latitude', y='longitude', hue='Ward', data=df, palette='Set1', alpha=0.7)
plt.title('Airbnb Listings: Latitude vs Longitude with Region Hue')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.legend(title='Ward Numbers')
plt.grid(True)
plt.show()


plt.figure(figsize=(16, 8))
sns.scatterplot(x='latitude', y='longitude', hue='host_identity_verified', data=data, palette='Set1', alpha=0.7)
plt.title('Latitude vs Longitude of an AirBnB with Host Identity')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.legend(title='Verified Identity(True/False)')
plt.grid(True)
plt.show()




    

#%%
#modelling - xgboost
#tra test split

#scaling

#accuracy metrics

#variable importance plot