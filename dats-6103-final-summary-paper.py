#%%

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance

#%% [markdown]
## 1. Introduction and Prior Research
# Founded in 2007, AirBnB has become one of the world’s leading companies for short-term rentals. The company boasts an “excess of six million listings, covering more than 100,000 cities and towns.” Among the cities hosting an array of AirBnB listings is Washington, DC. In 2022 alone, hosts in the DC area “made 29% more than the typical host in the U.S.” 
#
# Before we begin, we took into account a research done by Jessie Owens in May 2020, where she published an article titled "An Analysis of AirBnB in Washington, DC"(Owens, 2020).
# Jessie claimed that she finds hotels to be just too impersonal and touristy when looking for lodging.  
# Therefore, she discovered and started using AirBnB, it piqued her curiosity greatly.
# By avoiding the tourist and commercial portions of a city and staying in a local area, one might experience a place or area more like a native. 
# For travelers all around the world, it has become a well-liked 
# substitute for hostels, hotels, and resorts.  She found the 
# motto "belong anywhere" by AirBnB to be enticing since she wanted to experience travel like a native.
#
# The purpose of Jessie's effort was to uncover intriguing patterns in the data that could be instructive for a traveler, a homeowner (or host), an AirBnB decision-maker, or even a D.C. housing regulator.
# Jessie came up with a couple of inquiries that she hoped to address with the information:
#
# 1. What connections may be drawn between AirBnB data and significant occurrences in the nation's capital over the previous 10 years?
# 2. What elements play a major role in pricing prediction?
#
# According to Jessie, the following was the most crucial lesson discovered throughout this study and analytical project - 
#
# “Ask the right question.  Make it as specific as possible.”


#
# Despite the relative success of DC owners, hosts around the country have been troubled by declining revenue. Even more worrisome, almost 50% of hosts depend on the extra income to stay in their homes. Given the current threats to host revenue and the essentiality of the short-term rental market to the city of Washington, DC, we developed the following **SMART question**:
#
# "What are the most important variables for **predicting if an AirBnB’s rating** is above average or below average in the city of Washington, DC in 2023?""
#
# We aim to use classification models to predict if a listing’s rating is above or below average. The source of our data set is Inside AirBnB, where quarterly data is shared. We use a number of predictive variables  including the structural characteristics of the property, the location of the property, the host’s behavior, and the ratings of previous renters. The dataset contains 6542 rows of data and 75 columns including the id column. 
#
# This summary paper is organized as follows:   
# 1. Introduction and Prior Research (Current)
# 2. Data Preprocessing and Data Engineering 
# 3. Exploratory Data Analysis 
# 4. Modelling and variable importance plot
# 5. Interpretation of Data
# 6. References


#%% [markdown]
## 2. Data Preprocessing and Data Engineering 
#%%
data = pd.read_csv('DC-AirBnB-Listings.csv')
print(data.columns,'\n')
print(f'The file has {data.shape[0]} rows and {data.shape[1]} features.\n')
print(data.dtypes,'\n')
data.head()

#%% 
##### Subset the DataFrame to get the Features #####
df = data[['host_since', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost', 'neighbourhood_cleansed', 'neighbourhood', 'neighbourhood_group_cleansed', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'price', 'instant_bookable', 'review_scores_rating', 'latitude', 'longitude', 'amenities']]

#%% 
##### Check for Duplicates #####
print(df.duplicated().sum(),'\n')
df[df.duplicated()]

df.drop_duplicates(keep='last', inplace=True)

#%% 
##### Check for NAs #####
print(df.isna().sum(),'\n')
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
    checker = [x.lower() for x in df['amenities'][idx]]

    if any(x in el for x in streaming for el in checker):
        am_streaming.append(1)
    else:
        am_streaming.append(0)
        
    if any(x in el for x in outdoor for el in checker):
        am_outdoor.append(1)
    else:
        am_outdoor.append(0)
        
    if any(x in el for x in view for el in checker):
        am_view.append(1)
    else:
        am_view.append(0)
        
    if any(x in el for x in entertainment for el in checker):
        am_entertainment.append(1)
    else:
        am_entertainment.append(0)
        
    if any(x in el for x in kitchen_appl for el in checker):
        am_kitchenappl.append(1)
    else:
        am_kitchenappl.append(0)
        
    if any(x in el for x in other_appl for el in checker):
        am_otherappl.append(1)
    else:
        am_otherappl.append(0)
        
    if any(x in el for x in tech for el in checker):
        am_tech.append(1)
    else:
        am_tech.append(0)
    
    if any(x in el for x in luxury for el in checker):
        am_luxury.append(1)
    else:
        am_luxury.append(0)
        
    if any(x in el for x in security for el in checker):
        am_security.append(1)
    else:
        am_security.append(0)
        
    if any(x in el for x in parking for el in checker):
        am_parking.append(1)
    else:
        am_parking.append(0)
        
    if any(x in el for x in gym for el in checker):
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
##### Generate Plots to Show Distribution of Amenities Variables #####

fig, axes = plt.subplots(6, 2, figsize=(8,14))
fig.suptitle('Distribution of Engineered Amenities Variables', y=0.93)

plt.subplots_adjust(hspace=0.5, wspace=0.4)

axes[0,0].bar(df.groupby(by='Clothing', as_index=False)['price'].count()['Clothing'], df.groupby(by='Clothing', as_index=False)['price'].count()['price'])
axes[0,0].set_title('Clothing Variable')

axes[0,1].bar(df.groupby(by='Entertainment', as_index=False)['price'].count()['Entertainment'], df.groupby(by='Entertainment', as_index=False)['price'].count()['price'])
axes[0,1].set_title('Entertainment Variable')

axes[1,0].bar(df.groupby(by='Exercise', as_index=False)['price'].count()['Exercise'], df.groupby(by='Exercise', as_index=False)['price'].count()['price'])
axes[1,0].set_title('Exercise Variable')

axes[1,1].bar(df.groupby(by='Kitchen', as_index=False)['price'].count()['Kitchen'], df.groupby(by='Kitchen', as_index=False)['price'].count()['price'])
axes[1,1].set_title('Kitchen Variable')

axes[2,0].bar(df.groupby(by='Luxury', as_index=False)['price'].count()['Luxury'], df.groupby(by='Luxury', as_index=False)['price'].count()['price'])
axes[2,0].set_title('Luxury Variable')

axes[2,1].bar(df.groupby(by='Outdoor', as_index=False)['price'].count()['Outdoor'], df.groupby(by='Outdoor', as_index=False)['price'].count()['price'])
axes[2,1].set_title('Outdoor Variable')

axes[3,0].bar(df.groupby(by='Parking', as_index=False)['price'].count()['Parking'], df.groupby(by='Parking', as_index=False)['price'].count()['price'])
axes[3,0].set_title('Parking Variable')

axes[3,1].bar(df.groupby(by='Security', as_index=False)['price'].count()['Security'], df.groupby(by='Security', as_index=False)['price'].count()['price'])
axes[3,1].set_title('Security Variable')

axes[4,0].bar(df.groupby(by='Streaming', as_index=False)['price'].count()['Streaming'], df.groupby(by='Streaming', as_index=False)['price'].count()['price'])
axes[4,0].set_title('Streaming Variable')

axes[4,1].bar(df.groupby(by='Technology', as_index=False)['price'].count()['Technology'], df.groupby(by='Technology', as_index=False)['price'].count()['price'])
axes[4,1].set_title('Technology Variable')

axes[5,0].bar(df.groupby(by='Views', as_index=False)['price'].count()['Views'], df.groupby(by='Views', as_index=False)['price'].count()['price'])
axes[5,0].set_title('Views Variable')


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

review_mean = listings['review_scores_rating'].mean()
target = []

for rating in listings['review_scores_rating']:
    if rating >= review_mean:
        target.append(1)
    else:
        target.append(0)

listings['Target'] = target

#%% [markdown]
## 3. Exploratory Data Analysis

#%%
##### Generate Plot to Show Distribution of Neighborhoods Variables #####
plt.bar(df.groupby(by='Ward', as_index=False)['price'].count()['Ward'],df.groupby(by='Ward', as_index=False)['price'].count()['price'])
plt.title('Distribution of Engineered Neighborhood Variable')
plt.xlabel('Ward')
plt.ylabel('Count of Listings')

#%%
#barplot for hosts with more than 1 property

host_property_count = data['host_id'].value_counts()
host_property_count_df = pd.DataFrame({'host_id': host_property_count.index, 'property_count': host_property_count.values})
host_property_count_df = host_property_count_df.sort_values(by='property_count', ascending=False)

#range and range labels
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

#%% [markdown]------------------------------------------------------------
#There are times when customers want to book an Airbnb but it is not favorable to them due to a singular unfavorable factor in the long list of factors.
# The reason they decide to choose that airbnb that they could not book at the last minute was most likely because of its certain features that instantly got their attention, fit their price bracket and needs.
# One thing that can be done in this case as a product developer is to know how many bunch of properties exists that all come under a single host. it will allow us to take that information and utilize it to find 
# and design recommendation system to recommend other airbnb that might suite the customer if they were unable to finalize on an airbnb before.
#
# This bar plot shows the count of individual hosts that own such categories of properties on the x-axis. of course it is not a fixed number per host like every hosts in washington dc owns say 3 properties only. It varies based on the hosts personal situation.
#this plot contains bins for hosts and those bins begins from number of properties owned by hosts that own at least 2 properties all the way upto hosts that own 200 and more than 200 properties. This plot really helps categorize the airbnb market in washington dc based on hosts owning multiple properties. 

plt.figure(figsize=(10, 6))
grouped_counts.plot(kind='bar')
plt.xlabel('Host Groups')
plt.ylabel('Number of Properties')
plt.title('Number of Properties Listed by Host')
plt.xticks(rotation=40)  # To avoid overlap of host IDs on x-axis
plt.tight_layout()
plt.ylim(top = 100)
plt.show()

#%% [markdown]------------------------------------------------------------
# As visible from above plot, The hosts with atleast 2 properties dominate the airbnb market meaning there are majority number of hosts that own 2 properties or even 5 properties at best. 
#and the graph seems to be uneven for displaying number of properties owned by hosts that own higher number of properties as an individual. so in the second graph here, 
# we have tried to shrink down the scale to give a better visualization for the hosts that own larger amounts of properties by count.

#%%
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

#%% [markdown]
# Then we tried to group keywords from the property type variable to see how many properties belong to a certain type as it will help us visualize the most commonly available listings in the airbnbs of washington dc.
#
# We applied string operations to find counts of properties with such keywords and plotted a bar graph for the same which revealed us the information that most the airbnb's are listed on an **ENTIRE** basis and quite significantly so. 
# It was immediately followed by count of rooms and private rooms listed in a otherwise owned and in-use living space for the prior/existing owner/rentee. All the other types of properties were very less in number.
# The x-axis contains all the types of property while the y-axis contains the number of properties that particular property type.
# This plots shows us the inclination of people to stay in these three property type which is being taken into account by the hosts.


#%%
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

#%% [markdown]
# The Correlation plot essentially allows us to visualize 2 things - First, correlation of feature set variables with the target variable and Second, correlation amongst the features.
# It is important to know this information priorhand as it sets the foundation for solving the problem at hand. On both the x and y axis we have the features and their correlation values in the intersecting blocks of lines forming the squares.
# The values of the correlation varies from -1 to 1. 1 represents strong positive and -1 represents strong negative correlation. values closer to zero represent weaker to no correlation.
# Looking at correlation matrix/plot is part of the standard procedure to have a first look at the problems that may or may not arise with the large dimensional data like this Airbnb data. One of the very important thing that we need to worry about is the problem of multi-collinearity. 
# Multicollinearity occurs when we have variables that are showing correlation amongsts themselves in the feature set. While it may be bothersome in first glance but it should not be a basis for a conclusive decision that our data now has multicollinearity.
# 
# On visual inspection there is correlation present between feature set variables but we can conclude that it is in fact true based on variance inflation factor. Later on in the modelling phase, we will take a look at Variance Inflation factor(V.I.F.) to further determine if the feature space needs some reduction for the baseline modelling itself by cutting down on features that are showing high correlation with each other.
# The concept is simple, for the multicollinearity to be proven, the metric VIF number should be 5 and above in terms of its scaler value.
# In this project, we will analyze the VIF number when we are making models to see if we have to reduce our feature space and tune the models to remove multicollinearity and progress towards more accurate results. 

#%%
plt.figure(figsize=(16, 8))
sns.scatterplot(x='latitude', y='longitude', hue='Ward', data=df, palette='Set1', alpha=0.7)
plt.title('Airbnb Listings: Latitude vs Longitude with Region')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.legend(title='Ward Numbers')
plt.grid(True)
plt.show()

#%% [markdown]
#We, in this project, were able to feature engineer a variable called Ward where we grouped up the Washington DC's neighbourhoods into different wards as per the D.C City Authority website linked as follows -
# [Wards in Washington D.C.][https://publicsafety.fandom.com/wiki/List_of_neighborhoods_of_the_District_of_Columbia_by_ward]. This allowed us to explore another dimension of the EDA which is plotting the data over a pseudo map to have a look at the Airbnb distribution by ward variable.
# This allowed us to make a scatter plot as both latitude and longitude variables can be plotted best using that style of plot with latitude on x-axis and longitude on y-axis.
# With few outliers, we saw almost all regions in a ward grouped together on the pseudo map that we had generated to see which airbnbs belong to which ward which can be a useful information for someone local to the city to book the airbnbs for their acquantainces coming from around the country 
# and even around the world.  

#%%
plt.figure(figsize=(16, 8))
sns.scatterplot(x='latitude', y='longitude', hue='host_identity_verified', data=data, palette='Set2', alpha=0.7)
plt.title('Latitude vs Longitude of an AirBnB with Host Identity')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.legend(title='Verified Identity(True/False)')
plt.grid(True)
plt.show()

#%% [markdown]
#To build up on the previous scatterplot and to answer a question of security that is usually the first concern for someone trying to book an Airbnb, we went ahead and plotted another scatter plot that represented the pseudo map of the DC area.
# only this time, we had the security concern that needed to be answered at the top of our priority order. There are times when we see a listing with all the underwhelming and pixelated photos of the property which gives us the scary feeling that the listing might be fake
# It is not favorable for us as a customer to put money and trust in a listing that may very well not even exist at all and might just be fake at best.
# 
# That is where this scatter plot helps us visualize that data. Here we plot the same latitude and longitude but with a different hue and that being of whether the owner is verified airbnb host or not a verified host.
# This shows us and help us assure the user to steer them towards the verified hosts listing real properties to ensure the customer satisfaction at its best. We find that most of the hosts on Airbnb are real and verified in this data from 2013 to the present.


#%%
##### Plot the New Discretized Review Scores Variable #####
fig, ax = plt.subplots()
bars = ax.bar(listings.groupby(by=['Target'], as_index=False)['review_scores_rating'].count()['Target'], listings.groupby(by=['Target'], as_index=False)['review_scores_rating'].count()['review_scores_rating'])

plt.title('Distribution of Discretized Review Score Variable')
plt.xlabel('Rating Above Average: 1 - Yes, 0 - No')
plt.ylabel('Count of Reviews')

ax.bar_label(bars)


## Host Behavior Rates Box Plots
host_rates = ['host_response_rate', 'host_acceptance_rate']
host_rate_labels = ['Response', 'Acceptance']

plt.figure(figsize=(10, 6))
ax = sns.boxplot(data=df[host_rates])
plt.xticks(rotation=0) 
plt.title("Distributions of Host Behavior Rates")
plt.xlabel('Type of Rate')
plt.ylabel('Rate')

ax.set_xticklabels(host_rate_labels)

plt.tight_layout()
plt.show()
# These two box plots show the distributions of host response and acceptance rates. The types of host behavior are indicated on the x-axis and the response rate is indicated on the y-axis. Both distributions are left-skewed. Most hosts are very response and accept the majority of prospective customers. Box plots are ideal for concisely illustrating the distributions of variables such as these.

## Rental Price Box Plot
plt.figure(figsize=(10, 6))
ax = sns.boxplot(data=df[['price']], orient='h', color='lightgreen')
ax.set_yticklabels([])
plt.title("Distribution of Rental Prices")
plt.xlabel('Price ($)')
plt.ylabel(' ')
plt.tight_layout()
plt.show()
# Rental price is one of the listing characteristics most valued by AirBnB customers. This box plot shows the distribution of AirBnB rental prices in DC with price (in dollars) on the x-axis. The distribution is right-skewed, meaning that there are a few exceptionally expensive rentals available. However, most rentals are clusterd together around a lower price point.

## Mean Price per Ward Bar Plot
plt.figure(figsize=(12, 6))
sns.barplot(x='Ward', y='price', data=df.groupby('Ward')['price'].mean().reset_index())
plt.title("Mean Rental Price per Ward")
plt.xlabel("Ward Number")
plt.ylabel("Mean Price ($)")
plt.xticks()
plt.tight_layout()
plt.show()
# Digging deeper into the price data, we generate a bar plot showing the mean rental price per ward with mean rental price (in dollars) on the y-axis and ward number on the x-axis. Ward 2 is noticeably more expensive than other wards. Mean rental price seems to be somewhat effected by location. 

## Median Price per Ward Bar Plot
plt.figure(figsize=(12, 6))
sns.barplot(x='Ward', y='price', data=df.groupby('Ward')['price'].median().reset_index())
plt.title("Median Rental Price per Ward")
plt.xlabel("Ward Number")
plt.ylabel("Median Price ($)")
plt.xticks()
plt.tight_layout()
plt.show()
# When the median rental prices of each ward are compared with one another via a similar bar plot (this time for median instead of mean), we see that Ward 2 is no longer significantly more expensive than the other wards. This suggests that there are a few abnormally expensive listings available for rent in Ward 2 (which skew the distribution right and drag the mean up). This is similar to what we observed in the distribution displayed in the general rental price box plot. It seems that median rental price is less effected by location than mean rental price.

## Review Scores Box Plot
plt.figure(figsize=(10, 6))
ax = sns.boxplot(data=df[['review_scores_rating']], orient='h', color='yellow')  
ax.set_yticklabels([])
plt.title("Distribution of Review Scores")
plt.xlabel('Review Scores (Out of 5)')
plt.ylabel(' ')
plt.tight_layout()
plt.show()
# This box plot depicts the distribution of review scores with review score (out of 5) on the x-axis. The distribution is left-skewed with the majority of scores in the 4.7 - 4.9 range. This posed a challenge for us, considering that our initial plan had been to use multiple linear regression to model which variables most determined listings' review scores. However, as the above distribution shows, there is very little variation in the target variable (review score). Therefore, this isn't a suitable problem for multiple linear regression. Instead, we decided to treat our subsequent analysis as a classification problem. More specifically, we investigated whether it's possible to predict whether a listing will have an above average or a below average review score based on characteristics of the listing and the host's behavior. 


#%%
##### Initiate Train-Test Split #####

x = listings.drop(columns=['review_scores_rating','Target'])
y = listings['Target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state=42)

#%%
models = {'Logistic Regression':LogisticRegression(),'Decision Tree':DecisionTreeClassifier(random_state=1), 'Naive Bayes':GaussianNB(),
          'Support Vector':SVC(kernel='rbf'), 'Linear SVC':LinearSVC(), 'Nu-Support Vector':NuSVC(), 
          'KNN':KNeighborsClassifier(), 'Nearest Centroid':NearestCentroid(), 
          'Gradient Booster':GradientBoostingClassifier(), 
          'AdaBoost':AdaBoostClassifier(), 'XGBoost':xgb.XGBClassifier(),
          'Random Forest': RandomForestClassifier(n_estimators = 500)
          }

#models = {'Gradient Booster':GradientBoostingClassifier()}

for name, model in models.items():
    
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        vif = 1 / (1 - (r2_score(y_test, model.predict(x_test))))
        cf_matrix = confusion_matrix(y_test, y_pred)
        matrix_plot = ConfusionMatrixDisplay(cf_matrix)
        matrix_plot.plot()
        plt.suptitle(f'Confusion Matrix for {name}',y=1.05)
        plt.title('Class 0: Below Average \n Class 1: Above Average')
        print(f'VIF score for {name} is {vif}')
        # feature_importances = pd.DataFrame(model.feature_importances_, columns=['important_features'])
        # feature_importances.index = x.columns
        # plt.figure(figsize=(10, 6))
        # sns.barplot(data=feature_importances, x='important_features', y=feature_importances.index)
        # plt.title('Variable Importance')
        # plt.xlabel('Relative Importance')
        # plt.ylabel('Features')
        # plt.show()

def specificity1(y_test,y_pred):
    """The following function creates a scorer for the Specificity (True Negative) value. 
    The Specificity value is a measure of our model correctly identifying True Negatives, 
    and summarizes how well the negative class was predicted.""" 
    
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
    
    if (tn+fp) == 0 or tn == 0:
        true_negative_rate = 0
    else:
        true_negative_rate = tn/(tn+fp)
    
    return true_negative_rate

def neg_predictive_value1(y_test,y_pred):
    """The following function creates a scorer for the Negative Predictive value. 
    The negative predictive value is a measure of how well a model makes negative predicictions. 
    When our model predicts that a student will not be retained, 
    the negative predictive value provides the percentage of time our model makes a correct prediction.""" 

    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
    
    if (tn+fn) == 0 or tn == 0:
        neg_predictive_val = 0
    else:
        neg_predictive_val = tn/(tn+fn)
    
    return neg_predictive_val

def cross_validation_scores(model, cv_method, metrics, xm, ym):
    from sklearn.model_selection import cross_validate
    x_train, x_test, y_train, y_test = train_test_split(xm, ym, test_size=0.2, random_state=1)
    model.fit(x_train, y_train)
    cv_results = cross_validate(model, x_test, y_test, scoring=metrics, cv=cv_method)
    
    calculated_metrics = {}
    for name in metrics.keys():
        calculated_metrics[name] = cv_results[f'test_{name}']
    
    names = []
    mins = []
    means = []
    meds = []
    maxes = []
    stdvs = []

    for key, value in calculated_metrics.items():
        names.append(key)
        mins.append(value.min())
        means.append(value.mean())
        meds.append(np.median(value))
        maxes.append(value.max())
        stdvs.append(value.std())
        
    cv_df = pd.DataFrame({'Metric':names, 'Min':mins, 'Mean':means, 'Median':meds, 'Max':maxes, 'Stdv':stdvs})
    return cv_df

#%% [markdown]

#In the above dictionary of the model objects, we saw that for each model, we have the VIF number metric being generated as discussed in the Exploratory data analysis.
# As it is visible above, all the VIF number values are well below 1 which makes the features almost uncorrelated. This allows us to proceed with the overall feature set for the baseline modeling and comparision as per the scope of this project.

#%%

metrics = {'balanced_accuracy':make_scorer(balanced_accuracy_score), 'f1_score':make_scorer(f1_score), 'precision':make_scorer(precision_score), 'recall':make_scorer(recall_score), 'npv':make_scorer(neg_predictive_value1), 'tnr':make_scorer(specificity1)}

lgr_results = cross_validation_scores(model=models['Logistic Regression'], cv_method=StratifiedKFold(n_splits=10), metrics=metrics, xm=x, ym=y) 
dt_results = cross_validation_scores(model=models['Decision Tree'], cv_method=StratifiedKFold(n_splits=10), metrics=metrics, xm=x, ym=y)
rf_results = cross_validation_scores(model=models['Random Forest'], cv_method=StratifiedKFold(n_splits=10), metrics=metrics, xm=x, ym=y)
svc_results = cross_validation_scores(model=models['Support Vector'], cv_method=StratifiedKFold(n_splits=10), metrics=metrics, xm=x, ym=y)
nb_results = cross_validation_scores(model=models['Naive Bayes'], cv_method=StratifiedKFold(n_splits=10), metrics=metrics, xm=x, ym=y)
linsvc_results = cross_validation_scores(model=models['Linear SVC'], cv_method=StratifiedKFold(n_splits=10), metrics=metrics, xm=x, ym=y)
nusupp_results = cross_validation_scores(model=models['Nu-Support Vector'], cv_method=StratifiedKFold(n_splits=10), metrics=metrics, xm=x, ym=y)
knn_results = cross_validation_scores(model=models['KNN'], cv_method=StratifiedKFold(n_splits=10), metrics=metrics, xm=x, ym=y)
nrcctd_results = cross_validation_scores(model=models['Nearest Centroid'], cv_method=StratifiedKFold(n_splits=10), metrics=metrics, xm=x, ym=y)
gbc_results = cross_validation_scores(model=models['Gradient Booster'], cv_method=StratifiedKFold(n_splits=10), metrics=metrics, xm=x, ym=y)
ada_results = cross_validation_scores(model=models['AdaBoost'], cv_method=StratifiedKFold(n_splits=10), metrics=metrics, xm=x, ym=y)
xgb_results = cross_validation_scores(model=models['XGBoost'], cv_method=StratifiedKFold(n_splits=10), metrics=metrics, xm=x, ym=y)

print(f'Logistic regession accuracy metrics\n{lgr_results}\n')
print(f'Decision Tree accuracy metrics\n{dt_results}\n')
print(f'Random Forest Classifier accuracy metrics\n{rf_results}\n')
print(f'Support Vector Classifier accuracy metrics\n{svc_results}\n')
print(f'Naive Bayes Classifier accuracy metrics\n{nb_results}\n')
print(f'Linear SVC Classifier accuracy metrics\n{linsvc_results}\n')
print(f'Nu Support Classifier accuracy metrics \n{nusupp_results}\n')
print(f'K Nearest Neighbours accuracy metrics\n{knn_results}\n')
print(f'Nearest Centroid accuracy metrics\n{nrcctd_results}\n')
print(f'Gradient Boosting Classifier accuracy metrics\n{gbc_results}\n')
print(f'ADA Classifier accuracy metrics\n{ada_results}\n')
print(f'XG Boost accuracy metrics\n{xgb_results}\n')

#%% [markdown]
# We can now see two metrics per model for all the 12 models but the models to focus on after going through all the results are the ones as follows - 
#
# 1. Decision tree Classifier
# 2. Naive Bayes Classifier
# 3. Gradient Boosting Classifier
#
# The reason why these three models are chosen is based on the findings avaiable in the confusion matrix and the cross validation table generated above where we find these models to be relatively better 
# amongst the 12 models we originally began with. The reasoning for each of these 3 models being better than the rest along with the interpretation of their metrics is shown below - 
#
#Decision tree Classifier
#
# Content here
#
#Naive Bayes Classifier
#
# Content here
#
#Gradient Boosting Classifier
#
# Gradient Boosting classifiers are the classification machine learning models that uses the ensembling technique which is essentially combining several base models to make a one final model that is 
# able to optimally predict the target classes. the idea is divided into 3 steps or parts - 
# 
# 1. **Bagging** - The Bagging step fitting multiple decision trees on multiple data samples generated randomly from given data and averaging the predictions.
# 
# 2. **Stacking** - This step fits different variations of the model on the same data and uses yet another sub-model of sorts to learn how to combine the predictions in the most correct way possible.
# 
# 3. **Boosting** - This is where the combination or merging happens where the model where it adds the members of the ensemble before in a sequential manner that corrects the wrong predictions made in the previous step and outputs a weighted average of the predictions. 
#
#
# When our data was passed through this model, we found that it was able to predict the positive outcomes more accurately comapred to all the models ehich meant that it showed the highest mean precision score of 0.77 amongst all the models and amongst all the cross validations ran on this model alone. 
# In our case the positive outcome meant the ratings are above 4.5 for an Airbnb. As for the another important metric called recall rate which tells us the model's ability to identify all positive instances correctly and this model showed a recall score of 0.88 which was still the highest amongst the 
# models which meant that it was better able to predict the Airbnbs with higher than 4.5 ratings using the set of features passed into the model. It also had the highest accuracy score to further aid to our pursuit of finding the best model.
# Although this model had a slightly higher standard deviation while cross validating the true negative rate (which is lower in the 2 other better models - Decision Tree and Naive bayes), it still is able to help us answer the smart question which is to find which characteristics of Airbnbs are important in predicting the ratings(in the most accurate way possible). 
# 
#%%
#Feature importance plot for the best model

feat_imp = permutation_importance(models['Gradient Booster'], x_train, y_train)

# Create a DataFrame with the feature importances
dfimp = pd.DataFrame(feat_imp.importances_mean, columns=['important_features'])
dfimp.index = x_train.columns
dfimp = dfimp.sort_values('important_features',ascending = False)

# Plot the feature importances
plt.figure(figsize=(12, 8))
sns.barplot(data=dfimp, x='important_features', y=dfimp.index)
plt.title('Feature Importance')
plt.xlabel('Relative Importance')
plt.ylabel('Features')

plt.show()
#%% [markdown]
# 
# The plot for the feature importance shows that the features like price, host importance rate, beds, accomodation capacity, ward of the airbnb, response and acceptance rate of the host are the most valued core features in deciding the rating for an airbnb.
# Apart from them amenities like Parking Spaces, Washing/Drying, Gym, Kitchen, Entertainment and Privacy also contribute to the ratings being above and below 4.5 out of a rating scale of 5 respectively.
## 5. Conclusion  
# 1. The Three algorithms that have performed relatively better than the bunch of models that were tested in this project are - Gradient Boosting Classifier, Decision Tree and Naive Bayes Classifier.
#
#    **a**. The Naive Bayes Classifier had the highest true negative rate and precision score
#
#    **b**. The Decision Tree Classifier had relatively high metrics with the smallest standard deviations
#
#    **c**. The Gradient Boost Classifier had the highest balanced accuracy, recall, and f1 score. It also had the highest negative predictive value, although the standard deviation was relatively higher
#
# 2. To answer our SMART question, actionable factors influencing a listing’s rating include the listing price, the host’s response rate, the host’s response time, and amenities such as free parking, exercise and gym equipment, free washer and dryer, and extraordinary kitchen appliances. 
#
# 3. Hosts who want to increase their ratings should focus on fast response times and providing adequate parking, laundry facilities, exercise equipment, and additional kitchen appliances, among others
#
# 4. This project allowed us to learn, explore and perform predictive analytics on AirBnb's listings from 2013 till present day.
# It also has a lot of future Contingencies like analysis of the hyperparameter tuned models and cross verify the outcomes with stepwise feature reductions.
#
# 5. It is also possible to precdict the continuous variable price and predict the prices for the listings based on their features which will be helpful for the up and coming hosts signing up on AirBnb to list their properties in a way that it brings in more business and helps the business grow better. 
#
# 6. There is a room for us to Utilize Explainable AI libraries such as SHAP to gain a deeper understanding of our models by understanding the probability of each features ability to predict the target variable.
#
# 7. We can employ multi layer perceptrons to see if the predictive modelling results are improving through this deep learning approach and then we can draw a graphical compparision between the models through interactive plots.
#
#
#
## 6. References - 
#
# 1. Evan Lutins, (2017). Ensemble Methods in Machine Learning: What are They and Why Use Them?
# https://towardsdatascience.com/ensemble-methods-in-machine-learning-what-are-they-and-why-use-them-68ec3f9fef5f 
#
# 2. Jason Brownlee (2021). A Gentle Introduction to Ensemble Learning Algorithms
# https://machinelearningmastery.com/tour-of-ensemble-learning-algorithms/
#
# 3. Angelica Lo Duca (2021). How to Write a Scientific Paper from a Data Science Project.
# https://towardsdatascience.com/how-to-write-a-scientific-paper-from-a-data-science-project-62d7101c9057
#
# 4. Tony Yiu(2019). Understanding Random Forest.
# https://towardsdatascience.com/understanding-random-forest-58381e0602d2
#
# 5. http://insideairbnb.com/washington-dc/
#
# 6. Owens, J. (2020). An Analysis of AirBnB in Washington, DC. 
# https://medium.com/@jessie.owens2/an-analysis-of-airbnb-in-washington-dc-8013cfef7379

# %%
