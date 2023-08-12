#%% [markdown]
# Import Packages and Data 

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import GaussianNB

#%% [markdown]
# Load the Data and View Summary of Data Types and Columns
data = pd.read_csv('DC-AirBnB-Listings.csv')
print(data.columns)
print(f'The file has {data.shape[0]} rows and {data.shape[1]} features.')
print(data.dtypes)
data.head()

# %% [markdown]
# Subset the DataFrame to get the Features 
df = data[['host_since', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost', 'neighbourhood_cleansed', 'neighbourhood', 'neighbourhood_group_cleansed', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'price', 'instant_bookable', 'review_scores_rating', 'latitude', 'longitude', 'amenities']]

#%% [markdown]
# Check for Duplicates 
print(df.duplicated().sum())
df[df.duplicated()]

df.drop_duplicates(keep='last', inplace=True)

# %% [markdown]
# Check for NAs 
print(df.isna().sum())
df.drop(columns=['bathrooms', 'neighbourhood_group_cleansed', 'neighbourhood'], inplace=True) # Drop columns with all NA values
df = df[~df['review_scores_rating'].isna()] # Ensure there are no NAs for target variable

#%% [markdown]
# Check for Unique Values and Data Types for Columns with NAs 
nas = ['host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost', 'bedrooms', 'beds']
    
def get_value_counts(df, column):
    print(f"Column Name: {column}")
    print(f"Column Data Type: {df[column].dtype}")
    print(f"Number of Unique Values: {df[column].nunique()}")
    print(df[column].value_counts())
    
for column in nas:
    get_value_counts(df, column)

#%%
# Part 1: Imputation for Columns with NAs
#
####### First we will convert the columns to numeric variables 
#
####### Host Response Time: Categorical
#
####### Host Response Rate: Percentage as a String
#
####### Host Acceptance Rate: Percentage as a String
#
####### Host is Superhost: Boolean as a String
#
####### Bedrooms: Float
#
####### Beds: Float

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
        
#%% [markdown]
# Part 2: Imputation for Columns with NAs #####
####### Now we will replace the NA values with the column means. 

def impute_mean(df, column):
    df[column].fillna(df[~df[column].isna()][column].mean(), inplace=True)

impute_mean(df, 'host_response_time')
impute_mean(df, 'host_response_rate')
impute_mean(df, 'host_acceptance_rate')
impute_mean(df, 'host_is_superhost')
impute_mean(df, 'bedrooms')
impute_mean(df, 'beds')
#%% [markdown]
#Part 1: Feature Engineering for Amenities Variable #####
####### First we will create a list of all amenities and get the number of unique amenities. 

df['amenities'] = df['amenities'].apply(lambda x: ast.literal_eval(x))
all_amenities = []

for idx in df.index:
    for item in df['amenities'][idx]:
        if item not in all_amenities:
            all_amenities.append(item)

all_amenities_arr = np.array(all_amenities)
print(f'There are {len(np.unique(all_amenities_arr))} unique amenities listed.')
      
#%% [markdown]
# Part 2: Feature Engineering for Amenities Variable
####### Next we will generate amenity groups given there are over 2,000 amenities. The amenities groups represent groupings of common amenities by use-case. 

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

#%% [markdown]
#Part 3: Feature Engineering for Amenities Variable 
######## Finally we will create the variables for amenity groups and add them to the dataframe. The below for loop checks that for each cell in the Amenities column, any of the amenities mentioned are in the previously defined amenities groups. 

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

#%% [markdown]
# Distribution of Amenities Variables
####### The below code creates 12 subplots and generates a total of 11 bar plots to show the distribution of the new Amenities variables. 

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


#%% [markdown]
# Part 1: Feature Engineering for Neighbourhood Variable #
######## First we will create a list of all neighbourhoods and get the number of unique neighbourhoods. 

all_neighborhoods = []

for idx in df.index:
    neighborhood_list = [x.strip() for x in df['neighbourhood_cleansed'][idx].split(',')] 
    for item in neighborhood_list:
        if item not in all_neighborhoods:
            all_neighborhoods.append(item)

all_neighborhoods_arr = np.array(all_neighborhoods)
print(f'There are {len(np.unique(all_neighborhoods_arr))} unique neighborhoods listed.')

#%% [markdown]
# Part 2: Feature Engineering for Neighbourhood Variable 
####### Given that there are over 120 neighborhoods mentioned in the dataframe, we will sort the neighborhoods mentioned in one of DC's eight wards. 
####### Reference: https://publicsafety.fandom.com/wiki/List_of_neighborhoods_of_the_District_of_Columbia_by_ward#Ward_7

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

####### Get neighborhoods not currently assigned to any ward
list(set(all_neighborhoods) - set(all_wards)) 

ward1.append('cardozo/shaw')
ward2 = ward2 + ['connecticut avenue/k street', 'sheridan']
ward3.append('van ness')
ward4.append('north portal estates')
ward5 = ward5 + ['university heights', 'north capitol street']
ward6 = ward6 + ['union station', 'stanton park', 'buzzard point', 'fort mcnair']
ward7 = ward7 + ['mahaning heights', 'summit park', 'grant park', 'fairmount heights']
ward8 = ward8 + ['near southeast', 'historic anacostia', 'fort stanton']

#%% [markdown]
# Part 3: Feature Engineering for Neighbourhood Variable
####### Finally we will create the variable for wards and add them to the dataframe. The below for-loop takes each cell of the "neighborhood cleasned" column and checks if any of the neighborhoods mentiond in the cell are in any of the ward lists, starting from Ward 1 and ending with Ward 2. 

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

#%% [markdown]
# Generate Plot to Show Distribution of Neighborhoods Variables
####### The below plot shows the distrbution of the new Wards variable. 

plt.bar(df.groupby(by='Ward', as_index=False)['price'].count()['Ward'],df.groupby(by='Ward', as_index=False)['price'].count()['price'])
plt.title('Distribution of Engineered Neighborhood Variable')
plt.xlabel('Ward')
plt.ylabel('Count of Listings')

#%% [markdown]
# Convert Room Type to Ordinal Variable

df['Privacy'] = df['room_type'].replace({'Entire home/apt':1, 'Private room':2, 'Shared room': 3, 'Hotel room':4})

# Convert Instant Bookable to Boolean Integer
df['instant_bookable'] = df['instant_bookable'].replace({'t':1, 'f':0})

#%% [markdown]
# Convert Price to Float

prices = []
for price in df['price']:
    if "," in price:
        prices.append(float(''.join(price[1:].split(','))))
    else:
        prices.append(float(price[1:]))
        
df['price'] = prices

#%% [markdown]
# Final Dataframe
######## The final dataframe has 5,180 rows and 24 features. We are now ready to begin modeling. 

listings = df[['host_response_time', 'host_response_rate',
       'host_acceptance_rate', 'host_is_superhost', 'accommodates', 'bedrooms', 'beds', 'instant_bookable', 'price', 'Clothing', 'Entertainment', 'Exercise',
       'Kitchen', 'Luxury', 'Outdoor', 'Parking', 'Security', 'Streaming',
       'Technology', 'Views', 'Ward', 'Privacy', 'review_scores_rating']]


# %% [markdown]
# Review Scores Rating Discretization
####### The review scores rating variable had very little variation. Its standard deviation was 0.50. The lack of variation made it more difficult to predict review score ratings. Furthermore, the current variables in the listings file did not seem to be good predictors for the ratings, as they had very small coefficients and the r2 score was low.
#
####### To remedy this, the review scores rating variable was changed to a binary target. In turn, we will now try to look at the problem as a classification issue. 

listings['review_scores_rating'].plot(kind='hist', bins=35)
plt.title('Histogram of Review Scores Rating')
plt.xlabel('Review Scores Rating')
plt.xticks([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5])
plt.axvline(listings['review_scores_rating'].mean(), color='red', linestyle='dashed', linewidth=1)

review_mean = listings['review_scores_rating'].mean()
target = []

for rating in listings['review_scores_rating']:
    if rating >= review_mean:
        target.append(1)
    else:
        target.append(0)

listings['Target'] = target

#%% [markdown]
# Plot the New Discretized Review Scores Variable 
####### The new review scores target variable checks if a listing's review score is above the average score (1) or below the average score (0). 
#
####### As seen in the plot, there is a slight imbalance between the two classes, but not enough to make classification modeling impossible. 
fig, ax = plt.subplots()
bars = ax.bar(listings.groupby(by=['Target'], as_index=False)['review_scores_rating'].count()['Target'], listings.groupby(by=['Target'], as_index=False)['review_scores_rating'].count()['review_scores_rating'])

plt.title('Distribution of Discretized Review Score Variable')
plt.xlabel('Rating Above Average: 1 - Yes, 0 - No')
plt.ylabel('Count of Reviews')

ax.bar_label(bars)

#%% [markdown]
# Initiate the Train-Test Split
####### Here the train-test split is initaited. We will use an 80-20 ratio for the split. 
x = listings.drop(columns=['review_scores_rating', 'Target'])
y = listings['Target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

#%% [markdown]
# Generate Modeling Results and Confusion Matrices
####### The below code tests several classification models on our dataset. For each model, a confusion matrix is generated to allow us to see how well each model predicts each class. 

models = {'Decision Tree':DecisionTreeClassifier(random_state=1), 'Naive Bayes':GaussianNB(), 
          'Support Vector':SVC(), 'Linear SVC':LinearSVC(), 'Nu-Support Vector':NuSVC(), 
          'KNN':KNeighborsClassifier(), 'Nearest Centroid':NearestCentroid(), 
          'Gradient Booster':GradientBoostingClassifier(), 
          'AdaBoost':AdaBoostClassifier(), 'XGBoost':xgb.XGBClassifier()}

for name, model in models.items():
    
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        cf_matrix = confusion_matrix(y_test, y_pred)
        matrix_plot = ConfusionMatrixDisplay(cf_matrix)
        matrix_plot.plot()
        plt.suptitle(f'Confusion Matrix for {name}',y=1.05)
        plt.title('Class 0: Below Average \n Class 1: Above Average')



# %% [markdown]
####### The two best performing models based on the previous cell were the Decision Tree classifier and the Naive Bayes classifier. We will further assess these two models by generating cross-validation scores for them. 

dt = DecisionTreeClassifier(random_state=1)
nb = GaussianNB()

# %% [markdown]
# Define Functions to Calculate Specificity and Negative Predictive Value
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

#%% [markdown]
# Define Function to Calculate Cross Validation Scores
####### The below function calculates the cross validation scores for a specified list of metrics. 
def cross_validation_scores(model, cv_method, metrics, xm, ym):
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
# Calculation of Cross Validation Scores: Decision Tree
####### Here cross validation scores are calculated for the Decision Tree model. A Stratified KFold is used given the class imbalance for the target variable. 

metrics = {'balanced_accuracy':make_scorer(balanced_accuracy_score), 'f1_score':make_scorer(f1_score), 'precision':make_scorer(precision_score), 'recall':make_scorer(recall_score), 'npv':make_scorer(neg_predictive_value1), 'tnr':make_scorer(specificity1)}

dt_results = cross_validation_scores(model=dt, cv_method=StratifiedKFold(n_splits=10), metrics=metrics, xm=x, ym=y)
dt_results

# %% [markdown]
# Calculation of Cross Validation Scores: Naive Bayes
####### Here cross validation scores are calculated for the Naive Bayes model. A Stratified KFold is used given the class imbalance for the target variable. 

nb_results = cross_validation_scores(model=nb, cv_method=StratifiedKFold(n_splits=10), metrics=metrics, xm=x, ym=y)
nb_results
# %%
