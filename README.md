# PHASE 2 PROJECT
## FILM INDUSTRY ANALYSIS

## 1.0 BUSINESS UNDERSTANDING
The film industry is fast-paced,what's popular today might be a flop tomorrow. To succeed in this volatile market, our goal is to create movies with broad audience appeal that will be profitable. 

## 2.0 OBJECTIVES

### 2.1 MAIN OBJECTIVE
`To provide data-driven insights into key factors influencing movie production and profitability`.

### 2.2 SPECIFIC OBJECTIVE
* To determine the top 10 movie production studios by number of movies produced and the profit made by the movies produced.
* To determine the best season to release a movie.
* To determine the relationship between the production budget and profit made.
* to determine the genres with the most produced movies and those with the highest rated movies.
* To determine the top movie directors overall and per genre.
* to determine the highest rated movies


## 3.0 DATA UNDERSTANDING
# IMDb databa
The database has data on audience preferences through ratings and reviews,cast and crew that allows the study of successful projects to understand market trends and genres.
# The Numbers data 
Contains data that allows for the calculation of a movie's profitability by comparing its production budget to its worldwide gross.
This is helping in determining optimal budget ranges for financial success.
# Box Office Mojo
The data provides a detailed breakdown of a film's domestic and foreign gross revenue crucial for understanding a movie's market performance beyond its worldwide total, allowing a company to assess a film's success in specific geographical markets. 

## 4.0 EXPLORATORY DATA ANALYSIS

### 4.1 Importing Required Libraries


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import itertools 
from numbers import Number
from collections import defaultdict
import sqlite3
import os
import re
import warnings
warnings.filterwarnings('ignore')
```

### 4.2 Loading The Datasets


```python
#Loading box office mojo data
bom_movie= pd.read_csv('bom.movie_gross.csv')
bom_movie
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>studio</th>
      <th>domestic_gross</th>
      <th>foreign_gross</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Toy Story 3</td>
      <td>BV</td>
      <td>415000000.0</td>
      <td>652000000</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alice in Wonderland (2010)</td>
      <td>BV</td>
      <td>334200000.0</td>
      <td>691300000</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Harry Potter and the Deathly Hallows Part 1</td>
      <td>WB</td>
      <td>296000000.0</td>
      <td>664300000</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Inception</td>
      <td>WB</td>
      <td>292600000.0</td>
      <td>535700000</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Shrek Forever After</td>
      <td>P/DW</td>
      <td>238700000.0</td>
      <td>513900000</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3382</th>
      <td>The Quake</td>
      <td>Magn.</td>
      <td>6200.0</td>
      <td>NaN</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>3383</th>
      <td>Edward II (2018 re-release)</td>
      <td>FM</td>
      <td>4800.0</td>
      <td>NaN</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>3384</th>
      <td>El Pacto</td>
      <td>Sony</td>
      <td>2500.0</td>
      <td>NaN</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>3385</th>
      <td>The Swan</td>
      <td>Synergetic</td>
      <td>2400.0</td>
      <td>NaN</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>3386</th>
      <td>An Actor Prepares</td>
      <td>Grav.</td>
      <td>1700.0</td>
      <td>NaN</td>
      <td>2018</td>
    </tr>
  </tbody>
</table>
<p>3387 rows × 5 columns</p>
</div>




```python
#displaying tha data's information
bom_movie.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3387 entries, 0 to 3386
    Data columns (total 5 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   title           3387 non-null   object 
     1   studio          3382 non-null   object 
     2   domestic_gross  3359 non-null   float64
     3   foreign_gross   2037 non-null   object 
     4   year            3387 non-null   int64  
    dtypes: float64(1), int64(1), object(3)
    memory usage: 132.4+ KB
    


```python
#Loading the numbers data
movie_budgets=pd.read_csv('tn.movie_budgets.csv')
movie_budgets

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>release_date</th>
      <th>movie</th>
      <th>production_budget</th>
      <th>domestic_gross</th>
      <th>worldwide_gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Dec 18, 2009</td>
      <td>Avatar</td>
      <td>$425,000,000</td>
      <td>$760,507,625</td>
      <td>$2,776,345,279</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>May 20, 2011</td>
      <td>Pirates of the Caribbean: On Stranger Tides</td>
      <td>$410,600,000</td>
      <td>$241,063,875</td>
      <td>$1,045,663,875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Jun 7, 2019</td>
      <td>Dark Phoenix</td>
      <td>$350,000,000</td>
      <td>$42,762,350</td>
      <td>$149,762,350</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>May 1, 2015</td>
      <td>Avengers: Age of Ultron</td>
      <td>$330,600,000</td>
      <td>$459,005,868</td>
      <td>$1,403,013,963</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Dec 15, 2017</td>
      <td>Star Wars Ep. VIII: The Last Jedi</td>
      <td>$317,000,000</td>
      <td>$620,181,382</td>
      <td>$1,316,721,747</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5777</th>
      <td>78</td>
      <td>Dec 31, 2018</td>
      <td>Red 11</td>
      <td>$7,000</td>
      <td>$0</td>
      <td>$0</td>
    </tr>
    <tr>
      <th>5778</th>
      <td>79</td>
      <td>Apr 2, 1999</td>
      <td>Following</td>
      <td>$6,000</td>
      <td>$48,482</td>
      <td>$240,495</td>
    </tr>
    <tr>
      <th>5779</th>
      <td>80</td>
      <td>Jul 13, 2005</td>
      <td>Return to the Land of Wonders</td>
      <td>$5,000</td>
      <td>$1,338</td>
      <td>$1,338</td>
    </tr>
    <tr>
      <th>5780</th>
      <td>81</td>
      <td>Sep 29, 2015</td>
      <td>A Plague So Pleasant</td>
      <td>$1,400</td>
      <td>$0</td>
      <td>$0</td>
    </tr>
    <tr>
      <th>5781</th>
      <td>82</td>
      <td>Aug 5, 2005</td>
      <td>My Date With Drew</td>
      <td>$1,100</td>
      <td>$181,041</td>
      <td>$181,041</td>
    </tr>
  </tbody>
</table>
<p>5782 rows × 6 columns</p>
</div>



# 4.2 Data cleaning and Preparation.

# 4.2.1 Feature Analysis
This involves:
* Viewing the data to understand it
* Feature Engineering i.e removing commas, white spaces 
* Dropping unnecessary columns.
* Handling missing values.
* Checking for duplicates.
* Merging of dataframes.


```python
movie_budgets.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5782 entries, 0 to 5781
    Data columns (total 6 columns):
     #   Column             Non-Null Count  Dtype 
    ---  ------             --------------  ----- 
     0   id                 5782 non-null   int64 
     1   release_date       5782 non-null   object
     2   movie              5782 non-null   object
     3   production_budget  5782 non-null   object
     4   domestic_gross     5782 non-null   object
     5   worldwide_gross    5782 non-null   object
    dtypes: int64(1), object(5)
    memory usage: 271.2+ KB
    


```python
#Renaming the column named movie with title
movie_budgets= movie_budgets.rename(columns= {"movie": "title"})
movie_budgets.columns
```




    Index(['id', 'release_date', 'title', 'production_budget', 'domestic_gross',
           'worldwide_gross'],
          dtype='object')




```python
#Converting the 'domestic_gross' column in the movie_budgets to interger
movie_budgets['domestic_gross'] = pd.to_numeric(movie_budgets['domestic_gross'], errors='coerce')

#Converting the 'domestic_gross' column in the bom_movie DataFrame
bom_movie['domestic_gross'] = pd.to_numeric(bom_movie['domestic_gross'], errors='coerce')
```


```python
#Merging the 2 tables from movie_budgets and bom_movies
budgets_financials= pd.merge(movie_budgets, bom_movie, on=['title', 'domestic_gross'], how='outer')
```


```python
budgets_financials.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9167 entries, 0 to 9166
    Data columns (total 9 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   id                 5782 non-null   float64
     1   release_date       5782 non-null   object 
     2   title              9167 non-null   object 
     3   production_budget  5782 non-null   object 
     4   domestic_gross     3359 non-null   float64
     5   worldwide_gross    5782 non-null   object 
     6   studio             3382 non-null   object 
     7   foreign_gross      2037 non-null   object 
     8   year               3387 non-null   float64
    dtypes: float64(3), object(6)
    memory usage: 644.7+ KB
    


```python
#Statistical summary for categorical data
budgets_financials.describe(include='O')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>release_date</th>
      <th>title</th>
      <th>production_budget</th>
      <th>worldwide_gross</th>
      <th>studio</th>
      <th>foreign_gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5782</td>
      <td>9167</td>
      <td>5782</td>
      <td>5782</td>
      <td>3382</td>
      <td>2037</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2418</td>
      <td>7846</td>
      <td>509</td>
      <td>5356</td>
      <td>257</td>
      <td>1204</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Dec 31, 2014</td>
      <td>Unknown</td>
      <td>$20,000,000</td>
      <td>$0</td>
      <td>IFC</td>
      <td>1200000</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>24</td>
      <td>3</td>
      <td>231</td>
      <td>367</td>
      <td>166</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>




```python
#creating a copy of the data
financials_df= budgets_financials.copy(deep= True)

```


```python
#Feature engineering
#removing '$', '.00', and spaces  from the columnsand creating a list of the columns being modified
fin_modified= ['production_budget', 'domestic_gross', 'worldwide_gross','foreign_gross']

# Loop through the columns for modification
for col in fin_modified:
    # Remove commas, dollar signs, and the ".00" suffix
    # Then, convert the cleaned string values to a float
    financials_df[col] = (
        financials_df[col]
        .astype(str)
        .str.replace(",", "")
        .str.replace("$", "")
        .str.replace(".00", "")
        .astype(float)
    )

```


```python
#Showing skewness in the columns production_budget, domestic_gross, worldwide_gross
columns_to_plot = ['production_budget', 'domestic_gross', 'foreign_gross']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('KDE Plots to Show Data Skewness', fontsize=16)

    # Loop through the columns and create a plot for each
for i, col in enumerate(columns_to_plot):
        # The `kdeplot` function creates a density plot
        sns.kdeplot(data=financials_df, x=col, fill=True, ax=axes[i])
        axes[i].set_title(f'KDE Plot for {col.replace("_", " ").title()}')
        axes[i].set_xlabel(f'{col.replace("_", " ").title()}')
        axes[i].set_ylabel('Density')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
    
```


    
![png](FILM_COMPANY_files/FILM_COMPANY_21_0.png)
    



```python
#Filling in missing values using the median
# get the median
median_domestic_gross = financials_df["domestic_gross"].median()

financials_df["domestic_gross"].fillna(median_domestic_gross, inplace=True)

median_production = financials_df["production_budget"].median()
financials_df["production_budget"].fillna(median_production, inplace=True)

median_foreign = financials_df["foreign_gross"].median()
financials_df["foreign_gross"].fillna(median_foreign, inplace=True)
```


```python
financials_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9167 entries, 0 to 9166
    Data columns (total 9 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   id                 5782 non-null   float64
     1   release_date       5782 non-null   object 
     2   title              9167 non-null   object 
     3   production_budget  9167 non-null   float64
     4   domestic_gross     9167 non-null   float64
     5   worldwide_gross    5782 non-null   float64
     6   studio             3382 non-null   object 
     7   foreign_gross      9167 non-null   float64
     8   year               3387 non-null   float64
    dtypes: float64(6), object(3)
    memory usage: 644.7+ KB
    


```python
#filling missing values in worldwide gross by adding the sum of domestic and foreign gross

#Convert the 'domestic_gross' and 'foreign_gross' columns to numeric.
for col in ['domestic_gross', 'foreign_gross']:
    financials_df[col] = pd.to_numeric(
        financials_df[col].astype(str).str.replace('[$,]', '', regex=True),
        errors='coerce'
    )

#Identifying rows'worldwide_gross' is missing.
missing_gross= financials_df['worldwide_gross'].isnull()

# Filling missing 'worldwide_gross' values by adding foreign and domestic columns.
financials_df.loc[missing_gross, 'worldwide_gross'] = (
    financials_df.loc[missing_gross, 'domestic_gross'] +
    financials_df.loc[missing_gross, 'foreign_gross']
)
```


```python
#Converting 'release_date' to datetime objects
financials_df['release_date'] = pd.to_datetime(financials_df['release_date'])

# Mapping a month to a season
def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:  # December, January, February
        return 'Winter'

financials_df['release_season'] = financials_df['release_date'].dt.month.apply(get_season)

```


```python
#creating a new column named profit
financials_df['profit'] = financials_df['worldwide_gross'] - financials_df['production_budget']
```


```python
#dropping the id& year column
financials_df = financials_df.drop(['id','year'], axis=1)
```


```python
financials_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9167 entries, 0 to 9166
    Data columns (total 9 columns):
     #   Column             Non-Null Count  Dtype         
    ---  ------             --------------  -----         
     0   release_date       5782 non-null   datetime64[ns]
     1   title              9167 non-null   object        
     2   production_budget  9167 non-null   float64       
     3   domestic_gross     9167 non-null   float64       
     4   worldwide_gross    9167 non-null   float64       
     5   studio             3382 non-null   object        
     6   foreign_gross      9167 non-null   float64       
     7   release_season     9167 non-null   object        
     8   profit             9167 non-null   float64       
    dtypes: datetime64[ns](1), float64(5), object(3)
    memory usage: 644.7+ KB
    


```python
#filling the null values in studio and release date with unknown
my_col = {
    'studio': 'unknown',
    'release_date': 'unknown'
}

financials_df.fillna(value=my_col, inplace=True)
```


```python
financials_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9167 entries, 0 to 9166
    Data columns (total 9 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   release_date       9167 non-null   object 
     1   title              9167 non-null   object 
     2   production_budget  9167 non-null   float64
     3   domestic_gross     9167 non-null   float64
     4   worldwide_gross    9167 non-null   float64
     5   studio             9167 non-null   object 
     6   foreign_gross      9167 non-null   float64
     7   release_season     9167 non-null   object 
     8   profit             9167 non-null   float64
    dtypes: float64(5), object(4)
    memory usage: 644.7+ KB
    


```python
#Saving the data
financials_df.to_csv ('merged_financials_data.csv', index=False)
```

# 5.0 DATA ANALYSIS

## 5.1 UNIVARIATE DATA ANALYSIS 

### Top 10 Studios


```python
#top 10 studios that produced the most number of movies
filtered_studios = financials_df[financials_df['studio'] != 'unknown']

top_10_studios = filtered_studios['studio'].value_counts().head(10)

plt.figure(figsize=(8, 4))

colors = sns.color_palette('viridis', len(top_10_studios))
plt.bar(top_10_studios.index, top_10_studios.values, color=colors)
# title and labels for the axes.
plt.title('Top 10 Studios by Movie Count ', fontsize=16)
plt.xlabel('Studio', fontsize=12)
plt.ylabel('Number of Movies', fontsize=12)

plt.xticks(rotation=45, ha='right')
# grid lines for better readability.
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust plot layout to prevent labels from being cut off.
plt.tight_layout()

# Display the plot.
plt.show()


```


    
![png](FILM_COMPANY_files/FILM_COMPANY_35_0.png)
    



```python
#top 10 most protitable studios
filtered_df = financials_df[financials_df['studio'] != 'unknown']
studio_profit = filtered_df.groupby('studio')['profit'].sum().sort_values(ascending=False)

top_10_profitable_studios = studio_profit.head(10)


plt.figure(figsize=(7, 5))

colors = sns.color_palette('Blues_r', len(top_10_profitable_studios))
sns.barplot(x=top_10_profitable_studios.index, y=top_10_profitable_studios.values, palette=colors)

# Add a title and labels for the axes.
plt.title('Top 10 Most Profitable Studios', fontsize=16)
plt.xlabel('Studio', fontsize=10)
plt.ylabel('Total Profit', fontsize=12)

# Rotate x-axis labels to prevent them from overlapping.
plt.xticks(rotation=45, ha='right')

# Add grid lines for readability.
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust plot layout to prevent labels from being cut off.
plt.tight_layout()

# Display the plot.
plt.show()

most_profitable_studio_name = top_10_profitable_studios.index[0]
most_profitable_studio_profit = top_10_profitable_studios.values[0]

# Print the result.
print(f"The most profitable studio is '{most_profitable_studio_name}' with a total profit of ${most_profitable_studio_profit:,.2f}.")


```


    
![png](FILM_COMPANY_files/FILM_COMPANY_36_0.png)
    


    The most profitable studio is 'BV' with a total profit of $42,448,283,899.10.
    

## 5.2 BIVARIATE ANALYSIS


```python
#SEASON WITH THE HIGHEST GROSSING
season_gross = financials_df.groupby('release_season')['worldwide_gross'].sum().sort_values(ascending=False)
plt.figure(figsize=(7, 4))

colors = sns.color_palette('Greens', len(season_gross))
sns.barplot(x=season_gross.index, y=season_gross.values, palette=colors)
# Add a title and labels for the axes.
plt.title('Total Worldwide Gross by Release Season', fontsize=16)
plt.xlabel('Release Season', fontsize=12)
plt.ylabel('Total Worldwide Gross', fontsize=12)
plt.tight_layout()

plt.show()

```


    
![png](FILM_COMPANY_files/FILM_COMPANY_38_0.png)
    



```python
#scatter plot with a regression line showing the relation between the movie budget and profit
plt.figure(figsize=(8, 4))

sns.regplot(x='production_budget', y='profit', data=financials_df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})

plt.title('Relationship between Production Budget and Profit', fontsize=16)
plt.xlabel('Production Budget', fontsize=12)
plt.ylabel('Profit', fontsize=12)
plt.grid(True)

plt.gca().ticklabel_format(style='plain', axis='x')
plt.gca().ticklabel_format(style='plain', axis='y')
plt.show()

correlation, p_value = stats.pearsonr(financials_df['production_budget'], financials_df['profit'])

# Print the correlation result.
print(f"The Pearson correlation coefficient between Production Budget and Profit is: {correlation:.2f}")


```


    
![png](FILM_COMPANY_files/FILM_COMPANY_39_0.png)
    


    The Pearson correlation coefficient between Production Budget and Profit is: 0.44
    

# 6.0 IMBD DATA

## 6.1 EXPLORATORY DATA ANALYSIS OF THE IMDB DATA


```python
#RETRIEVING DATA FROM THE IMDB DATABASE AND THE COLUMN NAMES IN EACH TABLE
conn = sqlite3.connect('im.db')
cursor = conn.cursor()
 
# Get list of available tables
cursor.execute("""SELECT name
                  FROM sqlite_master
                  WHERE type='table';
                  """)
tables = cursor.fetchall()
print("Available tables:", tables)

# Loop through the tables and print the schema for each
for table_name in tables:
    table_name = table_name[0]  # The fetchall() result is a tuple, so we need the first element
    print(f"\n--- Schema for table: {table_name} ---")
    
    # Execute PRAGMA table_info() for the current table
    cursor.execute(f"PRAGMA table_info({table_name});")
    
    # Fetch all the column information
    schema = cursor.fetchall()
    
    # Print the schema details
    for column in schema:
        # The schema tuple contains (cid, name, type, notnull, dflt_value, pk)
        print(f"  - Column Name: {column[1]}, Type: {column[2]}, Not Null: {column[3]}, Primary Key: {column[5]}")


```

    Available tables: [('movie_basics',), ('directors',), ('known_for',), ('movie_akas',), ('movie_ratings',), ('persons',), ('principals',), ('writers',)]
    
    --- Schema for table: movie_basics ---
      - Column Name: movie_id, Type: TEXT, Not Null: 0, Primary Key: 0
      - Column Name: primary_title, Type: TEXT, Not Null: 0, Primary Key: 0
      - Column Name: original_title, Type: TEXT, Not Null: 0, Primary Key: 0
      - Column Name: start_year, Type: INTEGER, Not Null: 0, Primary Key: 0
      - Column Name: runtime_minutes, Type: REAL, Not Null: 0, Primary Key: 0
      - Column Name: genres, Type: TEXT, Not Null: 0, Primary Key: 0
    
    --- Schema for table: directors ---
      - Column Name: movie_id, Type: TEXT, Not Null: 0, Primary Key: 0
      - Column Name: person_id, Type: TEXT, Not Null: 0, Primary Key: 0
    
    --- Schema for table: known_for ---
      - Column Name: person_id, Type: TEXT, Not Null: 0, Primary Key: 0
      - Column Name: movie_id, Type: TEXT, Not Null: 0, Primary Key: 0
    
    --- Schema for table: movie_akas ---
      - Column Name: movie_id, Type: TEXT, Not Null: 0, Primary Key: 0
      - Column Name: ordering, Type: INTEGER, Not Null: 0, Primary Key: 0
      - Column Name: title, Type: TEXT, Not Null: 0, Primary Key: 0
      - Column Name: region, Type: TEXT, Not Null: 0, Primary Key: 0
      - Column Name: language, Type: TEXT, Not Null: 0, Primary Key: 0
      - Column Name: types, Type: TEXT, Not Null: 0, Primary Key: 0
      - Column Name: attributes, Type: TEXT, Not Null: 0, Primary Key: 0
      - Column Name: is_original_title, Type: REAL, Not Null: 0, Primary Key: 0
    
    --- Schema for table: movie_ratings ---
      - Column Name: movie_id, Type: TEXT, Not Null: 0, Primary Key: 0
      - Column Name: averagerating, Type: REAL, Not Null: 0, Primary Key: 0
      - Column Name: numvotes, Type: INTEGER, Not Null: 0, Primary Key: 0
    
    --- Schema for table: persons ---
      - Column Name: person_id, Type: TEXT, Not Null: 0, Primary Key: 0
      - Column Name: primary_name, Type: TEXT, Not Null: 0, Primary Key: 0
      - Column Name: birth_year, Type: REAL, Not Null: 0, Primary Key: 0
      - Column Name: death_year, Type: REAL, Not Null: 0, Primary Key: 0
      - Column Name: primary_profession, Type: TEXT, Not Null: 0, Primary Key: 0
    
    --- Schema for table: principals ---
      - Column Name: movie_id, Type: TEXT, Not Null: 0, Primary Key: 0
      - Column Name: ordering, Type: INTEGER, Not Null: 0, Primary Key: 0
      - Column Name: person_id, Type: TEXT, Not Null: 0, Primary Key: 0
      - Column Name: category, Type: TEXT, Not Null: 0, Primary Key: 0
      - Column Name: job, Type: TEXT, Not Null: 0, Primary Key: 0
      - Column Name: characters, Type: TEXT, Not Null: 0, Primary Key: 0
    
    --- Schema for table: writers ---
      - Column Name: movie_id, Type: TEXT, Not Null: 0, Primary Key: 0
      - Column Name: person_id, Type: TEXT, Not Null: 0, Primary Key: 0
    


```python
#joining tables movie_basics ,movie_ratings, movie_akas, directors and persons

sql_query = """
SELECT
    T1.primary_title,
    T1.start_year,
    T1.runtime_minutes,
    T1.genres,
    T2.averagerating,
    T2.numvotes,
    T3.ordering,
    T3.title AS akas_title,
    T3.region,
    T3.language,
    T3.types,
    T3.attributes,
    T3.is_original_title,
    T5.primary_name AS director_name,
    T5.birth_year AS director_birth_year,
    T5.primary_profession AS director_profession
FROM
    movie_basics AS T1
JOIN
    movie_ratings AS T2 ON T1.movie_id = T2.movie_id
JOIN
    movie_akas AS T3 ON T1.movie_id = T3.movie_id
JOIN
    directors AS T4 ON T1.movie_id = T4.movie_id
JOIN
    persons AS T5 ON T4.person_id = T5.person_id;
"""

imdb_df = pd.read_sql_query(sql_query, conn)

# Close the connection
conn.close()
 
```


```python
#checking data summary
imdb_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 722938 entries, 0 to 722937
    Data columns (total 16 columns):
     #   Column               Non-Null Count   Dtype  
    ---  ------               --------------   -----  
     0   primary_title        722938 non-null  object 
     1   start_year           722938 non-null  int64  
     2   runtime_minutes      695593 non-null  float64
     3   genres               720595 non-null  object 
     4   averagerating        722938 non-null  float64
     5   numvotes             722938 non-null  int64  
     6   ordering             722938 non-null  int64  
     7   akas_title           722938 non-null  object 
     8   region               612649 non-null  object 
     9   language             100918 non-null  object 
     10  types                421778 non-null  object 
     11  attributes           49407 non-null   object 
     12  is_original_title    722938 non-null  float64
     13  director_name        722938 non-null  object 
     14  director_birth_year  356093 non-null  float64
     15  director_profession  722826 non-null  object 
    dtypes: float64(4), int64(3), object(9)
    memory usage: 88.2+ MB
    


```python
#dropping columns that I dont  need
imdb_df.drop(columns=['start_year', 'ordering', 'akas_title', 'language', 'types', 'attributes', 'director_birth_year','director_profession', 'is_original_title',], inplace=True, errors='ignore')
imdb_df.columns
```




    Index(['primary_title', 'runtime_minutes', 'genres', 'averagerating',
           'numvotes', 'region', 'director_name'],
          dtype='object')



## 6.2 DATA CLEANING


```python
#renamimg columns
new_column_names = {
    'averagerating': 'average_rating',
    'numvotes': 'num_votes',
    'primary_title': 'title',
}
# Use the .rename() method to apply the new names.
imdb_df.rename(columns=new_column_names, inplace=True)
print(imdb_df.columns)
```

    Index(['title', 'runtime_minutes', 'genres', 'average_rating', 'num_votes',
           'region', 'director_name'],
          dtype='object')
    


```python
#Filling the missing runtime values with the mean
average_runtime = imdb_df['runtime_minutes'].mean()

imdb_df['runtime_minutes'].fillna(average_runtime, inplace=True)
```


```python
#dropping rows with missing values in genre column
imdb_df.dropna(subset=['genres'], inplace=True)
```


```python
#dropping duplicates in the title column
imdb_df.drop_duplicates(subset=['title'], inplace=True)
imdb_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>runtime_minutes</th>
      <th>genres</th>
      <th>average_rating</th>
      <th>num_votes</th>
      <th>region</th>
      <th>director_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sunghursh</td>
      <td>175.000000</td>
      <td>Action,Crime,Drama</td>
      <td>7.0</td>
      <td>77</td>
      <td>IN</td>
      <td>Harnam Singh Rawail</td>
    </tr>
    <tr>
      <th>20</th>
      <td>One Day Before the Rainy Season</td>
      <td>114.000000</td>
      <td>Biography,Drama</td>
      <td>7.2</td>
      <td>43</td>
      <td>XWW</td>
      <td>Mani Kaul</td>
    </tr>
    <tr>
      <th>24</th>
      <td>The Other Side of the Wind</td>
      <td>122.000000</td>
      <td>Drama</td>
      <td>6.9</td>
      <td>4517</td>
      <td>BR</td>
      <td>Orson Welles</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Sabse Bada Sukh</td>
      <td>102.207143</td>
      <td>Comedy,Drama</td>
      <td>6.1</td>
      <td>13</td>
      <td>IN</td>
      <td>Hrishikesh Mukherjee</td>
    </tr>
    <tr>
      <th>53</th>
      <td>The Wandering Soap Opera</td>
      <td>80.000000</td>
      <td>Comedy,Drama,Fantasy</td>
      <td>6.5</td>
      <td>119</td>
      <td>None</td>
      <td>Raoul Ruiz</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>722915</th>
      <td>The Agitation</td>
      <td>102.207143</td>
      <td>Drama,Thriller</td>
      <td>4.9</td>
      <td>14</td>
      <td>None</td>
      <td>Fereydoun Jeyrani</td>
    </tr>
    <tr>
      <th>722918</th>
      <td>Watching This Movie Is a Crime</td>
      <td>100.000000</td>
      <td>Drama,Thriller</td>
      <td>8.1</td>
      <td>7</td>
      <td>XWW</td>
      <td>Reza Zehtabchian</td>
    </tr>
    <tr>
      <th>722923</th>
      <td>BADMEN with a good behavior</td>
      <td>87.000000</td>
      <td>Comedy,Horror</td>
      <td>9.2</td>
      <td>5</td>
      <td>DE</td>
      <td>Loco Meisenkaiser</td>
    </tr>
    <tr>
      <th>722926</th>
      <td>Pengalila</td>
      <td>111.000000</td>
      <td>Drama</td>
      <td>8.4</td>
      <td>600</td>
      <td>None</td>
      <td>T.V. Chandran</td>
    </tr>
    <tr>
      <th>722929</th>
      <td>Padmavyuhathile Abhimanyu</td>
      <td>130.000000</td>
      <td>Drama</td>
      <td>8.4</td>
      <td>365</td>
      <td>None</td>
      <td>Vineesh Aaradya</td>
    </tr>
  </tbody>
</table>
<p>64929 rows × 7 columns</p>
</div>




```python
#Filling the null values in the region column with Unkown
imdb_df['region'].replace('', np.nan, inplace=True)

imdb_df['region'].fillna('UNKNOWN', inplace=True)
imdb_df.isna().sum()
```




    title              0
    runtime_minutes    0
    genres             0
    average_rating     0
    num_votes          0
    region             0
    director_name      0
    dtype: int64




```python
#splitting merged genres into individual genre
merged_exploded =imdb_df.assign(genre=imdb_df['genres'].str.split(',')).explode('genre')
genre_counts = merged_exploded['genre'].value_counts()
print(genre_counts)
```

    genre
    Drama          27484
    Documentary    15962
    Comedy         15689
    Thriller        7172
    Horror          6774
    Action          6270
    Romance         5884
    Crime           4189
    Biography       3589
    Adventure       3538
    Family          3117
    Mystery         2711
    History         2637
    Sci-Fi          1990
    Fantasy         1951
    Music           1793
    Animation       1576
    Sport           1042
    War              779
    Musical          636
    News             556
    Western          250
    Reality-TV        15
    Adult              3
    Game-Show          2
    Name: count, dtype: int64
    


```python
#saving the data
imdb_df.to_csv('cleaned_imbd.csv', index=False)
```

### 6.3 DATA ANALYSIS


```python
#our data is now clean
#we make a copy of the data
imdb_data= imdb_df.copy(deep= True)
```


```python

top_10_genres = genre_counts.nlargest(10)
# Create a horizontal bar chart for better readability
plt.figure(figsize=(8, 4))
plt.barh(top_10_genres.index, top_10_genres.values, color='purple')

# Add titles and labels for clarity
plt.title('Top 10 Most Produced Genres')
plt.xlabel('Number of Occurrences')
plt.ylabel('Genre')

# Invert the y-axis to display the most popular genre at the top
plt.gca().invert_yaxis()

# Ensure the labels don't get cut off
plt.tight_layout()

# Display the plot
plt.show()
```


    
![png](FILM_COMPANY_files/FILM_COMPANY_56_0.png)
    



```python
average_ratings_by_genre = merged_exploded.groupby('genre')['average_rating'].mean().sort_values(ascending=False)

top_10_genres = average_ratings_by_genre.head(10)
colors = plt.cm.Blues(np.linspace(0.4, 1, len(top_10_genres)))
#Create the bar chart
plt.figure(figsize=(12, 4))
plt.bar(top_10_genres.index, top_10_genres.values, color=colors)
plt.xlabel('Genre')
plt.ylabel('Average Rating')
plt.title('Top 10 Genres with the Highest Average Ratings')

```




    Text(0.5, 1.0, 'Top 10 Genres with the Highest Average Ratings')




    
![png](FILM_COMPANY_files/FILM_COMPANY_57_1.png)
    



```python
# percentage of top 10 highly rated movies genres produced vis a vis others
top_10_genres = genre_counts.head(10)
others_count = genre_counts.iloc[10:].sum()

pie_data = top_10_genres.copy()
pie_data['Others'] = others_count

genre_percentages = (pie_data / pie_data.sum()) * 100

plt.figure(figsize=(6, 6))
plt.pie(genre_percentages, labels=pie_data.index, autopct='%1.1f%%', startangle=140)
plt.title('Percentage of Top 10 Most Rated Genres ')
plt.axis('equal')  # Ensures the pie chart is a perfect circle.
#plt.savefig('Percentage of Top 10 Most Rated Genres ', bbox_inches='tight')


```




    (np.float64(-1.099999680050605),
     np.float64(1.0999999849777107),
     np.float64(-1.0999996539052144),
     np.float64(1.0999984168497667))




    
![png](FILM_COMPANY_files/FILM_COMPANY_58_1.png)
    



```python
#top 10 directors
highest_rated_directors= imdb_data.groupby('director_name')['average_rating'].mean().sort_values(ascending=False)
top_10 = highest_rated_directors.head(10)
colors = plt.cm.Greens(np.linspace(0.4, 1, len(top_10)))
plt.figure(figsize=(10, 4))
plt.bar(top_10.index, top_10.values, color=colors)
plt.xlabel('Director')
plt.ylabel('Average Rating')
plt.title('Top 10 Directors by Average Movie Rating')

plt.xticks(rotation=45, ha='right')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Display the plot
plt.show()
plt.savefig('Top 10 Directors by Average Movie Rating')
```


    
![png](FILM_COMPANY_files/FILM_COMPANY_59_0.png)
    



    <Figure size 640x480 with 0 Axes>



```python
#correlation between num votes and average ratings
#average rating: A measure of perceived quality
#num votes; total count of individual ratings a title has received. measure of popularity and reliability
correlation_coefficient = imdb_data['num_votes'].corr(imdb_df['average_rating'])

# Create a scatter plot
plt.figure(figsize=(8, 4))
plt.scatter(imdb_data['num_votes'], imdb_data['average_rating'], alpha=0.7, color='skyblue')

# Add a line of best fit (regression line)
z = np.polyfit(imdb_data['num_votes'], imdb_data['average_rating'], 1)
p = np.poly1d(z)
plt.plot(imdb_data['num_votes'], p(imdb_data['num_votes']), "r--", label='Line of Best Fit')

# Add titles and labels for clarity
plt.title(f'Correlation between Number of Votes and Average Rating\n(Correlation: {correlation_coefficient:.2f})')
plt.xlabel('Number of Votes')
plt.ylabel('Average Rating')
plt.legend()
plt.grid(True)
plt.savefig('correlation_plot.png') 

# Display the plot
plt.show()


```


    
![png](FILM_COMPANY_files/FILM_COMPANY_60_0.png)
    


### There is a correlation between the rating and the popularity of a movie


```python
# Individual movies with highest rating
#setting min vote threshold
min_votes = 1000
reliable_movies_df = imdb_data[imdb_data['num_votes'] > min_votes].copy()
highest_rated_movies = reliable_movies_df.sort_values(by='average_rating', ascending=False).head(10)

colors = ['#4B0082', '#6A5ACD', '#8A2BE2', '#9370DB', '#BA55D3', '#DA70D6', '#DDA0DD', '#EE82EE', '#FF00FF', '#FFB6C1']

plt.figure(figsize=(7, 5))
plt.bar(highest_rated_movies['title'], highest_rated_movies['average_rating'], color=colors)

# Set labels and title
plt.xlabel('Movie Title')
plt.ylabel('Average Rating')
plt.title('Top 10 Highest-Rated Movies')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
#plt.savefig('top_10_highest_rated_movies')
plt.show()

plt.show;

```


    
![png](FILM_COMPANY_files/FILM_COMPANY_62_0.png)
    



```python
#directors with most number of movies per genre
director_movie_counts = merged_exploded.groupby(['genre', 'director_name']).size().reset_index(name='movie_count')

top_directors_per_genre = director_movie_counts.loc[director_movie_counts.groupby('genre')['movie_count'].idxmax()]

top_directors_per_genre = top_directors_per_genre.sort_values(by='movie_count', ascending=False)

print("Directors with the highest number of movies per genre:")
print(top_directors_per_genre)
```

    Directors with the highest number of movies per genre:
                 genre               director_name  movie_count
    72680       Horror            Nayato Fio Nuala           24
    28542  Documentary                 Alex Gibney           19
    24162       Comedy             Wenn V. Deramas           16
    9077     Animation            William Winckler           15
    2485        Action             Koichi Sakamoto           15
    60837        Drama                 Tyler Perry           14
    27344        Crime             Ram Gopal Varma           14
    90808     Thriller                 Larry Rosen           12
    82355      Romance            Mae Czarina Cruz           10
    75489        Music                Paul Dugdale            9
    62337       Family          Arne Lindtner Næss            9
    6508     Adventure             Kunihiko Yuyama            8
    94400      Western          Christopher Forbes            8
    84962       Sci-Fi         Christopher R. Mihm            7
    66531      Fantasy              Umanosuke Iida            6
    86849        Sport               James Erskine            6
    9717     Biography              Clint Eastwood            5
    77145      Mystery            Denis Villeneuve            5
    79220         News                   Gary Null            5
    75972      Musical              Andrew Morahan            4
    68866      History            Steven Spielberg            4
    94103          War  Mohammad Ali Bashe Ahangar            3
    4829         Adult               Chaz Buchanan            1
    66635    Game-Show                 David Stern            1
    79608   Reality-TV                A.J. Shepard            1
    


```python
top_directors_per_genre['label'] = top_directors_per_genre['director_name'] + ' (' + top_directors_per_genre['genre'] + ')'
plt.figure(figsize=(15, 8))
plt.bar(top_directors_per_genre['label'], top_directors_per_genre['movie_count'], color='pink')
plt.xlabel('Director (Genre)')
plt.ylabel('Number of Movies')
plt.title('Top 10 Director Per Genre Combinations by Movie Count')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save and display the plot
plt.show()
```


    
![png](FILM_COMPANY_files/FILM_COMPANY_64_0.png)
    


## 7.0 FINDINGS
* The studios that created many movies are not necessarily te most profitable.
* The best time to release a movie is during the Winter season.
* Production budget has a positive correlation to the profit.
* The most produced genres are not the highest rated.
* The highest rated genres are non-fiction; documentaries,game shows and news

## 8.0 RECOMMENDATIONS
1.Targeting Profitable Genres: A recommendation to focus production of movies in the  documenatries and  drama genre since they have the highest profits. 

2.Optimizing Budget Allocation: Consider the balance of risk and reward.High-budget movies have a higher risk of losing money, while mid-budget films offer a more stable and predictable return.

3.Strategic Release Planning: it is good to consider a season to release a movie based on it's genre.We have seen that most movies do well  during winter.



```python

```


```python

```


```python

```
