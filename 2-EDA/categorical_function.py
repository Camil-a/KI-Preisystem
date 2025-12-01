import statsmodels.api as sm
from statsmodels.formula.api import ols

# Function to calculate mean prices for ordinal variables
def mean_pairwise(ordinal_variable, data):
    """
    This function calculates the mean price for each category of an ordinal variable
    and returns a DataFrame with the categories sorted by mean price.
    """
    mean_prices = data.groupby(ordinal_variable)['Price (EUR)'].mean()
    mean_prices = mean_prices.sort_values( ascending=True).reset_index()

    return mean_prices

# Function to print mean prices for a list of ordinal variables
def print_mean_prices(ordinal_variable, data):
    for col in ordinal_variable:
        mean_pairwise_df = mean_pairwise(col, data)
        print(f'\nMean prices for {col}:\n', mean_pairwise_df)


# Function to perform ANOVA for a list of ordinal features
def anova_categoricals_features(features, data, target="Price (EUR)"):
    """
    Performs ANOVA for each feature in 'features' on the target variable.
    
    Parameters:
    - features: list of column names (ordinal or nominal features)
    - data: pandas DataFrame containing the data
    - target: target column name (default "Price (EUR)")
    
    Prints feature name and its ANOVA p-value.
    """
    for feature in features:
        # Define the formula for ANOVA
        formula = f'Q("{target}") ~ C(Q("{feature}"))'
        
        # Fit the model
        model = ols(formula, data=data).fit()
        
        # Get ANOVA table
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        # Extract p-value
        p_value = anova_table["PR(>F)"][0]
        
        # Print result
        print(f'{feature}, p-value: {p_value}')
