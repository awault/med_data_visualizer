import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
def calc_overweight(height_cm, weight_kg):
    """
    Calculates the weight status based on BMI.

    Args:
    height_cm (float) - Height in cm
    weight_kg (float) - Weight in kg

    Returns:
    int: 1 if overweight, otherwise 0
    """
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    # Return 1 if True , 0 if False
    return int(bmi > 25)  

df['overweight'] = df.apply(lambda row: calc_overweight(row['height'],row['weight']), axis=1)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_filtered = df[['cholesterol','gluc','smoke','alco','active','overweight','cardio']]
    df_cat = pd.melt(df_filtered, value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'], id_vars='cardio', var_name='variable', value_name='value')

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.groupby(['variable', 'value', 'cardio']).size().reset_index(name='total')

    # Draw the catplot with 'sns.catplot()'
    g = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar')

    # Get the figure for the output
    fig = g.figure

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig

# Draw Heat Map
def draw_heat_map():
    # Remove Outliers
    bp_bool = df['ap_lo'] <= df['ap_hi']
    short_bool = df['height'] >= df['height'].quantile(0.025)
    tall_bool = df['height'] <= df['height'].quantile(0.975)
    uw_bool = df['weight']  >= df['weight'].quantile(.025)
    ow_bool = df['weight'] <= df['weight'].quantile(.975)

    df_heat = df[bp_bool & short_bool & tall_bool & uw_bool & ow_bool]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11,9))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', cmap='coolwarm', center=0,\
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
