# define a function to summarize a DataFrame
# defined a few functions for plotting in this file
# for numerical features: plot_hist, plot_kde_box
# for categorical features: plot_counts, plot_normalized_counts

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def summarize_df(df: pd.DataFrame):
    """
    Returns a summary of a dataframe's basic information.
    
    Parameters:
    - df: pd.DataFrame

    Returns:
    - pd.DataFrame
    """
    results_list = []
    
    for col in df:
        dtype = df[col].dtype
        count = len(df[col])
        unique_values = df[col].value_counts(dropna=False).index.tolist()[:5]
        n_unique = df[col].nunique(dropna=False)
        missing = df[col].isnull().sum()
        missing_percentage = df[col].isnull().sum() / len(df) * 100
        new_row = {'column': col, 
                   'dtype': dtype, 
                   'count': count,
                   'n_unique': n_unique, 
                   'top_5_values': unique_values, 
                   'missing count': missing,
                   'missing %': "{:.2f}".format(missing_percentage)
                   }
        results_list.append(new_row)

    results = pd.DataFrame(results_list).reset_index(drop=True)
    return results

# function to plot histgram of numerical columns
def plot_hist(
        df: pd.DataFrame, 
        feature_name: str, 
        figsize: tuple = (4, 3)
) -> None:
    """
    Plot a histogram of a numerical column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - feature_name (str): The name of the numerical column to plot.
    - figsize (tuple, optional): Figure size (width, height) in inches (default is (4, 3)).

    Returns:
    - None: Displays the histogram plot.
    """
    plt.figure(figsize=figsize)
    df[feature_name].hist(bins=50)
    plt.xlabel(feature_name, fontsize=14, fontweight='bold')
    plt.ylabel('Count', fontsize=14, fontweight='bold')
    plt.show()


# function to plot KDE boxplot of numerical columns
def plot_kde_box(
        df: pd.DataFrame, 
        feature_name: str, 
        target_label: str, 
        figsize: tuple = (8, 4)
) -> None:
    """
    Plot the distribution of numerical data in a binary target class using KDE and box plots.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - feature_name (str): The name of the numerical column to plot.
    - target_label (str): The name of the binary target class column.
    - figsize (tuple, optional): Figure size (width, height) in inches (default is (8, 4)).

    Returns:
    - None: Displays the KDE and box plots.
    """
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=figsize)  
    cols = feature_name 
    
    sns.kdeplot(data=df, 
                x=df[cols], 
                fill=True, 
                alpha=1, 
                hue=target_label,
                palette=('#54BAB9', '#E9DAC1'), 
                multiple='stack', 
                ax=ax[0])  

    sns.boxplot(data=df, 
                y=cols, 
                x=target_label, 
                ax=ax[1], 
                palette=('#54BAB9', '#E9DAC1'))  

    ax[0].set_xlabel(' ')
    ax[1].set_xlabel(target_label, fontsize=14, fontweight='bold')
    ax[1].set_ylabel(' ')
    ax[1].xaxis.set_tick_params(labelsize=14)
    ax[0].set_ylabel(cols, fontsize=14, fontweight='bold')
    
    plt.show()



# script to plot counts of categorical features
def plot_counts(df, category_col, length, height, fontsize, show_percentages=True):
    fig = plt.figure(figsize=(length, height))
    ax = sns.countplot(x=category_col, 
                       data=df)

    ax.set_xlabel(category_col, 
                  fontsize=fontsize, 
                  fontweight='bold')
    ax.set_ylabel('Count', 
                  fontsize=fontsize, 
                  fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), 
                       rotation=90, 
                       fontsize=fontsize, 
                       fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if show_percentages:
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height / len(df) * 100:.1f}%', 
                        xy=(p.get_x() + p.get_width() / 2., height), 
                        ha='center', 
                        va='center', 
                        xytext=(0, 5),
                        textcoords='offset points', 
                        fontsize=fontsize, 
                        fontweight='bold')

    plt.show()



# function to plot normalized count plot of categorical features 
def plot_normalized_counts(
    df: pd.DataFrame,
    category_col: str,
    target_col: str,
    length: float,
    height: float,
    fontsize: int,
    show_percentages: bool = True
) -> None:
    """
    Plot a normalized count plot of categorical features.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - category_col (str): The name of the categorical column to plot.
    - target_col (str): The name of the target column.
    - length (float): Figure width in inches.
    - height (float): Figure height in inches.
    - fontsize (int): Font size for labels and ticks.
    - show_percentages (bool, optional): Whether to display percentages on the plot (default is True).

    Returns:
    - None: Displays the normalized count plot.
    """
    
    # Calculate the normalized count for each category
    counts = df.groupby([category_col, target_col]).size().reset_index(name='count')
    counts['percent'] = counts.groupby(category_col)['count'].apply(lambda x: x / x.sum() * 100)

    # Plot a bar plot of the normalized counts
    fig = plt.figure(figsize = (length, height))
    ax = sns.barplot(x=category_col, 
                     y='percent', 
                     hue=target_col, 
                     data=counts)
    
    ax.set_xticklabels(ax.get_xticklabels(), 
                       rotation=90, 
                       fontsize=fontsize, 
                       fontweight='bold')
    ax.set_xlabel(category_col, 
                  fontsize=fontsize, 
                  fontweight='bold')
    ax.set_ylabel('Percentage', 
                  fontsize=fontsize, 
                  fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.rc('legend', fontsize=fontsize)
    
    # Add the percentage value on top of each bar
    if show_percentages:
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2.,
                    height + 1,
                    '{:.1f}%'.format(height),
                    ha="center",
                    fontsize=fontsize,
                    fontweight='bold')
            
    legend = ax.legend(loc='upper left', 
                       bbox_to_anchor=(1, 1), 
                       title=target_col)
    legend.set_title(target_col, 
                     prop={'size': fontsize})

    plt.show()