import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error

from statsmodels.tools.tools import add_constant
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.tools.eval_measures import meanabs
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

formatted_cols = {}

# ====================================================================================================================
# Pandas DataFrame manipulation functions
# ====================================================================================================================

def dataframe_percentages(data, target, categorical_var):
    """
    Example: 'target' is stroke (0 or 1), 'categorical_var' is gender ('male' or 'female')
        Will calculate percentage of females who had stroke, percentage of females who didn't. Do this for each gender
        Will calculate percentage of those with stroke who are female, how many without stroke are female. Do this for each gender
        More details below
    
    Parameters
    ----------
    data: pandas DataFrame
    target: string 
        Column name of target variable (in 'data')
        Must be discrete data
    categorical_var: string 
        Column name of categorical variable (in 'data')
    
    Returns
    -------
    df_grouped: pandas DataFrame
        First column name = function parameter 'target'
        Second column name = function parameter 'categorical_var'
            First two columns will include every unique combination of target and categorical_var
        Third column name = 'count'
            Represents the number of samples that fall in both categories in the first and second column (e.g. 'female' and 'stroke')
        Fourth column name = 'perc_of_target_cat'
            Represents the count as a percentage of the total number of samples in that target category (e.g. of all with stroke, how many were female)
        Fifth column name = 'percent_of_cat_var'
            Represents the count as a percentage of the total number of samples in that categorical_var category (e.g. of all with females, how many had a stroke)
    
    """
    
    # Create multi-index dataframe with primary index as the target, secondary index as categorical variable
    df_grouped = data.groupby([target, categorical_var])[categorical_var].count().to_frame()
    df_grouped = df_grouped.rename(columns = {categorical_var:'count'})

    # This code ensures that if a certain subcategory isn't present in the 'stroke' or 'no stroke' subset, 
    # it will be added, with a count of '0'
    df_grouped = df_grouped.unstack().fillna(0).stack().astype(int)

    # Add column which represents the categorical variable count as a percentage of target variable
    # (stroke vs. not stroke). I used range() to give them unique values for debugging purposes
    df_grouped['perc_of_target_cat'] = range(len(df_grouped))
    
    # Add column which represents the target variable count as a percentage of categorical variable
    df_grouped['percent_of_cat_var'] = range(len(df_grouped))

    # Loop through multi-index dataframe, giving the new columns they're appropriate values
    for target_value in df_grouped.index.levels[0]:
        for categorical_var_value in df_grouped.index.levels[1]:
            df_grouped.loc[(target_value, categorical_var_value), 'perc_of_target_cat'] = (df_grouped.loc[(target_value, categorical_var_value), 'count'] / df_grouped.loc[(target_value, slice(None)), :]['count'].sum()) * 100
            df_grouped.loc[(target_value, categorical_var_value), 'percent_of_cat_var'] = (df_grouped.loc[(target_value, categorical_var_value), 'count'] / df_grouped.loc[(slice(None), categorical_var_value), :]['count'].sum()) * 100

    # Convert from multi-index dataframe to two columns with those index values 
    # This will add columns for target and categorical variable value, as it makes it easier to create a boxplot
    return df_grouped.reset_index()


def df_shape_to_img(df, title_fontsize=12, number_fontsize=20, text_fontsize=10, h_spacing_between_numbers=0.4):
    """
    Creates image displaying shape of dataframe 'df'
    Credit: https://www.kaggle.com/dwin183287/covid-19-world-vaccination

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame shape to be display
    title_fontsize : int, optional
        Font size of title. The default is 12.
    number_fontsize : int, optional
        Font size of number. The default is 20.
    text_fontsize : int, optional
        Front size of text under numbers. The default is 10.
    h_spacing_between_numbers : int, optional
        Space (between 0 and 1) between centers of left and right numbers/text. The default is 0.4.

    Returns
    -------
    None.

    """
    
    # Y coordinates of numbers and text
    number_y_coord = 0.45
    text_v_spacing = 0.13
    text_y_coord = number_y_coord - text_v_spacing
    
    # X coordinates of numbers and text
    left_x = (1 - h_spacing_between_numbers) / 2
    right_x = left_x + h_spacing_between_numbers
    
    # Create figure, axis, and their properties
    fig = plt.figure()
    ax = plt.gca()
    plt.axis('off')
    fig.set_figwidth(3)
    fig.set_figheight(1)
    
    # Display text
    ax.text(0, 1, 'Dataset Overview:', fontsize=title_fontsize, weight='bold', ha='left', va='top')
    ax.text(left_x, number_y_coord, df.shape[0], color='#2693d7', fontsize=number_fontsize, weight='bold', ha='center', va='center')
    ax.text(left_x, text_y_coord, 'rows', color='dimgray', fontsize=text_fontsize, weight='bold', ha='center', va='top')
    ax.text(right_x, number_y_coord, df.shape[1], color='#2693d7', fontsize=number_fontsize, weight='bold', ha='center', va='center')
    ax.text(right_x, text_y_coord, 'columns', color='dimgray', fontsize=text_fontsize, weight='bold', ha='center', va='top')
    # plt.show()


def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14, header_color='#2693d7', 
                     row_colors=['#f1f1f2', 'w'], edge_color='w', bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, include_index=True, index_col_name='Index', **kwargs):
    """
    Create image (table) of pandas dataframe
    Credit: https://stackoverflow.com/questions/26678467/export-a-pandas-dataframe-as-a-table-image

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame to be displayed as image
    col_width : float64, optional
        DESCRIPTION. The default is 3.0.
    row_height : float64, optional
        DESCRIPTION. The default is 0.625.
    font_size : int, optional
        DESCRIPTION. The default is 14.
    header_color : str, optional
        DESCRIPTION. The default is '#2693d7'.
    row_colors : list of str, optional
        Alternating row colors. The default is ['#f1f1f2', 'w'].
    edge_color : str, optional
        Color of edge of cells. The default is 'w'.
    bbox : list, optional
        Bbox parameters for table. The default is [0, 0, 1, 1].
    header_columns : int, optional
        Index of lowest header column. The default is 0.
    ax : matplot.axes, optional
        Axis to display on. Will create one if one isn't passed. The default is None.
    include_index : boolean
        Whether or not to include DataFrame index in table
    index_col_name : string
        If index is included, the name of its column in the table
    **kwargs : 
        to be used for ax.table creation.

    Returns
    -------
    ax : matplot.axes
        Axis of table (if not passed in, then created in function).
        
    """
    #old header_color = '#40466e'
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    
    if not include_index:
        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    else:
        # Add 'data' indeces to 2D array of DataFrame values, so that the indeces are included in the table 
        # https://stackoverflow.com/questions/36878089/python-add-a-column-to-numpy-2d-array
        new_values = np.c_[np.array(data.index), data.values]
        
        # Add index column name to the array of column names
        all_col_names = np.append(index_col_name, data.columns)
        
        mpl_table = ax.table(cellText=new_values, bbox=bbox, colLabels=all_col_names)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    # Set cell edge colors, background color, font size, font color, etc.
    for k, cell in  mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors)])
    return ax
    
# ====================================================================================================================
# Matplotlib/Seaborn plot helper functions
# ====================================================================================================================

# Update global variable dict formatted_cols where the keys are the original dataset column names and the values
# are the same column names formatted for plots (capitalized, remove underscores, etc.)
# Parameter column_names is the list of dataset column names
# Parameter custom_dict is a dictionary of column names that need their own type of formatting that needs 
#  to be specified
def create_formatted_cols_dict(column_names, custom_dict=None):
    """
    Update the global variable formatted_cols: a dictionary where the keys are the original dataset column names and the values
    are the same column names formatted for plots (capitalized, remove underscores, etc.)
    
    Parameters
    ----------
    column_names: list of strings 
        Dataset column names
    custom_dict: dictionary, optional 
        Column names that need their own specificied formatting
        Keys are the original column name, values are the formatted column name. The default is None.

    """
    global formatted_cols
    for col in column_names:
        formatted_cols[col] = col.replace('_', ' ').title()
    
    if custom_dict is not None:
        for k in custom_dict.keys():
            formatted_cols[k] = custom_dict[k]

def add_edit_formatted_col(col_name, formatted_name):
    global formatted_cols
    formatted_cols[col_name] = formatted_name

def format_col(col_name):
    """
    Format col_name using already generated dictionary formatted_cols

    """
    global formatted_cols
    return formatted_cols[col_name]

# Create 2d array of given size, used for figures with gridspec (here, used with initialize_fig_gs_ax())
def create_2d_array(num_rows, num_cols):
    matrix = []
    for r in range(0, num_rows):
        matrix.append([0 for c in range(0, num_cols)])
    return matrix


def initialize_fig_gs_ax(num_rows, num_cols, figsize=(16, 8)):
    """
    Creates figure, gridspec, and 2d array of axes/subplots with given number of rows and columns

    Parameters
    ----------
    num_rows : int
        Specifies number of rows in new figure.
    num_cols : int
        Specifies number of columns in new figure.
    figsize : tuple with two integers, optional
        Specify the figure seize. The default is (16, 8).

    Returns
    -------
    fig :  matplotlib.pyplot.figure
        Number of axes = num_rows x num_cols, size = figsize.
    gs : matplotlib.gridspec.GridSpec
        Dimensions  = num_rows x num_cols.
    ax_array_flat : numpy.array
        Flattened array of axes, each representing a subplot on the gridspec (gs). Length = num_rows x num_cols.

    """
    
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    ax_array = create_2d_array(num_rows, num_cols)
    gs = fig.add_gridspec(num_rows, num_cols)

    # Map each subplot/axis to gridspec location
    for r in range(len(ax_array)):
        for c in range(len(ax_array[r])):
            ax_array[r][c] = fig.add_subplot(gs[r,c])

    # Flatten 2d array of axis objects to iterate through easier
    ax_array_flat = np.array(ax_array).flatten()
    
    return fig, gs, ax_array_flat

# Standardize image saving parameters
def save_image(filename, dir, dpi=300, bbox_inches='tight', pad_inches=0.1):
    plt.savefig(dir/filename, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    print("\nSaved image to '" + str(dir/filename) +"'\n")

def disp_y_val_in_barplot():
    """
    Adds y-value text above each bar in sns.barplot (works if parameter 'hue' is used as well)
    
    Returns
    -------
    None.

    """
    ax = plt.gca()
    for bar in ax.patches:
        value = bar.get_height().round(1)
        text = f'{value}'
        text_x = bar.get_x() + bar.get_width() / 2
        text_y = bar.get_y() + 2 + value
        ax.text(text_x, text_y, text, ha='center',size=10)

# ====================================================================================================================
# Regression model evaluation functions
# ====================================================================================================================

def plot_coefficient_df(top_df, middle_df, bottom_df, save_img=False, filename=None, save_dir=None):
    """
    Creates figure with three line graphs representing the change in variable coefficients from different linear regression models.
    Different graphs are used to allow for large different in scales of coefficients. Usually used to graph coefficients after 
    adding features, removing features, or removing outliers. The column names of the three dataframes that are passed in are 
    expected to be the same and represent the name of each model that is being graphed. The index of the dataframes represent 
    the name of the variables whose coefficients are being graphed

    Parameters
    ----------
    top_df : Pandas DataFrame
        Variable coefficients to be plotted on top graph.
    middle_df : DataFrame
        Variable coefficients to be plotted on middle graph.
    bottom_df : DataFrame
        Variable coefficients to be plotted on bottom graph.
    save_img : boolean, optional
        Whether or not to save the image. The default is False.
    filename : string, optional
        If file is saved, file name to be used. The default is None.
    save_dir : string, optional
        If file is saved, the directory to save it in. The default is None.

    Returns
    -------
    None.

    """
    # Plot combined
    fig, gs, ax_array_flat = initialize_fig_gs_ax(num_rows=3, num_cols=1, figsize=(9, 13))

    ax1 = ax_array_flat[0]
    for feature in top_df.index:
        ax1.plot(top_df.columns, top_df.loc[feature].to_list(), label=feature, linewidth=3)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Feature')
    plt.setp(ax1.get_xticklabels(), rotation=20, horizontalalignment='right')
    ax1.set_ylabel('Coefficient', fontsize=16)
    ax1.grid()

    ax2 = ax_array_flat[1]
    for feature in middle_df.index:
        ax2.plot(middle_df.columns, middle_df.loc[feature].to_list(), label=feature, linewidth=3)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Feature')
    plt.setp(ax2.get_xticklabels(), rotation=20, horizontalalignment='right')
    ax2.set_ylabel('Coefficient', fontsize=16)
    ax2.grid()

    ax3 = ax_array_flat[2]
    for feature in bottom_df.index:
        ax3.plot(bottom_df.columns, bottom_df.loc[feature].to_list(), label=feature, linewidth=3)
    ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Feature')
    plt.setp(ax3.get_xticklabels(), rotation=20, horizontalalignment='right')
    ax3.set_xlabel('Additional Features', fontsize=16)
    ax3.set_ylabel('Coefficient', fontsize=16)
    ax3.grid()

    fig.suptitle('Feature coeff w/ each additional feature', fontsize=24)
    fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
    if (save_img):
        save_image(filename, save_dir)
    plt.show()


def plot_model_metrics_df(top_l_df, top_r_df, bottom_l_df, bottom_r_df, save_img=False, filename=None, save_dir=None):
    """
    Creates figure with four line graphs representing the change in model performs metrics from different linear regression models.
    Different graphs are used to allow for large different in scales of model performs metrics.
    Usually used to graph metrics after adding features, removing features, or removing outliers. The column names of the 
    four dataframes that are passed in are expected to be the same and represent the name of each model that is being graphed.
    The index of the dataframes represent the name of the model performance metrics that are being graphed

    Parameters
    ----------
    top_l_df : Pandas DataFrame
        Model performs metrics to be graphed on top left plot.
    top_r_df : Pandas DataFrame
        Model performs metrics to be graphed on top right plot.
    bottom_l_df : Pandas DataFrame
        Model performs metrics to be graphed on bottom left plot.
    bottom_r_df : Pandas DataFrame
        Model performs metrics to be graphed on bottom right plot.
    save_img : boolean, optional
        Whether or not to save the image. The default is False.
    filename : string, optional
        If file is saved, file name to be used. The default is None.
    save_dir : string, optional
        If file is saved, the directory to save it in. The default is None.

    Returns
    -------
    None.

    """
    # Plot combined
    fig, gs, ax_array_flat = initialize_fig_gs_ax(num_rows=2, num_cols=2, figsize=(12, 7))

    ax1 = ax_array_flat[0]
    #df_to_plot = max_e_df
    for metric in top_l_df.index:
        ax1.plot(top_l_df.columns, top_l_df.loc[metric].to_list(), label=metric, linewidth=3)
    ax1.legend(loc='upper right', borderaxespad=0.5, title='Metric')
    plt.setp(ax1.get_xticklabels(), rotation=20, horizontalalignment='right')
    ax1.grid()

    ax2 = ax_array_flat[1]
    #df_to_plot = error_metrics
    for metric in top_r_df.index:
        ax2.plot(top_r_df.columns, top_r_df.loc[metric].to_list(), label=metric, linewidth=3)
    ax2.legend(loc='upper right', borderaxespad=0.5, title='Metric')
    plt.setp(ax2.get_xticklabels(), rotation=20, horizontalalignment='right')
    ax2.grid()

    ax3 = ax_array_flat[2]
    #df_to_plot = r_metrics
    for metric in bottom_l_df.index:
        ax3.plot(bottom_l_df.columns, bottom_l_df.loc[metric].to_list(), label=metric, linewidth=3)
    ax3.legend(loc='upper left', borderaxespad=0.5, title='Metric')
    plt.setp(ax3.get_xticklabels(), rotation=20, horizontalalignment='right')
    ax3.grid()

    ax4 = ax_array_flat[3]
    #df_to_plot = het_stats
    for metric in bottom_r_df.index:
        ax4.plot(bottom_r_df.columns, bottom_r_df.loc[metric].to_list(), label=metric, linewidth=3)
    ax4.legend(loc='upper left', borderaxespad=0.5, title='Metric')
    plt.setp(ax4.get_xticklabels(), rotation=20, horizontalalignment='right')
    ax4.grid()

    fig.suptitle('LR Performance w/ each additional feature', fontsize=24)
    fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
    
    if save_img:
        save_image(filename, save_dir)
    plt.show()
    

def evaluate_model_sk(y_valid, y_pred, X, round_results=3, print_results=False):  
    """
    Calculate regression model performance metrics. 
    Written for sklearn linear regression models, but only requires y, y_pred, and X, so could be used for any regression model

    Parameters
    ----------
    y_valid : array_like 
        1-D array of target values in validation set.
    y_pred : array_like
        1-D array of predicated target values from samples in validation set.
    X : Pandas DataFrame
        DataFrame of features and their values in validation set.
    round_results : int, optional
        How many decimal places to round results to. The default is 3.
    print_results : boolean, optional
        Whether or not to print metrics in readable format. The default is False.

    Returns
    -------
    metrics : Dictionary
        Dictionary where keys are name of metric, values are their values.

    """    
    metrics = {}
    metrics['max_e'] = max_error(y_valid, y_pred).round(round_results)
    metrics['mae'] = mean_absolute_error(y_valid, y_pred).round(round_results)
    metrics['mse'] = mean_squared_error(y_valid, y_pred).round(round_results)
    metrics['rmse'] = np.sqrt(metrics['mse']).round(round_results)
    metrics['med_abs_e'] = median_absolute_error(y_valid, y_pred).round(round_results)
    metrics['r2'] = r2_score(y_valid, y_pred).round(round_results)
    metrics['r2_adj'] = 1 - ((1-metrics['r2'])*(len(y_valid)-1)/(len(y_valid)-X.shape[1]-1)).round(round_results)
    
    if print_results:
        print('Max Error: ' + str(metrics['max_e']))
        print('Mean Absolute Error: ' + str(metrics['mae']))
        print('Mean Squared Error: ' + str(metrics['mse']))
        print('Root Mean Squared Error: ' + str(metrics['rmse']))
        print('Median Absolute Error: ' + str(metrics['med_abs_e']))
        print('R-squared: ' + str(metrics['r2']))
        print('R-squared (adj): ' + str(metrics['r2_adj']))
    return metrics

def evaluate_model_sm(y, y_pred, sm_lr_model, het_results=None, round_results=3, print_results=False):    
    """
    Calculate regression model performance metrics. 
    Written for statsmodels OLS model as it uses its built-in functions

    Parameters
    ----------
    y : array_like
        1-D array of target values.
    y_pred : array_like
        1-D array of predicated target values.
    sm_lr_model : statsmodels.regression.linear_model.OLS
        Statsmodels OLS model used to fit data.
    het_results : dictionary
        Dictionary of dictionaries containing heteroscedasticity results from
        fit_lr_model_results().
    round_results : int, optional
        How many decimal places to round results to. The default is 3.
    print_results : boolean, optional
        Whether or not to print metrics in readable format. The default is False.

    Returns
    -------
    metrics : Dictionary
        Dictionary where keys are name of metric, values are their values.

    """
    metrics = {}
    metrics['max_e'] = np.round(max(sm_lr_model.resid), round_results)
    metrics['mae'] = meanabs(y, y_pred).round(round_results)
    metrics['mse'] = sm_lr_model.mse_resid.round(round_results) # this is the closet metric to mse. It was within 3% of both my calculation and sklearn's. 'mse_model' and 'mse_total' were 99% and 75% different
    metrics['rmse'] = np.sqrt(metrics['mse']).round(round_results)
    metrics['med_abs_e'] = np.median(abs(sm_lr_model.resid)).round(round_results)
    metrics['r2'] = sm_lr_model.rsquared.round(round_results)
    metrics['r2_adj'] = sm_lr_model.rsquared_adj.round(round_results)
    
    # Quantify Heteroscedasticity using Breusch-Pagan test and White test 
    if het_results is None:
        bp_test = het_breuschpagan(sm_lr_model.resid, sm_lr_model.model.exog)
        white_test = het_white(sm_lr_model.resid, sm_lr_model.model.exog)
        labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
        bp_test_results = dict(zip(labels, bp_test))
        white_test_results = dict(zip(labels, white_test))
        metrics['bp_lm_p'] = '{:0.3e}'.format(bp_test_results['LM-Test p-value'])
        metrics['white_lm_p'] = '{:0.3e}'.format(white_test_results['LM-Test p-value'])
    else:
        metrics['bp_lm_p'] = '{:0.3e}'.format(het_results['BP']['LM-Test p-value'])
        metrics['white_lm_p'] = '{:0.3e}'.format(het_results['White']['LM-Test p-value'])
    
    if print_results:
        print('Max Error: ' + str(metrics['max_e']))
        print('Mean Absolute Error: ' + str(metrics['mae']))
        print('Mean Squared Error: ' + str(metrics['mse']))
        print('Root Mean Squared Error: ' + str(metrics['rmse']))
        print('Median Absolute Error: ' + str(metrics['med_abs_e']))
        print('R-squared: ' + str(metrics['r2']))
        print('R-squared (adj): ' + str(metrics['r2_adj']))
        print('Breusch-Pagan LM p-val: ' + metrics['bp_lm_p'])
        print('White LM p-val: ' + metrics['white_lm_p'])
    return metrics

    
def calulate_vif(data, numerical_cols):
    """
    Calculates variance inflation factor (VIF) based on the numerical features ('numerical_cols') in dataset 'data'
    Credit:
        https://www.statology.org/multiple-linear-regression-assumptions/
        https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python

    Parameters
    ----------
    data : Pandas DataFrame
        DataFrame containing all data.
    numerical_cols : list
        List of strings representing the column names of the numerical data in 'data'.

    Returns
    -------
    vif : Pandas DataFrame
        Indeces: numerical variable names
        First column: VIF for given variable

    """

    fxn_dataset = data[numerical_cols].copy()
    fxn_dataset = add_constant(fxn_dataset)
    vif = pd.Series([variance_inflation_factor(fxn_dataset.values, i) for i in range(fxn_dataset.shape[1])], index=fxn_dataset.columns)
    return vif


# ====================================================================================================================
# Fitting data to distributions
# ====================================================================================================================

def fit_to_dist_gof(my_data, dists_to_fit, ad_dists):
    """
    Uses package scipy to fit 'my_data' to all distributions in 'dists_to_fit' and performs 
    multiple GOF (goodness of fit) tests: Kolmogorov-Smirnov,  Anderson-Darling, Cramer-von Mises
    Stores fit parameters and GOF test results in a DataFrame which is returned
    
    Parameters
    ----------
    my_data: pandas DataFrame
        1D itertable of numerical values to be fit to the distributions 
    dists_to_fit: list of strings 
        Each string must represent the name of a scipy.stats.rv_continuous distribution object
    ad_dists: list of string 
        List of scipy continuous distributions that can be tested with scipy.stats.anderson()
    
    Returns
    -------
    complete_results_df: pandas DataFrame
        Contains fit parameters and GOF test results (specified in the function)
    """
    
    # Used for output while running loop
    num_dists = len(dists_to_fit)
    
    # Initialize all results DataFrame
    complete_results_df = pd.DataFrame(columns=['param', 'shape_param_name', 'mle', 'ks_stat', 'ks_pval', 'cm_stat', 'cm_pval', 'ad_stat', 'ad_critvals', 'ad_siglevels'])

    # Loop through dists_to_fit, fit to my data, perform GOF tests
    # SciPy fit() performs parameter estimation using MLE
    for i, dist_str in enumerate(dists_to_fit):    
        # Text updates as loop runs
        # '>' right-justifies, makes space for 3 digits
        print("{:>3} / {:<3}: {}".format(i+1, num_dists, dist_str))
        
        # Intialize df results row for this dist
        complete_results_df.loc[dist_str] = np.nan    
        
        # Create scipy distribution object based on its string name
        dist_obj = getattr(stats, dist_str)
        
        # Fit my data to the distribution (scipy uses MLE) and get fit disribution's parameters
        # The last two parameters are always locations ('loc') and scale ('scale')
        # If there are more parameters, they represent the shape parameters of the distribution
        # their names are accessed below with 'dist_obj.shapes
        param = dist_obj.fit(my_data)
        complete_results_df.loc[dist_str]['param'] = param
        
        # Add shape param names for each dist
        complete_results_df.loc[dist_str]['shape_param_name'] = dist_obj.shapes
        
        # Access and save the calculated MLE
        mle = dist_obj.nnlf(param, my_data)
        complete_results_df.loc[dist_str]['mle'] = mle
        
        # Perform Kolmogorov-Smirnov Test and save results
        ks = stats.kstest(my_data, dist_str, args=param)
        complete_results_df.loc[dist_str]['ks_stat'] = ks[0]
        complete_results_df.loc[dist_str]['ks_pval'] = ks[1]
        
        # Perform Anderson-Darling Test and save results
        if dist_str in ad_dists:
            ad = stats.anderson(my_data, dist=dist_str)
            complete_results_df.loc[dist_str]['ad_stat'] = ad[0]
            complete_results_df.loc[dist_str]['ad_critvals'] = ad[1]
            complete_results_df.loc[dist_str]['ad_siglevels'] = ad[2]
        
        # Perform Cramer-von Mises Test and save results
        cm = stats.cramervonmises(my_data, dist_str, param)
        complete_results_df.loc[dist_str]['cm_stat'] = cm.statistic
        complete_results_df.loc[dist_str]['cm_pval'] = cm.pvalue
    return complete_results_df


def calc_sse_dist(my_data, fit_results_df, bins=200):
    """
    Takes the scipy distribution function name and fit parameters from 'fit_results_df' that was generated using 
    fit_to_dist_gof() function above, and calculates the Squared Estimate of Errors (sse) between the fit function
    and 'my_data'

    Parameters
    ----------
    my_data : pandas DataFrame
        1D itertable of numerical values to be fit to the distributions 
    fit_results_df : pandas DataFrame
        Created with fit_to_dist_gof() function above
    bins : integer, optional
        Number of bins for histogram which will generate x-values for pdf calculations of comparison distributions
        The default is 200.

    Returns
    -------
    sse_results : list of floats 
        Each floats represents the sse of each distribution compared to my_data

    """
    
    # Get histogram of original data
    y, x = np.histogram(my_data, bins=bins, density=True)
    
    # In this case 'x' represents the bin edges, which has 1 more value in it than 'y'
    # Performing this calculation returns the center 'x' value of each bin
    x = (x + np.roll(x, -1))[:-1] / 2.0
    
    # Keep track of all calculated sse's
    sse_results = []
        
    # Loop through all dists
    for i, dist_str in enumerate(fit_results_df.index):         
        # Organize fit parameters
        fit_params = fit_results_df.loc[dist_str]['param']
        shape_params = fit_params[:-2]
        loc = fit_params[-2]
        scale = fit_params[-1]
        
        # Create object of distribution based on its string name
        dist_obj = getattr(stats, dist_str)
        
        # Calculate fitted PDF and error with fit in distribution
        pdf = dist_obj.pdf(x, loc=loc, scale=scale, *shape_params)
        sse = np.sum(np.power(y - pdf, 2.0))
        sse_results.append(sse)
    
    return sse_results


def rank_gof_stats(complete_results_df, gof_stat_col_names, gof_names):
    """
    Takes the DataFrame complete_results_df generated from fit_to_dist_gof() and adds 
    the rank for each GOF statistic. Also saves the top 10 of each GOF test and returns 
    as DataFrame top_10_df. 
    # gof_stat_col_names and gof_names have to represent the same gof test

    Parameters
    ----------
    complete_results_df : Pandas DataFrame
        Generated from fit_to_dist_gof().
    gof_stat_col_names : 1-D list
        The names of the columns that contain the values of the test statistic for each GOF test.
        Must be same length and correspond with names in gof_names.
    gof_names : 1-D list
        The abbreviated names of the GOF tests to be used in top_10_df. Must be same length and correspond
        with names in gof_stat_col_names.

    Returns
    -------
    top_10_df : Pandas DataFrame
        Columns are GOF test names, each column represents the top 10 ranked distributions, in order,
        of said GOF test.

    """
    # Number of dists in complete_results_df, used to calculate range of ranks
    num_dists = len(complete_results_df.index)
    
    # DataFrame of top 10 dists of each GOF
    top_10_df = pd.DataFrame(columns=gof_names)
    
    for i, gof_stat in enumerate(gof_stat_col_names):
        gof_name = gof_names[i]
        
        if gof_stat == 'ad_stat':
            ad_sorted_df = complete_results_df.sort_values(by='ad_stat')['ad_stat'].dropna().to_frame()
            ad_sorted_df['ad_rank'] = range(1, len(ad_sorted_df.index) + 1)
            complete_results_df['ad_rank'] = ad_sorted_df['ad_rank']
            top_10_df['ad'] = pd.Series(ad_sorted_df['ad_stat'].index)
        else:
            complete_results_df.sort_values(by=gof_stat, inplace=True)
            new_rank_col_name = gof_name + '_rank'
            complete_results_df[new_rank_col_name] = range(1, num_dists + 1)
            top_10_df[gof_name] = complete_results_df[:10][gof_stat].index
    
    return top_10_df

def top_10_counts(top_10_df):
    """
    Creates a DataFrame of number of times each dist appears in top_10_df (generated in rank_gof_stats()), 
    and for which GOF metrics specifically.

    Parameters
    ----------
    top_10_df : Pandas DataFrame
        Generated in rank_gof_stats(). Columns are GOF test names, each column represents the 
        top 10 ranked distributions, in order, of said GOF test.

    Returns
    -------
    top_10_counts_df : Pandas DataFrame
        Number of times each dist appears in top_10_df, and for which GOF metrics specifically.

    """
    # Provides a list of every distribution that appears at least once in top_10_df
    all_top_10_dists = top_10_df.apply(pd.value_counts).index
    
    # Create numpy array of all dists in top_10_df so I can easily count the occurance of each
    top_10_np_array = top_10_df.to_numpy()
    
    # Initialize results dataframe, the indeces represent each dist that appears in top_10_df
    top_10_counts_df = pd.DataFrame(index=all_top_10_dists.tolist())
    
    # Initialize results columns
    top_10_counts_df['Top 10 Count'] = np.nan
    top_10_counts_df['Top 10 In'] = '1' # needs to start as a string
       
    # Loop through top_10_counts_df indeces, and count the number of times the dist appears in top_10_np_array
    # Also determine which GOF tests the dist is a top 10 in
    for dist in top_10_counts_df.index:
        # Get count of dist in numpy array using np.sum (adds the boolean 1 for each appearance)
        top_10_counts_df.at[dist, 'Top 10 Count'] = np.sum(top_10_np_array==dist)
        
        # Keep track of GOF tests that dist is present in
        top_10_in_list = []
        top_10_in_str = ''
        
        # Determine which GOF tests the dist is a top 10 in
        # Loop through each column (a GOF test) of top_10_df and determine if dist is present
        for gof_test in top_10_df.columns:
            if dist in top_10_df[gof_test].values:
                top_10_in_list.append(gof_test)
                
        top_10_in_str = ', '.join(str(line) for line in top_10_in_list)
        top_10_counts_df.at[dist, 'Top 10 In'] = top_10_in_str
    
    # Sort counts in descending order
    top_10_counts_df.sort_values(by='Top 10 Count', inplace=True, ascending=False)
    
    return top_10_counts_df

# # Combine Q-Q plot and histogram/distribution comparisons plot
# def compare_dist_plots(dist_str, loc, scale, shape_params, comp_data, comp_data_str, rank_str, bins=200, save_img=False, img_filename=None, save_dir=None):
#     """    
#     This assumes scipy fit() has already been done on the distribution and dist paramaters have been generated
    
#     Input: dist_str - string representation of scipy distribution which my data will be compared to
#            loc - loc parameter of dist_str returned from fit() function
#            scale - scale parameter of dist_str returned from fit() function
#            shape_params - other shape parameters (if present) of dist_str returned from fit() function
#            comp_data - the data that I will be comparing to scipy distributions
#            comp_data_str - string name of comp_data for plot labeling
    
#     Returns: No return values
#     """
    
#     # Create scipy distribution object based on its name
#     dist_object = getattr(stats, dist_str)
    
#     # Specify scipy distribution shape, location, and scale based on the parameters calculated from fit()
#     rv = dist_object(*shape_params, loc, scale)
    
#     # Use the distribution to create x values for the plot
#     # ppf() is the inverse of cdf(). So if cdf(10) = 0.1, then ppf(0.1)=10
#     # ppf(0.1) is the x-value at which 10% of the values are less than or equal to it
#     x = np.linspace(rv.ppf(0.01), rv.ppf(0.99), 100)
    
#     # Create 1x2 figure
#     fig, gs, ax_array_flat = initialize_fig_gs_ax(num_rows=1, num_cols=2, figsize=(11, 5))
#     ax1 = ax_array_flat[0]
#     ax2 = ax_array_flat[1]
    
#     # Create Q-Q plot, fit=False because it uses the parameters we already calculated (loc, scale, distargs)
#     qqplot(comp_data, line='45', fit=False, dist=dist_object, loc=loc, scale=scale, distargs=shape_params, ax=ax1)
#     ax1.set_xlabel(f'Theoretical Quantiles ({dist_str})')
#     ax1.set_ylabel('Sample Quantiles')
#     ax1.set_title(f'{comp_data_str} Q-Q Plot', y=1.05)
    
#     # Plot distribution on top of histogram of charges in order to compare
#     ax2.hist(comp_data, bins=bins, density=True, histtype='stepfilled', alpha=0.9, label=comp_data_str)
#     ax2.plot(x, rv.pdf(x), 'r-', lw=2.5, alpha=1, label=dist_str)
#     ax2.set_title(f'{comp_data_str} histogram', y=1.05)
#     ax2.set_xlabel(f'{comp_data_str}')
#     ax2.legend()
    
#     # Include shape parameters
#     param_names = (dist_object.shapes + ', loc, scale').split(', ') if dist_object.shapes else ['loc', 'scale']
#     all_params = shape_params + (loc,) + (scale,) # Need to convert loc and scale to tuples
#     param_list = ['{}: {:0.2f}'.format(k,v) for k,v in zip(param_names, all_params)]
#     all_params_str = '\n'.join(str(line) for line in param_list)
#     textbox_text = 'Shape Params:\n' + all_params_str + '\n\nGOF Ranks: \n' + rank_str
#     box_style = {'facecolor':'white', 'boxstyle':'round', 'alpha':0.8}
#     ax2.text(1.05, 0.99, textbox_text, bbox=box_style, transform=ax2.transAxes, verticalalignment='top', horizontalalignment='left')  
    
#     # Figure formatting
#     fig.suptitle(f"{comp_data_str} vs. {dist_str} distribution", fontsize=22)
#     fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
    
#     if (save_img):
#         save_image(img_filename, save_dir)
#         print(f"saving file '{img_filename}' in '{save_dir}'")
#     plt.show()


def create_fit_param_str(dist_obj, fit_params):
    """
    Creates a string where each line includes the distribution fit parameter name and its value

    Parameters
    ----------
    dist_obj : scipy.stats.rv_continuous
        Scipy distribution object.
    fit_params : Tuple
        Parameters returned after with scipy function fit().

    Returns
    -------
    fit_params_str : String
        Each line includes the distribution fit parameter name and its value.

    """
    fit_param_names = (dist_obj.shapes + ', loc, scale').split(', ') if dist_obj.shapes else ['loc', 'scale']
    fit_param_list = ['{}: {:0.2f}'.format(k,v) for k,v in zip(fit_param_names, fit_params)]
    fit_params_str = '\n'.join(str(line) for line in fit_param_list)
    return fit_params_str


# Q-Q plot (default is normal dist)
def my_qq(data, my_data_str='Residuals', dist_obj=stats.norm, fit_params=None, dist_str='Normal Dist', 
          ax=None, y=1, save_img=False, img_filename=None, save_dir=None): 
    
    if not fit_params:
        # Fit my data to dist_obj and get fit parameters
        fit_params = dist_obj.fit(data)
    
    # Specify scipy distribution shape, location, and scale based on the parameters calculated from fit()
    loc = fit_params[-2]
    scale = fit_params[-1]
    shape_params = fit_params[:-2]
    
    # Q-Q Plot
    qqplot(data, line='45', fit=False, dist=dist_obj, loc=loc, scale=scale, distargs=shape_params, ax=ax)
    
    if not ax:
        ax = plt.gca()

    ax.set_xlabel(f'Theoretical Quantiles ({dist_str})')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title(f'Q-Q Plot {my_data_str} vs. {dist_str}', y=y)
    
    if save_img:
        save_image(img_filename, save_dir)

    if not ax:
        plt.show()

# Plots a scipy distribution vs. histogram of my_data
def hist_vs_dist_plot(my_data, my_data_str='Residuals', dist_obj=stats.norm, fit_params=None, dist_str='Normal Dist', 
                      bins=200, ax=None, textbox_str=None, save_img=False, img_filename=None, save_dir=None):    
    
    if not fit_params:
        # Fit my data to dist_obj and get fit parameters
        fit_params = dist_obj.fit(my_data)
    
    # Specify scipy distribution shape, location, and scale based on the parameters calculated from fit()
    loc = fit_params[-2]
    scale = fit_params[-1]
    shape_params = fit_params[:-2]
    
    # Specify scipy distribution shape, location, and scale based on the parameters calculated from fit() above
    rv = dist_obj(*shape_params, loc, scale)
    
    # Use the distribution to create x values for the plot
    # ppf() is the inverse of cdf(). So if cdf(10) = 0.1, then ppf(0.1)=10
    # ppf(0.1) is the x-value at which 10% of the values are less than or equal to it
    x = np.linspace(rv.ppf(0.01), rv.ppf(0.99), 100)
    
    if not ax:
        ax = plt.gca()
    
    # Plot distribution on top of histogram of charges in order to compare
    ax.hist(my_data, bins=bins, density=True, histtype='stepfilled', alpha=0.9, label=my_data_str)
    ax.plot(x, rv.pdf(x), 'r-', lw=2.5, alpha=1, label=dist_str)
    ax.set_title(f'{my_data_str} vs. {dist_str}', y=1.05)
    ax.set_xlabel(f'{my_data_str}')
    
    if textbox_str:
        # Add normality test interpretation text
        box_style = {'facecolor':'white', 'boxstyle':'round', 'alpha':0.8}
        ax.text(1.05, 0.99, textbox_str, bbox=box_style, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left') 
    
    
    ax.legend()
    
    if save_img:
        save_image(img_filename, save_dir)


def plot_qq_hist_dist_combined(my_data, my_data_str='Residuals', dist_obj=stats.norm, dist_str='Normal Dist',
                               fit_params=None, bins=50, textbox_str=None, fig_title=None, title_fontsize = 24, 
                               figsize=(10, 5), save_img=False, img_filename=None, save_dir=None):
    """
    Plot both qq and hist vs. dist plots in same figure. Utilizes hist_vs_dist_plot() and my_qq(). 

    Parameters
    ----------
    my_data : 1-D array-like
        Data which is being plotted against a specificed distribution.
    my_data_str : String, optional
        String labeling my_data, to be used in plots. The default is 'Residuals'.
    dist_obj : scipy.stats.rv_continuous, optional
        Scipy distribution object being compared to my_data. The default is stats.norm.
    dist_str : String, optional
        Represents name of dist_obj. Does not need to be the SciPy representation of the distribution as this String is just 
        used in the plots. The default is 'Normal Dist'.
    fit_params : tuple, optional
        Distribution fit patameters returned with SciPy fit() function. The default is None.
    bins : int, optional
        Number of bins for the hitsogram. The default is 50.
    textbox_str : String, optional
        Can be any string to be included in a textbox next to the plot. The default is None.
    fig_title : String, optional
        Title of figure. The default is None.
    title_fontsize : int, optional
        Font size of figure title. The default is 24.
    figsize : type, optional
        Figure size. The default is (10, 5).
    save_img : boolean, optional
        Whether or not to save image. The default is False.
    img_filename : String, optional
        Filed name if saving image. The default is None.
    save_dir : String, optional
        Directory to save image in. The default is None.

    Returns
    -------
    None.

    """
    # Create figure, gridspec, list of axes/subplots mapped to gridspec location
    fig, gs, ax_array_flat = initialize_fig_gs_ax(num_rows=1, num_cols=2, figsize=figsize)

    if not fit_params:    
        # Fit my data to dist_obj and get fit parameters
        fit_params = dist_obj.fit(my_data)
    
    # Plot Q-Q, add to figure
    my_qq(my_data, my_data_str=my_data_str, dist_obj=dist_obj, fit_params=fit_params, 
          dist_str=dist_str, ax=ax_array_flat[0], y=1.05, save_dir=save_dir)
    
    # Plot hist vs. dist, add to figure
    hist_vs_dist_plot(my_data, my_data_str=my_data_str, dist_obj=dist_obj, fit_params=fit_params, 
                      dist_str=dist_str, bins=bins, ax=ax_array_flat[1], textbox_str=textbox_str, save_dir=save_dir)
    
    # Figure title
    if fig_title:
        fig.suptitle(fig_title, fontsize=title_fontsize)
    else:
        fig.suptitle(f'{my_data_str} vs. {dist_str}', fontsize=title_fontsize)
    fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
    
    if save_img:
        save_image(img_filename, save_dir)
    plt.show()


def normality_tests(my_data):
    """
    Performs 6 tests for normality on my_data: Shapiro-Wilk, D'Agostino's K-squared, 
    Chi-Square, Jarque–Bera, Kolmogorov-Smirnov, Lilliefors, and Anderson-Darling.  
    Assumes an alpha of 0.05.

    Parameters
    ----------
    my_data : 1-D array-like
        Data which is being tested for normality.

    Returns
    -------
    norm_test_series : Pandas Series
        Contains test statistic and p-value (and crit values for ad test) for each test.
    test_interpret : Pandas Series
        Series that contains interpretation of normality test results: whether my_data 
        'passed' each of the normality tests or not.
    normal_interpret_txt : String
        String representing the same information in test_interpret Series.

    """
    alpha=0.05
    
    # List of normality test objects to use and their associated abbreviations
    test_obj_list = [stats.shapiro, stats.normaltest, stats.chisquare, stats.jarque_bera, stats.kstest, lilliefors]
    test_abbv = ['sw', 'dk', 'cs', 'jb', 'ks', 'lt']
    
    # Series to contain all test results (statistics, pvals, etc.)
    stat_indeces = [index+'_stat' for index in test_abbv]
    pval_indeces = [index+'_pval' for index in test_abbv]
    norm_test_series_index = [item for sublist in zip(stat_indeces, pval_indeces) for item in sublist]    
    norm_test_series = pd.Series(index=norm_test_series_index, dtype='float64', name='test_results')
    
    # Series with results interpretted (normal  vs. not)
    test_interpret = pd.Series(index=test_abbv, name='normal')
    
    # Determine critical value for ks test (assumes n > 40)
    ks_stat_cutoff = 1.36 / np.sqrt(len(my_data))
    
    # Loop through test objects, calculate their statistics/pvals, store in series,
    # and determine whether we can fail to reject H0 (i.e. that the dist may be normal)
    for i, test_obj in enumerate(test_obj_list):
        test_str = test_abbv[i]
                
        if test_str=='ks':
            test_results = test_obj(my_data, 'norm')
            norm_test_series[test_str+'_stat'] = test_results[0]
            norm_test_series[test_str+'_pval'] = test_results[1]
            test_interpret[test_str] = test_results[0] < ks_stat_cutoff
        else:
            test_results = test_obj(my_data)
            norm_test_series[test_str+'_stat'] = test_results[0]
            norm_test_series[test_str+'_pval'] = test_results[1]
            test_interpret[test_str] = test_results[1] > alpha
            
    # Add Anderson Darling test as it has more complex results
    ad_series = pd.Series(index=['ad_stat', 'ad_critvals', 'ad_siglevels'], dtype='object', name='test_results')
    ad_results = stats.anderson(my_data, dist='norm')
    ad_series['ad_stat'] = ad_results[0]
    ad_series['ad_critvals'] = ad_results[1]
    ad_series['ad_siglevels'] = ad_results[2]
    
    # Interpet Anderson Darling test results
    ad_interpret = pd.Series(index=['ad'], name='normal')
    # The third element in the critval array is for alpha = 0.05 when writing this code
    ad_interpret['ad'] = ad_results[0] < ad_results[1][2] 
    
    # Add Anderson Darling test results and interpretation to their respective Series
    norm_test_series = norm_test_series.append(ad_series)
    test_interpret = test_interpret.append(ad_interpret)
    
    return norm_test_series, test_interpret, normal_interpret_txt(test_interpret)


def normal_interpret_txt(interp_results):
    """
    Converts normality test interpretation (from normality_tests())  to text for plots

    Parameters
    ----------
    interp_results : Pandas Series
        Normality test interpretation from normality_tests().

    Returns
    -------
    output_txt_str : String
        String representing the same information in interp_results Series.

    """
    #test_names = ["Shapiro-Wilk", "D'Agostino's K-squared", "Chi-Square", "Jarque–Bera", "Kolmogorov-Smirnov", "Lilliefors", "Anderson-Darling"]
    test_abbv = ['SW', 'DK', 'CS', 'JB', 'KS', 'LT', 'AD']
    output_txt_str = 'Normality Test Results:\n\n'
    
    interp_results_df = interp_results.to_frame()
    interp_results_df['passed'] = ''
    
    for i in range(len(interp_results_df)):
        if interp_results_df.iat[i, 0]:
            interp_results_df.iat[i, 1] = 'passed'
        else:
            interp_results_df.iat[i, 1] = 'failed'
        
    output_txt_list = ['{}: {}'.format(k,v) for k,v in zip(test_abbv, interp_results_df['passed'])]
    output_txt_str = output_txt_str + '\n'.join(str(line) for line in output_txt_list)
    
    return output_txt_str