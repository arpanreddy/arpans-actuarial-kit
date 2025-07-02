import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from typing import Tuple
from typing import Union
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")


#maybe add something to take df directly off website 
def load_data(file_path_or_df: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """Load CSV file or accept existing DataFrame from repository and return DataFrame."""
    if isinstance(file_path_or_df, pd.DataFrame):
        df = file_path_or_df.copy()
        print(f"Using provided DataFrame with shape: {df.shape}")
        return df
    try:
        df = pd.read_csv(file_path_or_df)
        print(f"Loaded data from CSV with shape: {df.shape}")
        return df
    except Exception as e:
        raise ValueError(f"Error loading CSV: {e}")

#mean, variance, quartiles, # of missing values
#maybe make it look pretty later by putting it in a table
def summarize_data(df: pd.DataFrame):
    print("\nSummary Statistics:")
    print(df.describe(include='all').transpose())
    print("\nMissing Values:")
    print(df.isnull().sum())

#really basic +/- 1.5*IQR outlier detection
#might delete later
def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    outlier_summary = []
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_summary.append({"column": col, "outlier_count": outliers.shape[0]})
    return pd.DataFrame(outlier_summary)

#makes histograms and box plots for 'continuous' variables like integers and float.
#makes count plots for categorical variables
def plot_univariate(df: pd.DataFrame, output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)

   #Continuous Variables
    for col in df.select_dtypes(include=[np.number]).columns:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Histogram of {col}")
        plt.savefig(f"{output_dir}/hist_{col}.png")
        plt.show()

        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.savefig(f"{output_dir}/box_{col}.png")
        plt.show()

#Categorical Variables
    for col in df.select_dtypes(include=["object", "category"]).columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(y=col, data=df, order=df[col].value_counts().index)
        plt.title(f"Countplot of {col}")
        plt.savefig(f"{output_dir}/count_{col}.png")
        plt.show()

def run_eda(file_path_or_df: Union[str, pd.DataFrame]):
    df = load_data(file_path_or_df)
    summarize_data(df)
    outliers = detect_outliers(df)
    print("\nOutlier Summary:")
    print(outliers)
    plot_univariate(df)
    print("\nPlots saved in 'plots' directory.")





