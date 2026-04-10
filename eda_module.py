import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from imblearn.over_sampling import SMOTE
import os
import numpy as np

class DataExplorer:
    def __init__(self, data_path, sample_size=10000):
        """Initialize with dataset path and sample size for visualizations."""
        self.data = pd.read_csv(data_path)
        self.sample_data = self.data.sample(n=min(sample_size, len(self.data)), random_state=42)  # Downsample for plots
        self.output_dir = "eda_plots"
        os.makedirs(self.output_dir, exist_ok=True)
        plt.style.use('fast')  # Faster plotting style

    def compute_statistics(self, columns):
        """Compute descriptive statistics for specified columns."""
        stats = {}
        for col in columns:
            stats[col] = {
                "mean": self.data[col].mean(),
                "median": self.data[col].median(),
                "std": self.data[col].std(),
                "variance": self.data[col].var(),
                "min": self.data[col].min(),
                "max": self.data[col].max(),
                "skewness": skew(self.data[col]),
                "kurtosis": kurtosis(self.data[col])
            }
        stats_df = pd.DataFrame(stats).T
        print("Descriptive Statistics:\n", stats_df.round(2))
        stats_df.to_csv(os.path.join(self.output_dir, "statistics.csv"))
        return stats_df

    def plot_histogram(self, col):
        """Plot histogram for a numerical column."""
        plt.figure(figsize=(8, 5))
        sns.histplot(self.sample_data[col], kde=True, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {col}', fontsize=12)
        plt.xlabel(col, fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, f"{col}_histogram.png"), dpi=100, bbox_inches='tight')
        plt.show()  # Display in VS Code
        plt.close()

    def plot_boxplot(self, col):
        """Plot box plot for a numerical column."""
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=self.sample_data[col], color='lightgreen')
        plt.title(f'Box Plot of {col}', fontsize=12)
        plt.xlabel(col, fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, f"{col}_boxplot.png"), dpi=100, bbox_inches='tight')
        plt.show()  # Display in VS Code
        plt.close()

    def plot_bar(self, col):
        """Plot bar plot for a categorical column."""
        plt.figure(figsize=(8, 5))
        sns.countplot(x=col, data=self.sample_data, palette='viridis')
        plt.title(f'Frequency of {col}', fontsize=12)
        plt.xlabel(col, fontsize=10)
        plt.ylabel('Count', fontsize=10)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, f"{col}_barplot.png"), dpi=100, bbox_inches='tight')
        plt.show()  # Display in VS Code
        plt.close()

    def plot_class_balance(self, target_column):
        """Plot class distribution as a pie chart."""
        plt.figure(figsize=(6, 6))
        self.data[target_column].value_counts().plot.pie(
            autopct='%1.1f%%', colors=['salmon', 'lightblue'], startangle=90, textprops={'fontsize': 10}
        )
        plt.title(f'Class Distribution of {target_column}', fontsize=12)
        plt.ylabel('')
        plt.savefig(os.path.join(self.output_dir, f"{target_column}_piechart.png"), dpi=100, bbox_inches='tight')
        plt.show()  # Display in VS Code
        plt.close()
        class_counts = self.data[target_column].value_counts()
        print("Class Distribution:\n", class_counts)

    def plot_correlation(self, columns):
        """Plot correlation matrix as a heatmap."""
        plt.figure(figsize=(8, 6))
        corr = self.data[columns].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix', fontsize=12)
        plt.savefig(os.path.join(self.output_dir, "correlation_matrix.png"), dpi=100, bbox_inches='tight')
        plt.show()  # Display in VS Code
        plt.close()

    def plot_scatter(self, x_col, y_col, hue_col):
        """Plot scatter plot for dependencies."""
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=self.sample_data, palette='deep', alpha=0.6)
        plt.title(f'{x_col} vs {y_col} by {hue_col}', fontsize=12)
        plt.xlabel(x_col, fontsize=10)
        plt.ylabel(y_col, fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, f"{x_col}_vs_{y_col}_scatter.png"), dpi=100, bbox_inches='tight')
        plt.show()  # Display in VS Code
        plt.close()

    def handle_imbalance(self, target_column, sample_size=50000):
        """Apply SMOTE to handle class imbalance with limited output size."""
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        smote = SMOTE(random_state=42, sampling_strategy='auto')
        X_balanced, y_balanced = smote.fit_resample(X, y)
        # Limit output size to avoid memory issues
        balanced_data = pd.concat([pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced, name=target_column)], axis=1)
        balanced_data = balanced_data.sample(n=min(sample_size, len(balanced_data)), random_state=42)
        balanced_data.to_csv(os.path.join(self.output_dir, "balanced_data.csv"), index=False)
        print("Imbalanced data handled using SMOTE. Balanced data saved.")

    def compute_features(self):
        """Compute statistical features for ML (Section c)."""
        # Compute interaction feature
        self.data['Age_BMI_Interaction'] = self.data['Age'] * self.data['BMI']
        
        # Selected features
        selected_features = ['Age', 'Average Glucose Level', 'BMI', 'Chronic Stress', 'Sleep Hours', 'Age_BMI_Interaction']
        print("Selected Features for ML:\n", selected_features)
        print("Justification: Age, glucose levels, BMI, and stress are established stroke risk factors. Age_BMI_Interaction captures combined risk.")
        
        # Save dataset with new features
        self.data.to_csv(os.path.join(self.output_dir, "featured_data.csv"), index=False)
        print("Dataset with computed features saved to featured_data.csv")

# Example usage
if __name__ == "__main__":
    explorer = DataExplorer("preprocessed_data.csv")
    numerical_cols = ["Age", "Average Glucose Level", "BMI", "Chronic Stress"]
    categorical_cols = ["Gender"]
    
    # Compute statistics
    explorer.compute_statistics(numerical_cols)
    
    # Plot one of each type for speed
    explorer.plot_histogram("Age")
    explorer.plot_boxplot("BMI")
    explorer.plot_bar("Gender")
    explorer.plot_class_balance("Stroke Occurrence")
    explorer.plot_correlation(numerical_cols + ["Stroke Occurrence"])
    explorer.plot_scatter("Age", "Average Glucose Level", "Stroke Occurrence")
    
    # Handle class imbalance
    explorer.handle_imbalance("Stroke Occurrence")
    
    # Compute features for ML
    explorer.compute_features()
    
    print("All plots saved in 'eda_plots/' folder.")