import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
from eda_module import DataExplorer

# Suppress SVM ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

class UserInterface:
    def __init__(self, data_path="eda_plots/featured_data.csv", sample_size=20000, vis_sample_size=5000):
        """Initialize with dataset path and sample sizes."""
        if not os.path.exists(data_path):
            raise FileNotFoundError("Dataset file {} not found. Run eda_module.py first.".format(data_path))
        self.data = pd.read_csv(data_path)
        self.sample_size = min(sample_size, len(self.data))
        self.vis_sample_size = min(vis_sample_size, len(self.data))
        self.output_dir = "ui_plots"
        os.makedirs(self.output_dir, exist_ok=True)
        plt.style.use('fast')
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=15, random_state=42),
            'SVM': SVC(kernel='linear', max_iter=5000, random_state=42)
        }
        self.features = ['Age', 'Average Glucose Level', 'BMI', 'Sleep Hours', 'Age_BMI_Interaction']
        self.targets = ['Chronic Stress', 'Physical Activity', 'Income Level', 'Stroke Occurrence']
        self.explorer = DataExplorer(data_path, sample_size=vis_sample_size)
        missing_cols = [col for col in self.features + self.targets if col not in self.data.columns]
        if missing_cols:
            raise ValueError("Columns not found in dataset: {}".format(missing_cols))

    def preprocess_stress(self):
        """Preprocess Chronic Stress based on its type."""
        if pd.api.types.is_numeric_dtype(self.data['Chronic Stress']):
            bins = [0, 4, 7, 10]
            labels = [0, 1, 2]  # 0=Low, 1=Medium, 2=High
            self.data['Chronic Stress'] = pd.cut(self.data['Chronic Stress'], bins=bins, labels=labels, include_lowest=True)
            print("Chronic Stress discretized into Low (1-4), Medium (5-7), High (8-10).")
            class_counts = self.data['Chronic Stress'].value_counts()
            print("Chronic Stress distribution:\n", class_counts)
        else:
            print("Chronic Stress is categorical; using as-is.")

    def validate_target(self, target_column):
        """Validate target column data."""
        if self.data[target_column].isnull().all():
            return False, "All values are missing"
        unique_values = self.data[target_column].dropna().unique()
        if len(unique_values) < 2:
            return False, "Only one unique value"
        return True, ""

    def save_placeholder_output(self, target_column, reason):
        """Save placeholder output for invalid data."""
        results = {
            'Model': ['Logistic Regression', 'Random Forest', 'SVM'],
            'Accuracy': [0.0, 0.0, 0.0],
            'Precision': [0.0, 0.0, 0.0],
            'Recall': [0.0, 0.0, 0.0]
        }
        results_df = pd.DataFrame(results)
        print("\nPerformance for {} (Placeholder: {}):".format(target_column, reason))
        print(tabulate(results_df, headers='keys', tablefmt='psql', showindex=False, floatfmt='.2f'))
        print("Confusion Matrix: Not available due to {}".format(reason))

        plt.figure(figsize=(6, 4))
        results_df_melted = results_df.melt(id_vars='Model', value_vars=['Accuracy', 'Precision', 'Recall'])
        sns.barplot(x='Model', y='value', hue='variable', data=results_df_melted, palette='viridis')
        plt.title('Model Performance for {} (Placeholder)'.format(target_column), fontsize=10)
        plt.xlabel('Model', fontsize=8)
        plt.ylabel('Score', fontsize=8)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.2)
        plt.legend(title='Metric', fontsize=8)
        plt.savefig(os.path.join(self.output_dir, "{}_performance.png".format(target_column)), dpi=75, bbox_inches='tight')
        plt.close()
        results_df.to_csv(os.path.join(self.output_dir, "{}_results.csv".format(target_column)))
        print("Placeholder results and plot saved in '{}'.".format(self.output_dir))

    def train_and_evaluate(self, target_column):
        """Train and evaluate models for a target column."""
        # Preprocess Chronic Stress only if the target is Chronic Stress
        if target_column == 'Chronic Stress':
            try:
                self.preprocess_stress()
            except ValueError as e:
                print("Error in preprocessing Chronic Stress: {}".format(e))
                return False

        is_valid, reason = self.validate_target(target_column)
        if not is_valid:
            print("Warning: Cannot train models for {}. Reason: {}.".format(target_column, reason))
            self.save_placeholder_output(target_column, reason)
            return False

        sample_data = self.data.sample(n=self.sample_size, random_state=42, replace=True)
        X = sample_data[self.features]
        y = sample_data[target_column].dropna()

        if len(y) < 2:
            print("Warning: Not enough valid data for {}. Generating placeholder output.".format(target_column))
            self.save_placeholder_output(target_column, "insufficient data")
            return False

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled[:len(y)], y, test_size=0.2, random_state=42)

        results = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': []}
        for model_name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results['Model'].append(model_name)
                results['Accuracy'].append(accuracy_score(y_test, y_pred))
                results['Precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
                results['Recall'].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
                cm = confusion_matrix(y_test, y_pred)
                print("\n{} for {}:".format(model_name, target_column))
                print("Confusion Matrix:\n", cm)
            except ValueError as e:
                print("Error training {} for {}: {}".format(model_name, target_column, e))
                continue

        if not results['Model']:
            print("No models trained successfully for {}. Generating placeholder output.".format(target_column))
            self.save_placeholder_output(target_column, "model training failed")
            return False

        results_df = pd.DataFrame(results)
        print("\nPerformance for {}:".format(target_column))
        print(tabulate(results_df, headers='keys', tablefmt='psql', showindex=False, floatfmt='.2f'))

        plt.figure(figsize=(6, 4))
        results_df_melted = results_df.melt(id_vars='Model', value_vars=['Accuracy', 'Precision', 'Recall'])
        sns.barplot(x='Model', y='value', hue='variable', data=results_df_melted, palette='viridis')
        plt.title('Model Performance for {}'.format(target_column), fontsize=10)
        plt.xlabel('Model', fontsize=8)
        plt.ylabel('Score', fontsize=8)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.2)
        plt.legend(title='Metric', fontsize=8)
        plt.savefig(os.path.join(self.output_dir, "{}_performance.png".format(target_column)), dpi=75, bbox_inches='tight')
        plt.show()  # Display the plot interactively
        plt.close()
        results_df.to_csv(os.path.join(self.output_dir, "{}_results.csv".format(target_column)))
        print("Results and plot saved in '{}'.".format(self.output_dir))
        return True

    def run_eda_submenu(self):
        """Run the EDA submenu with options for statistics and plots."""
        numerical_cols = ['Age', 'Average Glucose Level', 'BMI', 'Chronic Stress']
        categorical_cols = ['Gender']
        target_column = 'Stroke Occurrence'

        while True:
            print("\n=== Exploratory Data Analysis (EDA) Menu ===")
            print("1. Show Descriptive Statistics")
            print("2. Show Histogram (Age)")
            print("3. Show Box Plot (BMI)")
            print("4. Show Bar Plot (Gender)")
            print("5. Show Class Balance (Stroke Occurrence)")
            print("6. Show Correlation Matrix")
            print("7. Show Scatter Plot (Age vs Glucose)")
            print("8. Return to Home")
            try:
                choice = int(input("Enter your choice (number): "))
                if choice == 1:
                    print("\nDisplaying Descriptive Statistics...")
                    self.explorer.compute_statistics(numerical_cols)
                elif choice == 2:
                    print("\nDisplaying Histogram for Age...")
                    self.explorer.plot_histogram('Age')
                elif choice == 3:
                    print("\nDisplaying Box Plot for BMI...")
                    self.explorer.plot_boxplot('BMI')
                elif choice == 4:
                    print("\nDisplaying Bar Plot for Gender...")
                    self.explorer.plot_bar('Gender')
                elif choice == 5:
                    print("\nDisplaying Class Balance for Stroke Occurrence...")
                    self.explorer.plot_class_balance(target_column)
                elif choice == 6:
                    print("\nDisplaying Correlation Matrix...")
                    self.explorer.plot_correlation(numerical_cols + [target_column])
                elif choice == 7:
                    print("\nDisplaying Scatter Plot (Age vs Average Glucose Level)...")
                    self.explorer.plot_scatter('Age', 'Average Glucose Level', target_column)
                elif choice == 8:
                    print("\nReturning to main menu...")
                    break
                else:
                    print("Invalid choice. Please enter a number between 1 and 8.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
            except Exception as e:
                print("Error during EDA execution: {}. Please check data and try again.".format(e))

    def run_ui(self):
        """Run user-friendly text-based UI with EDA option."""
        while True:
            print("\n=== Stroke Prediction System ===")
            print("Welcome! Choose an option below:")
            print("1. Exploratory Data Analysis (EDA)")
            print("2. Predict Chronic Stress")
            print("3. Predict Physical Activity")
            print("4. Predict Income Level")
            print("5. Predict Stroke Occurrence")
            print("6. Exit")
            try:
                choice = int(input("Enter your choice (number): "))
                if choice == 1:
                    print("\nEntering EDA Menu...")
                    self.run_eda_submenu()
                elif choice == 6:
                    print("Thank you for using the Stroke Prediction System. THANK YOU!")
                    break
                elif 2 <= choice <= 5:
                    target = self.targets[choice - 2]
                    print("\nAnalyzing {}...".format(target))
                    success = self.train_and_evaluate(target)
                    if not success:
                        print("Failed to analyze {}. Check data distribution.".format(target))
                else:
                    print("Invalid choice. Please enter a number between 1 and 6.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
            except Exception as e:
                print("Error during execution: {}. Please check data and try again.".format(e))

if __name__ == "__main__":
    try:
        ui = UserInterface()
        ui.run_ui()
    except Exception as e:
        print("Error: {}. Please ensure featured_data.csv exists and contains required columns.".format(e))