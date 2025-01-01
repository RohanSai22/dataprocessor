import streamlit as st
import pandas as pd
import numpy as np
import json
import xml.etree.ElementTree as ET
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.impute import KNNImputer, SimpleImputer
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Streamlit App Configuration
st.set_page_config(page_title="Advanced Data Processor", layout="wide")
st.title("Advanced Data Processor")

# Initialize session state for the dataset
if "df" not in st.session_state:
    st.session_state.df = None  # Placeholder for the dataset


class AdvancedDataProcessor:
    def __init__(self):
        self.df = st.session_state.df  # Use Streamlit's session state for persistence

    def load_data(self):
        """Load a dataset from file upload, sample datasets, or URL."""
        st.subheader("Load Dataset")
        data_source = st.radio("Select Data Source", ["Upload File", "Sample Dataset", "URL"])

        if data_source == "Upload File":
            uploaded_file = st.file_uploader("Upload your dataset (CSV/TSV/JSON/XML)", type=["csv", "tsv", "json", "xml"])
            file_type = st.selectbox("Select File Type", ["CSV", "TSV", "JSON", "XML"]).lower()

            if st.button("Load Dataset"):
                if uploaded_file is not None:
                    try:
                        if file_type == "csv":
                            separator = st.text_input("Enter Separator (default is ',')", value=",")
                            self.df = pd.read_csv(uploaded_file, sep=separator)
                        elif file_type == "tsv":
                            self.df = pd.read_csv(uploaded_file, sep="\t")
                        elif file_type == "json":
                            self.df = pd.read_json(uploaded_file)
                        elif file_type == "xml":
                            tree = ET.parse(uploaded_file)
                            root = tree.getroot()
                            data = [{elem.tag: elem.text for elem in child} for child in root]
                            self.df = pd.DataFrame(data)
                        st.session_state.df = self.df  # Save to session state
                        st.success("Dataset loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading file: {str(e)}")
                else:
                    st.warning("Please upload a file before clicking 'Load Dataset'.")

        elif data_source == "Sample Dataset":
            dataset_choice = st.selectbox("Choose Sample Dataset", ["Iris", "Wine Quality", "Boston Housing"])

            if st.button("Load Sample Dataset"):
                try:
                    if dataset_choice == "Iris":
                        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
                        columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
                        self.df = pd.read_csv(url, names=columns)
                    elif dataset_choice == "Wine Quality":
                        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
                        self.df = pd.read_csv(url, sep=";")
                    elif dataset_choice == "Boston Housing":
                        from sklearn.datasets import load_boston  # Deprecated; replace with alternatives if needed.
                        boston = load_boston()
                        self.df = pd.DataFrame(boston.data, columns=boston.feature_names)
                        self.df["PRICE"] = boston.target
                    st.session_state.df = self.df  # Save to session state
                    st.success("Sample dataset loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading sample dataset: {str(e)}")

        elif data_source == "URL":
            url = st.text_input("Enter Dataset URL")
            file_type = st.selectbox("Select File Type (CSV/TSV/JSON/XML)", ["CSV", "TSV", "JSON", "XML"]).lower()

            if st.button("Load Dataset from URL"):
                try:
                    if url:
                        if file_type in ["csv", "tsv"]:
                            separator = "," if file_type == "csv" else "\t"
                            self.df = pd.read_csv(url, sep=separator)
                        elif file_type == "json":
                            self.df = pd.read_json(url)
                        elif file_type == "xml":
                            # Add XML parsing logic here if needed.
                            pass
                        st.session_state.df = self.df  # Save to session state
                        st.success("Dataset loaded successfully from URL!")
                    else:
                        st.warning("Please enter a valid URL before clicking 'Load Dataset from URL'.")
                except Exception as e:
                    st.error(f"Error loading dataset from URL: {str(e)}")

    def display_data_info(self):
        """Display basic information about the loaded dataset."""
        if self.df is not None:
            st.subheader("Dataset Information")
            st.write(f"Shape: {self.df.shape}")
            st.write(f"Data Types:")
            st.write(self.df.dtypes)

            missing_values_summary = self.df.isnull().sum()
            if missing_values_summary.any():
                st.write(f"Missing Values:")
                st.write(missing_values_summary)

            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.write(f"Basic Statistics for Numeric Columns:")
                st.write(self.df[numeric_cols].describe())

            # Display first few rows of the dataset
            st.write(f"First Few Rows:")
            st.dataframe(self.df.head())
        else:
            st.warning("No dataset loaded yet!")
    def handle_missing_values(self):
        """Handle missing values in the dataset with user-selected options."""
        if self.df is None:
            st.warning("No dataset loaded! Please load a dataset first.")
            return

        st.subheader("Handle Missing Values")
        
        # Display missing value summary
        missing_summary = self.df.isnull().sum()
        if missing_summary.sum() == 0:
            st.success("No missing values found in the dataset!")
            return
        
        st.write("Missing Values Summary:")
        st.write(missing_summary[missing_summary > 0])
        
        # Step 1: Choose an imputation method
        methods = ['Mean', 'Median', 'Most Frequent', 'KNN Imputation', 'Drop Rows']
        chosen_method = st.selectbox("Select Imputation Method", methods)
        
        # Step 2: Apply selected method when the button is clicked
        if st.button("Apply Missing Value Treatment"):
            try:
                if chosen_method == 'Drop Rows':
                    self.df.dropna(inplace=True)
                    st.success("Rows with missing values dropped successfully!")
                
                elif chosen_method == 'KNN Imputation':
                    n_neighbors = st.number_input("Number of Neighbors for KNN", min_value=1, max_value=20, value=5)
                    imputer = KNNImputer(n_neighbors=n_neighbors)
                    numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                    self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
                    st.success("Missing values imputed using KNN successfully!")
                
                else:
                    strategy = chosen_method.lower()  # Convert to lowercase for SimpleImputer
                    imputer = SimpleImputer(strategy=strategy)
                    self.df = pd.DataFrame(imputer.fit_transform(self.df), columns=self.df.columns)
                    st.success(f"Missing values handled using {chosen_method} strategy!")
                
                # Display updated dataset preview
                st.write("Updated Dataset Preview:")
                st.dataframe(self.df.head())
            
            except Exception as e:
                st.error(f"Error during missing value treatment: {str(e)}")

    def handle_outliers(self):
        """Detect and handle outliers in the dataset with user-selected options."""
        if self.df is None:
            st.warning("No dataset loaded! Please load a dataset first.")
            return

        st.subheader("Handle Outliers")
        
        # Step 1: Identify numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns found in the dataset.")
            return
        
        # Step 2: Choose an outlier detection method
        methods = ['Z-Score', 'IQR']
        chosen_method = st.selectbox("Select Outlier Detection Method", methods)
        
        # Step 3: Choose an outlier treatment method
        treatment_options = ["Remove", "Cap", "Winsorize"]
        chosen_treatment = st.radio("Select Outlier Treatment", treatment_options)
        
        # Step 4: Apply selected method when the button is clicked
        if st.button("Apply Outlier Handling"):
            try:
                for col in numeric_cols:
                    # Detect outliers using Z-Score or IQR
                    if chosen_method == 'Z-Score':
                        z_scores = np.abs(stats.zscore(self.df[col]))
                        outliers = z_scores > 3
                    elif chosen_method == 'IQR':
                        Q1 = self.df[col].quantile(0.25)
                        Q3 = self.df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = (self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))
                    
                    # Apply the chosen treatment
                    if chosen_treatment == "Remove":
                        self.df = self.df[~outliers]
                    elif chosen_treatment == "Cap":
                        if chosen_method == 'Z-Score':
                            self.df[col] = self.df[col].clip(
                                lower=self.df[col].mean() - 3 * self.df[col].std(),
                                upper=self.df[col].mean() + 3 * self.df[col].std()
                            )
                        elif chosen_method == 'IQR':
                            self.df[col] = self.df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
                    elif chosen_treatment == "Winsorize":
                        from scipy.stats.mstats import winsorize
                        self.df[col] = winsorize(self.df[col], limits=[0.05, 0.05])
                
                st.success(f"Outliers handled using {chosen_method} and treated with {chosen_treatment}.")
                
                # Display updated dataset preview
                st.write("Updated Dataset Preview:")
                st.dataframe(self.df.head())
            
            except Exception as e:
                st.error(f"Error during outlier handling: {str(e)}")

    def scale_features(self):
        """Scale numeric features using various scaling techniques."""
        if self.df is None:
            st.warning("No dataset loaded! Please load a dataset first.")
            return

        st.subheader("Scale Features")
        
        # Step 1: Identify numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns found in the dataset.")
            return
        
        # Step 2: Choose a scaling method
        methods = ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'PowerTransformer']
        chosen_method = st.selectbox("Select Scaling Method", methods)
        
        # Step 3: Apply selected method when the button is clicked
        if st.button("Apply Scaling"):
            try:
                scaler = None
                if chosen_method == 'StandardScaler':
                    scaler = StandardScaler()
                elif chosen_method == 'MinMaxScaler':
                    scaler = MinMaxScaler()
                elif chosen_method == 'RobustScaler':
                    scaler = RobustScaler()
                elif chosen_method == 'PowerTransformer':
                    scaler = PowerTransformer()

                if scaler is not None:
                    # Scale the numeric columns
                    scaled_data = scaler.fit_transform(self.df[numeric_cols])
                    scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols)

                    # Replace original columns with scaled ones
                    for col in numeric_cols:
                        self.df[col] = scaled_df[col]
                    
                    st.success(f"Features scaled using {chosen_method}.")
                    
                    # Display updated dataset preview
                    st.write("Scaled Data Preview:")
                    st.dataframe(self.df.head())
            
            except Exception as e:
                st.error(f"Error during feature scaling: {str(e)}")

    def select_features(self):
        """Select important features based on statistical techniques."""
        if self.df is None:
            st.warning("No dataset loaded! Please load a dataset first.")
            return

        st.subheader("Feature Selection")
        
        # Step 1: Identify numeric columns and ensure target column exists
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2 or "target" not in self.df.columns:
            st.warning("Feature selection requires at least one numeric column and a target column.")
            return

        # Step 2: Choose a feature selection method
        methods = ['Variance Threshold', 'Mutual Information', 'Chi-Square Test', 'ANOVA F-Test']
        chosen_method = st.selectbox("Select Feature Selection Method", methods)
        
        # Step 3: Choose the number of features to select
        n_features_to_select = st.slider(
            "Number of Features to Select",
            min_value=1,
            max_value=len(numeric_cols),
            value=min(5, len(numeric_cols))
        )
        
        # Step 4: Apply selected method when the button is clicked
        if st.button("Apply Feature Selection"):
            try:
                X, y = self.df[numeric_cols], self.df['target']
                selector, selected_features = None, []

                if chosen_method == "Variance Threshold":
                    from sklearn.feature_selection import VarianceThreshold
                    selector = VarianceThreshold(threshold=0.01)
                    selector.fit(X)
                    selected_features = X.columns[selector.get_support()]
                
                elif chosen_method == "Mutual Information":
                    selector = SelectKBest(score_func=mutual_info_classif, k=n_features_to_select)
                    selector.fit(X, y)
                    selected_features = X.columns[selector.get_support()]
                
                elif chosen_method == "Chi-Square Test":
                    selector = SelectKBest(score_func=chi2, k=n_features_to_select)
                    selector.fit(X, y)
                    selected_features = X.columns[selector.get_support()]
                
                elif chosen_method == "ANOVA F-Test":
                    selector = SelectKBest(score_func=f_classif, k=n_features_to_select)
                    selector.fit(X, y)
                    selected_features = X.columns[selector.get_support()]

                # Update the dataframe with selected features and target column
                selected_columns_list = list(selected_features) + ['target']
                self.df = self.df[selected_columns_list]
                
                st.success(f"Selected top {n_features_to_select} features using {chosen_method}.")
                st.write(f"Selected Features: {selected_features}")
            
            except Exception as e:
                st.error(f"Error during feature selection: {str(e)}")
    
    def reduce_dimensionality(self):
        """Reduce dimensionality using PCA, ICA, or NMF."""
        if self.df is None:
            st.warning("No dataset loaded! Please load a dataset first.")
            return

        st.subheader("Dimensionality Reduction")
        
        # Step 1: Identify numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns available for dimensionality reduction.")
            return

        # Step 2: Choose a dimensionality reduction method
        methods = ['PCA', 'ICA', 'NMF']
        chosen_method = st.selectbox("Select Dimensionality Reduction Method", methods)
        
        # Step 3: Choose the number of components to keep
        n_components = st.slider(
            "Number of Components to Keep",
            min_value=1,
            max_value=len(numeric_cols),
            value=min(5, len(numeric_cols))
        )
        
        # Step 4: Apply selected method when the button is clicked
        if st.button("Apply Dimensionality Reduction"):
            try:
                reducer = None
                
                if chosen_method == 'PCA':
                    reducer = PCA(n_components=n_components)
                elif chosen_method == 'ICA':
                    reducer = FastICA(n_components=n_components)
                elif chosen_method == 'NMF':
                    reducer = NMF(n_components=n_components)

                reduced_data = reducer.fit_transform(self.df[numeric_cols])
                reduced_df = pd.DataFrame(
                    reduced_data,
                    columns=[f"{chosen_method}_Component_{i+1}" for i in range(n_components)]
                )

                # Add non-numeric columns back to the reduced dataset
                categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
                if categorical_cols:
                    reduced_df = pd.concat([reduced_df, self.df[categorical_cols].reset_index(drop=True)], axis=1)

                self.df = reduced_df
                
                st.success(f"Dimensionality reduction completed using {chosen_method}.")
                st.write("Reduced Dataset Preview:")
                st.dataframe(self.df.head())
            
            except Exception as e:
                st.error(f"Error during dimensionality reduction: {str(e)}")
                
    def encode_categories(self):
        """Encode categorical columns using various encoding techniques."""
        if self.df is None:
            st.warning("No dataset loaded! Please load a dataset first.")
            return

        st.subheader("Category Encoding")

        # Step 1: Identify categorical columns
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        if not categorical_cols:
            st.warning("No categorical columns found!")
            return

        # Step 2: Choose an encoding method
        methods = ['Label Encoding', 'One-Hot Encoding', 'Target Encoding', 'Frequency Encoding']
        chosen_method = st.selectbox("Select Encoding Method", methods)

        # Step 3: Apply selected method when the button is clicked
        if st.button("Apply Encoding"):
            try:
                for col in categorical_cols:
                    if chosen_method == 'Label Encoding':
                        self.df[col] = self.df[col].astype('category').cat.codes

                    elif chosen_method == 'One-Hot Encoding':
                        one_hot_encoded = pd.get_dummies(self.df[col], prefix=col)
                        self.df.drop(col, axis=1, inplace=True)
                        self.df = pd.concat([self.df, one_hot_encoded], axis=1)

                    elif chosen_method == 'Target Encoding':
                        if 'target' not in self.df.columns:
                            st.warning(f"Skipping {col} - Target column not found for target encoding.")
                            continue
                        target_mean_map = self.df.groupby(col)['target'].mean().to_dict()
                        self.df[col] = self.df[col].map(target_mean_map)

                    elif chosen_method == 'Frequency Encoding':
                        freq_map = self.df[col].value_counts(normalize=True).to_dict()
                        self.df[col] = self.df[col].map(freq_map)

                st.success(f"Categories encoded using {chosen_method}.")
                st.write("Encoded Dataset Preview:")
                st.dataframe(self.df.head())

            except Exception as e:
                st.error(f"Error during category encoding: {str(e)}")
    def transform_features(self):
        """Transform features using log, Box-Cox, or Yeo-Johnson transformations."""
        if self.df is None:
            st.warning("No dataset loaded! Please load a dataset first.")
            return

        st.subheader("Feature Transformation")

        # Step 1: Identify numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns available for transformation.")
            return

        # Step 2: Choose a transformation method
        methods = ['Log Transformation', 'Box-Cox Transformation', 'Yeo-Johnson Transformation']
        chosen_method = st.selectbox("Select Transformation Method", methods)

        # Step 3: Apply selected method when the button is clicked
        if st.button("Apply Transformation"):
            try:
                for col in numeric_cols:
                    if chosen_method == 'Log Transformation':
                        # Handle negative or zero values before applying log transformation
                        min_val = self.df[col].min()
                        if min_val <= 0:
                            self.df[col] += abs(min_val) + 1
                        self.df[col] = np.log1p(self.df[col])

                    elif chosen_method == 'Box-Cox Transformation':
                        # Box-Cox requires positive values
                        if (self.df[col] > 0).all():
                            self.df[col], _ = stats.boxcox(self.df[col])
                        else:
                            st.warning(f"Skipping {col} - Box-Cox requires positive values.")

                    elif chosen_method == 'Yeo-Johnson Transformation':
                        # Yeo-Johnson can handle both positive and negative values
                        transformer = PowerTransformer(method='yeo-johnson')
                        transformed_col = transformer.fit_transform(self.df[[col]])
                        self.df[col] = transformed_col.flatten()

                st.success(f"Features transformed using {chosen_method}.")
                st.write("Transformed Dataset Preview:")
                st.dataframe(self.df.head())

            except Exception as e:
                st.error(f"Error during feature transformation: {str(e)}")
    def visualize_data(self):
        """Visualize the dataset with various plots."""
        if self.df is None:
            st.warning("No dataset loaded! Please load a dataset first.")
            return

        st.subheader("Data Visualization")

        # Step 1: Choose a visualization type
        visualization_types = [
            "Distribution Plots",
            "Correlation Heatmap",
            "Box Plots",
            "Scatter Matrix",
            "Feature Importance (if target exists)"
        ]
        choice = st.selectbox("Choose Visualization Type", visualization_types)

        # Step 2: Identify numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # Step 3: Generate the selected visualization when the button is clicked
        if st.button("Generate Visualization"):
            try:
                if choice == "Distribution Plots":
                    col_to_plot = st.selectbox("Select Column to Plot Distribution", numeric_cols)
                    fig = px.histogram(self.df, x=col_to_plot, title=f"Distribution of {col_to_plot}")
                    st.plotly_chart(fig)

                elif choice == "Correlation Heatmap":
                    correlation_matrix = self.df[numeric_cols].corr()
                    fig = px.imshow(correlation_matrix, text_auto=True, color_continuous_scale="RdBu_r")
                    fig.update_layout(title="Correlation Heatmap")
                    st.plotly_chart(fig)

                elif choice == "Box Plots":
                    col_to_plot = st.selectbox("Select Column for Box Plot", numeric_cols)
                    fig = px.box(self.df, y=col_to_plot, title=f"Box Plot of {col_to_plot}")
                    st.plotly_chart(fig)

                elif choice == "Scatter Matrix":
                    fig = px.scatter_matrix(self.df[numeric_cols], title="Scatter Matrix")
                    fig.update_layout(height=800, width=800)
                    st.plotly_chart(fig)

                elif choice == "Feature Importance (if target exists)":
                    if 'target' in self.df.columns:
                        X = self.df.drop('target', axis=1).select_dtypes(include=[np.number])
                        y = self.df['target']
                        from sklearn.ensemble import RandomForestRegressor
                        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                        rf_model.fit(X, y)
                        importance_df = pd.DataFrame({
                            'Feature': X.columns,
                            'Importance': rf_model.feature_importances_
                        }).sort_values(by="Importance", ascending=False)
                        fig = px.bar(importance_df, x='Feature', y='Importance', title="Feature Importance")
                        st.plotly_chart(fig)
                    else:
                        st.warning("Target column not found in the dataset!")

            except Exception as e:
                st.error(f"Error during visualization: {str(e)}")

    def save_processed_data(self):
        """Save the processed dataset and provide a download option."""
        if self.df is None:
            st.warning("No dataset loaded! Please load or process a dataset first.")
            return

        st.subheader("Save Processed Data")

        # Step 1: Choose file format
        save_format = st.selectbox("Select File Format", ["CSV", "Excel", "JSON", "Pickle"])

        # Step 2: Enter file name
        file_name = st.text_input("Enter File Name (without extension)", value="processed_data")

        # Step 3: Generate the downloadable file when the button is clicked
        if st.button("Generate Download Link"):
            try:
                if save_format == "CSV":
                    csv_data = self.df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"{file_name}.csv",
                        mime="text/csv"
                    )

                elif save_format == "Excel":
                    import io
                    excel_buffer = io.BytesIO()
                    self.df.to_excel(excel_buffer, index=False, engine='openpyxl')
                    excel_buffer.seek(0)
                    st.download_button(
                        label="Download Excel",
                        data=excel_buffer,
                        file_name=f"{file_name}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                elif save_format == "JSON":
                    json_data = self.df.to_json(indent=4).encode('utf-8')
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"{file_name}.json",
                        mime="application/json"
                    )

                elif save_format == "Pickle":
                    import pickle
                    pickle_buffer = pickle.dumps(self.df)
                    st.download_button(
                        label="Download Pickle",
                        data=pickle_buffer,
                        file_name=f"{file_name}.pkl",
                        mime="application/octet-stream"
                    )

                st.success(f"Your {save_format} file is ready for download!")

            except Exception as e:
                st.error(f"Error generating download link: {str(e)}")


# Main Function for Streamlit App
def main():
    # Initialize the processor
    processor = AdvancedDataProcessor()

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    menu_options = [
        "Load Data",
        "View Dataset Info",
        "Handle Missing Values",
        "Handle Outliers",
        "Scale Features",
        "Select Features",
        "Reduce Dimensionality",
        "Encode Categories",
        "Transform Features",
        "Visualize Data",
        "Save Processed Data",
    ]
    choice = st.sidebar.radio("Choose an Option", menu_options)

    # Menu Options
    if choice == "Load Data":
        processor.load_data()
    elif choice == "View Dataset Info":
        processor.display_data_info()
    elif choice == "Handle Missing Values":
        processor.handle_missing_values()
    elif choice == "Handle Outliers":
        processor.handle_outliers()
    elif choice == "Scale Features":
        processor.scale_features()
    elif choice == "Select Features":
        processor.select_features()
    elif choice == "Reduce Dimensionality":
        processor.reduce_dimensionality()
    elif choice == "Encode Categories":
        processor.encode_categories()
    elif choice == "Transform Features":
        processor.transform_features()
    elif choice == "Visualize Data":
        processor.visualize_data()
    elif choice == "Save Processed Data":
        processor.save_processed_data()

# Run the Streamlit App
if __name__ == "__main__":
    main()
