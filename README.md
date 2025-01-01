# Advanced Data Processor - README

Welcome to the **Advanced Data Processor**, an interactive Streamlit web application designed for advanced data preprocessing and visualization tasks. This tool allows users to load, clean, preprocess, transform, and visualize datasets with ease.

## Try it Now!

👉 [Launch Advanced Data Processor](https://dataproces.streamlit.app/)

---

## **Features**

The application provides the following features:

### **1. Load Data**
- Load datasets from:
  - File upload (CSV, TSV, JSON, XML)
  - Sample datasets (Iris, Wine Quality, Boston Housing)
  - URL
- Supports multiple file formats and custom separators.

### **2. View Dataset Info**
- Displays:
  - Dataset shape
  - Data types
  - Missing values summary
  - Descriptive statistics for numeric columns
  - A preview of the first few rows

### **3. Handle Missing Values**
- Impute missing values using:
  - Mean
  - Median
  - Most Frequent Value
  - KNN Imputation
- Option to drop rows with missing values.

### **4. Handle Outliers**
- Detect outliers using:
  - Z-Score
  - Interquartile Range (IQR)
- Treat outliers by:
  - Removing them
  - Capping them within a range
  - Winsorizing them

### **5. Scale Features**
- Scale numeric features using:
  - StandardScaler
  - MinMaxScaler
  - RobustScaler
  - PowerTransformer

### **6. Select Features**
- Select important features based on:
  - Variance Threshold
  - Mutual Information
  - Chi-Square Test
  - ANOVA F-Test

### **7. Reduce Dimensionality**
- Reduce dataset dimensions using:
  - Principal Component Analysis (PCA)
  - Independent Component Analysis (ICA)
  - Non-Negative Matrix Factorization (NMF)

### **8. Encode Categories**
- Encode categorical columns using:
  - Label Encoding
  - One-Hot Encoding
  - Target Encoding
  - Frequency Encoding

### **9. Transform Features**
- Transform numeric features using:
  - Log Transformation
  - Box-Cox Transformation
  - Yeo-Johnson Transformation

### **10. Visualize Data**
- Generate visualizations such as:
  - Distribution plots for individual columns
  - Correlation heatmaps for numeric columns
  - Box plots for outlier detection
  - Scatter matrix for pairwise relationships
  - Feature importance visualization (if a target column exists)

### **11. Save Processed Data**
- Save the processed dataset in various formats:
  - CSV, Excel, JSON, or Pickle.
- Download the dataset directly to your system.

---

## **How to Use**

1. **Access the Application**: Visit [https://dataproces.streamlit.app/](https://dataproces.streamlit.app/) to launch the app.
2. **Navigate Through the Sidebar**: Use the sidebar menu to select a task (e.g., Load Data, Handle Missing Values).
3. **Perform Operations**: Follow on-screen instructions to interactively process your dataset.
4. **Save and Download**: After processing, go to "Save Processed Data" to download the dataset in your preferred format.

---

## **Technical Details**

The application is built using Python and leverages the following libraries:

- **Streamlit**: For building an interactive web application.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Scikit-learn**: For machine learning preprocessing techniques.
- **SciPy**: For statistical transformations.
- **Plotly**: For generating interactive visualizations.

---

## **Installation (For Local Use)**

To run this application locally:

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/advanced-data-processor.git
   cd advanced-data-processor
   ```

2. Install dependencies:
   ```bash
   pip install streamlit pandas numpy scikit-learn scipy plotly openpyxl lxml 
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

4. Open `http://localhost:8501` in your browser.

---

## **Screenshots**

### Sidebar Navigation:
Sidebar Navigation

### Example Visualization (Correlation Heatmap):
Correlation Heatmap

---

## **Contributing**

We welcome contributions to enhance this project! To contribute:

1. Fork this repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

---

## **License**

This project is licensed under the MIT License.

---


