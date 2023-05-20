import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def main():
    st.title("Hello World 😎 ")
    st.markdown("## This is streamlit project")

    input_name = st.text_input("Input your name")
    if input_name:
        st.write(f"Hello {input_name} ")
            
    # show_btn = st.button("Show Code!")
    # if show_btn:

        # with st.echo():
            
    st.markdown("## CSV Data Selection")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Display column selection multi-select
        selected_columns = st.multiselect("Select columns", df.columns)

        # Create new DataFrame with selected columns
        new_df = df[selected_columns]

        # Display new DataFrame
        st.write("New DataFrame:")
        st.write(new_df)

        # Prepare subplots
        st.write("Show Data Distribution:")
        fig, axs = plt.subplots(1, min(4, len(selected_columns)), figsize=(20, 5))  # Adjust figure size here

        # Plotting histograms of each selected column
        for i, column in enumerate(selected_columns[:4]):  # Limiting to first 4 columns
            axs[i].hist(new_df[column])
            axs[i].set_title(f"Histogram for {column}")

        # Show the plot
        st.pyplot(fig)

        models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
}
        # Select target variable
        st.write("Develop Machine Learning Model :")
        target = st.selectbox("Select target variable", new_df.columns)

        X = new_df.drop(target, axis=1)
        y = new_df[target]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train and evaluate models
        st.write("Model Evaluation:")
        best_model_name = ""
        best_model_score = 0.0
        for name, model in models.items():
            model.fit(X_train, y_train)
            cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()
            st.write(f"{name} cross-validation score: {cv_score}")
            if cv_score > best_model_score:
                best_model_name = name
                best_model_score = cv_score

        st.write(f"Best Model: {best_model_name} with score: {best_model_score}")
                
            

if __name__ == "__main__":
    main()
