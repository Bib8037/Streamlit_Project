import streamlit as st
import pandas as pd


def main():
    st.title("Hello World 😎 ")
    st.markdown("## This is streamlit project")

    input_name = st.text_input("Input your name")
    if input_name:
        st.write(f"Hello {input_name} ")
            
    # show_btn = st.button("Show Code!")
    # if show_btn:

        # with st.echo():
            
    st.title("CSV Data Selection")

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
        
            

if __name__ == "__main__":
    main()
