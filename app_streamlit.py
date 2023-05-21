import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LassoLars, BayesianRidge, PassiveAggressiveRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as rmse 
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import itertools
import openai
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import numpy as np
import seaborn as sns
import plotly.express as px
# import shap
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI




def Auto_ML():
    st.title("Hello World 😎 ")
    st.markdown("## This is streamlit project to help develop Machine Learning")

    input_name = st.text_input("Input your name")
    if input_name:
        st.write(f"Hello {input_name} ")
            
    # show_btn = st.button("Show Code!")
    # if show_btn:

        # with st.echo():
    #Slidebar - Specify parameter settings
    st.sidebar.header('Set Parameters')
    split_size = st.sidebar.slider('Data split ratio (% forTraining Set)',10,90,80,5)
    seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 70, 1)
                
    st.markdown("## 1. CSV Data Uploading")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Display column selection multi-select
        selected_columns = st.multiselect("Select columns", df.columns)

        # Create new DataFrame with selected columns
        new_df = df[selected_columns]

        # Display new DataFrame
        st.markdown("## 2. Selected DataFrame:")
        st.write(new_df)
        st.write('---')
        st.markdown("## 3. Filter/clearn data")
        modify = st.checkbox("Add filters")

        def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
            """
            Adds a UI on top of a dataframe to let viewers filter columns

            Args:
                df (pd.DataFrame): Original dataframe

            Returns:
                pd.DataFrame: Filtered dataframe
            """
            if not modify:
                return df

            df = df.copy()

            for col in df.columns:
                if is_object_dtype(df[col]):
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except Exception:
                        pass

                if is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].dt.tz_localize(None)

            modification_container = st.container()

            with modification_container:
                to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
                for column in to_filter_columns:
                    left, right = st.columns((1, 20))
                    # Treat columns with < 10 unique values as categorical
                    if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                        user_cat_input = right.multiselect(
                            f"Values for {column}",
                            df[column].unique(),
                            default=list(df[column].unique()),
                        )
                        df = df[df[column].isin(user_cat_input)]
                    elif is_numeric_dtype(df[column]):
                        _min = float(df[column].min())
                        _max = float(df[column].max())
                        step = (_max - _min) / 100
                        user_num_input = right.slider(
                            f"Values for {column}",
                            min_value=_min,
                            max_value=_max,
                            value=(_min, _max),
                            step=step,
                        )
                        df = df[df[column].between(*user_num_input)]
                    elif is_datetime64_any_dtype(df[column]):
                        user_date_input = right.date_input(
                            f"Values for {column}",
                            value=(
                                df[column].min(),
                                df[column].max(),
                            ),
                        )
                        if len(user_date_input) == 2:
                            user_date_input = tuple(map(pd.to_datetime, user_date_input))
                            start_date, end_date = user_date_input
                            df = df.loc[df[column].between(start_date, end_date)]
                    else:
                        user_text_input = right.text_input(
                            f"Substring or regex in {column}",
                        )
                        if user_text_input:
                            df = df[df[column].astype(str).str.contains(user_text_input)]

            return df
        
        new_df = filter_dataframe(new_df)
        
        # Prepare subplots
        st.write('---')
        st.markdown("## 4. Check Data Distribution:")
        # fig, axs = plt.subplots(1, min(4, len(selected_columns)), figsize=(20, 5))  # Adjust figure size here

        # # Plotting histograms of each selected column
        # for i, column in enumerate(selected_columns[:4]):  # Limiting to first 4 columns
        #     axs[i].hist(new_df[column])
        #     axs[i].set_title(f"Histogram for {column}")

        # # Show the plot
        # st.pyplot(fig)
        # Determine the number of rows for the subplot grid
        num_cols = len(selected_columns)
        num_rows = num_cols // 4
        if num_cols % 4: 
            num_rows += 1  # Add an extra row if columns don't divide evenly by 4

        # Prepare subplots
        fig, axs = plt.subplots(num_rows, 4, figsize=(20, 5 * num_rows))  # Adjust figure size here
        axs = axs.ravel()  # Flatten the array to iterate easily

        # Plotting histograms of each selected column
        for i, column in enumerate(selected_columns):
            axs[i].hist(new_df[column])
            axs[i].set_title(f"Histogram for {column}")

        # If less than num_rows*4 plots, remove the empty subplots
        if num_cols < num_rows * 4:
            for i in range(num_cols, num_rows * 4):
                fig.delaxes(axs[i])
        # Show the plot
        st.pyplot(fig)

        # Calculate the number of scatter plots
        num_plots = len(selected_columns) * (len(selected_columns) - 1) // 2

        # Calculate the number of rows needed for the subplot grid
        num_rows = num_plots // 4
        if num_plots % 4: 
            num_rows += 1  # Add an extra row if the plots don't divide evenly by 4

        # Prepare subplots
        fig, axs = plt.subplots(num_rows, 4, figsize=(20, 5 * num_rows))
        axs = axs.ravel()  # Flatten the array to iterate easily

        # Plotting scatter plots of each pair of selected columns
        i = 0
        for pair in itertools.combinations(selected_columns, 2):
            axs[i].scatter(new_df[pair[0]], new_df[pair[1]])
            axs[i].set_title(f"Scatter plot: {pair[0]} vs {pair[1]}")
            i += 1

        # If less than num_rows*4 plots, remove the empty subplots
        if num_plots < num_rows * 4:
            for i in range(num_plots, num_rows * 4):
                fig.delaxes(axs[i])

        # Show the plot
        st.write('---')
        st.markdown("## 5. Check Data Correlation:")
        st.pyplot(fig)
        st.write('---')

        #HeatMap correlation
        st.markdown("## 6. Check Intercorrelation Matrix Heatmap")
        corr = new_df.corr()
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(7, 5))
            ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
        st.pyplot(f)
        st.write('---')



        #Check accuracy of model
        algo = [
        [LinearRegression(), 
        'LinearRegression', 'This is a simple linear model that fits a straight line through the data to predict the target variable. It assumes that the relationship between the input features and the target variable is linear.'],
        [RandomForestRegressor(), 
        'RandomForestRegressor', 'This is also an ensemble model that combines multiple decision trees to make predictions. It fits each tree to a randomly selected subset of the features and tries to minimize the mean squared error'],
        [SVR(), 
        'SVR', 'Support Vector Regression is a model that tries to find a hyperplane that maximizes the margin around the predicted values. It is effective in high-dimensional spaces and when the number of features is larger than the number of samples.'],
        [NuSVR(), 
        'NuSVR', 'This is similar to SVR but uses a slightly different formulation that allows for more control over the number of support vectors.'],
        [KNeighborsRegressor(),
        'KNeighborsRegressor', 'This is a non-linear model that predicts the target variable based on the mean value of the k-nearest neighbors in the training data.'],
        [Ridge(),
        'Ridge', 'This is a linear model that adds an L2 regularization term to the cost function to prevent overfitting.'],
        [Lasso(),
        'Lass', ' This is a linear model that adds an L1 regularization term to the cost function to select only the most important features.'],
        [ElasticNet(),
        'ElasticNet', 'This is a linear model that adds both L1 and L2 regularization terms to the cost function to balance between feature selection and preventing overfitting.'],
        # [LassoLars(),'LassoLars'],
        [BayesianRidge(),
        'BayesianRidge', 'This is a linear model that uses Bayesian methods to perform regression and estimates the posterior distribution over the model parameters.'],
        [PassiveAggressiveRegressor(),
        'PassiveAggressiveRegressor', ' This is a linear model that updates the model parameters based on whether the prediction error exceeds a threshold.'],
        [MLPRegressor(),
        'MLPRegressor', ' This is a neural network model that consists of multiple layers of interconnected nodes that can model non-linear relationships between the input features and the target variable.'],
        [BaggingRegressor(), 
        'BaggingRegressor', 'This is another ensemble model that combines multiple decision trees to make predictions. It fits each tree to a random subset of the data and tries to minimize the mean squared error.'],
        [AdaBoostRegressor(), 
        'AdaBoostRegressor', 'This is also an ensemble model that combines multiple decision trees to make predictions. It fits each tree to the residuals of the previous tree and tries to minimize the exponential loss function.'],
        [ExtraTreesRegressor(), 
        'ExtraTreesRegressor', 'This is another ensemble model that combines multiple decision trees to make predictions. It fits each tree to a random subset of the data and the features, and tries to minimize the mean squared error.'],
        [GradientBoostingRegressor(), 
        'GradientBoostingRegressor', 'This is an ensemble model that combines multiple decision trees to make predictions. It fits each tree to the residuals of the previous tree and tries to minimize the mean squared error.']
        ]
        
        # Select target variable
        st.markdown("## 7. Develop Machine Learning Model :")
        target = st.selectbox("Select target variable", new_df.columns)

        X = new_df.drop(target, axis=1)
        y = new_df[target]

        
        #slite train and test
        test_size= 1- (split_size/100)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state= seed_number)

        # Train and evaluate models
        st.write('---')
        st.markdown("## 8. Model Evaluation:")
        best_model_name = ""
        best_model_score = 0.0

       
        y_mean = y_train.mean()
        r2_threshold = st.sidebar.slider(f"R2 Score Threshold", 0.0, 1.0, 0.9, key=f'r2_threshold_slide')
        mae_threshold_percentage = st.sidebar.slider(f"MAE Threshold as a percentage of Y value", 0.0, 1.0, 0.05, key=f'mae_threshold_percentage_slide')
        rmse_threshold = st.sidebar.slider(f"RMSE Threshold", 0, 100, 15, key=f'rmse_threshold_slide')
        mae_threshold = mae_threshold_percentage  * y_mean

        for a in algo:
            model = a[0]
            model.fit(X_train,y_train)
            
            st.subheader(str(a[1])+" model : " )
            st.write(str(a[2]))
          
            #Train
            y_prediction_train = model.predict(X_train)
            st.write('Train : Root Mean squared error (RMSE): %.2f'
            % rmse(y_prediction_train, y_train))
            st.write('Train : Mean absolute error (MAE): %.2f'
            % mae(y_prediction_train, y_train))
            st.write('Train : Coefficient of determination (R^2): %.2f'
            % r2_score(y_prediction_train, y_train))
            #Test
            y_prediction_test = model.predict(X_test)
            st.write('Test : Root Mean squared error (RMSE): %.2f'
            % rmse(y_prediction_test, y_test))
            st.write('Test : Mean absolute error (MAE): %.2f'
            % mae(y_prediction_test, y_test))
            st.write('Test : Coefficient of determination (R^2): %.2f'
            % r2_score(y_prediction_test, y_test))

            cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()
            st.write(str(a[1]) +": cross-validation score: " +str(cv_score))
            
            if cv_score > best_model_score:
                best_model_name = str(a[1])
                best_model_score = cv_score

           
            y_mean = y_train.mean()
            mae_threshold = mae_threshold_percentage  * y_mean
    
            # Continue with the rest of the model evaluation as before
            r2_train = r2_score(y_train, y_prediction_train)
            r2_test = r2_score(y_test, y_prediction_test)
            mae_train = mae(y_train, y_prediction_train)
            mae_test = mae(y_test, y_prediction_test)
            rmse_train = rmse(y_train, y_prediction_train)
            rmse_test = rmse(y_test, y_prediction_test)

            # Check if the model is underfitting
            if r2_train < r2_threshold or r2_test < r2_threshold or mae_train > mae_threshold or rmse_train > rmse_threshold or rmse_test > rmse_threshold:
                st.write('**❌ Model Underfit**')

            # Check if the model is a good fit
            elif abs(mae_train - mae_test) <= mae_threshold and abs(rmse_train - rmse_test) <= rmse_threshold:
                st.write('**✅ Good Model**')

            # Otherwise, the model is overfitting
            else:
                st.write('**❌ Model Overfit**')

            #Plotly.express
            fig = px.scatter(df, x = y_prediction_test, y= y_test, labels=dict(x="y_prediction_test", y="y_test"))
            st.write(fig)
            st.write('---')

        st.write(f"Best Model: {best_model_name} with score: {best_model_score}")

        st.markdown("## 9. Feature Importance")
        # Get feature importances
        importances = model.feature_importances_

        # Sort the features by importance
        sorted_indices = importances.argsort()[::-1]
        sorted_features = X.columns[sorted_indices]
        # Create a dataframe with the feature importances and sorted features
        df = pd.DataFrame({
            'Feature': sorted_features,
            'Importance': importances[sorted_indices]
        })
        # Plot the feature importances using plotly.express
        fig = px.bar(df, x='Feature', y='Importance', orientation='v')
        fig.update_layout(xaxis_tickangle=-90)
        st.plotly_chart(fig)

        


      
            
                        
def Open_AI(): 
    # st.title("Hello World  ")
    st.title("AI Assistant : openAI + Streamlit 😎")
    st.markdown("## This is streamlit project connect to OpenAI")
    
    # """
    # This is a Python script that serves as a frontend for a conversational AI model built with the `langchain` and `llms` libraries.
    # The code creates a web application using Streamlit, a Python library for building interactive web apps.
    # # Author: Avratanu Biswas
    # # Date: March 11, 2023
    # """

    # Import necessary libraries
    

    # Set Streamlit page configuration
    # st.set_page_config(page_title='🧠MemoryBot🤖', layout='wide')
    # Initialize session states
    # if "generated" not in st.session_state:
    #     st.session_state["generated"] = []
    # if "past" not in st.session_state:
    #     st.session_state["past"] = []
    # if "input" not in st.session_state:
    #     st.session_state["input"] = ""
    # if "stored_session" not in st.session_state:
    #     st.session_state["stored_session"] = []

    # # Define function to get user input
    # def get_text():
    #     """
    #     Get the user input text.

    #     Returns:
    #         (str): The text entered by the user
    #     """
    #     input_text = st.text_input("You: ", st.session_state["input"], key="input",
    #                             placeholder="Your AI assistant here! Ask me anything ...", 
    #                             label_visibility='hidden')
    #     return input_text

    # # Define function to start a new chat
    # def new_chat():
    #     """
    #     Clears session state and starts a new chat.
    #     """
    #     save = []
    #     for i in range(len(st.session_state['generated'])-1, -1, -1):
    #         save.append("User:" + st.session_state["past"][i])
    #         save.append("Bot:" + st.session_state["generated"][i])        
    #     st.session_state["stored_session"].append(save)
    #     st.session_state["generated"] = []
    #     st.session_state["past"] = []
    #     st.session_state["input"] = ""
    #     st.session_state.entity_memory.entity_store = {}
    #     st.session_state.entity_memory.buffer.clear()

    # # Set up sidebar with various options
    # with st.sidebar.expander("🛠️ ", expanded=False):
    #     # Option to preview memory store
    #     if st.checkbox("Preview memory store"):
    #         with st.expander("Memory-Store", expanded=False):
    #             st.session_state.entity_memory.store
    #     # Option to preview memory buffer
    #     if st.checkbox("Preview memory buffer"):
    #         with st.expander("Bufffer-Store", expanded=False):
    #             st.session_state.entity_memory.buffer
    #     MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo','text-davinci-003','text-davinci-002','code-davinci-002'])
    #     K = st.number_input(' (#)Summary of prompts to consider',min_value=3,max_value=1000)

    # # Set up the Streamlit app layout
    # st.title("🤖 Chat Bot with 🧠")
    # st.subheader(" Powered by 🦜 LangChain + OpenAI + Streamlit")
    #input text
    st.text_input("Your question to ChatGPT about furnace ", key="question")
    question = st.session_state.question

    prompt = f"""<|im_start|>systemAssistant is a highly intelligent chatbot designed to help users answer heater technical questions. 
    <|im_end|><|im_start|>user{question}
    <|im_end|><|im_start|>assistant"""
    # Ask the user to enter their OpenAI API key
    API_O = st.sidebar.text_input("API-KEY", type="password")
    openai.api_type = "azure"
    openai.api_base = "https://chem-dm-openai-dev01.openai.azure.com/"
    openai.api_version = "2022-12-01"
    openai.api_key = API_O


    response = openai.Completion.create(
    engine="gpt-35",#"davinci-003",
    prompt=prompt,
    temperature=1,
    max_tokens=800,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["<|im_end|>"])

    st.subheader("Your anwser from ChatGPT (engine :"+str(response.engine)+") about furnace : ")
    st.write(response['choices'][0]['text'])
        # You can access the value at any point with:
    

    # # Session state storage would be ideal
    # if API_O:
    #     # Create an OpenAI instance
    #     llm = OpenAI(temperature=0,
    #                 openai_api_key=API_O, 
    #                 model_name=MODEL, 
    #                 verbose=False) 


    #     # Create a ConversationEntityMemory object if not already created
    #     if 'entity_memory' not in st.session_state:
    #             st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=K )
            
    #         # Create the ConversationChain object with the specified configuration
    #     Conversation = ConversationChain(
    #             llm=llm, 
    #             prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    #             memory=st.session_state.entity_memory
    #         )  
    # else:
    #     st.sidebar.warning('API key required to try this app.The API key is not stored in any form.')
    #     # st.stop()


    # # Add a button to start a new chat
    # st.sidebar.button("New Chat", on_click = new_chat, type='primary')

    # # Get the user input
    # user_input = get_text()

    # # Generate the output using the ConversationChain object and the user input, and add the input/output to the session
    # if user_input:
    #     output = Conversation.run(input=user_input)  
    #     st.session_state.past.append(user_input)  
    #     st.session_state.generated.append(output)  

    # # Allow to download as well
    # download_str = []
    # # Display the conversation history using an expander, and allow the user to download it
    # with st.expander("Conversation", expanded=True):
    #     for i in range(len(st.session_state['generated'])-1, -1, -1):
    #         st.info(st.session_state["past"][i],icon="🧐")
    #         st.success(st.session_state["generated"][i], icon="🤖")
    #         download_str.append(st.session_state["past"][i])
    #         download_str.append(st.session_state["generated"][i])
        
    #     # Can throw error - requires fix
    #     download_str = '\n'.join(download_str)
    #     if download_str:
    #         st.download_button('Download',download_str)

    # # Display stored conversation sessions in the sidebar
    # for i, sublist in enumerate(st.session_state.stored_session):
    #         with st.sidebar.expander(label= f"Conversation-Session:{i}"):
    #             st.write(sublist)

    # # Allow the user to clear all stored conversation sessions
    # if st.session_state.stored_session:   
    #     if st.sidebar.checkbox("Clear-all"):
    #         del st.session_state.stored_session
        
    #     st.title("Hello World 😎 ")
    #     st.markdown("## This is streamlit project to help summarize content")
    #     st.title("AI Assistant : openAI + Streamlit")
    #     prompt = st.text_input("Enter your message:", key='prompt')
    #     api_secret = "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"
    #     openai.api_key = api_secret#st.secrets[api_secret]

    #     # This function uses the Ope"nAI Completion API to generate a 
    #     # response based on the given prompt. The temperature parameter controls 
    #     # the randomness of the generated response. A higher temperature will result 
    #     # in more random responses, 
    #     # while a lower temperature will result in more predictable responses.

    #     def generate_response(prompt):
    #         completions = openai.Completion.create (
    #             engine="text-davinci-003",
    #             prompt=prompt,
    #             max_tokens=1024,
    #             n=1,
    #             stop=None,
    #             temperature=0.5,
    #         )

    #         message = completions.choices[0].text
    #         return message

        
    #     if st.button("Submit", key='submit'):
    #         response = generate_response(prompt)
    #         st.success(response)
    

if __name__ == "__main__":


    page_names_to_funcs = {
            "Auto Machine Learning": Auto_ML,
            "Open AI Tools": Open_AI,
    }
    demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
    page_names_to_funcs[demo_name]()
