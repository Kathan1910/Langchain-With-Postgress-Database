import os
import pandas as pd
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_sql_agent, AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool
import time

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Database connection
db_user = "postgres"
db_password = "upforce123"
db_host = "localhost"
db_name = "schoolDB"
db_port = 5432
SQLALCHEMY_DATABASE_URL = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
db = SQLDatabase.from_uri(SQLALCHEMY_DATABASE_URL)
# LLM and SQL agent setup
llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106",temperature=0.5)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)

prompt_template = """
Your task is to assist users with their queries. When a query is related to the 'sql_bot' database, refine it for clarity and accuracy, and provide detailed responses in a tabular format. For queries that might result in large data sets, like listing all employees in a particular role, limit the response to a manageable subset (e.g., the first 10 entries) and inform the user about this limitation. For queries that are not related to the database, respond in a conversational and friendly manner. Your responses should be informative and tailored to the type of query. Here are your specific instructions:

1. For database-related queries: Provide detailed answers in a tabular format. If a query could return a large amount of data, limit the response to a subset (like the top 10 entries) and let the user know only a partial list is shown.

2. For non-database-related queries: Engage in a helpful and friendly manner, providing answers or assistance as a conversational chatbot would.

3. For employee experience calculations: Use the join date and leave date in the calculation. If the leave date is earlier than the join date, highlight this as a potential data inconsistency. If there is no leave date, then use the current date.
   
Your role is to communicate effectively with the user, ensuring responses are appropriate to the query type and manageably sized.
"""

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        # Combine the prompt template with the user's query
        full_query = prompt_template + "\nUser query: " + query

        # Process the combined query
        try:
            result = chain.run(full_query)
            st.text(f'Spent a total of {cb} tokens')
        except Exception as e:
            result = "I don't have suffiecient resource to process your query !"
            return e
    return result

# Streamlit app
def main():
    st.title("Universal Chatbot: SQL Database")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    user_query = st.chat_input("Enter your query:")

    if user_query:
        # Displaying the User Message
        with st.chat_message("user"):
            st.markdown(user_query)

        # Process the user's query using the SQL agent
        start_time = time.time()
        response = count_tokens(agent_executor, user_query)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(elapsed_time)

        # Displaying and Storing the Assistant Message
        with st.chat_message("assistant"):
            st.markdown(response)
            st.markdown(elapsed_time)

        # Storing Messages
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()