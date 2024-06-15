import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_chat import message
import tempfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from huggingface_hub import hf_hub_download
import os



DB_FAISS_PATH = 'vstore/db_faiss'


# Loading the model
# def load_llm2():
#
#     llm = CTransformers(
#         model="llama-2-7b-chat.ggmlv3.q8_0.bin",
#         model_type="llama",
#         max_new_tokens=512,
#         temperature=0.5
#     )
#     return llm

def load_llm2():
    model_filename = "llama-2-7b-chat.ggmlv3.q8_0.bin"
    model_dir = "./models"
    model_path = os.path.join(model_dir, model_filename)

    # Create the model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Download the model if it doesn't exist locally
    if not os.path.exists(model_path):
        st.write("Downloading model, please wait...")
        try:
            hf_hub_download(
                repo_id="TheBloke/Llama-2-7B-Chat-GGML",
                filename=model_filename,
                local_dir=model_dir
            )
            st.write("Model downloaded successfully.")
        except Exception as e:
            st.write(f"Failed to download model: {e}")
            st.stop()

    llm = CTransformers(
        model=model_path,
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm



# Function to read and parse CSV files
def read_csv(file_path):
    return pd.read_csv(file_path)


# Function to perform statistical analysis
def perform_statistical_analysis(df):
    numeric_df = df.select_dtypes(include='number')
    statistics = {
        'mean': numeric_df.mean(),
        'median': numeric_df.median(),
        'mode': numeric_df.mode().iloc[0],
        'std_dev': numeric_df.std(),
        'correlation': numeric_df.corr()
    }
    return statistics


# Functions to generate plots
def plot_histogram(df, column):
    plt.figure()
    df[column].hist()
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    st.pyplot(plt)


def plot_scatter(df, column1, column2):
    plt.figure()
    plt.scatter(df[column1], df[column2])
    plt.title(f'Scatter Plot of {column1} vs {column2}')
    plt.xlabel(column1)
    plt.ylabel(column2)
    st.pyplot(plt)


def plot_line(df, column):
    plt.figure()
    df[column].plot.line()
    plt.title(f'Line Plot of {column}')
    plt.xlabel('Index')
    plt.ylabel(column)
    st.pyplot(plt)


# Function for conversational chat
def conversational_chat_with_llm(query, chain):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]


uploaded_file = st.sidebar.file_uploader("Upload your File", type="csv")

if uploaded_file:
    if 'uploaded_file_name' not in st.session_state or st.session_state['uploaded_file_name'] != uploaded_file.name:
        # Reset session state when a new file is uploaded
        st.session_state['uploaded_file_name'] = uploaded_file.name
        st.session_state['history'] = []
        st.session_state['generated'] = ["Hello! Ask me anything about " + uploaded_file.name + "🤖"]
        st.session_state['past'] = ["Hey! 👋"]

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)
    llm = load_llm2()
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Chat with CSV", "Statistical Analysis", "Generate Plots"],
            icons=["chat", "bar-chart-line", "graph-up"],
            menu_icon="cast",
            default_index=0,
        )

    if selected == "Chat with CSV":
        st.title("💬 Chat with CSV using Llama2")

        response_container = st.container()
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Talk to your csv data here 😃", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = conversational_chat_with_llm(user_input, chain)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")



    elif selected == "Statistical Analysis":
        st.title("📊 Statistical Analysis")
        df = read_csv(tmp_file_path)
        stats = perform_statistical_analysis(df)
        st.subheader("Statistical Analysis Results")
        st.markdown("### 📈 Mean")
        st.table(stats['mean'].reset_index().rename(columns={0: 'Mean'}))
        st.markdown("### 📉 Median")
        st.table(stats['median'].reset_index().rename(columns={0: 'Median'}))
        st.markdown("### 📊 Mode")
        st.table(stats['mode'].reset_index().rename(columns={0: 'Mode'}))
        st.markdown("### 📉 Standard Deviation")
        st.table(stats['std_dev'].reset_index().rename(columns={0: 'Std Dev'}))
        st.markdown("### 🔄 Correlation Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(stats['correlation'], annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)


    elif selected == "Generate Plots":
        st.title("📈 Generate Plots")
        df = read_csv(tmp_file_path)
        plot_type = st.selectbox("Select plot type", ["Histogram", "Scatter Plot", "Line Plot"])
        if plot_type == "Histogram":
            column = st.selectbox("Select column", df.columns)
            if st.button("Generate Histogram"):
                plot_histogram(df, column)

        elif plot_type == "Scatter Plot":
            column1 = st.selectbox("Select column 1", df.columns)
            column2 = st.selectbox("Select column 2", df.columns)
            if st.button("Generate Scatter Plot"):
                plot_scatter(df, column1, column2)

        elif plot_type == "Line Plot":
            column = st.selectbox("Select column", df.columns)
            if st.button("Generate Line Plot"):
                plot_line(df, column)

else:
    st.markdown("""
        <style>
            .welcome {
                text-align: center;
                font-size: 24px;
                color: #333;
            }
            .instructions {
                margin: 20px 0;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            .instructions h3 {
                color: #444;
            }
            .instructions p {
                color: #666;
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div class="welcome">
            <h1>🤗 Welcome to CSV Data Analysis with Llama2</h1>
        </div>
        <div class="instructions">
            <h3>How to use this app:</h3>
            <p>1. <strong>Upload a CSV file</strong> using the sidebar.</p>
            <p>2. Navigate to:</p>
            <ul>
                <li><strong>Chat with CSV:</strong> Interact with your data using natural language queries.</li>
                <li><strong>Statistical Analysis:</strong> Get descriptive statistics of your data.</li>
                <li><strong>Generate Plots:</strong> Visualize your data using different plot types.</li>
            </ul>
            <p>3. Start exploring and analyzing your data effortlessly!</p>
        </div>
    """, unsafe_allow_html=True)
