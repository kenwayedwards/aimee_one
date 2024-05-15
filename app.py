import os
import streamlit as st
import speech_recognition as sr
#import pyautogui
from langchain import embeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings


import toml

# Load the TOML file
config = toml.load('config.toml')

# Access the API key
OPENAI_API_KEY = config['OPENAI_API_KEY']

# Now you can use the api_key in your application

# Set OpenAI API key
#os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Function for speech-to-text conversion
def speech_to_text():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            st.write("Speak Anything...")
            audio_data = recognizer.listen(source)
            text = recognizer.recognize_google(audio_data, language='en-US')
            st.write("You said:", text)
            return text
    except sr.UnknownValueError:
        st.error("Speech Recognition could not understand audio")
    except sr.RequestError as e:
        st.error("Could not request results from Google Speech Recognition service; {0}".format(e))
    except Exception as e:
        st.error("An error occurred: {0}".format(e))
    return None

# Input email ID to identify the client
email = st.text_input("Please provide your registered email ID:")

# Define client details based on email input
if email:
    if email == "beck@gmail.com":
        client_name = 'Mr. Beck'
        option_taken = 'New Cash'
        loader = DirectoryLoader("content/new_articles_beck/", glob="*.txt", loader_cls=TextLoader)
    elif email == "maverick@gmail.com":
        client_name = 'Mr. Maverick'
        option_taken = 'New Finance'
        loader = DirectoryLoader("content/new_articles_maverick/", glob="*.txt", loader_cls=TextLoader)
    else:
        st.error("Sorry, the provided email ID is not recognized. Please enter a valid email ID.")
else:
    st.warning("Please enter your registered email ID.")

if email in ["beck@gmail.com", "maverick@gmail.com"]:
    # Load documents
    document = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text = text_splitter.split_documents(document)

    # Initialize OpenAI embeddings
    embedding = OpenAIEmbeddings()

    # Create and persist vector database
    persist_directory = 'db'
    vectordb = Chroma.from_documents(documents=text, embedding=embedding, persist_directory=persist_directory)
    vectordb.persist()

    # Initialize OpenAI language model
    llm = OpenAI(temperature=0.4, max_tokens=512)

    # Initialize retriever
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    retriever = vectordb.as_retriever()

    # Main Streamlit app loop
    try:
        while True:
            ask = speech_to_text()

            if ask is not None:
                checker = "AFFIRMATIVE"
                qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

                if "QUIT" in ask.upper():
                    st.write("Conversation ended by admin")
                    break

                else:
                    query = f"""You are 'Katie' which is a chatbot system designed for human-like realistic conversation with clients.
                    {client_name} is your client.
                    Client visited the Automotive Service Center just now to discuss and understand the {option_taken} option for his car.
                    Your objectives are: State and provide the details of 'Service Contract'.
                    Note that you will only state one paragraph in one response covering only one objective at a time. Also if the client wants to continue the conversation with maintenance upgrade, just say "AFFIRMATIVE".
                    So your overall objective is to state the details for the client.
                    Now here is the client's response for you to reply : {ask} """
                    llm_response = qa_chain.invoke(query)
                    if checker in llm_response['result']:
                        query = f"""You are 'Katie' which is a chatbot system designed for human-like realistic conversation with clients.
                        {client_name} is your client.
                        Client visited the Automotive Service Center just now to discuss and understand the {option_taken} option for his car.
                        Your objectives are: State and provide the details of 'Maintenance Upgrade'.
                        Note that you will only state one paragraph in one response covering only one objective at a time.
                        So your overall objective is to state the details for the client.
                        Now here is the client's response for you to reply : {ask} """
                        llm_response = qa_chain.invoke(query)
            
                st.write(llm_response['result'])

                # Simulate pressing the down arrow key, logic is pending to be finalised
                #pyautogui.press('down')
                #pyautogui.press('down')
                #pyautogui.press('down')
                #pyautogui.press('down')
                #pyautogui.press('down')
                #pyautogui.press('down')
                #pyautogui.press('down')
                #pyautogui.press('down')
                #pyautogui.press('down')
                #pyautogui.press('down')
                #pyautogui.press('down')
                #pyautogui.press('down')
                #pyautogui.press('down')
                #pyautogui.press('down')
                #pyautogui.press('down')
                #pyautogui.press('down')
                #pyautogui.press('down')
                #pyautogui.press('down')

            else:
                st.write("No speech input received.")

        # Deleting the DB
        vectordb.delete_collection()
        vectordb.persist()

    except KeyboardInterrupt:
        st.write("\nProgram terminated by user.")
