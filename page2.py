Page1.py
import streamlit as st
import requests
from openai import OpenAI
import json
import time

# Initialize session state for chat history and search history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'search_history' not in st.session_state:
    st.session_state['search_history'] = []

# Streamlit app title and sidebar filters
st.title("🌍 **Interactive Travel Guide Chatbot** 🤖")
st.markdown("Your personal travel assistant to explore amazing places.")

with st.sidebar:
    st.markdown("### Filters")
    min_rating = st.slider("Minimum Rating", 0.0, 5.0, 3.5, step=0.1)
    max_results = st.number_input("Max Results to Display", min_value=1, max_value=20, value=10)
    st.markdown("___")
    st.markdown("### Search History")
    selected_query = st.selectbox("Recent Searches", options=[""] + st.session_state['search_history'])

# API keys
api_key = st.secrets["api_key"]
openai_api_key = st.secrets["key1"]


functions = [
            {
            "name": "multi_Func",
            "description": "Call two functions in one call",
            "parameters": {
                "type": "object",
                "properties": {
                    "get_Weather": {
                        "name": "get_Weather",
                        "description": "Get the weather for the location.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA",
                                }
                            },
                            "required": ["location"],
                        }
                    },
                    "get_places_from_google": {
                        "name": "get_places_from_google",
                        "description": "Get details of places like hotels, restaurants, tourism locations, lakes, mountain etc. from Google Places API.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                               "query": {"type": "string", "description": "Search query for Google Places API."}
                            },
                            "required": ["query"],
                        }
                    }
                }, "required": ["get_Weather", "get_places_from_google"],
            }
        }
]

# Weather data function
def get_Weather(location, API_key):
    if "," in location:
        location = location.split(",")[0].strip()

    urlbase = "https://api.openweathermap.org/data/2.5/"
    urlweather = f"weather?q={location}&appid={API_key}"
    url = urlbase + urlweather
    response = requests.get(url)
    data = response.json()
    
    return data

# Function to fetch places from Google Places API
def fetch_places_from_google(query):
    base_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": query,
        "key": api_key
    }
    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            # Filter by minimum rating and limit results
            filtered_results = [place for place in results if place.get("rating", 0) >= min_rating]
            return filtered_results[:max_results]
        else:
            return {"error": f"API error {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}


# Function for interacting with OpenAI's API
def chat_completion_request(messages):
    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            functions = functions,
            function_call="auto"
        )
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None


# Handle function calls from GPT response
def handle_function_calls(response_message):
    function_call = response_message.function_call
    if function_call:
        function_name = function_call.name
        function_args = json.loads(function_call.arguments)
        
        weather_data, places_data = None, None

        # Process get_Weather if provided
        if function_args.get("get_Weather"):
            location = function_args["get_Weather"].get("location")
            if location:
                st.markdown(f"Fetching weather for: **{location}**")
                open_api_key = st.secrets['OpenWeatherAPIkey']
                weather_data = get_Weather(location, open_api_key)
                messages = [
                    {"role": "user", "content": "Explain in normal English in few words including what kind of clothing can be worn and what tips need to be taken based on the following weather data."},
                    {"role": "user", "content": json.dumps(weather_data)}
                ]
                client = OpenAI(api_key=openai_api_key)
                stream = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    stream = True
                )
                message_placeholder = st.empty()
                full_response = ""
                if stream:
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                
        # Process get_places_from_google if provided
        if function_args.get("get_places_from_google"):
            query = function_args["get_places_from_google"].get("query")
            if query:
                st.markdown(f"Searching for: **{query}**")
                places_data = fetch_places_from_google(query)

                if isinstance(places_data, dict) and "error" in places_data:
                    st.error(f"Error: {places_data['error']}")
                elif not places_data:
                    st.warning("No places found matching your criteria.")
                else:
                    st.markdown("### 📍 Top Recommendations")
                    for idx, place in enumerate(places_data):
                        with st.expander(f"{idx + 1}. {place.get('name', 'No Name')}"):
                            st.write(f"📍 **Address**: {place.get('formatted_address', 'No address available')}")
                            st.write(f"🌟 **Rating**: {place.get('rating', 'N/A')} (Based on {place.get('user_ratings_total', 'N/A')} reviews)")
                            st.write(f"💲 **Price Level**: {place.get('price_level', 'N/A')}")
                            if "photos" in place:
                                photo_ref = place["photos"][0]["photo_reference"]
                                photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_ref}&key={api_key}"
                                st.image(photo_url, caption=place.get("name", "Photo"), use_column_width=True)
                            lat, lng = place["geometry"]["location"].values()
                            map_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lng}"
                            st.markdown(f"[📍 View on Map]({map_url})", unsafe_allow_html=True)

    else:
        st.error("Function call is incomplete.")

# Display chat history
for message in st.session_state['messages']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# Display chat history and handle user input
user_query = st.text_input("🔍 What are you looking for? (e.g., 'restaurants in Los Angeles'):", value=selected_query)

if user_query:
    if user_query not in st.session_state["search_history"]:
        st.session_state["search_history"].append(user_query)

    st.session_state['messages'].append({"role": "user", "content": user_query})

    # Get response from OpenAI
    with st.spinner("Generating response..."):
        response = chat_completion_request(st.session_state['messages'])

    if response:
        response_message = response.choices[0].message
        
        # Handle function call from GPT
        if response_message.function_call:
            handle_function_calls(response_message)
        else:
            st.session_state['messages'].append({"role": "assistant", "content": response_message.content})
            with st.chat_message("assistant"):
                st.markdown(response_message.content)

Page2.py
import streamlit as st
import requests
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from datetime import date
from PIL import Image
import io

# Function to fetch places from Google Places API
def fetch_places_from_google(query):
    base_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": query, "key": api_key}
    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            # Filter by minimum rating and limit results
            filtered_results = [place for place in results if place.get("rating", 0) >= min_rating]
            return filtered_results[:max_results]
        else:
            return {"error": f"API error {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

# Helper function to resize images
def fetch_and_resize_image(url, size=(200, 200)):
    try:
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        img = img.resize(size)  # Resize to uniform dimensions
        return img
    except Exception as e:
        return None  # Return None if fetching or resizing fails

# Display places in 3x3 grid layout with uniform image sizes and consistent spacing
def display_places_grid(places):
    cols = st.columns(3, gap="medium")  # Adjust gap for spacing between columns
    for idx, place in enumerate(places):
        with cols[idx % 3]:  # Distribute places evenly across 3 columns
            name = place.get("name", "No Name")
            lat, lng = place["geometry"]["location"].values()
            map_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lng}"
            photo_url = None
            if "photos" in place:
                photo_ref = place["photos"][0]["photo_reference"]
                photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_ref}&key={api_key}"

            # Fetch and display image
            if photo_url:
                img = fetch_and_resize_image(photo_url, size=(200, 200))  # Set uniform size
                if img:
                    st.image(img, caption=name, use_column_width=False)
                else:
                    st.write(name)
            else:
                st.write(name)

            # Link to map
            st.markdown(f"[📍 View on Map]({map_url})", unsafe_allow_html=True)
            
            # Manage itinerary bucket
            if name in st.session_state['itinerary_bucket']:
                st.button("Added", disabled=True, key=f"added_{idx}")
            else:
                if st.button("Add to Itinerary", key=f"add_{idx}"):
                    st.session_state['itinerary_bucket'].append(name)

        # Add vertical spacing between rows
        if (idx + 1) % 3 == 0:  # After every 3 places
            st.write("")  # Empty line for spacing between rows

# Function to generate an itinerary using LangChain
def plan_itinerary_with_langchain():
    if not st.session_state['itinerary_bucket']:
        st.warning("No places in itinerary bucket!")
        return

    st.markdown("### 🗺️ AI-Generated Itinerary")
    places_list = "\n".join(st.session_state['itinerary_bucket'])

    if selected_date:
        st.info(f"Planning itinerary for {selected_date.strftime('%A, %B %d, %Y')} 🎉")
    else:
        st.info("No specific date chosen. Starting from 9:00 AM by default.")

    prompt_template = PromptTemplate(
        input_variables=["places", "date"],
        template="""Plan a travel itinerary for the following places:
        {places}
        Date of travel: {date}
        Provide a detailed plan including the best order to visit, time at each location, transportation time, and meal breaks.
        """
    )

    date_str = selected_date.strftime('%A, %B %d, %Y') if selected_date else "Not specified"
    formatted_prompt = prompt_template.format(places=places_list, date=date_str)

    with st.spinner("Generating your itinerary..."):
        response = llm([HumanMessage(content=formatted_prompt)])
        st.markdown(response.content)

# Initialize session state for itinerary bucket and search history
if 'itinerary_bucket' not in st.session_state:
    st.session_state['itinerary_bucket'] = []
if 'search_history' not in st.session_state:
    st.session_state['search_history'] = []

# Streamlit app title and sidebar filters
st.title("🌍 **Travel Planner with AI** ✈️")
st.markdown("Discover amazing places and plan your trip effortlessly!")

# Sidebar with filters, search history, and saved itineraries
with st.sidebar:
    st.markdown("### Filters")
    min_rating = st.slider("Minimum Rating", 0.0, 5.0, 3.5, step=0.1)
    max_results = st.number_input("Max Results to Display", min_value=1, max_value=20, value=9)
    st.markdown("___")
    st.markdown("### Search History")
    selected_query = st.selectbox("Recent Searches", options=[""] + st.session_state['search_history'])
    
# API key for Google Places API
api_key = st.secrets["api_key"]
openai_api_key = st.secrets["openai_api_key"]

# Initialize LangChain ChatOpenAI model
llm = ChatOpenAI(temperature=0.3, model="gpt-4o-mini", openai_api_key=openai_api_key, verbose=True)

# Handle search input
user_query = st.text_input("🔍 Search for places (e.g., 'restaurants in Paris'):", value=selected_query)
selected_date = st.date_input("Choose a date for your trip (optional):", value=None)
if user_query:
    if user_query not in st.session_state["search_history"]:
        st.session_state["search_history"].append(user_query)

    st.markdown(f"### Results for: **{user_query}**")
    with st.spinner("Fetching places..."):
        places_data = fetch_places_from_google(user_query)

    if isinstance(places_data, dict) and "error" in places_data:
        st.error(f"Error: {places_data['error']}")
    elif not places_data:
        st.warning("No places found matching your criteria.")
    else:
        display_places_grid(places_data)

    # Show itinerary bucket
    # Show itinerary bucket
    st.markdown("### 📋 Itinerary Bucket")
            # Button to clear the entire itinerary bucket
    if st.button("Clear Itinerary Bucket"):
        st.session_state['itinerary_bucket'] = []  # Clear the list
        st.success("Itinerary bucket cleared!")
    if st.session_state['itinerary_bucket']:
        # Display itinerary items with remove buttons
        for place in st.session_state['itinerary_bucket']:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(place)
            with col2:
                if st.button("Remove", key=f"remove_{place}"):
                    st.session_state['itinerary_bucket'].remove(place)
    
    else:
        st.write("Your itinerary bucket is empty.")

    # Generate itinerary button
    if st.button("Generate AI Itinerary"):
        plan_itinerary_with_langchain()

Page3.py
# page3-whisper.py
import streamlit as st
from openai import OpenAI
import os
from audio_recorder_streamlit import audio_recorder
import base64
import time

# Dictionary of countries and their primary languages
COUNTRY_LANGUAGES = {
    "Spain": "Spanish",
    "France": "French",
    "Germany": "German",
    "Italy": "Italian",
    "Japan": "Japanese",
    "China": "Chinese",
    "Brazil": "Portuguese",
    "North India": "Hindi",
    "Telangana / Andhra Pradesh": "Telugu",
}

# Setup OpenAI client
if 'openai_client' not in st.session_state:
    api_key = st.secrets["openai_api_key"]
    st.session_state.openai_client = OpenAI(api_key=api_key)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'last_recorded_audio' not in st.session_state:
    st.session_state.last_recorded_audio = None
if 'target_language' not in st.session_state:
    st.session_state.target_language = None

# Function to transcribe audio
def transcribe_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = st.session_state.openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcript.text

# Function to convert text to audio
def text_to_audio(text, audio_path, voice="nova"):
    response = st.session_state.openai_client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text
    )
    response.stream_to_file(audio_path)

# Function to play audio
def auto_play_audio(audio_file):
    if os.path.exists(audio_file):
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        audio_html = f'<audio src="data:audio/mp3;base64,{base64_audio}" controls autoplay>'
        st.markdown(audio_html, unsafe_allow_html=True)

# Function to translate text
def translate_text(text, target_language):
    messages = [
        {"role": "system", "content": f"You are a translator. Translate the following text to {target_language}. Maintain the tone and meaning of the original text. Only respond with the translation, no additional text. Also, do not talk too fast"},
        {"role": "user", "content": text}
    ]
    
    response = st.session_state.openai_client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.75
    )
    
    return response.choices[0].message.content

# Function to process input
def process_input(text, target_language, is_voice=False):
    translated_text = translate_text(text, target_language)
    
    if is_voice:
        audio_file = f"audio_response_{int(time.time())}.mp3"
        text_to_audio(translated_text, audio_file)
        return translated_text, audio_file
    
    return translated_text, None

# Main page content
st.title("Travel Translation Assistant")

# Country/Language Selection
st.sidebar.header("Translation Settings")
country_selection = st.sidebar.selectbox(
    "Where are you traveling to?",
    options=list(COUNTRY_LANGUAGES.keys()),
    key="country_selection"
)

# Update target language based on country selection
st.session_state.target_language = COUNTRY_LANGUAGES[country_selection]
st.sidebar.write(f"Translation will be provided in: {st.session_state.target_language}")

# Chat interface
chat_container = st.container()

# Display chat history
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "translation" in message:
                st.write(f"🔄 {message['translation']}")
            if "audio" in message:
                auto_play_audio(message["audio"])

# Voice and text input
col1, col2 = st.columns([8, 2])

with col2:
    recorded_audio = audio_recorder(
        text="",
        recording_color="#e74c3c",
        neutral_color="#95a5a6",
        key="voice_recorder"
    )

with col1:
    text_input = st.chat_input("Type your message or use voice input...")

# Handle text input
if text_input:
    translation, _ = process_input(text_input, st.session_state.target_language)
    
    st.session_state.messages.append({
        "role": "user",
        "content": text_input,
        "translation": translation
    })
    st.rerun()

# Handle voice input
if recorded_audio is not None and recorded_audio != st.session_state.last_recorded_audio:
    st.session_state.last_recorded_audio = recorded_audio
    
    # Save and transcribe audio
    audio_file = f"audio_input_{int(time.time())}.mp3"
    with open(audio_file, "wb") as f:
        f.write(recorded_audio)

    # Transcribe the audio
    transcribed_text = transcribe_audio(audio_file)
    os.remove(audio_file)

    # Get translation and audio response
    translation, response_audio = process_input(
        transcribed_text, 
        st.session_state.target_language, 
        is_voice=True
    )

    # Update chat history
    st.session_state.messages.append({
        "role": "user",
        "content": f"🎤 {transcribed_text}",
        "translation": translation,
        "audio": response_audio
    })
    
    st.rerun()

Page4.py
import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

# Initialize OpenAI client
if 'openai_client' not in st.session_state:
    api_key = st.secrets['key1']
    st.session_state.openai_client = OpenAI(api_key=api_key)

# Function to add PDF content to ChromaDB collection
def add_to_collection(collection, text, filename):
    openai_client = st.session_state.openai_client
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    collection.add(
        documents=[text],
        ids=[filename],
        embeddings=[embedding]
    )
    return collection

# Function to set up VectorDB if not already created
def setup_vectordb():
    if 'travelfaq_vectorDB' not in st.session_state:
        client = chromadb.PersistentClient()
        collection = client.get_or_create_collection(
            name="travelfaq_collection",
            metadata={"hnsw:space": "cosine", "hnsw:M": 32}
        )
        
        datafiles_path = os.path.join(os.getcwd(), "datafiles")
        pdf_files = [f for f in os.listdir(datafiles_path) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            file_path = os.path.join(datafiles_path, pdf_file)
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                collection = add_to_collection(collection, text, pdf_file)
        
        st.session_state.travelfaq_vectorDB = collection
        st.success(f"Welcome to Trip Assistor")
    else:
        st.info("Welcome to Trip Assistor Dear!!")

# Function to query the VectorDB and retrieve relevant documents
def query_vectordb(query, k=3):
    if 'travelfaq_vectorDB' in st.session_state:
        collection = st.session_state.travelfaq_vectorDB
        openai_client = st.session_state.openai_client
        response = openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
        results = collection.query(
            query_embeddings=[query_embedding],
            include=['documents', 'distances', 'metadatas'],
            n_results=k
        )
        return results
    else:
        st.error("VectorDB not set up. Please set up the VectorDB first.")
        return None

# Function to get a response from OpenAI using the retrieved context
def get_ai_response(query, context):
    openai_client = st.session_state.openai_client
    messages = [
        {"role": "system", "content": "You are a helpful assistant with knowledge about the trips and safety of people! You politely answer the questions."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
    ]
    for msg in st.session_state.messages:
        messages.append(msg)
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=150
    )
    return response.choices[0].message.content

# Main Streamlit app
st.title("Your Trip Assistant")

# Set up the VectorDB if it's not already set up
setup_vectordb()

# Initialize chat history if not already in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input and respond
if prompt := st.chat_input("Hello Travelor, How can I help you?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query VectorDB for relevant documents
    results = query_vectordb(prompt)
    
    # Set a distance threshold (adjust as needed)
    DISTANCE_THRESHOLD = 0.7
    
    if results and results['documents'][0] and results['distances'][0][0] < DISTANCE_THRESHOLD:
        # Retrieve document content from the vector DB and use it as context
        context = " ".join([doc for doc in results['documents'][0]])
        response = get_ai_response(prompt, context)
        # Indicate that the bot is using context from the RAG pipeline
        st.session_state.messages.append({"role": "system", "content": response})
        with st.chat_message("system"):
            st.markdown(response)
    else:
        # If no relevant documents were found, generate response without document context
        response = get_ai_response(prompt, "")
        st.session_state.messages.append({"role": "system", "content":response})
        with st.chat_message("system"):
            st.markdown(response)
