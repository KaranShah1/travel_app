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

# Set background image and apply blur
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://github.com/KaranShah1/travel_app/blob/main/background.jpg?raw=true');
        background-size: cover;
        background-position: center;
        filter: blur(8px);
    }
    .st-container {
        position: relative;
    }
    .st-container > .st-cm {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 1;
    }
    </style>
    """, unsafe_allow_html=True)

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
                    "description": "Get details of places like hotels, restaurants, tourism locations, lakes, mountain, parks etc. from Google Places API. As long as it is some information about Cities or Towns, any minute details of facilities or places in cities, we can get that information here e.g places in New York, Tourist places in Syracuse etc give details about places in that cities",
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
            functions=functions,
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
                    stream=True
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
        
        st.session_state['messages'].append({"role": response_message.role, "content": response_message.content})

