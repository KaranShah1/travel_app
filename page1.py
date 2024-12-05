import streamlit as st
import requests
from openai import OpenAI
import json
from PIL import Image
from io import BytesIO

# API keys
api_key = st.secrets["api_key"]
openai_api_key = st.secrets["key1"]

# Create two columns (left for chat and right for the image)
col1, col2 = st.columns([2, 1])  # Adjust ratio as needed (2 parts left, 1 part right)

# Initialize session state for chat history and search history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'search_history' not in st.session_state:
    st.session_state['search_history'] = []

with col1:
    # Displaying chat history
    for message in st.session_state['messages']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input for chat
    user_query = st.text_input("🔍 What are you looking for? (e.g., 'restaurants in Los Angeles'):")

    if user_query:
        # Add user query to messages history
        if user_query not in st.session_state["search_history"]:
            st.session_state["search_history"].append(user_query)
        
        st.session_state['messages'].append({"role": "user", "content": user_query})

        # Get response from OpenAI (function call or chat)
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


with col2:
    # Display the image in the second column (right side)
    image_url = "https://github.com/KaranShah1/travel_app/blob/main/explore.jpg?raw=true"  # Make sure to use the raw URL for GitHub
    response = requests.get(image_url)

    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        st.image(image, use_column_width=True)
    else:
        st.error(f"Failed to load image. Status code: {response.status_code}")

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

# Chat completion function (using OpenAI API)
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

# Handle function calls (weather and places)
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
                weather_data = get_Weather(location, openai_api_key)  # Replace with actual weather API call
                
                # Provide weather info to GPT for response generation
                messages = [
                    {"role": "user", "content": "Explain in normal English in a few words what kind of clothing can be worn."},
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
