import streamlit as st
import requests
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from datetime import date
from PIL import Image
from io import BytesIO
import io

# Use the raw URL for the image
image_url = "https://raw.githubusercontent.com/KaranShah1/travel_app/main/cover_image.jpg"

# Fetch and display the image
response = requests.get(image_url)

if response.status_code == 200:
    image = Image.open(BytesIO(response.content))
    st.image(image)
else:
    st.error(f"Failed to load image. Status code: {response.status_code}")


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
