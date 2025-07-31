
import streamlit as st
import pandas as pd
import requests
import google.generativeai as genai
from io import StringIO

# --- Page Configuration ---
st.set_page_config(
    page_title="Instant Site Analysis",
    page_icon="🗺️",
    layout="wide"
)

# --- API Key Configuration ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    API_KEY_CONFIGURED = True
except (KeyError, AttributeError):
    API_KEY_CONFIGURED = False

# --- Helper Functions ---
def geocode_address(address):
    """Converts a street address to latitude and longitude."""
    url = f"https://geocode.maps.co/search?q={address}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        if response.json():
            data = response.json()[0]
            return float(data['lat']), float(data['lon']), data.get('display_name')
    except requests.exceptions.RequestException as e:
        st.error(f"Geocoding API failed: {e}")
    return None, None, None

def get_nearby_earthquakes(lat, lon, radius_km=100):
    """Fetches recent earthquake data from the USGS API."""
    url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&latitude={lat}&longitude={lon}&maxradiuskm={radius_km}&orderby=time"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()['features']
        earthquakes = []
        for feature in data:
            properties = feature['properties']
            coords = feature['geometry']['coordinates']
            earthquakes.append({
                'place': properties['place'],
                'magnitude': properties['mag'],
                'time': pd.to_datetime(properties['time'], unit='ms'),
                'lat': coords[1],
                'lon': coords[0]
            })
        return pd.DataFrame(earthquakes)
    except requests.exceptions.RequestException as e:
        st.error(f"USGS Earthquake API failed: {e}")
    return pd.DataFrame()

def generate_ai_summary(address_name, df):
    """Generates a summary of findings using the Gemini AI model."""
    if not API_KEY_CONFIGURED:
        return "AI model is not configured. Please add your GOOGLE_API_KEY to the Streamlit secrets."
    
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    if df.empty:
        prompt = f"Write a brief, one-paragraph site analysis summary for the location: {address_name}. State that no recent seismic activity was found in the immediate vicinity, which is a positive sign for geological stability."
    else:
        num_quakes = len(df)
        max_mag = df['magnitude'].max()
        prompt = (f"Write a brief, one-paragraph site analysis summary for the location: {address_name}. "
                  f"The analysis found {num_quakes} recent seismic events nearby. "
                  f"The largest magnitude was {max_mag:.2f}. "
                  f"Briefly explain what this might mean for a site risk assessment in a professional but easy-to-understand tone.")
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred with the AI model: {e}"

# --- Streamlit App UI ---
st.title("🗺️ Instant Site Analysis Report")
st.write("Enter a U.S. address to pull live data and generate an AI-powered summary. This demonstrates rapid prototyping with real-time data analysis.")

address = st.text_input("Enter a U.S. Address:", "1600 Amphitheatre Parkway, Mountain View, CA")

if st.button("Analyze Site"):
    with st.spinner("Analyzing..."):
        lat, lon, full_address = geocode_address(address)

        if lat and lon:
            st.session_state.lat = lat
            st.session_state.lon = lon
            st.session_state.full_address = full_address
            
            df = get_nearby_earthquakes(lat, lon)
            st.session_state.df = df
            st.success(f"Successfully analyzed: **{full_address}**")
        else:
            # Error is handled inside the functions now
            pass

if 'df' in st.session_state:
    st.subheader("Seismic Activity Report")
    
    if not st.session_state.df.empty:
        st.write(f"Found **{len(st.session_state.df)}** recent seismic events within 100km.")
        st.dataframe(st.session_state.df)
        
        map_df = st.session_state.df[['lat', 'lon']].copy()
        analyzed_point = pd.DataFrame([{'lat': st.session_state.lat, 'lon': st.session_state.lon}])
        st.map(pd.concat([map_df, analyzed_point]), zoom=7)

        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button(
           label="Download Report as CSV",
           data=csv,
           file_name='seismic_report.csv',
           mime='text/csv',
        )
    else:
        st.info("No recent seismic activity found in the vicinity.")
        st.map(pd.DataFrame([{'lat': st.session_state.lat, 'lon': st.session_state.lon}]), zoom=7)

    st.subheader("🤖 AI-Powered Summary")
    if not API_KEY_CONFIGURED:
        st.warning("AI features are disabled. Please configure your Google API Key in secrets.toml.")
    elif st.button("Generate AI Summary"):
        with st.spinner("AI is thinking..."):
            summary = generate_ai_summary(st.session_state.full_address, st.session_state.df)
            st.session_state.summary = summary

if 'summary' in st.session_state:
    st.markdown(st.session_state.summary)
