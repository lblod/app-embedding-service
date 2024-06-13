import streamlit as st
import requests
import json 

# Define the URL of the API
API_URL = "http://localhost:2000/search"

# Create a text input for the search query
search_query = st.text_input("Enter your search query:")

# When the user enters a search query, send a POST request to the API
if search_query:
    # Send the POST request
    try:
        response = requests.post(API_URL, headers={'Content-Type': 'application/json'}, data=json.dumps(search_query))
        response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx

        # If the request was successful, display the response
        data = response.json().get("response",{}).get('data', [])
        for item in data:
            st.write({
                'type': item.get('type'),
                'id': item.get('id'),
                'file': item.get('attributes', {}).get('data')
            })

    # Handle exceptions
    except requests.exceptions.HTTPError as errh:
        st.write(f"Http Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        st.write(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        st.write(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        st.write(f"Something went wrong: {err}")