import streamlit as st
import requests
import signal 

def sigterm_handler(signum, frame):
    sys.exit(0)

#signal.signal(signal.SIGTERM, sigterm_handler)

def fetch_messages():
    try:
        response = requests.get('http://localhost:5000/get_messages')
        if response.status_code == 200:
            return response.json()["messages"]
        else:
            return []
    except Exception as e:
        st.error(f"Failed to fetch messages: {str(e)}")
        return []

def display_messages(messages):
    for message in messages:
        st.info(message)

def main():
    st.title('Terminal Text Display in Word Bubbles')

    displayed_messages = []
    while True:
        messages = fetch_messages()
        new_messages = [msg for msg in messages if msg not in displayed_messages]
        
        if new_messages:
            display_messages(new_messages)
            displayed_messages.extend(new_messages)
        st.rerun()

if __name__ == "__main__":
    main()

