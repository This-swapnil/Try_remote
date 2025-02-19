from flask import Flask, jsonify, request
import threading
import logging
import subprocess
import sys

app = Flask(__name__)

# List to store messages
messages = []

stt = subprocess.Popen(["streamlit", "run" ,"streamlit_app.py"])

def input_thread():
    while True:
        text = input("Enter text to display on the web UI: ")
        messages.append(text)

@app.route('/get_messages', methods=['GET'])
def get_messages():
    return jsonify({"messages": messages})

try:
    if __name__ == '__main__':
        # Starting a thread for terminal input to not block the Flask app
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.WARNING)
        
        thread = threading.Thread(target=input_thread)
        thread.daemon = True
        thread.start()

        # Starting Flask app
        app.run(port=5000)

except KeyboardInterrupt:
    stt.terminate()
    sys.exit(0)