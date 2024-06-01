from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
pulse_data = []
emotion_data = "Unknown"

def read_serial():
    global pulse_data
    global emotion_data
    # Add your serial reading logic here

@app.route('/')
def index():
    return render_template('index.html', data=pulse_data)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/view_data')
def view_data():
    global pulse_data
    global emotion_data
    app.logger.debug("Accessed view_data route")
    app.logger.debug("Pulse data: %s", pulse_data)
    app.logger.debug("Emotion data: %s", emotion_data)
    if emotion_data:
      return render_template('view_data.html', pulse_data=pulse_data, emotion_data=emotion_data)


    else:
        return "Sorry, there is no data!"

@app.route('/other')
def other():
    return render_template('other.html')

@app.route('/update_emotion', methods=['POST'])
def update_emotion():
    global emotion_data
    emotion = request.json.get('emotion')
    emotion_data = emotion
    app.logger.debug("Updated emotion data: %s", emotion_data)
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
