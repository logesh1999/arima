from flask import Flask, request, render_template
import
app = Flask(__name__, template_folder='templates')

@app.route('/pedict')
def predict():

    model =



if __name__ == "__main__":
    app.run(debug=True)