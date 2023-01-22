
from flask import Flask, request, render_template

from model.main import getPrediction

app = Flask(__name__)

@app.route('/', methods=["GET","POST"])
def login_page():
    if request.method == "POST":
        input = request.form['input']
        cleaned_text, pred_sentiment = getPrediction(input)
        return render_template("index.html", text=input, cleaned_text=cleaned_text, pred_sentiment=pred_sentiment)
    else:
        return render_template("index.html")
    

# flask --app flaskr --debug run




