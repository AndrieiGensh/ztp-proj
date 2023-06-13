from flask import Flask, render_template
from flask import request
from csv import DictWriter

from flask import jsonify

import pandas as pnd
import numpy as np

from server import model

import sys, os
from  pathlib import Path

template_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
print(template_dir)
template_dir = os.path.join(template_dir, 'templates')
print(template_dir)


trained_model = model.get_model() 

app = Flask(__name__, template_folder=template_dir)

@app.route("/books", methods = ["POST"])
def books():
    json_data = request.get_json()
    with open('./data/stream_data.csv', 'a') as file:
        dict_writer = DictWriter(file, fieldnames=["User_id", "Date"])
        data = {}
        data["User_id"] = json_data.get("User_id")
        data["Date"] = json_data.get("Date")

        dict_writer.writerow(data)

        file.close()
    print(f"Book data = {json_data.get('Book')}")
    book = model.scale(np.array(json_data.get('Book')).reshape(1,-1))
    print(f"Book data = {book}")
    result = trained_model.predict(book)
    print(f"result = {result}")
    return jsonify(result = {"class": result[0]})

@app.route("/stats", methods = ["GET"])
def stats():
    stats = model.get_stats(trained_model)
    return render_template("metrics.html", Metrics = stats)

@app.route("/time_stats", methods = ["GET", "POST"])
def time_stats():
    if request.method == "GET":
        return render_template("time_stats_base.html")
    else:
        stats = []

        from datetime import datetime

        from_date = datetime.strptime(request.form['from_date'], '%Y-%m-%dT%H:%M')
        to_date = datetime.strptime(request.form['to_date'], '%Y-%m-%dT%H:%M')
        df = pnd.read_csv('./data/stream_data.csv')
        #limit by date, group by user_id, return user_id and how much request they sent
        df["Date"] = df["Date"].apply(pnd.to_datetime)
        df = df.loc[(df["Date"] >= from_date) & (df["Date"] <= to_date)]
        unique_user_ids = df["User_id"].unique() 
        for user_id in unique_user_ids:
            user_rows = df.loc[df["User_id"] == user_id]
            stats.append({"User": int(user_id), "Visits": int(len(user_rows))})
        return render_template("time_stats.html", stats = stats)

@app.route("/models", methods = ["GET","POST"])
def models():
    if request.method == "GET":
        return render_template("new_model_get.html")
    else:
        trained_model = model.get_model()
        import datetime
        time = datetime.datetime.now()
        return render_template("new_model_post.html", time = time)


app.run(host="0.0.0.0")