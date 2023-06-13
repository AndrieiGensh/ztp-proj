import time
import random
from datetime import datetime
import requests
import sys

while True:
    data = {
            "User_id": random.randint(0, 5), 
            "Date": str(datetime(2023, random.randint(4, 6), random.randint(1, 30))),
            "Book": [random.randint(100, 500) for _ in range(5)]
           }
    print(data["Book"])
    requests.post("http://server:5000/books", json=data)
    time.sleep(2)
