import requests

url = 'http://localhost:5000/predict_api'
#r = requests.post(url,json={'experience':2, 'test_score':9, 'interview_score':6})

#r = requests.post(url,json={'NUMBER_SPECIAL_CHARACTERS':7, 'REMOTE_APP_BYTES':832, 'APP_PACKETS':9})

r = requests.post(url,json={'URL LINK':'https://www.spit.ac.in/'})

print(r.json()) # -*- coding: utf-8 -*-

