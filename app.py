import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    
    for url in request.form.values():
    
    #url="https://classroom.google.com/u/0/h"
   
        #VARIABLES
        content_length = 0        
        
        #URL_LENGTH
        length_feature = len(url)        
    
        #NUMBER OF SPECIAL CHARACTERS
        import re
        new = re.sub('[^\^&#%*$]+','',url)
        spec_Char = len(new)        
        
        
        
        #CONTENT LENGTH
        import requests
        x = requests.head(url)
        dict = x.headers;
        #print(dict)
        key='Content-Length'
        
        def checkKey(dict, key): 
      
            if key in dict.keys(): 
                content_length = x.headers['Content-Length']        
            else: 
                content_length = 0   
            #print(content_length)           
    
        checkKey(dict, key)  
        
        """
        
        
       
        #SOURCE APP BYTES - BYTES DURING A REQUEST         
        #use the 'auth' parameter to send requests with HTTP Basic Auth:
        r = requests.get(url, auth = ('user', 'pass'))
        
        src_app_bytes=len(r.content)
        #print(a)  
        
        """      
       
        #SERVER NAME
        y = requests.head(url)  
        server_name = y.headers['Server']
        
        from cal_server_val import calculate_ser_val
        #calling functions
        server_val1 = calculate_ser_val(server_name) 
        
        
        #final_features = [np.array([length_feature,spec_Char,content_length,server_val1,src_app_bytes])]
        #final_features = [np.array([length_feature,spec_Char])]
        final_features = [np.array([length_feature,spec_Char,content_length,server_val1])]
        prediction = model.predict(final_features) 
        #print(final_features)    
    
        if(prediction == 1):    
            output = "MALICIOUS"    
        else:
            output = "NOT MALICIOUS" 

    
    return render_template('index.html', prediction_text="THE URL IS " + output)  

if __name__ == "__main__":
    app.run(debug=True)# -*- coding: utf-8 -*-

