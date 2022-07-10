# import essential libraries
from django.shortcuts import redirect
from flask import Flask,render_template,request, url_for
import pickle
import numpy as np

# load the Random forest Classifier model
filename='IPL_rf.pkl'
clf=pickle.load(open(filename,'rb'))

app=Flask(__name__,template_folder='template')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    temp_array=list()

    if request.method=='POST':
        batting_team=request.form['BattingTeam']
        if batting_team=='Chennai Super Kings':
            temp_array=temp_array+[1,0,0,0,0,0,0,0,0,0]
        elif batting_team=='Mumbai Indians':
            temp_array=temp_array+[0,1,0,0,0,0,0,0,0,0]
        elif batting_team=='Royal Challengers Bengalore':
            temp_array=temp_array+[0,0,1,0,0,0,0,0,0,0]
        elif batting_team=='Rajasthan Royal':
            temp_array=temp_array+[0,0,0,1,0,0,0,0,0,0]
        elif batting_team=='Punjab Kings':
            temp_array=temp_array+[0,0,0,0,1,0,0,0,0,0]
        elif batting_team=='Kolkata Knight Riders':
            temp_array=temp_array+[0,0,0,0,0,1,0,0,0,0]
        elif batting_team=='Lucknow Super Giants':
            temp_array=temp_array+[0,0,0,0,0,0,1,0,0,0]
        elif batting_team=='Gujarat titans':
            temp_array=temp_array+[0,0,0,0,0,0,0,1,0,0]
        elif batting_team=='Delhi Capitals':
            temp_array=temp_array+[0,0,0,0,0,0,0,0,1,0]
        elif batting_team=='Sunrisers Hyderabad':
            temp_array=temp_array+[0,0,0,0,0,0,0,0,0,1]
        
        bowling_team=request.form['BowlingTeam']
        if bowling_team=='Chennai Super Kings':
            temp_array=temp_array+[1,0,0,0,0,0,0,0,0,0]
        elif bowling_team=='Mumbai Indians':
            temp_array=temp_array+[0,1,0,0,0,0,0,0,0,0]
        elif bowling_team=='Royal Challengers Bengalore':
            temp_array=temp_array+[0,0,1,0,0,0,0,0,0,0]
        elif bowling_team=='Rajasthan Royal':
            temp_array=temp_array+[0,0,0,1,0,0,0,0,0,0]
        elif bowling_team=='Punjab Kings':
            temp_array=temp_array+[0,0,0,0,1,0,0,0,0,0]
        elif bowling_team=='Kolkata Knight Riders':
            temp_array=temp_array+[0,0,0,0,0,1,0,0,0,0]
        elif bowling_team=='Lucknow Super Giants':
            temp_array=temp_array+[0,0,0,0,0,0,1,0,0,0]
        elif bowling_team=='Gujarat titans':
            temp_array=temp_array+[0,0,0,0,0,0,0,1,0,0]
        elif bowling_team=='Delhi Capitals':
            temp_array=temp_array+[0,0,0,0,0,0,0,0,1,0]
        elif bowling_team=='Sunrisers Hyderabad':
            temp_array=temp_array+[0,0,0,0,0,0,0,0,0,1]

        city=request.form['City']
        if city=='Ahmedabad':
            temp_array=temp_array+[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif city=='Kolkata':
            temp_array=temp_array+[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif city=='Mumbai':
            temp_array=temp_array+[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif city=='Navi Mumbai':
            temp_array=temp_array+[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif city=='Pune':
            temp_array=temp_array+[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif city=='Dubai':
            temp_array=temp_array+[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif city=='Abu Dhabi':
            temp_array=temp_array+[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif city=='Delhi':
            temp_array=temp_array+[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif city=='Chennai':
            temp_array=temp_array+[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif city=='Hyderabad':
            temp_array=temp_array+[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif city=='Vishakapatnam':
            temp_array=temp_array+[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif city=='Chandigarh':
            temp_array=temp_array+[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif city=='Bengaluru':
            temp_array=temp_array+[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif city=='Jaipur':
            temp_array=temp_array+[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif city=='Indore':
            temp_array=temp_array+[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif city=='Bangalore':
            temp_array=temp_array+[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif city=='Raipur':
            temp_array=temp_array+[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif city=='Ranchi':
            temp_array=temp_array+[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
        elif city=='Cuttack':
            temp_array=temp_array+[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
        elif city=='Dharamsala':
            temp_array=temp_array+[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
        elif city=='Nagpur':
            temp_array=temp_array+[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
        elif city=='Johannesburg':
            temp_array=temp_array+[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
        elif city=='Centurion':
            temp_array=temp_array+[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
        elif city=='Durban':
            temp_array=temp_array+[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
        elif city=='Bloemfontein':
            temp_array=temp_array+[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
        elif city=='Port Elizabeth':
            temp_array=temp_array+[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
        elif city=='Kimberley':
            temp_array=temp_array+[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
        elif city=='East London':
            temp_array=temp_array+[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
        elif city=='Cape Town':
            temp_array=temp_array+[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        elif city=='Sharjah':
            temp_array=temp_array+[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]


        runs_left=int(request.form['runs_left'])
        balls_left=int(request.form['balls_left'])
        wickets_left=int(request.form['wickets_left'])
        current_run_rate=float(request.form['current_run_rate'])
        required_run_rate=float(request.form['required_run_rate'])
        target=int(request.form['Target'])

        temp_array=temp_array+[runs_left,balls_left,wickets_left,current_run_rate,required_run_rate,target]

        data=np.array([temp_array])
        my_prediction=int(clf.predict(data))

        return render_template('result.html',prediction=my_prediction)
if __name__=='__main__':
        app.run(debug=True)
