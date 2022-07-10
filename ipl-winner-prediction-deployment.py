from fileinput import filename
import numpy as np
import pandas as pd
import pickle

ball=pd.read_csv('IPL_Ball_by_Ball_2008_2022.csv')
match=pd.read_csv('IPL_Matches_2008_2022.csv')
ball=ball.sample(50000)

# Removing columns having more than 90% nan values
ball.drop(['fielders_involved','kind','player_out','extra_type'],axis=1,inplace=True)
# Removing columns from the match datasets
match.drop(['Season','SuperOver','Team1Players','Team2Players','Umpire1','Umpire2','Margin','method','MatchNumber'],axis=1,inplace=True)

total_score=ball.groupby(['ID','innings'])['total_run'].sum().reset_index()

# I need only First Innings Score 
total_score=total_score[total_score['innings']==1]

# add "target" column in total_score
total_score['Target']=total_score['total_run']+1

match_df=match.merge(total_score[['ID','Target']],on='ID')

match_df['Team1'].unique()

teams = [
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad', 
    'Delhi Capitals', 
    'Chennai Super Kings',
    'Gujarat Titans', 
    'Lucknow Super Giants', 
    'Kolkata Knight Riders',
    'Punjab Kings', 
    'Mumbai Indians'
]

match_df['Team1']=match_df['Team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['Team2']=match_df['Team2'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['WinningTeam']=match_df['WinningTeam'].str.replace('Delhi Daredevils','Delhi Capitals')

match_df['Team1']=match_df['Team1'].str.replace('Deccan Chargers' ,'Sunrisers Hyderabad')
match_df['Team2']=match_df['Team2'].str.replace('Deccan Chargers' ,'Sunrisers Hyderabad')
match_df['WinningTeam']=match_df['WinningTeam'].str.replace('Deccan Chargers' ,'Sunrisers Hyderabad')

match_df['Team1']=match_df['Team1'].str.replace('Kings XI Punjab','Punjab Kings')
match_df['Team2']=match_df['Team2'].str.replace('Kings XI Punjab','Punjab Kings')
match_df['WinningTeam']=match_df['WinningTeam'].str.replace('Kings XI Punjab','Punjab Kings')

match_df=match_df[match_df['Team1'].isin(teams)]
match_df=match_df[match_df['Team2'].isin(teams)]
match_df=match_df[match_df['WinningTeam'].isin(teams)]

ball['BattingTeam']=ball['BattingTeam'].str.replace('Deccan Chargers' ,'Sunrisers Hyderabad')
ball['BattingTeam']=ball['BattingTeam'].str.replace('Kings XI Punjab' ,'Punjab Kings')
ball['BattingTeam']=ball['BattingTeam'].str.replace('Delhi Daredevils','Delhi Capitals')

ball=ball[ball['BattingTeam'].isin(teams)]

match_df=match_df[['ID','Team1','Team2','City','WinningTeam','Target']].dropna()

ball_df=match_df.merge(ball,on='ID')
ball_df.drop(['batter','bowler','non-striker','batsman_run','extras_run'],axis=1,inplace=True)

# Now we need to focus on Second Innings only
ball_df=ball_df[ball_df['innings']==2]

ball_df['current_score']=ball_df.groupby(['ID'])['total_run'].cumsum()

ball_df['runs_left']=np.where(ball_df['Target']-ball_df['current_score']>=0,ball_df['Target']-ball_df['current_score'],0)

ball_df['wickets_left']=10-ball_df.groupby(['ID'])['isWicketDelivery'].cumsum()

ball_df['balls_left']=np.where(120-ball_df['overs']*6-ball_df['ballnumber']>=0,120-ball_df['overs']-ball_df['ballnumber'],0)

ball_df['current_run_rate']=(ball_df['current_score']*6/(120-ball_df['balls_left']))

ball_df['required_run_rate']=np.where(ball_df['balls_left']>0, (ball_df['runs_left']*6)/ball_df['balls_left'],0)

id1=ball_df[ball_df['Team1']==ball_df['BattingTeam']]['Team2'].index
id2=ball_df[ball_df['Team2']==ball_df['BattingTeam']]['Team1'].index

def result(row):
    return 1 if row['BattingTeam']==row['WinningTeam'] else 0

ball_df['result']=ball_df.apply(result,axis=1)
ball_df.loc[id1,'BowlingTeam']=ball_df.loc[id1,'Team2']
ball_df.loc[id2,'BowlingTeam']=ball_df.loc[id2,'Team1']

# Final dataframe to predict
final_df=ball_df[['BattingTeam','BowlingTeam','City','Target','runs_left','balls_left','wickets_left','current_run_rate','required_run_rate','result']]


# category Encoding
final_df=pd.get_dummies(columns=['BattingTeam','BowlingTeam','City'],data=final_df)

# Model Building
X=final_df.drop('result',axis=1)
y=final_df['result']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

# feature scalling
from sklearn.preprocessing import StandardScaler
st=StandardScaler()
X_train=st.fit_transform(X_train)
X_test=st.fit_transform(X_test)

# Creating Random Forest Model
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=200)
rf.fit(X_train,y_train)

# creating a pickle file for the classifier
filename='IPL-winner-prediction-rfc-model.pkl'
pickle.dump(rf,open(filename,'wb'))