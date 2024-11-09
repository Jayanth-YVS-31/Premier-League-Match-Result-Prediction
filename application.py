from flask import Flask, request, jsonify
import pandas as pd
import joblib  # or pickle if you're using that to load models

app = Flask(__name__)

meta_model_3 = joblib.load('models/meta_model_3.pkl')
tuned_meta_model_2 = joblib.load('models/tuned_meta_model_2.pkl')

@app.route('/') 
def home(): 
    return "Premier League Match Prediction API is running  use https://premier-league-match-result-prediction.onrender.com/predict" 

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        # For GET requests, we'll retrieve data from query parameters
        if request.method == 'GET':
            week = request.args.get('week')
            date = request.args.get('matchDate')
            home_team = request.args.get('homeTeam')
            away_team = request.args.get('awayTeam')
            lastHomeResult = int(request.args.get('lastHomeResult'))
            lastAwayResult = int(request.args.get('lastAwayResult'))
            homeStreak = int(request.args.get('homeStreak'))
            homeRestDays = int(request.args.get('homeRestDays'))
            awayStreak = int(request.args.get('awayStreak'))
            awayRestDays = int(request.args.get('awayRestDays'))
        
        # For POST requests, we'll retrieve data from JSON body
        elif request.method == 'POST':
            data = request.json
            week = data.get('week')
            date = request.args.get('matchDate')
            home_team = data.get('homeTeam')
            away_team = data.get('awayTeam')
            lastHomeResult = data.get('lastHomeResult')
            lastAwayResult = data.get('lastAwayResult')
            homeStreak = data.get('homeStreak')
            homeRestDays = data.get('homeRestDays')
            awayStreak = data.get('awayStreak')
            awayRestDays = data.get('awayRestDays')

        week_mapping = {'Early': 0, 'Mid': 1, 'End': 2}
        home_team_mapping = {
    'Coventry City': 0, 'Leeds United': 1, 'Sheffield Utd': 2, 'Crystal Palace': 3,
    'Arsenal': 4, 'Ipswich Town': 5, 'Everton': 6, 'Southampton': 7, 'Chelsea': 8,
    "Nott'ham Forest": 9, 'Manchester City': 10, 'Blackburn': 11, 'Wimbledon': 12,
    'Tottenham': 13, 'Liverpool': 14, 'Aston Villa': 15, 'Oldham Athletic': 16,
    'Middlesbrough': 17, 'Norwich City': 18, 'QPR': 19, 'Manchester Utd': 20,
    'Sheffield Weds': 21, 'Newcastle Utd': 22, 'West Ham': 23, 'Swindon Town': 24,
    'Leicester City': 25, 'Bolton': 26, 'Sunderland': 27, 'Derby County': 28,
    'Barnsley': 29, 'Charlton Ath': 30, 'Watford': 31, 'Bradford City': 32, 'Fulham': 33,
    'West Brom': 34, 'Birmingham City': 35, 'Wolves': 36, 'Portsmouth': 37,
    'Wigan Athletic': 38, 'Reading': 39, 'Stoke City': 40, 'Hull City': 41, 'Burnley': 42,
    'Blackpool': 43, 'Swansea City': 44, 'Cardiff City': 45, 'Bournemouth': 46,
    'Huddersfield': 47, 'Brighton': 48, 'Brentford': 49
}  # Add your home team mapping here

        season_phase = week_mapping.get(week, -1)

        input_data = {
            'HomeWinStreak_5': homeStreak,
            'AwayWinStreak_5': awayStreak,
            'LastHomeResult': lastHomeResult,
            'LastAwayResult': lastAwayResult,
            'SeasonPhase_Early': 1 if season_phase == 0 else 0,
            'SeasonPhase_Mid': 1 if season_phase == 1 else 0,
            'SeasonPhase_End': 1 if season_phase == 2 else 0,
            'DaysSinceLastHomeMatch': homeRestDays,
            'DaysSinceLastAwayMatch': awayRestDays,
            'Home_team': home_team_mapping.get(home_team, -1),
            'Away_team': home_team_mapping.get(away_team, -1)
        }

        input_df = pd.DataFrame([input_data])

        prediction_3 = meta_model_3.predict(input_df)
        prediction_2 = tuned_meta_model_2.predict(input_df)

        return jsonify({
            'prediction_3': prediction_3[0],
            'prediction_2': prediction_2[0]
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__": 
    app.run(debug=True)
