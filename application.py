from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load models
meta_model_3 = joblib.load('models/meta_model_3.pkl')
tuned_meta_model_2 = joblib.load('models/tuned_meta_model_2.pkl')

@app.route('/')
def home(): 
    return """<html>
                <h2>Premier League Match Prediction API</h2>
                <p>Welcome! Use the /predict endpoint with either GET or POST methods to make predictions.</p>
                <h5>For GET requests:</h5>
                <p>Pass parameters as query parameters in the URL.</p>
                <h5>For POST requests:</h5>
                <p>Send a JSON payload with the required fields.</p>
              </html>"""

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        # Retrieve data based on request method
        if request.method == 'GET':
            week = request.args.get('week', '')
            date = request.args.get('matchDate', '')
            home_team = request.args.get('homeTeam', '')
            away_team = request.args.get('awayTeam', '')
            lastHomeResult = request.args.get('lastHomeResult', '')
            lastAwayResult = request.args.get('lastAwayResult', '')
            homeStreak = request.args.get('homeStreak', '')
            homeRestDays = request.args.get('homeRestDays', '')
            awayStreak = request.args.get('awayStreak', '')
            awayRestDays = request.args.get('awayRestDays', '')
        
        elif request.method == 'POST':
            data = request.json or {}
            week = data.get('week', '')
            date = data.get('matchDate', '')
            home_team = data.get('homeTeam', '')
            away_team = data.get('awayTeam', '')
            lastHomeResult = data.get('lastHomeResult', '')
            lastAwayResult = data.get('lastAwayResult', '')
            homeStreak = data.get('homeStreak', '')
            homeRestDays = data.get('homeRestDays', '')
            awayStreak = data.get('awayStreak', '')
            awayRestDays = data.get('awayRestDays', '')

        # Mapping for season phases and team names
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
        }

        # Determine season phase and prepare input data
        season_phase = week_mapping.get(week, -1)
        input_data = {
            'HomeWinStreak_5': int(homeStreak) if homeStreak else 0,
            'AwayWinStreak_5': int(awayStreak) if awayStreak else 0,
            'LastHomeResult': int(lastHomeResult) if lastHomeResult else 0,
            'LastAwayResult': int(lastAwayResult) if lastAwayResult else 0,
            'SeasonPhase_Early': 1 if season_phase == 0 else 0,
            'SeasonPhase_Mid': 1 if season_phase == 1 else 0,
            'SeasonPhase_End': 1 if season_phase == 2 else 0,
            'DaysSinceLastHomeMatch': int(homeRestDays) if homeRestDays else 0,
            'DaysSinceLastAwayMatch': int(awayRestDays) if awayRestDays else 0,
            'Home_team': home_team_mapping.get(home_team, -1),
            'Away_team': home_team_mapping.get(away_team, -1)
        }

        # Convert input data to DataFrame and make predictions
        input_df = pd.DataFrame([input_data])
        prediction_3 = meta_model_3.predict(input_df)
        prediction_2 = tuned_meta_model_2.predict(input_df)

        return jsonify({
            'prediction_3': int(prediction_3[0]),
            'prediction_2': int(prediction_2[0])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__": 
    app.run(debug=True)