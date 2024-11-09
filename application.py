from flask import Flask, request, jsonify
import pandas as pd
import joblib  # or pickle if you're using that to load models

app = Flask(__name__)

meta_model_3 = joblib.load('models/meta_model_3.pkl')
tuned_meta_model_2 = joblib.load('models/tuned_meta_model_2.pkl')
@app.route('/')
def home():
    return "Premier League Match Prediction API is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        week = data.get('week')
        home_team = data.get('homeTeam')
        away_team = data.get('awayTeam')
        lastHomeResult = data.get('lastHomeResult')
        lastAwayResult = data.get('lastAwayResult')
        homeStreak = data.get('homeStreak')
        homeRestDays = data.get('homeRestDays')
        awayStreak = data.get('awayStreak')
        awayRestDays = data.get('awayRestDays')

        week_mapping = {'Early': 0, 'Mid': 1, 'End': 2}
        home_team_mapping = { ... }  

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