@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.json
        
        # Extract relevant fields from the data
        week = data.get('week')
        home_team = data.get('homeTeam')
        away_team = data.get('awayTeam')
        lastHomeResult = data.get('lastHomeResult')
        lastAwayResult = data.get('lastAwayResult')
        homeStreak = data.get('homeStreak')
        homeRestDays = data.get('homeRestDays')
        awayStreak = data.get('awayStreak')
        awayRestDays = data.get('awayRestDays')

        # Preprocessing (mapping weeks and teams)
        week_mapping = {'Early': 0, 'Mid': 1, 'End': 2}
        home_team_mapping = {
            'Coventry City': 0, 'Leeds United': 1, 'Sheffield Utd': 2, 'Crystal Palace': 3,
            'Arsenal': 4, 'Ipswich Town': 5, 'Everton': 6, 'Southampton': 7, 'Chelsea': 8, 
            "Nott'ham Forest": 9, 'Manchester City': 10, 'Blackburn': 11, 'Wimbledon': 12,
            'Tottenham': 13, 'Liverpool': 14, 'Aston Villa': 15, 'Oldham Athletic': 16,
            'Middlesbrough': 17, 'Norwich City': 18, 'QPR': 19, 'Manchester Utd': 20,
            'Sheffield Weds': 21, 'Newcastle Utd': 22, 'West Ham': 23, 'Swindon Town': 24,
            'Leicester City': 25, 'Bolton': 26, 'Sunderland': 27, 'Derby County': 28,
            'Barnsley': 29, 'Charlton Ath': 30, 'Watford': 31, 'Bradford City': 32,
            'Fulham': 33, 'West Brom': 34, 'Birmingham City': 35, 'Wolves': 36, 'Portsmouth': 37,
            'Wigan Athletic': 38, 'Reading': 39, 'Stoke City': 40, 'Hull City': 41,
            'Burnley': 42, 'Blackpool': 43, 'Swansea City': 44, 'Cardiff City': 45,
            'Bournemouth': 46, 'Huddersfield': 47, 'Brighton': 48, 'Brentford': 49
        }

        # Map the 'week' to its corresponding value
        season_phase = week_mapping.get(week, -1)

        # Prepare the input data for the model
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

        # Convert to DataFrame (if needed by the model)
        input_df = pd.DataFrame([input_data])

        # Predict using the loaded model
        prediction = model.predict(input_df)

        # Return the prediction result in JSON format
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})
