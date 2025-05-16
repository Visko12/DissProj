from flask import Flask, render_template, jsonify
import pandas as pd
import json
from datetime import datetime, timedelta
import os
from collections import defaultdict
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

app = Flask(__name__)

def load_message_logs():
    logs = []
    log_dir = 'logs'
    
    #
    for filename in os.listdir(log_dir):
        if filename.startswith('message_log_') and filename.endswith('.json'):
            with open(os.path.join(log_dir, filename), 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        # Set platform to 'unknown' if missing
                        if 'platform' not in entry:
                            entry['platform'] = 'unknown'
                        # Set author_name to 'unknown' if missing
                        if 'author_name' not in entry:
                            entry['author_name'] = 'unknown'
                        logs.append(entry)
                    except json.JSONDecodeError:
                        continue
    
    return pd.DataFrame(logs)

def create_warning_chart(df):
    #filter by platform
    platforms = ['discord', 'twitch']
    data = []
    for platform in platforms:
        platform_df = df[df['platform'] == platform]
        warning_counts = platform_df[platform_df['action_taken'] == 'deleted']['author_name'].value_counts()
        data.append(go.Bar(
            x=warning_counts.index,
            y=warning_counts.values,
            name=platform.capitalize(),
            text=warning_counts.values,
            textposition='auto',
        ))
    fig = go.Figure(data=data)
    fig.update_layout(
        barmode='group',
        title='Warning Distribution by User and Platform',
        xaxis_title='User',
        yaxis_title='Number of Warnings',
        template='plotly_dark'
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_toxicity_trend(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    platforms = ['discord', 'twitch']
    fig = go.Figure()
    for platform in platforms:
        platform_df = df[df['platform'] == platform]
        daily_toxicity = platform_df.groupby('date')['toxicity_score'].mean().reset_index()
        fig.add_trace(go.Scatter(
            x=daily_toxicity['date'],
            y=daily_toxicity['toxicity_score'],
            mode='lines+markers',
            name=platform.capitalize()
        ))
    fig.update_layout(
        title='Toxicity Score Trend Over Time by Platform',
        xaxis_title='Date',
        yaxis_title='Average Toxicity Score',
        template='plotly_dark'
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_message_stats(df):
    total_messages = len(df)
    toxic_messages = len(df[df['toxicity_score'] > 0.9])
    warnings_issued = len(df[df['action_taken'] == 'deleted'])
    
    stats = {
        'total_messages': total_messages,
        'toxic_messages': toxic_messages,
        'warnings_issued': warnings_issued,
        'toxicity_percentage': (toxic_messages / total_messages * 100) if total_messages > 0 else 0
    }
    
    return stats

@app.route('/')
def index():
    try:
        df = load_message_logs()
        stats_discord = create_message_stats(df[df['platform'] == 'discord'])
        stats_twitch = create_message_stats(df[df['platform'] == 'twitch'])
        #create visuilisations
        warning_chart = create_warning_chart(df)
        toxicity_trend = create_toxicity_trend(df)
        return render_template(
            'dashboard.html',
            warning_chart=warning_chart,
            toxicity_trend=toxicity_trend,
            stats_discord=stats_discord,
            stats_twitch=stats_twitch
        )
    except Exception as e:
        return f"Error loading dashboard: {str(e)}"

@app.route('/api/stats')
def get_stats():
    try:
        df = load_message_logs()
        return jsonify({
            'discord': create_message_stats(df[df['platform'] == 'discord']),
            'twitch': create_message_stats(df[df['platform'] == 'twitch'])
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000) 