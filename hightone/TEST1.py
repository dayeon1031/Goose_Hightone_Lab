from flask import Flask, render_template, request, jsonify
import sounddevice as sd
import numpy as np
import librosa
import scipy.io.wavfile as wavfile
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pymysql
from sklearn.preprocessing import StandardScaler
import json
import os

app = Flask(__name__)

# MySQL 연결 설정
connection = pymysql.connect(
    host='localhost',  # MySQL 서버 주소
    user='root',       # 사용자 이름
    password='Hello192!',  # 비밀번호
    database='music_db',  # 데이터베이스 이름
)

# 녹음 설정
fs = 44100  # 샘플링 주파수 (Hz)
duration_low = 5  # 최저음을 녹음할 시간 (초)
duration_high = 5  # 최고음을 녹음할 시간 (초)

# 최저음과 최고음 녹음 및 저장을 위한 경로 설정
LOW_PITCH_FILE = "lowest_pitch.wav"
HIGH_PITCH_FILE = "highest_pitch.wav"

# 곡 리스트 초기화
@app.route('/', methods=['GET', 'POST'])
def index():
    song_list = []
    
    # URL에서 song_list 데이터를 받아서 처리
    song_list_param = request.args.get('song_list')
    if song_list_param:
        song_list = json.loads(song_list_param)

    return render_template('index.html', song_list=song_list)

#record 페이지
@app.route('/record')
def record():
    return render_template('record.html')


# JSON 파일에서 musicId를 키로 하는 매핑을 생성
json_path = r"C:\Users\나현준\Desktop\semina\hightone\music_maching.json"
with open(json_path, 'r', encoding='utf-8') as f:
    music_mapping = json.load(f)
music_list = music_mapping[0]['voiceWaveMatchingResponseDtoList']
music_dict = {item['musicId']: (item['singer'], item['title']) for item in music_list}
music_ids_in_json = list(music_dict.keys())

def handle_missing_values(df):
    df = df.fillna(0)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    return df

def record_audio(duration, filename):
    print(f"녹음 시작 ({duration}초)")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()
    wavfile.write(filename, fs, audio)
    print(f"녹음 완료: {filename}")

def analyze_audio(file):
    y, sr = librosa.load(file)
    df = pd.DataFrame()

    zero_crossings = librosa.zero_crossings(y=y, pad=False)
    df.loc[0, 'zero_crossing_rate_mean'] = zero_crossings.mean()
    df.loc[0, 'zero_crossing_rate_var'] = zero_crossings.var()

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    df.loc[0, 'tempo'] = tempo

    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    df.loc[0, 'spectral_centroid_mean'] = spectral_centroids.mean()
    df.loc[0, 'spectral_centroid_var'] = spectral_centroids.var()

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    df.loc[0, 'rolloff_mean'] = spectral_rolloff.mean()
    df.loc[0, 'rolloff_var'] = spectral_rolloff.var()

    chromagram = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    df.loc[0, 'chroma_stft_mean'] = chromagram.mean()
    df.loc[0, 'chroma_stft_var'] = chromagram.var()

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    df.loc[0, 'spectral_bandwidth_mean'] = spectral_bandwidth.mean()
    df.loc[0, 'spectral_bandwidth_var'] = spectral_bandwidth.var()

    spectral_contrasts = librosa.feature.spectral_contrast(y=y, sr=sr)
    df.loc[0, 'spectral_contrast_mean'] = spectral_contrasts.mean()
    df.loc[0, 'spectral_contrast_var'] = spectral_contrasts.var()

    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    df.loc[0, 'melspectrogram_mean'] = melspectrogram.mean()
    df.loc[0, 'melspectrogram_var'] = melspectrogram.var()

    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    for i in range(len(mfccs)):
        df.loc[0, f'mfcc{i}_mean'] = mfccs[i].mean()
        df.loc[0, f'mfcc{i}_var'] = mfccs[i].var()

    return df

def fetch_database_data():
    query = "SELECT * FROM music_table WHERE music_id IN %(music_ids)s"
    df_db = pd.read_sql(query, connection, params={"music_ids": tuple(music_ids_in_json)})

    feature_columns = [
        'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'tempo',
        'spectral_centroid_mean', 'spectral_centroid_var', 'rolloff_mean', 'rolloff_var',
        'chroma_stft_mean', 'chroma_stft_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var',
        'spectral_contrast_mean', 'spectral_contrast_var', 'melspectrogram_mean', 'melspectrogram_var'
    ] + [f'mfcc{i}_mean' for i in range(20)] + [f'mfcc{i}_var' for i in range(20)]
    
    return df_db, feature_columns

def find_similar_songs(user_features, df_db, feature_columns):
    db_features = df_db[feature_columns].copy()
    db_features = handle_missing_values(db_features)

    common_columns = db_features.columns.intersection(user_features.columns)

    db_features = db_features[common_columns]
    user_features = user_features[common_columns]

    scaler = StandardScaler()
    db_features_scaled = scaler.fit_transform(db_features)
    user_features_scaled = scaler.transform(user_features)

    similarity = cosine_similarity(db_features_scaled, user_features_scaled)

    df_db['similarity'] = similarity.flatten()

    top_5_songs = df_db.nlargest(5, 'similarity')[['music_id', 'similarity']].drop_duplicates()

    final_results = []
    for _, row in top_5_songs.iterrows():
        music_id = row['music_id']
        similarity_score = row['similarity']
        
        if music_id in music_dict:
            singer, title = music_dict[music_id]
            final_results.append({'singer': singer, 'title': title, 'similarity': similarity_score})
    
    return final_results


@app.route('/process_record', methods=['POST'])
def process_record():
    pitch = request.json.get('pitch')
    
    if pitch == 'low':
        record_audio(duration_low, LOW_PITCH_FILE)
        return jsonify({'message': '최저음 녹음 완료'})
    elif pitch == 'high':
        record_audio(duration_high, HIGH_PITCH_FILE)
        return jsonify({'message': '최고음 녹음 완료'})
    else:
        return jsonify({'error': '올바른 pitch 값이 필요합니다.'}), 400

@app.route('/analyze', methods=['POST'])
def analyze():
    user_lowest_pitch = analyze_audio(LOW_PITCH_FILE)
    user_highest_pitch = analyze_audio(HIGH_PITCH_FILE)

    user_mean_features = (user_lowest_pitch + user_highest_pitch) / 2

    df_db, feature_columns = fetch_database_data()

    top_5_songs = find_similar_songs(user_mean_features, df_db, feature_columns)

    return jsonify({'songs': top_5_songs})
    
# ABOUT 페이지
@app.route('/about')
def about():
    return render_template('about.html')

# HELP 페이지
@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/high_pitch_challenge')
def high_pitch_challenge():
    return render_template('high_pitch_challenge.html')

@app.route('/low_pitch_challenge')
def low_pitch_challenge():
    return render_template('low_pitch_challenge.html')


if __name__ == '__main__':
    app.run(debug=True)
