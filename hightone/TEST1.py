from flask import Flask, render_template, request, jsonify, send_file
import os
import sounddevice as sd
import numpy as np
import librosa
import scipy.io.wavfile as wavfile
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pymysql
from sklearn.preprocessing import StandardScaler
import json

# Flask 앱 설정
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# MySQL 연결 설정
connection = pymysql.connect(
    host='localhost',  # MySQL 서버 주소
    user='root',       # 사용자 이름
    password='dayeon',  # 비밀번호
    database='music_db',  # 데이터베이스 이름
)

# 곡 리스트 초기화
song_list = [
    "곡 1: 예시곡 1",
    "곡 2: 예시곡 2",
    "곡 3: 예시곡 3",
]

# 녹음 설정
fs = 44100  # 샘플링 주파수 (Hz)
duration_low = 10  # 최저음을 녹음할 시간 (초)
duration_high = 10  # 최고음을 녹음할 시간 (초)

def handle_missing_values(df):
    df = df.fillna(0)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    return df

def record_audio(duration, filename):
    """녹음 후 파일로 저장"""
    print(f"녹음 시작 ({duration}초) 파일명: {filename}")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()  # 녹음이 끝날 때까지 대기
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    wavfile.write(file_path, fs, audio)  # 녹음한 데이터를 .wav 파일로 저장
    print(f"녹음 완료: {filename}")

def analyze_audio(file):
    """녹음된 .wav 파일을 분석하여 특징 추출"""
    print(f"분석 중: {file}")
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

def fetch_database_data(music_ids_in_json):
    """MySQL 데이터베이스에서 musicId에 해당하는 곡 데이터 가져오기"""
    query = "SELECT * FROM music_table WHERE music_id IN %(music_ids)s"
    df_db = pd.read_sql(query, connection, params={"music_ids": tuple(music_ids_in_json)})

    feature_columns = [
        'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'tempo',
        'spectral_centroid_mean', 'spectral_centroid_var', 'rolloff_mean', 'rolloff_var',
        'chroma_stft_mean', 'chroma_stft_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var',
        'spectral_contrast_mean', 'spectral_contrast_var', 'melspectrogram_mean', 'melspectrogram_var'
    ] + [f'mfcc{i}_mean' for i in range(20)] + [f'mfcc{i}_var' for i in range(20)]

    print(f"필터링된 데이터베이스 크기: {df_db.shape}")
    return df_db, feature_columns

def find_similar_songs(user_features, df_db, feature_columns, music_dict):
    """사용자의 특징과 데이터베이스의 곡들을 비교하여 유사한 곡 찾기"""
    db_features = df_db[feature_columns].copy()
    db_features = handle_missing_values(db_features)

    if not isinstance(user_features, pd.DataFrame):
        user_features = pd.DataFrame(user_features, columns=feature_columns)

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
    print("유사도 계산 후 상위 5개의 music_id와 유사도 값:")
    print(top_5_songs)

    for _, row in top_5_songs.iterrows():
        music_id = row['music_id']
        similarity_score = row['similarity']
        if music_id in music_dict:
            singer, title = music_dict[music_id]
            final_results.append({'singer': singer, 'title': title, 'similarity': similarity_score})
        else:
            print(f"매핑되지 않은 music_id: {music_id}")

    return final_results

@app.route('/', methods=['GET', 'POST'])
def index():
    """메인 페이지"""
    global song_list
    if request.method == 'POST':
        if 'song_file' in request.files:
            file = request.files['song_file']
            if file.filename.endswith('.txt'):
                lines = file.read().decode('utf-8').splitlines()
                song_list.extend(lines)

    return render_template('index.html', song_list=song_list)

@app.route('/record')
def record():
    """녹음 페이지"""
    return render_template("record.html")

@app.route('/analyze_pitch', methods=['POST'])
def analyze_pitch():
    """최저음과 최고음을 녹음하고 분석한 후 유사한 곡을 추천"""
    try:
        # 최저음 녹음 및 분석
        print("최저음을 녹음하고 있습니다...")
        record_audio(duration_low, "lowest_pitch.wav")
        user_lowest_pitch = analyze_audio(os.path.join(UPLOAD_FOLDER, "lowest_pitch.wav"))

        # 최고음 녹음 및 분석
        print("최고음을 녹음하고 있습니다...")
        record_audio(duration_high, "highest_pitch.wav")
        user_highest_pitch = analyze_audio(os.path.join(UPLOAD_FOLDER, "highest_pitch.wav"))

        # 평균값 계산
        user_mean_features = (user_lowest_pitch + user_highest_pitch) / 2

        # JSON 파일 경로 지정 및 불러오기
        json_path = r"C:\Users\USER\Desktop\jakpum3-2\music_maching.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            music_mapping = json.load(f)

        music_list = music_mapping[0]['voiceWaveMatchingResponseDtoList']
        music_dict = {item['musicId']: (item['singer'], item['title']) for item in music_list}
        music_ids_in_json = list(music_dict.keys())

        # 데이터베이스에서 음악 데이터를 가져오기
        df_db, feature_columns = fetch_database_data(music_ids_in_json)
        df_db = handle_missing_values(df_db)

        # 유사한 노래 찾기
        final_results = find_similar_songs(user_mean_features, df_db, feature_columns, music_dict)

        return jsonify(final_results)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/upload_recording', methods=['POST'])
def upload_recording():
    """녹음된 오디오 파일을 저장하는 엔드포인트"""
    if 'audio' not in request.files or 'filename' not in request.form:
        return jsonify({'error': '파일 업로드 실패'}), 400
    
    file = request.files['audio']
    filename = request.form['filename']
    
    # 파일 저장 경로 설정
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    return jsonify({'status': '파일 저장 완료'})

@app.route('/process_analysis', methods=['GET'])
def process_analysis():
    """업로드된 파일을 분석하여 결과를 반환하는 엔드포인트"""
    lowest_pitch_file = os.path.join(UPLOAD_FOLDER, 'lowest_pitch.wav')
    highest_pitch_file = os.path.join(UPLOAD_FOLDER, 'highest_pitch.wav')

    if not os.path.exists(lowest_pitch_file) or not os.path.exists(highest_pitch_file):
        return jsonify({'error': '녹음 파일을 찾을 수 없습니다.'}), 400

    # 최저음 및 최고음 파일 분석
    user_lowest_pitch = analyze_audio(lowest_pitch_file)
    user_highest_pitch = analyze_audio(highest_pitch_file)
    user_mean_features = (user_lowest_pitch + user_highest_pitch) / 2

    # JSON 파일 경로 지정 및 불러오기
    json_path = r"C:\Users\USER\Desktop\jakpum3-2\music_maching.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        music_mapping = json.load(f)

    music_list = music_mapping[0]['voiceWaveMatchingResponseDtoList']
    music_dict = {item['musicId']: (item['singer'], item['title']) for item in music_list}
    music_ids_in_json = list(music_dict.keys())

    # 데이터베이스에서 음악 데이터를 가져오기
    df_db, feature_columns = fetch_database_data(music_ids_in_json)
    df_db = handle_missing_values(df_db)

    # 유사한 노래 찾기
    final_results = find_similar_songs(user_mean_features, df_db, feature_columns, music_dict)

    return jsonify({'result': final_results})

if __name__ == '__main__':
    app.run(debug=True)
