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
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler

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
    return render_template('index.html')

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

    # 중앙 부분만 사용: 시작과 끝의 20%를 제외한 중앙 60% 분석
    start_sample = int(len(y) * 0.2)
    end_sample = int(len(y) * 0.8)
    y = y[start_sample:end_sample]
    
    df = pd.DataFrame()

    # 피치 검출 개선: 매그니튜드가 높은 구간만 사용하여 노이즈 필터링
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    threshold = np.percentile(magnitudes, 95)  # 상위 5%의 강한 신호
    strong_pitches = pitches[magnitudes > threshold]

    # 유효한 피치만 필터링 (사람 목소리 범위인 80Hz ~ 500Hz)
    valid_pitches = strong_pitches[(strong_pitches > 80) & (strong_pitches < 1300)]

    # 노이즈 필터링 후 중앙값으로 피치 결정
    max_pitch = np.median(valid_pitches) if valid_pitches.size > 0 else 0
    df.loc[0, 'max_pitch'] = max_pitch

    # 나머지 특성 추출
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

# 1. music_table에서 파형 데이터를 가져오는 함수
def fetch_waveform_data(music_ids_in_json):
    # MySQL에서 music_table에서 파형 분석 관련 데이터를 가져오기
    query = "SELECT * FROM music_table WHERE music_id IN %(music_ids)s"
    try:
        df_waveform_db = pd.read_sql(query, connection, params={"music_ids": tuple(music_ids_in_json)})
        
        # MySQL에 없는 music_id 찾기
        fetched_ids = df_waveform_db['music_id'].tolist()
        missing_ids = set(music_ids_in_json) - set(fetched_ids)
        
        # MySQL에서 가져오지 못한 music_id 출력
        if missing_ids:
            print(f"MySQL에서 찾을 수 없는 music_id: {missing_ids}")

        # 필요한 특성만 선택 (파형 분석 관련 컬럼들)
        feature_columns = [
            'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'tempo',
            'spectral_centroid_mean', 'spectral_centroid_var', 'rolloff_mean', 'rolloff_var',
            'chroma_stft_mean', 'chroma_stft_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var',
            'spectral_contrast_mean', 'spectral_contrast_var', 'melspectrogram_mean', 'melspectrogram_var'
        ] + [f'mfcc{i}_mean' for i in range(20)] + [f'mfcc{i}_var' for i in range(20)]

        print(f"파형 데이터베이스 크기: {df_waveform_db.shape}")
        return df_waveform_db, feature_columns
    except Exception as e:
        print(f"Error fetching waveform data: {e}")
        return None, []

# 2. music_pitch_table에서 피치 데이터를 가져오는 함수
def fetch_pitch_data(music_ids_in_json):
    # MySQL에서 music_pitch_table에서 피치 분석 관련 데이터를 가져오기
    query = "SELECT * FROM music_pitch_table WHERE music_id IN %(music_ids)s"
    try:
        df_pitch_db = pd.read_sql(query, connection, params={"music_ids": tuple(music_ids_in_json)})
        
        # MySQL에 없는 music_id 찾기
        fetched_ids = df_pitch_db['music_id'].tolist()
        missing_ids = set(music_ids_in_json) - set(fetched_ids)
        
        # MySQL에서 가져오지 못한 music_id 출력
        if missing_ids:
            print(f"MySQL에서 찾을 수 없는 music_id: {missing_ids}")
        
        print(f"피치 데이터베이스 크기: {df_pitch_db.shape}")
        
        return df_pitch_db
    except Exception as e:
        print(f"Error fetching pitch data: {e}")
        return None
    

def find_similar_songs(user_features, df_waveform_db, df_pitch_db, feature_columns, music_dict):
    # 공통된 music_id를 기준으로 데이터프레임 필터링
    common_music_ids = set(df_waveform_db['music_id']).intersection(df_pitch_db['music_id'])

    # 두 데이터 프레임에서 공통된 music_id만 남기기
    df_waveform_filtered = df_waveform_db[df_waveform_db['music_id'].isin(common_music_ids)].copy()
    df_pitch_filtered = df_pitch_db[df_pitch_db['music_id'].isin(common_music_ids)].copy()

    # db_features만 추출하여 결측치 및 무한대 값 처리
    db_waveform_features = df_waveform_filtered[feature_columns].copy()
    db_waveform_features = handle_missing_values(db_waveform_features)

    # 공통된 컬럼만 사용하여 파형 분석 점수 계산
    common_columns = db_waveform_features.columns.intersection(user_features.columns)
    db_waveform_features = db_waveform_features[common_columns]
    user_waveform_features = user_features[common_columns]

    # 스케일링 추가 (데이터 정규화)
    scaler = MinMaxScaler()
    db_waveform_scaled = scaler.fit_transform(db_waveform_features)
    user_waveform_scaled = scaler.transform(user_waveform_features)

    # 코사인 유사도 계산 (파형 분석 점수)
    waveform_similarity = cosine_similarity(db_waveform_scaled, user_waveform_scaled)

    # 피치 데이터 유사도 계산 (스케일링 없이)
    db_pitch_features = df_pitch_filtered[['max_pitch']].copy().values
    user_pitch_features = user_features[['max_pitch']].copy().values

    # 유클리드 거리 기반 피치 유사도 계산
    pitch_similarity = np.array([1 / (1 + euclidean(db_pitch, user_pitch_features[0])) for db_pitch in db_pitch_features])

    # 파형과 피치 점수에 가중치 적용 (4:6 비율)
    df_waveform_filtered['waveform_similarity'] = waveform_similarity.flatten()
    df_waveform_filtered['pitch_similarity'] = pitch_similarity
    df_waveform_filtered['similarity'] = (0.4 * waveform_similarity.flatten()) + (0.6 * pitch_similarity)

    # 상위 5개의 유사한 곡 추출 (중복 제거 후)
    top_5_songs = df_waveform_filtered.nlargest(5, 'similarity')[['music_id', 'similarity', 'waveform_similarity', 'pitch_similarity']].drop_duplicates()

    # music_id를 singer와 title로 매핑하여 최종 결과 출력
    final_results = []
    for _, row in top_5_songs.iterrows():
        music_id = row['music_id']
        similarity_score = row['similarity']
        waveform_sim = row['waveform_similarity']
        pitch_sim = row['pitch_similarity']
        
        if music_id in music_dict:
            singer, title = music_dict[music_id]
            max_pitch = df_pitch_filtered[df_pitch_filtered['music_id'] == music_id]['max_pitch'].values[0]
            final_results.append({
                'singer': singer,
                'title': title,
                'similarity': similarity_score,
                'waveform_similarity': waveform_sim,
                'pitch_similarity': pitch_sim,
                'max_pitch': max_pitch
            })

    # 결과 출력
    if final_results:
        print("\n가장 유사한 5개의 노래 (singer와 title):")
        for result in final_results:
            print(f"Singer: {result['singer']}, Title: {result['title']}, Similarity: {result['similarity']:.6f}")
    else:
        print("매핑된 결과가 없습니다.")

    # 사용자 음역대 출력
    user_max_pitch = user_features['max_pitch'].iloc[0]
    print(f"\n사용자 음악의 최고 피치 (max_pitch): {user_max_pitch:.2f}")

    # 추가 기능: 더 높은 음역대에 도전할 사람을 위한 추천 (바로 위 음역대 3곡)
    higher_pitch_songs = df_pitch_db[df_pitch_db['max_pitch'] > user_max_pitch].sort_values(by='max_pitch', ascending=True)
    
    if len(higher_pitch_songs) >= 3:
        # 바로 위의 3개의 노래를 추천
        recommended_high_pitch_songs = higher_pitch_songs.head(3)
    else:
        # 3개 미만인 경우에는 가능한 곡만 모두 추천
        recommended_high_pitch_songs = higher_pitch_songs
    
    # music_id를 singer와 title로 매핑하여 추천 결과 출력
    high_pitch_results = []
    for _, row in recommended_high_pitch_songs.iterrows():
        music_id = row['music_id']
        if music_id in music_dict:
            singer, title = music_dict[music_id]
            high_pitch_results.append({'singer': singer, 'title': title, 'max_pitch': row['max_pitch']})
    
    # 높은 음역대 도전 추천 결과 출력
    if high_pitch_results:
        print("\n높은 음역대 도전을 위한 추천 노래 (바로 위 3곡):")
        for result in high_pitch_results:
            print(f"Singer: {result['singer']}, Title: {result['title']}, Max Pitch: {result['max_pitch']:.2f}")
    else:
        print("높은 음역대 도전 노래를 찾지 못했습니다.")
        
    return final_results, high_pitch_results


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

    # 사용자 피치 정보 계산
    user_mean_features = (user_lowest_pitch + user_highest_pitch) / 2
    user_max_pitch = float(user_mean_features['max_pitch'].iloc[0])  # float으로 변환

    # 데이터베이스에서 파형 및 피치 데이터 가져오기
    df_waveform_db, feature_columns = fetch_waveform_data(music_ids_in_json)
    df_pitch_db = fetch_pitch_data(music_ids_in_json)

    # 데이터베이스에서 데이터를 제대로 불러왔는지 확인
    if df_waveform_db is None or df_pitch_db is None:
        return jsonify({'error': '데이터베이스에서 데이터를 불러오지 못했습니다.'}), 500

    # 유사한 노래 찾기
    top_5_songs, recommended_high_pitch_songs = find_similar_songs(user_mean_features, df_waveform_db, df_pitch_db, feature_columns, music_dict)

    # 모든 float32 값을 float으로 변환
    top_5_songs = [
        {
            'singer': song['singer'],
            'title': song['title'],
            'similarity': float(song['similarity']),
            'waveform_similarity': float(song['waveform_similarity']),
            'pitch_similarity': float(song['pitch_similarity']),
            'max_pitch': float(song['max_pitch']),
        }
        for song in top_5_songs
    ]

    recommended_high_pitch_songs = [
        {
            'singer': song['singer'],
            'title': song['title'],
            'max_pitch': float(song['max_pitch']),
        }
        for song in recommended_high_pitch_songs
        
    ]

    # 결과를 JSON 형식으로 반환, 사용자 피치 추가
    return jsonify({
        'user_max_pitch': user_max_pitch,
        'songs': top_5_songs,
        'recommended_high_pitch_songs': recommended_high_pitch_songs
    })


# ABOUT 페이지
@app.route('/about')
def about():
    return render_template('about.html')

# HELP 페이지
@app.route('/help')
def help():
    return render_template('help.html')

if __name__ == '__main__':
    app.run(debug=True)
