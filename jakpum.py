import sounddevice as sd
import numpy as np
import librosa
import scipy.io.wavfile as wavfile
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pymysql
from sklearn.preprocessing import StandardScaler
import json

# MySQL 연결 설정
connection = pymysql.connect(
    host='localhost',  # MySQL 서버 주소
    user='root',       # 사용자 이름
    password='dayeon',  # 비밀번호
    database='music_db',  # 데이터베이스 이름
)

# 녹음 설정
fs = 44100  # 샘플링 주파수 (Hz)
duration_low = 10  # 최저음을 녹음할 시간 (초)
duration_high = 10  # 최고음을 녹음할 시간 (초)

def handle_missing_values(df):
    df = df.fillna(0)  # 결측치(NaN)를 0으로 대체
    df.replace([np.inf, -np.inf], 0, inplace=True)  # 무한대 값 처리를 위해 매우 큰 값은 np.inf로 처리 후 0으로 대체
    return df

def record_audio(duration, filename):
    print(f"녹음 시작 ({duration}초)")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()  # 녹음이 끝날 때까지 대기
    wavfile.write(filename, fs, audio)  # 녹음한 데이터를 .wav 파일로 저장
    print(f"녹음 완료: {filename}")

def analyze_audio(file):
    print(f"분석 중: {file}")
    y, sr = librosa.load(file)

    # 특성 데이터 추출
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
    # MySQL에서 JSON 파일에 있는 musicId만 필터링하여 데이터 가져오기
    query = "SELECT * FROM music_table WHERE music_id IN %(music_ids)s"
    df_db = pd.read_sql(query, connection, params={"music_ids": tuple(music_ids_in_json)})
    
    # 필요한 특성만 선택
    feature_columns = [
        'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'tempo',
        'spectral_centroid_mean', 'spectral_centroid_var', 'rolloff_mean', 'rolloff_var',
        'chroma_stft_mean', 'chroma_stft_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var',
        'spectral_contrast_mean', 'spectral_contrast_var', 'melspectrogram_mean', 'melspectrogram_var'
    ] + [f'mfcc{i}_mean' for i in range(20)] + [f'mfcc{i}_var' for i in range(20)]
    
    print(f"필터링된 데이터베이스 크기: {df_db.shape}")

    return df_db, feature_columns

def find_similar_songs(user_features, df_db, feature_columns, music_dict):
    # db_features만 추출하여 결측치 및 무한대 값 처리
    db_features = df_db[feature_columns].copy()
    db_features = handle_missing_values(db_features)

    # user_features를 DataFrame으로 변환 (필요한 경우)
    if not isinstance(user_features, pd.DataFrame):
        user_features = pd.DataFrame(user_features, columns=feature_columns)

    # 공통된 컬럼만 사용하여 유사도 계산
    common_columns = db_features.columns.intersection(user_features.columns)

    db_features = db_features[common_columns]
    user_features = user_features[common_columns]

    # 스케일링 추가 (데이터 정규화)
    scaler = StandardScaler()
    db_features_scaled = scaler.fit_transform(db_features)
    user_features_scaled = scaler.transform(user_features)

    # 코사인 유사도 계산
    similarity = cosine_similarity(db_features_scaled, user_features_scaled)

    df_db['similarity'] = similarity.flatten()  # 유사도 값을 DataFrame에 추가

    # 상위 5개의 유사한 곡 추출 (중복 제거 후)
    top_5_songs = df_db.nlargest(5, 'similarity')[['music_id', 'similarity']].drop_duplicates()

    # music_id를 singer와 title로 매핑하여 최종 결과 출력
    final_results = []
    print("유사도 계산 후 상위 5개의 music_id와 유사도 값:")
    print(top_5_songs)  # 유사도와 music_id 확인
    
    for _, row in top_5_songs.iterrows():
        music_id = row['music_id']
        similarity_score = row['similarity']
        
        # 매핑 확인
        if music_id in music_dict:
            singer, title = music_dict[music_id]
            final_results.append({'singer': singer, 'title': title, 'similarity': similarity_score})
        else:
            print(f"매핑되지 않은 music_id: {music_id}")  # 매핑되지 않은 music_id 확인

    # 결과 출력
    if final_results:
        print("가장 유사한 5개의 노래 (singer와 title):")
        for result in final_results:
            print(f"Singer: {result['singer']}, Title: {result['title']}, Similarity: {result['similarity']:.6f}")
    else:
        print("매핑된 결과가 없습니다.")
    
    return final_results


def main():
    # music_matching.json 파일 경로
    json_path = r"C:\Users\USER\Desktop\jakpum3-2\music_maching.json"

    # JSON 파일 읽기
    with open(json_path, 'r', encoding='utf-8') as f:
        music_mapping = json.load(f)

    # JSON이 리스트로 시작하므로 첫 번째 요소에 접근 후, 'voiceWaveMatchingResponseDtoList'를 추출
    music_list = music_mapping[0]['voiceWaveMatchingResponseDtoList']

    # musicId를 키로 매핑
    music_dict = {item['musicId']: (item['singer'], item['title']) for item in music_list}
    
    # JSON에 있는 musicId 목록 추출
    music_ids_in_json = list(music_dict.keys())

    # 최저음 녹음 (장덕철 - 그날처럼)
    print("최저음을 불러주세요: 장덕철-그날처럼 한 소절 (참 많은 시간이 흘러가고 넌 어떻게 사는지 참 궁금해)")
    record_audio(duration_low, "lowest_pitch.wav")
    
    # 최고음 녹음 (아이유 - 너랑나)
    print("최고음을 불러주세요: 아이유-너랑나 한 소절 (너랑 나랑은 조금 남았지 몇날 몇실진 모르겠지만)")
    record_audio(duration_high, "highest_pitch.wav")

    # 최저음, 최고음 파일 분석 후 평균화
    user_lowest_pitch = analyze_audio("lowest_pitch.wav")
    user_highest_pitch = analyze_audio("highest_pitch.wav")
    
    # 평균값 계산
    user_mean_features = (user_lowest_pitch + user_highest_pitch) / 2

    # 데이터베이스에서 JSON 파일의 musicId에 해당하는 곡들만 필터링하여 가져오기
    df_db, feature_columns = fetch_database_data(music_ids_in_json)
    df_db = handle_missing_values(df_db)

    # 유사한 노래 찾기 및 결과 출력 (music_id -> singer, title 매핑)
    find_similar_songs(user_mean_features, df_db, feature_columns, music_dict)

if __name__ == "__main__":
    main()
