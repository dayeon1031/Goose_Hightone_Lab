from flask import Flask, render_template, request, jsonify
import time #분석을 위한 딜레이 시뮬레이션

app = Flask(__name__)

# 곡 리스트 초기화
song_list = [
    "곡 1: 예시곡 1",
    "곡 2: 예시곡 2",
    "곡 3: 예시곡 3",
]

@app.route('/', methods=['GET', 'POST'])
def index():
    global song_list
    if request.method == 'POST':
        # 파일 업로드 처리
        if 'song_file' in request.files:
            file = request.files['song_file']
            if file.filename.endswith('.txt'):
                # 파일 내용을 읽어 리스트에 추가
                lines = file.read().decode('utf-8').splitlines()
                song_list.extend(lines)

    return render_template('index.html', song_list=song_list)

#record 페이지

@app.route('/record')
def record():
    return render_template("record.html")

@app.route('/process_record', methods=['POST'])
def process_record():
    # 여기서 녹음 데이터 처리 및 분석을 수행합니다.
    time.sleep(5)  # 분석 지연 시뮬레이션
    with open('analysis_result.txt', 'w') as f:
        f.write('분석결과: 녹음된 목소리는 440Hz입니다.')
    return jsonify({"result": "분석 완료! 이 결과를 받아보세요."})

# 분석 결과 파일 불러오기
@app.route('/get_analysis_result')
def get_analysis_result():
    return send_file('analysis_result.txt', as_attachment=True)

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