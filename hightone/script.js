// 버튼과 메시지 요소를 가져옵니다
const button = document.getElementById('changeColorButton');
const message = document.getElementById('colorMessage');

// 배경 색상 리스트
const colors = ['#FFDDC1', '#FFABAB', '#FFC3A0', '#D5AAFF', '#6D28D9'];

// 현재 색상 인덱스
let currentColorIndex = 0;

// 버튼 클릭 이벤트 핸들러를 추가합니다
button.addEventListener('click', function() {
    // 배경 색상을 변경합니다
    document.body.style.backgroundColor = colors[currentColorIndex];

    // 메시지를 업데이트합니다
    message.textContent = `현재 배경 색상: ${colors[currentColorIndex]}`;

    // 색상 인덱스를 업데이트합니다
    currentColorIndex = (currentColorIndex + 1) % colors.length;
});
