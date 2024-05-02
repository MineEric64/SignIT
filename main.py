"""
[데이터 만들기]
- 파이썬 이용
- mediapipe, opencv를 이용해 데이터 크롤링 및 인식
- 한국수어사전 웹사이트에서 영상을 추출 (웹크롤링 이용)
- csv나 json 파일로 데이터 구조화
- 데이터 너무 커지면 압축 (MessagePack, LZ4 같은 압축 기술)

[앱 만들기]
- 코틀린 & 안드로이드 스튜디오 이용
- 직관적인 UI
- 카메라를 이용해서 코틀린에 있는 손 인식 라이브러리 사용
- 기존에 있던 데이터 대조
"""

import math

import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image

from HandExpression import HandExpression

def get_nanum_font(size: int):
    return ImageFont.truetype("fonts/NanumGothicExtraBold.ttf", size)

FONT_KR = get_nanum_font(16)
    
def putTextKR(image, x: int, y: int, text: str, color):
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    draw.text((x, y), text, color, font=FONT_KR)

    return np.array(image_pil)

def main():
    mpHands = mp.solutions.hands
    my_hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    while True:
        print("1. 크롤링")
        print("2. 데이터 구조화")
        print("3. 실시간 손 인식 & 종료")
        print("4. 종료")

        select = input("숫자를 입력하세요: ")

        match select:
            case "1":
                crawling()
            
            case "2":
                structured()
            
            case "3":
                hand_recognition_realtime([
                    HandExpression.from_str("ㅎㅇ,0.5598243,[0.0843156,0.0974272,0.0833473,0.0684212,0.1401812,0.107723,0.0704284,0.0596023,0.235142,0.1252078,0.0800656,0.0665435,0.2884362,0.1188639,0.0793848,0.0718104,0.3053236,0.0837295,0.0594721,0.0552487]")

                ],
                                          mpHands, my_hands, mpDraw)
                return
            
            case _:
                return
            
        print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ\n")

def crawling():
    return

def structured():
    return

def hand_recognition_realtime(expressions_defined: list[HandExpression], mpHands, my_hands, mpDraw):
    print("잠시만 기다려주세요... C를 눌러 프로그램을 탈출하시기 바랍니다.")

    cap = cv2.VideoCapture(0)

    while True:
        success, image = cap.read()
        height, width, c = image.shape

        image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        expressions: list[HandExpression] = hand_recognition(image2, mpHands, my_hands, mpDraw)
        highest: tuple[HandExpression, float] = None

        if len(expressions) > 0: # 임시 코드
            highest = expressions[0].get_highest_similar_expression(expressions_defined)

        #print(expressions)
        
        image3 = putTextKR(image, 10, 10,
                           f"이름: {highest[0].mean}, 정확도: {round(highest[1] * 100, 2)}%" if highest != None else "N/A",
                           (0, 0, 0))

        cv2.imshow("Realtime Hand Recognition", image3)
        key = cv2.waitKey(1)

        if key == 67 or key == 99: # C
            cv2.destroyWindow("Realtime Hand Recognition")
            break
        elif key == 68 or key == 100: # D
            import pyperclip

            pyperclip.copy(str(expressions[0]))
            print("Copied successfully to clipboard!")

def hand_recognition(image: cv2.Mat, mpHands, my_hands, mpDraw) -> list[HandExpression]:
    """각 벡터 사이의 거리를 측정해 기록 (0-1, 1-2, 2-3 등)"""
    expressions: list[HandExpression] = []
    results = my_hands.process(image)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            distances: list[float] = []

            for i in range(0, 20):
                distance: float = get_distance(handLms.landmark[i].x,
                                               handLms.landmark[i].y,
                                               handLms.landmark[i + 1].x,
                                               handLms.landmark[i + 1].y
                                               )
                distances.append(distance)

            size = get_distance(handLms.landmark[0].x,
                                handLms.landmark[0].y,
                                handLms.landmark[12].x,
                                handLms.landmark[12].y
                                )
            expression = HandExpression("", size, distances)
            expressions.append(expression)

    return expressions

def get_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

main()