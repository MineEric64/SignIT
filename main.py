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
    return ImageFont.truetype("fonts/NEXON_Lv2_Gothic_Bold.ttf", size)

FONT_KR = get_nanum_font(24)
    
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
                    HandExpression.from_str("ㅎㅇ,0.6406905,[0.4725127,1.5315198,3.2986618,2.8136746,0.143839,4.0155325,4.0021206,5.4761856,2.0531518,18.0388412,13.8271807,14.2644774,4.3260188,23.2946829,109.2975133,70.7574846,7.5708637,5.1003054,6.0474333,9.0296814]")
                    #,HandExpression.from_str("1,0.2223344,[0.6755338,1.152393,4.4093682,0.1962716,2.8488999,9.6950757,8.6599624,21.047235,3.7263703,6.0490975,9.5211018,116.2321022,0.9713795,12.2594004,4.2110002,6.887117,0.698449,17.5658216,1.9175382,2.9172917]")
                    #,HandExpression.from_str("2,0.8243095,[0.2256337,1.0853725,0.7982021,0.250031,2.4152246,5.2488346,5.2165964,9.4606201,2.3714011,6.4926514,5.0478258,5.9509572,211.4361289,1.6336957,3.0592839,2.5396329,1.0033144,40.4620905,2.005398,1.5274805]")
                    #,HandExpression.from_str("3,0.7885674,[0.076907,0.9866234,0.8799837,0.6224894,1.1363086,4.0497789,3.583756,3.5299615,1.9467752,35.8440869,14.0633193,7.6884738,4.8353204,4.0584768,6.6235237,13.312251,52.4675931,3.0343994,3.8539529,3.6207117]")
                    #,HandExpression.from_str("4,0.7346906,[0.3963562,1.299773,3.8067493,0.0454077,2.4596015,3.1575902,3.2037101,4.1208758,1.7757997,6.9882596,6.3378274,6.4115772,3.159628,42.3562466,34.8547842,112.521374,7.1030914,4.4844438,4.2595729,5.2181802]")
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
        expressions: list[HandExpression] = hand_recognition(image2, mpHands, my_hands, mpDraw, image)
        highest: tuple[HandExpression, float] = None

        if len(expressions) > 0: # 임시 코드
            highest = expressions[0].get_highest_similar_expression(expressions_defined, 0.0)

        #print(expressions)
        
        image3 = putTextKR(image, 10, 10,
                           f"이름: {highest[0].mean}, 정확도: {round(highest[1] * 100, 2)}%" if highest != None else "N/A",
                           (0, 0, 0))
        

        cv2.imshow("Realtime Hand Recognition", image3)
        key = cv2.waitKey(1)

        if key == 67 or key == 99: # C
            cv2.destroyAllWindows()
            break
        elif key == 68 or key == 100: # D
            import pyperclip

            pyperclip.copy(str(expressions[0]))
            print("Copied successfully to clipboard!")

def hand_recognition(image: cv2.Mat, mpHands, my_hands, mpDraw, draw_hands_image = None) -> list[HandExpression]:
    """각 벡터 사이의 거리를 측정해 기록 (0-1, 1-2, 2-3 등)"""
    expressions: list[HandExpression] = []
    results = my_hands.process(image)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            distances: list[float] = []

            for i in range(0, 20):
                distance: float = get_abs_slope(handLms.landmark[i].x,
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

            if draw_hands_image is not None:
                mpDraw.draw_landmarks(draw_hands_image, handLms, mpHands.HAND_CONNECTIONS)

    return expressions

def get_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

def get_abs_slope(x1: int, y1: int, x2: int, y2: int) -> float:
    return abs((y2 - y1) / (x2 - x1))

main()