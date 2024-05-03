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
                    HandExpression.from_str("ㅎㅇ,0.5598243,[0.0843156,0.0974272,0.0833473,0.0684212,0.1401812,0.107723,0.0704284,0.0596023,0.235142,0.1252078,0.0800656,0.0665435,0.2884362,0.1188639,0.0793848,0.0718104,0.3053236,0.0837295,0.0594721,0.0552487]")
                    ,HandExpression.from_str("1,0.5,[0.2931913,0.3165028,0.1944873,0.2044441,0.4768443,0.5580896,0.3427008,0.2857942,1.2808742,0.1544044,0.3965392,0.3002323,0.4762013,0.0592539,0.3738107,0.2292974,0.5543905,0.1109274,0.2839462,0.1604281]")
                    ,HandExpression.from_str("2,0.5,[0.0536647,0.070482,0.0400955,0.0468711,0.1334506,0.1081482,0.0684415,0.0594558,0.2454123,0.1140796,0.074059,0.0656833,0.2803742,0.0125798,0.0848934,0.0553908,0.0985087,0.0452633,0.0618816,0.0369753]")
                    ,HandExpression.from_str("3,0.5,[0.0535173,0.0626672,0.0474047,0.0535444,0.1286592,0.0957054,0.0613603,0.0575249,0.2144125,0.1107039,0.0698701,0.0642281,0.2659302,0.089128,0.0707904,0.068111,0.271052,0.0116794,0.0535463,0.0478691]")
                    ,HandExpression.from_str("4,0.5,[0.0663368,0.0750401,0.0538337,0.0509664,0.1218373,0.0948103,0.0633608,0.0570704,0.2168962,0.1127399,0.0744925,0.0653096,0.2776038,0.1039465,0.0701251,0.0612241,0.2755609,0.0743275,0.0508699,0.0466991]")
                    ,HandExpression.from_str("주먹인사,0.5,[0.1916886,0.2704857,0.1951443,0.1629875,0.1359368,0.1256382,0.2108611,0.0692245,0.3033398,0.1326806,0.240156,0.0474212,0.2693309,0.1228179,0.2372553,0.0341531,0.2808129,0.1312756,0.1679444,0.0274294]")
                    ,HandExpression.from_str("ㅗ,0.5,[0.0741572,0.1026253,0.075473,0.053877,0.1348913,0.1337885,0.0285074,0.0258773,0.0898391,0.1514374,0.0834244,0.0641995,0.2906201,0.1016679,0.0261563,0.0328285,0.0417054,0.055574,0.03088,0.0233348]")
                    ,HandExpression.from_str("SpiderMan!,0.5,[0.2186643,0.1823301,0.1689262,0.1373482,0.3409354,0.260012,0.1497391,0.1154849,0.5489959,0.2020163,0.0750271,0.0715503,0.1805877,0.1684457,0.0615834,0.0664832,0.1609216,0.2107869,0.0935568,0.0741588]")
                    ,HandExpression.from_str("CEI,0.5,[0.2397161,0.2133339,0.2104135,0.1029913,0.1891404,0.195695,0.0884321,0.0832815,0.1756405,0.1827458,0.0763587,0.0551052,0.1904978,0.1783351,0.0522262,0.0651359,0.1701711,0.2192659,0.1138786,0.0785328]")
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
            highest = expressions[0].get_highest_similar_expression(expressions_defined, 0.5)

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

            if draw_hands_image is not None:
                mpDraw.draw_landmarks(draw_hands_image, handLms, mpHands.HAND_CONNECTIONS)

    return expressions

def get_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

main()