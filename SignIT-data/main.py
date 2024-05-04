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
    return ImageFont.truetype("SignIT-data/fonts/NEXON_Lv2_Gothic_Bold.ttf", size)

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
                    '"ㅎㅇ",0.6925683,[0.1129616,0.1260465,0.1103281,0.0964167,0.1150161,0.1379517,0.087125,0.0800134,0.3090716,0.1575691,0.1012851,0.0921667,0.3776921,0.1469689,0.0975114,0.0922074,0.3855936,0.1040011,0.0746073,0.0731322],[0.5333872,1.7705291,5.7391258,3.7659482,0.3880106,4.3758458,4.4044875,6.1019931,2.2034548,27.4656007,18.1021349,16.4024156,4.9974375,23.7104455,39.1589886,149.2906292,9.0087717,4.0762021,3.9424096,4.4911405]'
                    ,'"1",0.608535,[0.0553133,0.0633672,0.0474752,0.0459115,0.1136335,0.1186251,0.0730621,0.0652188,0.2762317,0.0414357,0.0860746,0.0531663,0.0896003,0.0133628,0.0885023,0.0465827,0.1046299,0.0142741,0.0687667,0.0323509],[0.418652,1.7438583,0.9497448,0.8089875,1.9905492,68.6222871,40.1766386,85.6398482,5.8483758,358.8104326,7.2667254,6.7153543,1.1304006,6.5955943,3.7242698,3.2445147,1.0749283,0.9048849,2.7198237,2.2170374]'
                    ,'"2",0.548207,[0.0530631,0.0514913,0.0366352,0.0470713,0.1551243,0.106911,0.068146,0.0609404,0.2439469,0.1128132,0.0820032,0.0776696,0.2955362,0.0095564,0.0854891,0.0543703,0.1060166,0.0412092,0.0595524,0.0385739],[0.1929256,1.1525765,0.4175466,0.4907668,2.2232745,4.3598343,4.6948582,7.1596825,2.2344161,4.3239575,3.5100617,3.8561805,11.2809163,0.4294134,3.6916669,3.1518497,1.522154,5.672041,2.3161767,1.8115857]'
                    ,'"3",0.5409826,[0.0516313,0.0497075,0.0481686,0.0536647,0.1457289,0.0978593,0.0648009,0.0593418,0.2171569,0.1114882,0.0726154,0.066365,0.273791,0.0859023,0.0702734,0.0669642,0.265095,0.0081128,0.0508558,0.0428538],[0.3715228,1.5790024,0.5119083,0.5762028,1.3866126,3.9110921,4.1028243,5.3670366,1.9836862,134.57368,104.942715,30.6934715,7.7236947,2.6837174,3.1752059,3.4685864,6.5903449,0.8631014,5.042949,8.4957827]'
                    ,'"4",0.767566,[0.1020844,0.1209203,0.0814302,0.0840326,0.1659198,0.1452811,0.0978196,0.0895776,0.3279035,0.1716875,0.1166601,0.1038958,0.4202884,0.1600477,0.1085256,0.0973984,0.4206238,0.1138838,0.0803235,0.0766266],[0.4211107,1.4609876,2.8235219,0.313311,2.773162,4.092686,4.8012213,8.6706081,2.2411668,26.8554062,26.7853807,48.8936958,5.6100782,9.8564047,23.0416297,29.2165501,13.9796003,2.9568161,2.8958133,3.2013269]'
                    ,'"SpiderMan!",0.471227,[0.0945603,0.0944129,0.0839053,0.0688463,0.1221155,0.1209636,0.0721289,0.0583099,0.2624769,0.0925443,0.0381255,0.0379567,0.0904894,0.0784667,0.0358697,0.0365873,0.0861258,0.0987977,0.0513866,0.0446378],[1.1393174,2.3777556,4.6705898,3.4907684,0.5536484,14.0468789,16.0576622,35.1866982,4.2670577,6.8702857,1.8472457,3.8647499,0.6009529,8.4080128,1.2870775,2.7137922,0.5717209,3.6267979,5.0187941,4.5059427]'
                    ,'"CEI",0.5546288,[0.1304959,0.0986448,0.0919682,0.052431,0.0507699,0.0456932,0.0538605,0.038034,0.0753948,0.0374551,0.0579109,0.0411155,0.0722665,0.0429537,0.0472675,0.0456096,0.0654091,0.1351415,0.0714451,0.055485],[1.090081,2.9473339,20.5379136,5.2581566,0.5471073,4.0823862,20.106356,9.7441207,1.1455889,12.826901,733.8240181,13.9269275,0.8014212,23.7830151,13.3416227,19.1250125,0.4618516,11.2274567,13.243525,98.5428224]'
                    ,'"주먹인사",0.2761646,[0.1149947,0.0911447,0.0689242,0.0414446,0.0339988,0.054613,0.0436036,0.0424445,0.0570623,0.0382143,0.0411857,0.0414571,0.0565474,0.0223716,0.03651,0.0395569,0.0619752,0.0175588,0.031881,0.0319318],[1.4242655,4.9552242,3.8599201,0.7104285,0.0986035,2.6012805,4.9254607,71.2244173,1.3362174,4.7518159,9.3623258,18.1511461,0.6644572,18.8420219,24.3376745,8.6301906,0.4817934,6.7686448,4.3831742,4.7912521]'
                    ,'"ㅗ",0.6496833,[0.1413413,0.0984515,0.080587,0.0551033,0.0534278,0.0906197,0.029765,0.0353989,0.0525138,0.1770769,0.1094256,0.0752828,0.3730528,0.1322537,0.0406616,0.0536732,0.0946746,0.0793406,0.032689,0.0368555],[1.2614317,3.248215,3.1119275,0.5366807,1.199323,5.23682,20.6084783,18.4992045,0.0442913,1473.6393849,24.1003613,7.7533392,4.5651629,8.8252789,3.6857527,10.0248372,0.9081684,2.2424585,1.3277616,2.7685017]'
                    ,'"가위",0.5887874,[0.0933133,0.131522,0.0998781,0.0765435,0.1216635,0.1314644,0.0807606,0.0697828,0.3060293,0.0647938,0.0952216,0.0712954,0.0901596,0.0324934,0.1036481,0.0601176,0.1136175,0.0311501,0.071201,0.0423774],[0.682124,1.8665331,2.9974185,2.3426318,0.3062867,14.187733,14.1271284,11.3193276,3.8455738,14.3574559,26.0088957,12.0984665,1.1916257,3.8153036,5.409149,3.260543,1.06471,9.1179816,2.9658776,2.1845212]'
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

def hand_recognition_realtime(expressions_defined: list[HandExpression | str], mpHands, my_hands, mpDraw):
    print("잠시만 기다려주세요... C를 눌러 프로그램을 탈출하시기 바랍니다.")

    if len(expressions_defined) > 0 and isinstance(expressions_defined[0], str):
        expressions_defined_str = expressions_defined.copy()
        expressions_defined: list[HandExpression] = []

        for i in range(0, len(expressions_defined_str)):
            expressions_defined.append(HandExpression.from_str(expressions_defined_str[i]))

    cap = cv2.VideoCapture(0)

    while True:
        success, image = cap.read()
        height, width, c = image.shape

        image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        expressions: list[HandExpression] = hand_recognition(image2, mpHands, my_hands, mpDraw, image)
        highest: tuple[HandExpression, float] = None

        if len(expressions) > 0: # 임시 코드
            highest = expressions[0].get_highest_similar_expression(expressions_defined, 0.0)
            for expression in expressions_defined:
                if expression.mean == "CEI":
                    break
                    #expression.get_similarity(expressions[0], True)

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
            slopes: list[float] = []

            for i in range(0, 20):
                distance: float = get_distance(handLms.landmark[i].x,
                                               handLms.landmark[i].y,
                                               handLms.landmark[i + 1].x,
                                               handLms.landmark[i + 1].y
                                               )
                slope: float = get_abs_slope(
                    handLms.landmark[i].x,
                    handLms.landmark[i].y,
                    handLms.landmark[i + 1].x,
                    handLms.landmark[i + 1].y
                )
                distances.append(distance)
                slopes.append(slope)

            sizes_index = [4, 8, 12, 16, 20]
            size = 0.0

            for i in sizes_index:
                size = max(size, get_distance(handLms.landmark[0].x,
                                    handLms.landmark[0].y,
                                    handLms.landmark[i].x,
                                    handLms.landmark[i].y
                                    ))
            expression = HandExpression("", size, distances, slopes)
            expressions.append(expression)

            if draw_hands_image is not None:
                mpDraw.draw_landmarks(draw_hands_image, handLms, mpHands.HAND_CONNECTIONS)

    return expressions

def get_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

def get_abs_slope(x1: int, y1: int, x2: int, y2: int) -> float:
    dx = x2 - x1

    if dx == 0:
        return float("inf")
    
    return abs((y2 - y1) / dx)

main()