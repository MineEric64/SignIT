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
                    HandExpression.from_str('"ㅎㅇ",0.6739595,[0.0791045,0.0922721,0.0745494,0.0611431,0.1202845,0.1030444,0.0678884,0.0602232,0.235015,0.1116826,0.0707002,0.0635938,0.261882,0.1081187,0.073,0.0708641,0.2836459,0.0820573,0.0565222,0.0566783],[0.4004375,1.3661445,2.0251043,1.5127314,0.1294259,3.3300438,3.5143321,4.4051354,1.8589493,19.7217023,22.9081346,18.3700541,4.4961447,40.9300809,70.1490587,62.8048642,7.2279658,3.8852951,3.8335813,5.4640404]')
                    ,HandExpression.from_str('"1",0.6300495,[0.0586795,0.0921313,0.0508867,0.0431062,0.0849849,0.1187819,0.0658929,0.0576148,0.2558391,0.0282515,0.1041358,0.0617816,0.1368775,0.0120929,0.0956748,0.045149,0.1341544,0.0162633,0.0674127,0.0298701],[0.7838961,2.5575617,1.9263521,0.2343893,2.5716901,11.9951803,25.9450719,10.5155666,4.1774816,2.6739703,5.5672366,6.6164282,1.2314576,0.0812822,3.0883825,2.6090555,0.9186905,0.9057229,1.6814973,1.1324882]')
                    ,HandExpression.from_str('"2",0.8162767,[0.0453978,0.0431412,0.0324768,0.0577368,0.1574817,0.1144106,0.0705244,0.0677403,0.2596637,0.1172121,0.07552,0.0859965,0.2987671,0.0229508,0.0985829,0.044394,0.1329714,0.0531123,0.0705307,0.0363744],[0.1088992,0.888507,0.4012603,0.4323957,2.5254732,4.7736377,4.3644463,8.4077776,2.2761478,4.4302727,3.0717984,3.2939752,9.3312186,3.1823874,2.5551826,1.9478956,1.322978,6.5398892,1.7995622,1.1220955]')
                    ,HandExpression.from_str('"3",0.8651076,[0.0303004,0.0356963,0.0516591,0.0640285,0.152677,0.1038393,0.0669235,0.0639266,0.2305919,0.1183053,0.0757819,0.0744148,0.2885187,0.092298,0.0808278,0.079482,0.2937008,0.0098902,0.0614825,0.0552225],[0.0458206,2.0945384,0.5276795,0.5954246,1.2957803,3.6768097,3.8095896,4.9986539,1.9380791,18.9494687,19.1711799,25.3938811,12.1210989,2.729764,5.0425528,5.9044649,9.9749447,266.3209647,3.3133986,4.3800364]')
                    ,HandExpression.from_str('"4",0.6773919,[0.0656919,0.0798571,0.0407839,0.0448589,0.1210498,0.1009363,0.0686058,0.0608043,0.2343966,0.1100395,0.0707232,0.0661806,0.2689497,0.109308,0.0733783,0.0687911,0.2895447,0.0827491,0.0561102,0.0557114],[0.4813025,1.3203091,5.2309973,0.29186,28.5489015,4.4176723,4.9394865,6.3739003,2.2036336,165.1325102,62.2115072,32.361935,5.3000209,7.9697463,9.2141075,14.9011701,20.6424924,2.1939376,2.0413559,2.6446895]')
                    ,HandExpression.from_str('"ㅗ",0.7695208,[0.1012039,0.1100129,0.0863897,0.0503823,0.0899339,0.1329605,0.039012,0.0458849,0.0698541,0.1538904,0.0783054,0.0594863,0.3008585,0.1187643,0.0242858,0.0347556,0.105699,0.0906929,0.0247815,0.0253434],[1.3637942,6.6258427,3.012565,0.6264393,4.3825282,10.9876659,13.8293049,49.7353151,1.120988,50.5999185,10.7950681,5.9730073,11.49022,3.9420125,0.938972,8.9878622,0.9054756,1.7289534,0.8788148,4.1112235]')
                    ,HandExpression.from_str('"주먹인사",0.3626844,[0.149806,0.1608494,0.1593106,0.0804876,0.1210992,0.1094259,0.0514712,0.0394762,0.0495034,0.0931,0.0442877,0.0401646,0.0699187,0.0614066,0.0336713,0.0359226,0.0804852,0.0450815,0.0326408,0.0247417],[1.5795812,5.1380217,38.5592881,0.9065673,17.575367,1.8813644,12.5621152,3.0012215,0.5960689,2.9250016,8.4627334,4.140103,0.6743353,7.4867416,3.8411058,4.1780825,0.4807587,3.6890885,4.0170493,4.9926691]')
                    ,HandExpression.from_str('"SpiderMan!",0.8009251,[0.0914257,0.1049604,0.0883705,0.0658005,0.1298441,0.1306421,0.0781756,0.0627831,0.2809243,0.1040941,0.0442465,0.0409251,0.054537,0.0917502,0.03667,0.0364223,0.0568243,0.1039667,0.0521665,0.0447216],[1.7633379,4.8696972,7.0570478,4.4547355,1.3178728,5.8719789,10.2109277,7.8174273,40.6355183,3.5986288,5.3229902,3.889492,1.4988942,3.6741185,3.7438864,3.1737422,1.4465871,2.5439477,2.1909785,2.4455255]')
                    ,HandExpression.from_str('"CEI",0.5921735,[0.1269537,0.1200785,0.0925126,0.0519006,0.0467797,0.111821,0.0417361,0.0400048,0.0739618,0.1100756,0.0424003,0.033313,0.0869761,0.1096033,0.0335668,0.0365234,0.0891744,0.1231025,0.0624335,0.0524278],[1.3194192,7.6906602,2.1272729,0.0597251,1.3941808,16.2073799,3.7857578,29.4062037,0.4309487,11.479301,3.0156766,15.736041,0.6933241,7.6638388,1.8044779,25.5439431,0.9162492,10.7400384,8.0324449,3.3803705]')
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
    return abs((y2 - y1) / (x2 - x1))

main()