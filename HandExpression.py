from __future__ import annotations
import json
import numpy as np

def list_to_str_with_round(arr: list[float], round_to: int) -> str:
    info = "["
    
    for i in range(0, len(arr)):
        info += str(round(arr[i], round_to))
        info += "," if i != len(arr) - 1 else "]"

    return info

class HandExpression:
    mean: str # 뜻
    distances: list[float]
    distances_calibrated: list[float] # calibrated size: 0.5
    slopes: list[float]
    size: float # 0.0 ~ 1.0

    def __init__(self, mean: str, size: float, distances: list[float], slopes: list[float]):
        self.distances = distances
        self.distances_calibrated = distances.copy()
        self.slopes = slopes
        self.size = size
        self.mean = mean
        
        self.calibrate()

    @staticmethod
    def from_str(command: str) -> HandExpression:
        expression = HandExpression("", 0, [], [])
        j = json.loads(f"[{command}]")

        expression.mean = str(j[0])
        expression.size = float(j[1])
        expression.distances = list(j[2])
        expression.distances_calibrated = list(j[2])
        expression.slopes = list(j[3])

        expression.calibrate()
        return expression

    def get_similarity(self, expression: HandExpression, verbose=False) -> tuple[float, float, float]:
        def sigmoid(x: float) -> float:
            return 1 / (1 + np.exp(x))

        sum = 0.0 # 차이 절댓값의 합
        sumd = 0.0
        sums = 0.0

        for i in range(0, len(self.distances_calibrated)):
            distance1: float = expression.distances_calibrated[i]
            distance2: float = self.distances_calibrated[i]
            slope1: float = expression.slopes[i]
            slope2: float = self.slopes[i]
            dd = abs(distance1 - distance2)
            ds = sigmoid(-np.log(abs(slope1 - slope2))) / 50

            if verbose:
                print(f"{dd}, {ds}")

            sumd += dd
            sums += ds

        return (sigmoid(np.log((sumd + sums) / 2)), sumd, sums)
    
    def get_highest_similar_expression(self, expressions_defined: list[HandExpression], standard: float = 0.0) -> tuple[HandExpression, float]:
        index: int = -1
        max_: float = 0.0
        slope: float = 1.0

        for i in range(0, len(expressions_defined)):
            expression_defined = expressions_defined[i]
            info = self.get_similarity(expression_defined)
            similarity = info[0]
            sumd = info[1]
            sums = info[2]

            if max_ < similarity and similarity >= standard and slope > sums and abs(slope - sums) > 0.003:
                index = i
                max_ = similarity
                slope = sums

        return (expressions_defined[index], max_) if index >= 0 else None
    
    def calibrate(self):
        if self.size == 0:
            return
        
        ratio: float = 0.5 / self.size
        
        for i in range(0, len(self.distances)):
            self.distances_calibrated[i] = self.distances[i] * ratio

    def __repr__(self):
        info = f'"{self.mean}",{round(self.size, 7)},'
        info += list_to_str_with_round(self.distances, 7)
        info += ","
        info += list_to_str_with_round(self.slopes, 7)

        return info
    
    def to_simple_info(self, compare_to: HandExpression) -> str:
        return f"이름: {self.mean}, 정확도: {round(self.get_similarity(compare_to) * 100, 2)}%"