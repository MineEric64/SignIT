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
        self.distances_calibrated = distances
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

    def get_similarity(self, expression: HandExpression) -> float:
        def sigmoid(x: float) -> float:
            return 1 / (1 + np.exp(x))

        sum = 0.0 # 차이 절댓값의 합

        for i in range(0, len(self.distances_calibrated)):
            distance1: float = expression.distances_calibrated[i]
            distance2: float = self.distances_calibrated[i]
            slope1: float = expression.slopes[i]
            slope2: float = self.slopes[i]

            sum += min(abs(distance1 - distance2), abs(slope1 - slope2))

        return sigmoid(np.log(sum))
    
    def get_highest_similar_expression(self, expressions_defined: list[HandExpression], standard: float = 0.0) -> tuple[HandExpression, float]:
        index: int = -1
        max_: float = 0.0

        for i in range(0, len(expressions_defined)):
            expression_defined = expressions_defined[i]
            similarity = self.get_similarity(expression_defined)

            if max_ < similarity and similarity >= standard:
                index = i
                max_ = similarity

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