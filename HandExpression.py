from __future__ import annotations
import numpy as np

class HandExpression:
    mean: str # 뜻
    distances: list[float]
    distances_calibrated: list[float]
    size: float # 0.0 ~ 1.0

    def __init__(self, mean: str, size: float, distances: list[float]):
        self.distances = distances
        self.distances_calibrated = distances
        self.size = size
        self.mean = mean
        
        self.calibrate()

    @staticmethod
    def from_str(command: str) -> HandExpression:
        expression = HandExpression("", 0, [])
        value = ""
        count = 0

        for i in range(0, len(command)):
            ch = command[i]

            if ch == ",":
                match count:
                    case 0:
                        expression.mean = value
                    case 1:
                        expression.size = float(value)

                value = ""
                count += 1

            elif ch == "[": # distances
                values = command[i + 1:len(command) - 2].split(",")
                
                for value2 in values:
                    expression.distances.append(float(value2))

                break

            else:
                value += ch

        expression.calibrate()
        return expression

    def get_similarity(self, expression: HandExpression) -> float:
        def sigmoid(x: float) -> float:
            return 1 / (1 + np.exp(x))

        sum = 0.0 # 차이 절댓값의 합

        for i in range(0, len(self.distances_calibrated)):
            distance1: float = expression.distances_calibrated[i]
            distance2: float = self.distances_calibrated[i]

            sum += abs(distance1 - distance2)

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

        self.size = 0.5

    def __repr__(self):
        info = f"{self.mean},{round(self.size, 7)},["

        for i in range(0, len(self.distances)):
            info += str(round(self.distances[i], 7))
            info += "," if i != len(self.distances) - 1 else "]"

        return info