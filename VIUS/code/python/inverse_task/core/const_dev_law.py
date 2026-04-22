from core.base_deviation_law import *
class ConstantDeviation(DeviationLaw):
    def __init__(self, tan_theta: float = 0.0):
        self._tan = tan_theta

    def tan_theta(self, s: float) -> float:
        return self._tan

    def d_tan_theta_ds(self, s: float) -> float:
        return 0.0