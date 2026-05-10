# geometry/exceptions.py
class GeometryOutOfBoundsError(ValueError):
    """Выход параметров за границы области определения поверхности."""
    def __init__(self, param_name, value, bounds):
        self.param_name = param_name
        self.value = value
        self.bounds = bounds
        msg = f"{param_name}={value} вне диапазона [{bounds[0]}, {bounds[1]}]"
        super().__init__(msg)