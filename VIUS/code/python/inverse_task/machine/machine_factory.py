# # machine/machine_factory.py
# from .machine3axis_exact import Machine3AxisExact_ODE
# # from .machine5axis_exact_ode import Machine5AxisExact_ODE  # раскомментировать, когда будет готов

# _REGISTRY = {
#     'Machine3AxisExact_ODE': Machine3AxisExact_ODE,
#     # 'Machine5AxisExact_ODE': Machine5AxisExact_ODE,
# }

# def create_machine(class_name: str, params: dict):
#     klass = _REGISTRY.get(class_name)
#     if klass is None:
#         raise ValueError(f"Неизвестный тип станка: {class_name}")
#     return klass(params)
from .machine3axis_exact import Machine3AxisExact_ODE
# from .machine5axis_exact_ode import Machine5AxisExact_ODE

_REGISTRY = {
    'Machine3AxisExact_ODE': Machine3AxisExact_ODE,
    # 'Machine5AxisExact_ODE': Machine5AxisExact_ODE,  # когда будет готов
}

def create_machine(class_name: str, params: dict):
    klass = _REGISTRY.get(class_name)
    if klass is None:
        raise ValueError(f"Неизвестный тип станка: {class_name}")
    # Распаковываем словарь в именованные аргументы
    return klass(**params)