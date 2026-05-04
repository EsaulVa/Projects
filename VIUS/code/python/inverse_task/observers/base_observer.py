# observers/base_observer.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class Observer(ABC):
    """Базовый наблюдатель за любым вычислительным процессом."""
    
    @abstractmethod
    def update(self, event: str, data: Dict[str, Any]) -> None:
        """Принять порцию данных о событии (event)."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Сбросить накопленную информацию."""
        pass

    def report(self) -> Dict[str, Any]:
        """Вернуть сводку (по умолчанию пустую)."""
        return {}