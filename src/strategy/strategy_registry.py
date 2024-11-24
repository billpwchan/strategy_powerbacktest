from typing import Dict, Type
from .base_strategy import BaseStrategy
from .macd_strategy import MACDStrategy

class StrategyRegistry:
    """
    Registry for managing and accessing trading strategies.
    Implements the Singleton pattern.
    """
    _instance = None
    _strategies: Dict[str, Type[BaseStrategy]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StrategyRegistry, cls).__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, strategy_name: str, strategy_class: Type[BaseStrategy]) -> None:
        """
        Register a new strategy class.
        
        Args:
            strategy_name: Name of the strategy
            strategy_class: Strategy class to register
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise ValueError("Strategy must inherit from BaseStrategy")
        cls._strategies[strategy_name] = strategy_class

    @classmethod
    def get_strategy(cls, strategy_name: str) -> Type[BaseStrategy]:
        """
        Get a strategy class by name.
        
        Args:
            strategy_name: Name of the strategy to retrieve
            
        Returns:
            Strategy class
            
        Raises:
            KeyError: If strategy is not found
        """
        if strategy_name not in cls._strategies:
            raise KeyError(f"Strategy '{strategy_name}' not found in registry")
        return cls._strategies[strategy_name]

    @classmethod
    def list_strategies(cls) -> list[str]:
        """
        List all registered strategies.
        
        Returns:
            List of strategy names
        """
        return list(cls._strategies.keys()) 