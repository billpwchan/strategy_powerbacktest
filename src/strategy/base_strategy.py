from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
import numpy as np

class BaseStrategy(ABC):
    def __init__(self, parameters: Dict[str, Any] = None):
        self.parameters = parameters or {}
        self.position = 0
        self.signals = []
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on the strategy logic
        
        Args:
            data: Historical market data
            
        Returns:
            Series of trading signals (1: Buy, -1: Sell, 0: Hold)
        """
        pass
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators used by the strategy
        
        Args:
            data: Historical market data
            
        Returns:
            DataFrame with additional technical indicators
        """
        return data
        
    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters
        
        Returns:
            Boolean indicating if parameters are valid
        """
        return True