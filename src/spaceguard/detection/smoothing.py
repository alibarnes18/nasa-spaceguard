import numpy as np
import pandas as pd


def exponential_moving_average(
    errors: np.ndarray, window_size: int = 10
) -> np.ndarray:
    
    
    error_series = pd.Series(errors.flatten())
    smoothed = error_series.ewm(span=window_size, adjust=False).mean().values
    return smoothed