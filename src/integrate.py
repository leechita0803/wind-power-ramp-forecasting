import numpy as np

def integrate_predictions(predictions):
    """
    Combine IMF predictions into one final forecast.
    """
    final_pred = (
        np.array(predictions["imf1"]) +
        np.array(predictions["imf2"]) +
        np.array(predictions["imf3"])
    )
    return final_pred