import pandas as pd

def decompose_signal(series):
    """
    Temporary placeholder decomposition.
    Splits the signal into 3 simple components.
    Later, this can be replaced with real VMD decomposition.
    """
    imf1 = series * 0.3
    imf2 = series * 0.3
    imf3 = series * 0.4

    imf_df = pd.DataFrame({
        "imf1": imf1,
        "imf2": imf2,
        "imf3": imf3
    }, index=series.index)

    return imf_df