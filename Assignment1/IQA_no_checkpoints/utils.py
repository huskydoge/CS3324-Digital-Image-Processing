# PLCC and KRCC functions
import numpy as np
def get_PLLC(df1, df2):
    """
    Calculates the PLCC between two dataframes.
    y = df1; y_hat = df2
    """

    y = df1["score"].to_numpy()
    y_hat = df2["score"].to_numpy()
    return np.corrcoef(y, y_hat)[0,1]


def get_SRCC(df1, df2):
    """
    Calculates the SRCC between two dataframes.
    y = df1; y_hat = df2
    """
    y = df1["score"].to_numpy()
    y_hat = df2["score"].to_numpy()
    return np.spearmanr(y, y_hat)[0]
