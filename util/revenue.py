import numpy as np
import pandas as pd


def revenue_expected(df: pd.DataFrame):
    total_revenue = np.sum(df['query_total_rev'].values)
    total_pl = np.sum(df['query_total_rev_pl_slots'].values)
    total_ol: float = max(0, total_revenue - total_pl)
    return total_revenue, total_pl, total_ol
