import pandas as pd
import plotly.express as px
import os


class DataModule:
    def __init__(self, path):
        self.path = path
        self.df = None
        self.read_data()

    def read_data(self):
        if os.path.exists(self.path):
            self.df = pd.read_csv(self.path)
        else:
            self.df = pd.DataFrame()

    def head(self, n=5):
        return self.df.head(n)

    def describe(self):
        return self.df.describe()

    def visualize_corr(self):
        numeric = self.df.select_dtypes(include=['number'])
        if numeric.shape[1] == 0:
            return None
        corr = numeric.corr()
        fig = px.imshow(corr, text_auto=True, aspect='auto', title='Correlation matrix')
        return fig

    def visualize_dist(self, cols=None):
        if cols is None:
            cols = self.df.select_dtypes(include=['number']).columns.tolist()[:4]
        if not cols:
            return None
        fig = px.histogram(self.df, x=cols[0], nbins=40, title=f'Distribution of {cols[0]}')
        return fig

    def append_row(self, row_dict):
        row = pd.DataFrame([row_dict])
        self.df = pd.concat([self.df, row], ignore_index=True)
        # try to persist
        try:
            self.df.to_csv(self.path, index=False)
            return True
        except Exception:
            return False
