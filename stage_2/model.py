from catboost import CatBoostRegressor
import pandas as pd


MODEL_PATH = 'old_code_train_160625.cbm'


class BullInTheBushes():

    def __init__(self, path: str = MODEL_PATH):
        self.model = CatBoostRegressor()
        self.model.load_model(path)
    
    def get_predictions(self, df, features):
        # print('Getting model predictions')
        pred = self.model.predict(df[features])
        res = pd.DataFrame()
        res['timestamp'] = df['timestamp']
        res['ticker'] = df['ticker']
        res['predict'] = pred
        latest_pred = (
            res.sort_values(['ticker', 'timestamp'])
            .drop_duplicates('ticker', keep='last')
            .sort_values('predict', ascending=False)
            .reset_index(drop=True)
        )

        return latest_pred
