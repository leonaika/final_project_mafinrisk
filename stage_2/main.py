import md
from feat_generator import FeatGen
from model import BullInTheBushes
from portfolio_manager import PortfolioManager
import pandas as pd
from sdk.oms_client import OmsClient
import os
from datetime import datetime, timedelta
from time import sleep


OMS_URL = "https://quant-competition-oms-test.yorkapp.com"
OMS_ACCESS_TOKEN = ""
client = OmsClient(OMS_URL, OMS_ACCESS_TOKEN)

dates = []
cur = datetime(2025, 6, 16)
end = datetime(2025, 9, 16)
step = timedelta(days=3)

while cur <= end:
    dates.append(cur)
    cur += step

models = []
for f in os.listdir("LIVETRADING/3d_models/"):
    f = "LIVETRADING/3d_models/" + f
    models.append(f)

model_idx = 0
cur_model_date = dates[model_idx]

gen = FeatGen()
model = BullInTheBushes(models[model_idx])
pm = PortfolioManager(client, simulation_flag=True)


if __name__ == "__main__":
    if True:
        df = md.get_data()
        print(df.head())
        df, features = gen.build_features(df)
        pred = model.get_predictions(df, features)
        # pred.to_parquet('cache_pred.parquet', index=False)

        print("Sleeping...")
        sleep(10)
    else:
        pred = pd.read_parquet("cache_pred.parquet")

    pm.manage(pred)

    print("Done")
