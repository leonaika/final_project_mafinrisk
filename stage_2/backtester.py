import os
from datetime import datetime, timedelta
from time import time

import pandas as pd
from constants import stable_feats
from feat_generator import FeatGen
from model import BullInTheBushes
from portfolio_manager import PortfolioManager
from sdk.oms_client import OmsClient
from tqdm import tqdm

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

models_dir = "LIVETRADING/3d_models/"
# models_dict = {}
# for path in sorted(os.listdir(models_dir)):
#     assert path.endswith(".cbm")
#     assert path.startswith("cutoff_")

#     cutoff_date = path.split(".")[0].split("_")[1]
#     print("Parsed cutoff_date:", cutoff_date)
#     models_dict[cutoff_date] = BullInTheBushes(models_dir + path)
# model_dates = sorted(models_dict.keys())

models = [models_dir + m for m in sorted(os.listdir(models_dir))]

model_idx = 0
model = BullInTheBushes(models[model_idx])
gen = FeatGen()
pm = PortfolioManager(client, simulation_flag=True)


# df = pd.read_parquet('LIVETRADING/data/market_data_0125_0925.parquet')
# df.sort_values(['ticker', 'timestamp'], inplace=True)
# df, features = gen.build_features(df)
# df.to_parquet('df_with_feat.parquet', index=False)
# print(df.head())
# exit()

print("Reading data...")
df = pd.read_parquet("LIVETRADING/data/features_for_weekly_train_1609.parquet")
print(df["timestamp"].min(), df["timestamp"].max())

cur_date = datetime(2025, 6, 16)
end_date = datetime(2025, 9, 16)
time_step = timedelta(hours=12)
time_acc = timedelta(hours=0)

# model_name = "before_big_boom_2024-10-01"
# model_name = "after_big_boom_2024-12-31"
# model_name = "baseline_2025-06-16_iter8k_depth3"
# model_name = "baseline_2025-06-16"

model_name = "3d_models_stoploss_4"
# model_name = "7d_models"
# model_name = 'baseline_2025-06-16_24h'
SAVE_NAVLOGS = f"LIVETRADING/navlogs/{model_name}.csv"

t0 = time()
navlog = []
reslog = []
try:
    while cur_date <= end_date:
        data = df.loc[
            (df["timestamp"] <= (cur_date + timedelta(days=1)))
            & (df["timestamp"] >= cur_date)
        ].copy()
        data.reset_index(drop=True, inplace=True)

        if cur_date > dates[model_idx + 1]:
            print("SWITCH")
            model_idx += 1
            model = BullInTheBushes(models[model_idx])
            # model = BullInTheBushes(f"models/baseline_2025-06-16.cbm")

        # model trigger
        # if time_acc.seconds % (3600 * 12):
        if True:
            pred = model.get_predictions(data, stable_feats)
            # print(pred.head())
            # print(pred.tail())
            res = pm.manage_simulator(pred, data)
            reslog.append(res)
        else:
            # market trigger
            pm.manage_simulator(data=data, md_trigger_flag=True)

        nav = pm.get_nav_simulation(data)
        print("NAV:", cur_date, nav)
        navlog.append((cur_date, nav))

        cur_date += time_step
        # time_acc += time_step
except (KeyboardInterrupt, IndexError):
    pass

print(time() - t0)

navdf = pd.DataFrame(navlog, columns=["date", "nav"])
navdf.to_csv(SAVE_NAVLOGS, index=False)
