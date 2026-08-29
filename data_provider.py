import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from timefeatures import time_features
import warnings

class DLAM_Dataset(Dataset):
    def __init__(self, root_path=None, data_path=None, flag='train', size=None,
                 scale=True, timeenc=1, freq='h', id_col='series_id', cycle=None, 
                 train_timestamp_length=4320, val_timestamp_length=336, num_series=96): 
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24
            self.est_horizon = 24 * 7
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            self.est_horizon = size[3]
            
        assert flag in ['train', 'val', 'pred']
        type_map = {'train': 0, 'val': 1, 'pred':2}
        self.set_type = type_map[flag]
        
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.id_col = id_col
        self.cycle = cycle
        self.num_series = num_series
        self.train_timestamp_length = train_timestamp_length
        self.val_timestamp_length = val_timestamp_length
        self._phase_offset = (0 if self.set_type == 0 else (self.train_timestamp_length - self.seq_len) % self.cycle)

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        print(f"Loading AIML-TUDA/dlam-ts-project-data-2026 from HuggingFace...")
        splits = {'train': 'train.csv', 'val': 'validation_input.csv'}
        df_raw_train = pd.read_csv("hf://datasets/AIML-TUDA/dlam-ts-project-data-2026/" + splits["train"])
        df_raw_train = df_raw_train.sort_values(["series_id", "timestamp"])


        continuous_cols = [
            "workload_intensity",
            "demand_forecast",
            "staffing_forecast",
            "upstream_quality_forecast",
            "promotion_intensity",
            "shock_risk",
            "unit_reliability_forecast",
            "queue_pressure_forecast",
            "network_pressure_forecast",
            "event_load_forecast",
            "service_irregularity_risk_forecast",
            "throughput_disruption_risk_forecast",
        ]

        df_raw_train[continuous_cols] = (
            df_raw_train.groupby(self.id_col)[continuous_cols]
            .transform(lambda x: x.interpolate(method="linear"))
        )
        df_raw_train[continuous_cols] = (
            df_raw_train.groupby(self.id_col)[continuous_cols]
            .transform(lambda x: x.ffill().bfill())
        )
        features_without_target = [c for c in df_raw_train.columns if c != self.id_col]
        df_raw_train[features_without_target] = df_raw_train.groupby(self.id_col)[features_without_target].ffill().bfill()

        border1s = [0, self.train_timestamp_length - self.val_timestamp_length - self.seq_len]
        border2s = [self.train_timestamp_length - self.val_timestamp_length, self.train_timestamp_length]
        

        if self.set_type == 0:
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            df_l = df_raw_train
            self.windows_per_unit = self.train_timestamp_length - self.val_timestamp_length - self.seq_len - self.pred_len + 1
        elif self.set_type == 1:
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            df_l = df_raw_train
            self.windows_per_unit = self.val_timestamp_length - self.pred_len + 1
        elif self.set_type == 2:
            df_raw_val = pd.read_csv("hf://datasets/AIML-TUDA/dlam-ts-project-data-2026/" + splits["val"])

            border1 = 0
            border2 = self.seq_len + self.val_timestamp_length
            
            features_without_target = [c for c in df_raw_val.columns if c != "target" and c != self.id_col]
            df_raw_val[features_without_target] = df_raw_val.groupby(self.id_col)[features_without_target].ffill().bfill()

            df_t = df_raw_train.groupby(self.id_col, sort=False).tail(self.seq_len)
            df_l = pd.concat([df_t, df_raw_val], ignore_index=True)

            assert self.val_timestamp_length % self.pred_len == 0
            assert self.est_horizon % self.pred_len == 0
            assert self.val_timestamp_length >= self.est_horizon

            self.windows_per_unit = self.val_timestamp_length - self.est_horizon + 1
            df_l = df_l.sort_values(["series_id", "timestamp"])


        self.unit_list = [] # (length per unit, time_stamp, data, cycle_index)

        if self.scale:
            train_for_scaling = (df_raw_train.groupby(self.id_col, group_keys=False).apply(lambda x: x.iloc[:-self.val_timestamp_length])
            )
            self.scaler.fit(train_for_scaling[df_raw_train.columns[2:]])

        if self.set_type == 2:
            timestamp_length = self.val_timestamp_length + self.seq_len
        else:
            timestamp_length = self.windows_per_unit + self.seq_len + self.pred_len - 1
            
        for s_i, df_raw in df_l.groupby(self.id_col, sort=True):
            df_raw = df_raw[border1:border2]
            assert df_raw.shape[0] == timestamp_length

            cols_data = df_raw.columns[2:]
            df_data = df_raw[cols_data]

            if self.scale:
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values
            
            df_stamp = df_raw[['timestamp']].copy()
            df_stamp['timestamp'] = pd.to_datetime(df_stamp.timestamp)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.timestamp.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.timestamp.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.timestamp.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.timestamp.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(['timestamp'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['timestamp'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)

            self.unit_list.append((int(s_i.split('_')[-1]), data_stamp, data, 
                                   ((np.arange(len(data))  + self._phase_offset) % self.cycle)))

    def __getitem__(self, index):

        unit_index = index // self.windows_per_unit
        time_index = index % self.windows_per_unit
        if self.set_type == 2:
            time_index *= self.pred_len

        s_begin = time_index
        s_end = time_index + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        unit_id, data_stamp, data, cycle_index = self.unit_list[unit_index]

        data_x = data
        data_y = data

        seq_x = data_x[s_begin:s_end]
        seq_y = data_y[r_begin:r_end]

        time_x_mark = data_stamp[s_begin:s_end]
        time_y_mark = data_stamp[r_begin:r_end]

        one_hot = np.zeros((1, self.num_series), dtype=np.float32)
        one_hot[0, unit_id] = 1

        seq_x_mark = np.concatenate([time_x_mark, np.repeat(one_hot, len(time_x_mark), axis=0)], axis=-1)
        seq_y_mark = np.concatenate([time_y_mark, np.repeat(one_hot, len(time_y_mark), axis=0)], axis=-1)
    
        cycle_index = torch.tensor(cycle_index[s_end])


        return seq_x.astype(np.float32), seq_y.astype(np.float32), seq_x_mark.astype(np.float32), seq_y_mark.astype(np.float32), cycle_index

    def __len__(self):
        return self.num_series * self.windows_per_unit

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


data_dict = {
    'DLAM': DLAM_Dataset,
}

def data_provider(configs, flag):
    Data = data_dict[configs.data]
    timeenc = 0 if configs.embed != 'timeF' else 1

    if flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = configs.batch_size
        freq = configs.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = configs.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = configs.batch_size
        freq = configs.freq

    data_set = Data(
        root_path=configs.root_path,
        data_path=configs.data_path,
        flag=flag,
        size=[configs.seq_len, configs.label_len, configs.pred_len, configs.est_horizon],
        scale=configs.scale,
        timeenc=timeenc,
        freq=freq,
        cycle=configs.cycle
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=configs.num_workers,
        drop_last=drop_last)
    return data_set, data_loader