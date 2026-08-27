from data_provider import data_provider
from models.combined_model import Combined_Model
import metrics

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from tools import EarlyStopping, adjust_learning_rate, visual


import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

class Exp(object):
    def __init__(self, configs):
        self.configs = configs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Combined_Model(configs).to(self.device)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.configs, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.configs.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.HuberLoss(delta=0.5)
        return criterion

    def vali(self, vali_data, vali_loader, metric):
        total_loss = []
        preds = []
        trues = []
        inputx = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # encoder - decoder
                if self.configs.use_amp:
                    with torch.amp.autocast(device_type="cuda"): 
                        outputs = self.model(batch_x, batch_cycle, batch_x_mark)
                else:
                    outputs = self.model(batch_x, batch_cycle, batch_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                loss = metric(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.configs.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.configs.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.configs.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.configs.pct_start,
                                            epochs=self.configs.train_epochs,
                                            max_lr=self.configs.learning_rate)

        for epoch in range(self.configs.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # max_memory = 0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # encoder - decoder
                if self.configs.use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        outputs = self.model(batch_x, batch_cycle, batch_x_mark)

                        f_dim = -1 if self.configs.features == 'MS' else 0
                        outputs = outputs[:, -self.configs.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.configs.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_cycle, batch_x_mark)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.configs.features == 'MS' else 0
                    outputs = outputs[:, -self.configs.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.configs.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.configs.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.configs.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                # current_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
                # max_memory = max(max_memory, current_memory)

                if self.configs.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.configs, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, metrics.WAPE)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.configs.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.configs)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # print(f"Max Memory (MB): {max_memory}")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='val')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # encoder - decoder
                if self.configs.use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        outputs = self.model(batch_x, batch_cycle, batch_x_mark)
                else:
                    outputs = self.model(batch_x, batch_cycle, batch_x_mark)

                f_dim = -1 if self.configs.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.configs.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.configs.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        # inputx = np.concatenate(inputx, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        ### denorm ###
        denorm_preds = np.stack([test_data.inverse_transform(pred) for pred in preds])
        denorm_trues = np.stack([test_data.inverse_transform(true) for true in trues])

        ### denorm ###

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # wape, mae, mse, rmse, mape, mspe, rse, corr = metrics.metric(preds, trues)
        wape, mae, mse, rmse, mape, mspe, rse, corr = metrics.metric(denorm_preds[:,:,-1], denorm_trues[:,:,-1])

        print('mse:{}, mae:{}, wape:{}'.format(mse, mae, wape))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, wape:{}'.format(mse, mae, wape))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                if self.args.use_amp:
                    with torch.amp.autocast(device_type="cuda"): 
                        outputs = self.model(batch_x, batch_cycle, batch_x_mark)
                else:
                    outputs = self.model(batch_x, batch_cycle, batch_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        denorm_preds = np.stack([pred_data.inverse_transform(pred) for pred in preds])


        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', denorm_preds)

        return