import keras
import matplotlib.pyplot as plt
import numpy as np


def f1_score(true, pred, f1_score_class):
    correct_positives = 0
    pred_positives = 0
    true_positives = 0

    for t, p in zip(true, pred):
        if t[f1_score_class] > 0.5 and p[f1_score_class] > 0.5:
            correct_positives += 1
        if t[f1_score_class] > 0.5:
            true_positives += 1
        if p[f1_score_class] > 0.5:
            pred_positives += 1

    if pred_positives > 0:
        precision = correct_positives / pred_positives
    else:
        precision = 0
    if true_positives > 0:
        recall = correct_positives / true_positives
    else:
        recall = 0
    if precision == 0 and recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


class PlotLosses(keras.callbacks.Callback):
    def __init__(self, loss_image_path, accuracy_image_path, f1_image_path=None, validation_data=None, f1_score_class=0):
        super().__init__()
        self.loss_image_path = loss_image_path
        self.accuracy_image_path = accuracy_image_path
        self.f1_image_path = f1_image_path
        self.validation_data = validation_data
        self.f1_score_class = f1_score_class
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []

        self.fig = plt.figure()

        self.logs = []

        self.val_f1s = []

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []

        self.fig = plt.figure()

        self.logs = []

        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1

        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.savefig(self.loss_image_path)
        plt.clf()

        plt.plot(self.x, self.acc, label="acc")
        plt.plot(self.x, self.val_acc, label="val_acc")
        plt.legend()
        plt.savefig(self.accuracy_image_path)
        plt.clf()

        if self.f1_image_path is not None and self.validation_data is not None:
            val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
            val_targ = self.validation_data[1]
            _val_f1 = f1_score(val_targ, val_predict, self.f1_score_class)
            self.val_f1s.append(_val_f1)

            plt.plot(self.x, self.val_f1s, label="f1")
            plt.legend()
            plt.savefig(self.f1_image_path)
            plt.clf()
        else:
            self.val_f1s.append(0)

        print("Iter %04d: loss=%02.2f acc=%02.2f val_loss=%02.2f val_acc=%02.2f f1=%02.2f" % (self.i, self.losses[-1], self.acc[-1], self.val_losses[-1], self.val_acc[-1], self.val_f1s[-1]))
