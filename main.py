#########################################
# BCI competition III dataset IVa
# Subject al
# HW 1
# Jurio.
# 10/16, 2023
##########################################
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import ShuffleSplit, cross_val_score

import mne
from mne.decoding import CSP

from scipy.io import loadmat
from sklearn.pipeline import Pipeline

# Path
data_path = './EEG dataset/'
result_path = './Results/'

if os.path.exists(result_path) is False:
    os.makedirs(result_path)

# Tasks
__task_1__ = True
__task_2__ = False

__task_3__ = False
__decode__ = True
__search_filter__ = False

__search_t__ = False


def draw_line(iters, datas, name_of_alg, colors):
    avg = np.mean(datas, axis=-1)
    std = np.std(datas, axis=-1)
    r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))  # 上方差
    r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))  # 下方差
    plt.plot(iters, avg, color=colors, label=name_of_alg, linewidth=3.5)
    plt.fill_between(iters, r1, r2, color=colors, alpha=0.2)


def task1_time_freq_pos(eeg_data, label_y, eeg_fs, cues_pos, event_id, ch_name, pos):
    """
    EEG time features visualization
    :param pos:
    :param ch_name:
    :param eeg_fs:
    :param eeg_data
    :return:
    """
    # MNE object
    info = mne.create_info(
        ch_names=[i[0] for i in ch_name],
        sfreq=eeg_fs,
        ch_types='eeg')
    pos_dic = dict(zip(info.ch_names, pos))
    montage = mne.channels.make_dig_montage(pos_dic)

    info.set_montage(montage)

    raw = mne.io.RawArray(eeg_data.T, info)

    # Apply band-pass filter
    raw.filter(12, 28, fir_design="firwin", skip_by_annotation="edge")

    # Events and epochs
    raw.crop(tmax=100)
    events = np.vstack((cues_pos, np.zeros(len(cues_pos)), label_y[0, :])).T.astype(int)
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

    # Epochs
    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        0.5,
        3.5,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
    )

    # # plot raw eeg data
    # raw.plot(n_channels=10, scalings='auto', title='Raw EEG Signals')

    # # plot electrode position
    # raw.plot_sensors(title='Channel positions')
    #
    # plot events top map
    # epochs[3].compute_psd().plot_topomap(ch_type="eeg", agg_fun=np.median, normalize=True)

    # # plot frequency spectrum
    # raw.compute_psd().plot()
    # raw.compute_psd().plot_topo(color="k", fig_facecolor="w", axis_facecolor="w")

    # # plot time-frequency features
    # freqs = np.logspace(*np.log10([12, 28]), num=8)
    # n_cycles = freqs / 2.0  # different number of cycle per frequency
    # power, itc = mne.time_frequency.tfr_morlet(
    #     epochs,
    #     freqs=freqs,
    #     n_cycles=n_cycles,
    #     use_fft=True,
    #     return_itc=True,
    #     decim=3,
    #     n_jobs=None,
    # )
    # power.plot_joint(
    #     baseline=(0, 0.5), mode="mean", tmin=0, tmax=3.5,
    #     timefreqs=[(1, 20), (2, 16), (2.8, 14)]
    # )
    # itc.plot_topo(title="Inter-Trial coherence", vmin=0.0, vmax=1.0, cmap="Reds")


def task3_CSP_decoding(eeg_data, label_y, cues_pos, event_id, eeg_fs, ch_name, pos,
                       search=True):
    """
    Extract EEG CSP features and decoding (search BP filter)
    :param search: hyperparameter search
    :param event_id:
    :param cues_pos:
    :param label_y:
    :param eeg_data:
    :param eeg_fs:
    :param ch_name:
    :param pos:
    :return:
    """
    # Set
    __plot_csp__ = True
    fr_win_len = 16  # 8-24Hz \mu & \beta
    fr_step = 2
    test_r = 0.1

    # Cue onset
    tMin, tMax = 0.5, 3.5

    # Search or best result
    if search:
        # Filter
        l_fr = np.array([i for i in range(0, 40 - fr_win_len, fr_step)])
        h_fr = fr_win_len + l_fr
    else:
        l_fr, h_fr = [12.0], [28.0]

    # MNE object
    info = mne.create_info(
        ch_names=[i[0] for i in ch_name],
        sfreq=eeg_fs,
        ch_types='eeg')
    pos_dic = dict(zip(info.ch_names, pos))
    montage = mne.channels.make_dig_montage(pos_dic)

    info.set_montage(montage)
    raw = mne.io.RawArray(eeg_data.T, info)

    # Decoding
    scores = []
    for i in range(len(l_fr)):
        # Apply band-pass filter
        raw.filter(l_fr[i], h_fr[i], fir_design="firwin", skip_by_annotation="edge")

        # Events
        events = np.vstack((cues_pos, np.zeros(len(cues_pos)), label_y[0, :])).T.astype(int)
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

        # Epochs
        epochs = mne.Epochs(
            raw,
            events,
            event_id,
            tMin,
            tMax,
            proj=True,
            picks=picks,
            baseline=None,
            preload=True,
        )

        # Prepare data for training
        x = epochs.get_data()
        y = label_y[0, :] - 1

        # ten-fold cross-validation
        cv = ShuffleSplit(10, test_size=test_r, random_state=42)
        x_split = cv.split(x)

        # Classification with LDA on CSP features
        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components=10, reg=None, log=True, norm_trace=False)

        clf = Pipeline([("CSP", csp), ("LDA", lda)])
        scores.append(cross_val_score(clf, x, y, cv=cv, n_jobs=None))

        # Printing the results
        print(f'Classification accuracy on BP filter [{l_fr[i]}, {h_fr[i]}]: '
              f'{np.mean(scores[i]):.2f}')

        # plot CSP patterns estimated on full data for visualization
        if __plot_csp__:
            csp.fit_transform(x, y)
            csp.plot_patterns(epochs.info, ch_type="eeg", units="Patterns (AU)", size=1.5)

    # Printing the chance level
    chance_level = np.mean(label_y[0, :] == 2)
    chance_level = max(chance_level, 1.0 - chance_level)
    print(f'Classification accuracy of Chance level: {chance_level:.2f}')

    # Plot
    if search:
        plt.style.use('seaborn-whitegrid')
        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 30,
                 }
        plt.figure(figsize=(12, 8))
        xx = np.arange(len(scores))
        draw_line(xx, np.array(scores), 'Band width(16Hz)', 1)

        chance = np.zeros(len(l_fr)) + chance_level
        plt.plot(xx, chance, color='k', label='Chance', linewidth=3.5)

        plt.xticks(ticks=xx, labels=l_fr, fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlabel('Left Frequency', fontsize=30)
        plt.ylabel('Accuracy', fontsize=30)
        plt.legend(prop=font1, loc='lower right')
        plt.title(f'Classification Performance on BP filter (test={test_r})', fontsize=35)
        plt.savefig(result_path + f't3_search_BPfilter(test={test_r}).png', dpi=300)
        plt.show()


def task3_search_t(eeg_data, label_y, cues_pos, event_id, eeg_fs, ch_name, pos):
    """
    hyperparameter search time windows
    :param event_id:
    :param cues_pos:
    :param label_y:
    :param eeg_data:
    :param eeg_fs:
    :param ch_name:
    :param pos:
    :return:
    """
    # Set
    t_win_len = [10, 20, 30, 35]
    # t_win_len = [25, 30]
    t_step = 1
    test_r = 0.1

    # BP Filter
    l_fr, h_fr = 12.0, 28.0

    # Search or best result
    tMin, tMax = [], []
    for i in range(len(t_win_len)):
        tMin.append(np.array([i for i in range(0, 36 - t_win_len[i], t_step)]) / 10)
        tMax.append(tMin[i] + t_win_len[i] / 10)

    # MNE object
    info = mne.create_info(
        ch_names=[i[0] for i in ch_name],
        sfreq=eeg_fs,
        ch_types='eeg')
    pos_dic = dict(zip(info.ch_names, pos))
    montage = mne.channels.make_dig_montage(pos_dic)

    info.set_montage(montage)
    raw = mne.io.RawArray(eeg_data.T, info)
    # Apply band-pass filter
    raw.filter(l_fr, h_fr, fir_design="firwin", skip_by_annotation="edge")

    # Decoding
    try:
        scores = np.load(result_path + f't3_search_tWin(test={test_r}).npy', allow_pickle=True)
    except:
        scores = []
        for i in range(len(tMin)):
            scores_this_win = []
            for j in range(len(tMin[i])):
                # Events
                events = np.vstack((cues_pos, np.zeros(len(cues_pos)), label_y[0, :])).T.astype(int)
                picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

                # Epochs
                epochs = mne.Epochs(
                    raw,
                    events,
                    event_id,
                    tMin[i][j],
                    tMax[i][j],
                    proj=True,
                    picks=picks,
                    baseline=None,
                    preload=True,
                )

                # Prepare data for training
                x = epochs.get_data()
                y = label_y[0, :] - 1

                # ten-fold cross-validation
                cv = ShuffleSplit(10, test_size=test_r, random_state=42)

                # Classification with LDA on CSP features
                lda = LinearDiscriminantAnalysis()
                csp = CSP(n_components=10, reg=None, log=True, norm_trace=False)

                clf = Pipeline([("CSP", csp), ("LDA", lda)])
                scores_this_win.append(cross_val_score(clf, x, y, cv=cv, n_jobs=None))

                # Printing the results
                print(f'Classification accuracy on time windows [{tMin[i][j]}, {tMax[i][j]}]: '
                      f'{np.mean(scores_this_win[j]):.2f}')

            scores.append(scores_this_win)

    # Plot
    plt.style.use('seaborn-whitegrid')
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 30,
             }
    plt.figure(figsize=(14, 8))
    colors = plt.get_cmap('Set1')
    for k in range(len(scores)):
        if len(scores[k]) > 1:
            xx = np.arange(len(scores[k]))
            draw_line(xx,
                      np.array(scores[k]),
                      f'{(0.1 * t_win_len[k]):.1f}s',
                      colors(k + 1))
        else:
            plt.scatter(0, np.mean(scores[k]), s=100, c=colors(k + 1), label='3.5s')
            plt.errorbar(0, np.mean(scores[k]), yerr=np.std(scores[k]), fmt='-',
                         elinewidth=3, ecolor=colors(k + 1), capsize=20, capthick=3)

    x_tick = np.arange(0, len(scores[0]), 5)
    x_tick_label = np.arange(0, len(scores[0]), 5) / 10
    plt.xticks(ticks=x_tick, labels=x_tick_label, fontsize=25)
    plt.yticks(fontsize=25)
    plt.ylim(0.9, 1.01)
    plt.xlabel('Windows Left', fontsize=30)
    plt.ylabel('Accuracy', fontsize=30)
    plt.legend(prop=font1, loc='lower right')
    # plt.legend(prop=font1)
    plt.title(f'Classification Performance on time windows (test={test_r})', fontsize=35)
    plt.savefig(result_path + f't3_search_tWin(test={test_r}).png', dpi=300)
    plt.show()
    np.save(result_path + f't3_search_tWin(test={test_r}).npy', np.array(scores))


if __name__ == '__main__':
    # load EEG.mat
    data = loadmat(data_path + 'data_set_IVa_al.mat')
    labels = loadmat(data_path + 'true_labels_al.mat')

    # EEG signals
    eeg = data['cnt']

    # Target y
    target = data['mrk']['y'][0][0][0]
    real_label = labels['true_y']
    test_id = labels['test_idx']
    events_id = dict(right=1, foot=2)

    # time pos
    cues = data['mrk']['pos'][0][0][0]

    # Info
    data_name = data['nfo']['name'][0][0][0]
    fs = data['nfo']['fs'][0][0][0][0]
    ele_name = data['nfo']['clab'][0][0][0]
    ele_x = data['nfo']['xpos'][0][0]
    ele_y = data['nfo']['ypos'][0][0]

    ele_pos = 0.1 * np.hstack((ele_x, ele_y, np.zeros(ele_x.shape)))

    # Task 1: EEG signals analysis
    if __task_1__:
        task1_time_freq_pos(eeg, real_label, fs, cues, events_id, ele_name, ele_pos)

    # Task 3: EEG CSP decoding
    if __task_3__:
        if __decode__:
            task3_CSP_decoding(eeg, real_label, cues, events_id, fs, ele_name, ele_pos,
                               search=__search_filter__)
        if __search_t__:
            task3_search_t(eeg, real_label, cues, events_id, fs, ele_name, ele_pos)

    sys.exit(0)
