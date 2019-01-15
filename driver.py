'''
Created on Nov 10, 2018

@author: mroch
'''

import os
from keras.layers import Dense, Dropout, LSTM, GRU, Masking, TimeDistributed,Bidirectional, CuDNNLSTM
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt

from dsp.features import Features
from timit.corpus import Corpus
from dsp.utils import Timer
import pathlib
from myclassifier.batchgenerator import PaddedBatchGenerator
from myclassifier.crossvalidator import CrossValidator
import myclassifier.recurrent
from sklearn.metrics import confusion_matrix
from keras.layers.normalization import BatchNormalization

plt.switch_backend("TkAgg")


def main():
    adv_ms = 10  # frame advance and length
    len_ms = 20

    TimitBaseDir = 'C:\\Users\\rashm\\Documents\\CS682\\Lab2\\timit-for-students'

    corpus = Corpus(TimitBaseDir, os.path.join(TimitBaseDir, 'wav'))

    phonemes = corpus.get_phonemes()  # List of phonemes
    phonemesN = len(phonemes)  # Number of categories

    # Get utterance keys
    devel = corpus.get_utterances('train')  # development corpus
    eval = corpus.get_utterances('test')  # evaluation corpus

    #For testing on smaller dataset
    if True:
        truncate_to_N = 50

        print("Truncating t %d files" % (truncate_to_N))
        devel = devel[:truncate_to_N]  # truncate test for speed
        eval = eval[:truncate_to_N]

    features = Features(adv_ms, len_ms, corpus.get_audio_dir())
    # set features storage location
    features.set_cacheroot(
        os.path.join(TimitBaseDir, 'feature_cache').replace("\\", "/"))
    corpus.set_feature_extractor(features)

    f = corpus.get_features(devel[0])
    input_dim = f.shape[1]

    #Model specification
    models_rnn = [
        lambda dim, width, dropout, l2:
        [(Masking, [], {"mask_value": 0.,
                        "input_shape": [None, dim]}),
         (LSTM, [width], {
             "return_sequences": True,
             "kernel_regularizer": regularizers.l2(l2),
             "recurrent_regularizer": regularizers.l2(l2)
         }),
         (BatchNormalization, [], {}),
         (Dropout, [dropout], {}),
         (LSTM, [width*2], {
             "return_sequences": True,
             "kernel_regularizer": regularizers.l2(l2),
             "recurrent_regularizer": regularizers.l2(l2)
         }),
         (BatchNormalization, [], {}),
         (Dropout, [dropout], {}),
         (Dense, [phonemesN], {'activation': 'softmax',
                               'kernel_regularizer': regularizers.l2(l2)},
          (TimeDistributed, [], {}))
         ],
        lambda dim, width, dropout, l2:
        [
         (CuDNNLSTM, [width], {
             "return_sequences": True,
             "kernel_regularizer": regularizers.l2(l2),
             "recurrent_regularizer": regularizers.l2(l2),
             "input_shape": [None, dim]
         }),
         (Dropout, [dropout], {}),
         ((BatchNormalization, [], {})),
         (CuDNNLSTM, [width], {
             "return_sequences": True,
             "kernel_regularizer": regularizers.l2(l2),
             "recurrent_regularizer": regularizers.l2(l2)
         }),
         (Dropout, [dropout], {}),
         ((BatchNormalization, [], {})),
         (Dense, [phonemesN], {'activation': 'softmax',
                               'kernel_regularizer': regularizers.l2(l2)},
          (TimeDistributed, [], {}))
         ],
        lambda dim, width, dropout, l2:
        [(Masking, [], {"mask_value": 0.,
                        "input_shape": [None, dim]}),
         (LSTM, [width], {
             "return_sequences": True,
             "kernel_regularizer": regularizers.l2(l2),
             "recurrent_regularizer": regularizers.l2(l2)
         }),
         (BatchNormalization, [], {}),
         (Dropout, [dropout], {}),
         (LSTM, [width * 2], {
             "return_sequences": True,
             "kernel_regularizer": regularizers.l2(l2),
             "recurrent_regularizer": regularizers.l2(l2)
         }),
         (BatchNormalization, [], {}),
         (Dropout, [dropout], {}),
         (LSTM, [width * 3], {
             "return_sequences": True,
             "kernel_regularizer": regularizers.l2(l2),
             "recurrent_regularizer": regularizers.l2(l2)
         }),
         (BatchNormalization, [], {}),
         (Dropout, [dropout], {}),
         (Dense, [width], {'activation': 'relu',
                               'kernel_regularizer': regularizers.l2(l2)},
          (TimeDistributed, [], {})),
         (Dense, [phonemesN], {'activation': 'softmax',
                               'kernel_regularizer': regularizers.l2(l2)},
          (TimeDistributed, [], {}))
         ],
    ]

    #List of epochs for Grid Search
    epoch_search = [2,5,10]
    search_results_epoch = []

    n_folds = 3

    timer = Timer()
    for i in epoch_search:
        cv = CrossValidator(corpus, devel, models_rnn[2](input_dim, 30,0.1, 0.001),
                            myclassifier.recurrent.train_and_evaluate,
                            batch_size=100,
                            epochs=i,
                            n_folds=n_folds)
        search_results_epoch.append(cv.get_errors())
        print('Training Results for {} epoch model is {} \n'.format(i,
                                                                    cv.get_errors()))

    print("Total time to train: {}",timer.elapsed())
    min_err_epoch = []
    for i in range(len(search_results_epoch)):
        min_err_epoch.append(np.amin(search_results_epoch[i]))

    min_err_epoch_index = np.argmin(min_err_epoch)
    best_epoch = epoch_search[min_err_epoch_index]

    print("Best accuracy observed when number of epochs equals {}".format(
        best_epoch))
    print("Minimum error while training is {}".format(
        min_err_epoch[min_err_epoch_index]))

    #Plot the Loss grapgh during training and validation
    plt.figure()
    loss_values = cv.get_losses()
    for i in range(0, len(loss_values)):
        plt.plot(np.arange(len(cv.get_losses()[i])), cv.get_losses()[i],
                 label='Fold - {}'.format(i))
    plt.xlabel("Loss History Index per fold")
    plt.ylabel("Loss Value")
    savedir = os.path.join('.', 'loss_graph')
    pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
    plt.savefig("{}/Loss_Graph".format(savedir))
    plt.clf()

    models = cv.get_models()
    least_error_model_values = np.amin(np.asarray(cv.get_errors()))
    least_error_model_index = np.argmin(least_error_model_values)
    least_error_model = models[least_error_model_index]

    #Testing with the Test Data (eval) on the least_error_model
    test_gen = PaddedBatchGenerator(corpus, eval, len(eval))
    test_examples, test_labels = next(test_gen)

    loss, acc = least_error_model.evaluate(test_examples, test_labels,
                                           verbose=True)

    print("Total time to complete training and testing of the model: {}", timer.elapsed())
    print("Accuracy on Test Data", acc)
    y_pred = least_error_model.predict_generator(test_gen, steps=1, verbose=0)
    for i in range(0, y_pred.shape[0]):
        c = confusion_matrix(test_labels.argmax(axis=2)[i],
                             y_pred.argmax(axis=2)[i])
    print('Confusion Matrix : \n', c)


if __name__ == '__main__':
    plt.ion()

    main()
