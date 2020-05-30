import pickle
import keras
import numpy as np
import tensorflow as tf
from keras.layers import Input, Bidirectional, GRU, Masking, Dense, Dropout, TimeDistributed, concatenate
from keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from keras_transformer.transformer import double_trans_modality_part

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def calc_test_result(result, test_label, test_mask, print_detailed_results=False):
    """
    # Arguments
        predicted test labels, gold test labels and test mask

    # Returns
        accuracy of the predicted labels
    """
    true_label = []
    predicted_label = []

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if test_mask[i, j] == 1:
                true_label.append(np.argmax(test_label[i, j]))
                predicted_label.append(np.argmax(result[i, j]))

    if print_detailed_results:
        print("Confusion Matrix :")
        print(confusion_matrix(true_label, predicted_label))
        print("Classification Report :")
        print(classification_report(true_label, predicted_label))
    print("Accuracy ", accuracy_score(true_label, predicted_label))
    return accuracy_score(true_label, predicted_label)


def create_one_hot_labels(train_label, test_label):
    """
    # Arguments
        train and test labels (2D matrices)

    # Returns
        one hot encoded train and test labels (3D matrices)
    """

    maxlen = int(max(train_label.max(), test_label.max()))

    train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen + 1))
    test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen + 1))

    for i in range(train_label.shape[0]):
        for j in range(train_label.shape[1]):
            train[i, j, train_label[i, j]] = 1

    for i in range(test_label.shape[0]):
        for j in range(test_label.shape[1]):
            test[i, j, test_label[i, j]] = 1

    return train, test


def create_mask(train_data, test_data, train_length, test_length):
    '''
    # Arguments
        train, test data (any one modality (text, audio or video)), utterance lengths in train, test videos

    # Returns
        mask for train and test data
    '''

    train_mask = np.zeros((train_data.shape[0], train_data.shape[1]), dtype='float')
    for i in range(len(train_length)):
        train_mask[i, :train_length[i]] = 1.0

    test_mask = np.zeros((test_data.shape[0], test_data.shape[1]), dtype='float')
    for i in range(len(test_length)):
        test_mask[i, :test_length[i]] = 1.0

    return train_mask, test_mask


def load_data(name):
    if name == 'MOSI':
        (train_text, train_label, test_text, test_label, max_utt_len, train_len, test_len) = pickle.load(
            open('./input/text.pickle', 'rb'))
        (train_audio, _, test_audio, _, _, _, _) = pickle.load(open('./input/audio.pickle', 'rb'))
        (train_video, _, test_video, _, _, _, _) = pickle.load(open('./input/video.pickle', 'rb'))
        train_label, test_label = create_one_hot_labels(train_label.astype('int'), test_label.astype('int'))
        train_mask, test_mask = create_mask(train_text, test_text, train_len, test_len)
        category_num = 2
    elif name == 'IEMOCAP':
        train_text, train_audio, train_video, train_label, \
        test_text, test_audio, test_video, test_label, \
        max_len, train_len, test_len = pickle.load(open("input/IEMOCAP.pkl", 'rb'))
        train_label, test_label = create_one_hot_labels(train_label.astype('int'), test_label.astype('int'))
        train_mask, test_mask = create_mask(train_text, test_text, train_len, test_len)
        category_num = 7
    elif name == 'MELD-sent':
        train_text, train_audio, train_label, train_sent, \
        test_text, test_audio, test_label, test_sent, \
        max_len, train_len, test_len = pickle.load(open("input/MELD.pkl", 'rb'))
        train_video = train_audio
        test_video = test_audio
        train_label, test_label = create_one_hot_labels(train_sent.astype('int'), test_sent.astype('int'))
        train_mask, test_mask = create_mask(train_text, test_text, train_len, test_len)
        category_num = 3
    elif name == 'MELD-emotion':
        train_text, train_audio, train_label, train_sent, \
        test_text, test_audio, test_label, test_sent, \
        max_len, train_len, test_len = pickle.load(open("input/MELD.pkl", 'rb'))
        train_video = train_audio
        test_video = test_audio
        train_label, test_label = create_one_hot_labels(train_label.astype('int'), test_label.astype('int'))
        train_mask, test_mask = create_mask(train_text, test_text, train_len, test_len)
        category_num = 6
    else:
        raise LookupError

    return train_text, train_audio, train_video, train_label, train_len, train_mask, \
        test_text, test_audio, test_video, test_label, test_len, test_mask, \
        category_num


dataset_name = 'MOSI'

train_text, train_audio, train_video, train_label, train_len, train_mask, \
    test_text, test_audio, test_video, test_label, test_len, test_mask, \
    category_num = load_data(dataset_name)

text_dim = train_text.shape[-1]
video_dim = train_video.shape[-1]
audio_dim = train_audio.shape[-1]


def trans_modality():
    in_text = Input(shape=(train_text.shape[1], train_text.shape[2]))
    in_audio = Input(shape=(train_audio.shape[1], train_audio.shape[2]))
    in_video = Input(shape=(train_video.shape[1], train_video.shape[2]))

    masked_text = Masking(mask_value=0)(in_text)
    masked_audio = Masking(mask_value=0)(in_audio)
    masked_video = Masking(mask_value=0)(in_video)

    drop_rnn = 0.3
    gru_units = 300

    rnn_text = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                             merge_mode='concat')(masked_text)
    rnn_audio = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                              merge_mode='concat')(masked_audio)
    rnn_video = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                              merge_mode='concat')(masked_video)

    rnn_text = Dropout(drop_rnn)(rnn_text)
    rnn_audio = Dropout(drop_rnn)(rnn_audio)
    rnn_video = Dropout(drop_rnn)(rnn_video)

    drop_dense = 0.3
    dense_units = 100

    dense_text = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_text))
    dense_audio = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_audio))
    dense_video = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_video))

    # text - video - text
    tv_encode, tv_decode, vt_encode, vt_decode = double_trans_modality_part(main_input=dense_text,
                                                                            main_dim=text_dim,
                                                                            middle_input=dense_video,
                                                                            middle_dim=video_dim,
                                                                            encoder_num=10,
                                                                            decoder_num=5,
                                                                            head_num=5,
                                                                            hidden_dim=200,
                                                                            attention_activation='relu',
                                                                            feed_forward_activation='relu',
                                                                            dropout_rate=0.5,
                                                                            prefix="text-video")

    # text - audio - text
    ta_encode, ta_decode, at_encode, at_decode = double_trans_modality_part(main_input=dense_text,
                                                                            main_dim=text_dim,
                                                                            middle_input=dense_audio,
                                                                            middle_dim=audio_dim,
                                                                            encoder_num=10,
                                                                            decoder_num=5,
                                                                            head_num=5,
                                                                            hidden_dim=120,
                                                                            attention_activation='relu',
                                                                            feed_forward_activation='relu',
                                                                            dropout_rate=0.5,
                                                                            prefix="text-audio")

    concat_feature = concatenate([tv_encode, vt_encode, ta_encode, at_encode, dense_text, dense_audio, dense_video],
                                 axis=-1)

    classify_score = TimeDistributed(Dense(category_num, activation="softmax"))(concat_feature)

    model = Model(inputs=[in_text, in_audio, in_video],
                  outputs=[vt_decode, at_decode, tv_decode, ta_decode, classify_score])

    return model


def train():
    model = trans_modality()

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=[keras.losses.mae,
              keras.losses.mae,
              keras.losses.mae,
              keras.losses.mae,
              keras.losses.categorical_crossentropy,
              keras.losses.mae],
        loss_weights=[10, 10, 10, 10, 1, 0.5],
        metrics=[keras.metrics.mae,
                 keras.metrics.categorical_accuracy,
                 keras.metrics.categorical_crossentropy],
        sample_weight_mode='temporal'
    )

    history = model.fit(
        batch_size=32,
        x=[train_text, train_audio, train_video],
        y=[train_text, train_text, train_video, train_audio, train_label],
        sample_weight=[train_mask, train_mask, train_mask, train_mask, train_mask],
        validation_data=(
            [test_text, test_audio, test_video], [test_text, test_text, test_video, test_audio, test_label],
            [test_mask, test_mask, test_mask, test_mask, test_mask]),
        epochs=100,
        verbose=2,
    )


if __name__ == "__main__":
    train()
