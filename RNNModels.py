from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

from tensorflow.keras import layers as Layers
from tensorflow.keras import backend as K

from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D

def RNNSpeechModel(nCategories, samplingrate=16000, inputLength=16000):
    # simple LSTM
    sr = samplingrate
    iLen = inputLength

    inputs = Layers.Input((iLen,))

    x = Layers.Reshape((1, -1))(inputs)

    x = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, iLen),
                       padding='same', sr=sr, n_mels=80,
                       fmin=40.0, fmax=sr / 2, power_melgram=1.0,
                       return_decibel_melgram=True, trainable_fb=False,
                       trainable_kernel=False,
                       name='mel_stft')(x)

    x = Normalization2D(int_axis=0)(x)

    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    x = Layers.Permute((2, 1, 3))(x)

    x = Layers.Conv2D(10, (5, 1), activation='relu', padding='same')(x)
    x = Layers.BatchNormalization()(x)
    x = Layers.Conv2D(1, (5, 1), activation='relu', padding='same')(x)
    x = Layers.BatchNormalization()(x)

    # x = Reshape((125, 80)) (x)
    # keras.backend.squeeze(x, axis)
    x = Layers.Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim')(x)

    x = Layers.Bidirectional(Layers.LSTM(64, return_sequences=True))(x)  # [b_s, seq_len, vec_dim]
    x = Layers.Bidirectional(Layers.LSTM(64))(x)

    x = Layers.Dense(64, activation='relu')(x)
    x = Layers.Dense(32, activation='relu')(x)

    output = Layers.Dense(nCategories, activation='softmax')(x)

    model = Model(inputs=[inputs], outputs=[output])

    return model

def my_RNNSpeechModel(nCategories, samplingrate=16000, inputLength=16000):
    model = Sequential([ 
        # TODO: Input init
        # In Sequential probably should not be used
        # Layers.InputLayer(input_tensor=Layers.Input((inputLength,)),),
        # Layers.Input((inputLength,)),
        Layers.Reshape((1, -1)),
        # TODO: Melspectogram
        Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, inputLength),
                       padding='same', sr=samplingrate, n_mels=80,
                       fmin=40.0, fmax=samplingrate / 2, power_melgram=1.0,
                       return_decibel_melgram=True, trainable_fb=False,
                       trainable_kernel=False,
                       name='mel_stft'),
        # TODO: Normalization2D
        Normalization2D(int_axis=0),

        # Layers
        Layers.Permute((2, 1, 3)),

        Layers.Conv2D(10, (5, 1), activation='relu', padding='same'),
        Layers.BatchNormalization(),
        Layers.Conv2D(1, (5, 1), activation='relu', padding='same'),
        Layers.BatchNormalization(),

        Layers.Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim'),

        Layers.Bidirectional(Layers.LSTM(64, return_sequences=True)),
        Layers.Bidirectional(Layers.LSTM(64)),

        Layers.Dense(64, activation='relu'),
        Layers.Dense(32, activation='relu'),

        # Output
        Layers.Dense(nCategories, activation='softmax')
    ])

    return model

def my2_SimpleRNNSpeechModel(nCategories, samplingrate=16000, inputLength=16000):
    model = Sequential()
    model.add(Layers.Reshape((1, -1)))
    model.add(Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, inputLength),
                padding='same', sr=samplingrate, n_mels=80,
                fmin=40.0, fmax=samplingrate / 2, power_melgram=1.0,
                return_decibel_melgram=True, trainable_fb=False,
                trainable_kernel=False,
                name='mel_stft'))
    model.add(Normalization2D(int_axis=0))
    model.add(Layers.Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim'))
    model.add(Layers.SimpleRNN(64, return_sequences=True))
    model.add(Layers.Flatten())
    model.add(Layers.Dense(64, activation='relu'))
    model.add(Layers.Dense(32, activation='relu'))
    model.add(Layers.Dense(nCategories, activation='softmax'))
    return model

def my_SimpleLSTMSpeechModel(nCategories, samplingrate=16000, inputLength=16000):
    model = Sequential()
    model.add(Layers.Reshape((1, -1)))
    model.add(Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, inputLength),
                padding='same', sr=samplingrate, n_mels=80,
                fmin=40.0, fmax=samplingrate / 2, power_melgram=1.0,
                return_decibel_melgram=True, trainable_fb=False,
                trainable_kernel=False,
                name='mel_stft'))
    model.add(Normalization2D(int_axis=0))
    model.add(Layers.Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim'))
    model.add(Layers.LSTM(64, return_sequences=True))
    model.add(Layers.Flatten())
    model.add(Layers.Dense(64, activation='relu'))
    model.add(Layers.Dense(32, activation='relu'))
    model.add(Layers.Dense(nCategories, activation='softmax'))
    return model

def my_SimpleStackedLSTMSpeechModel(nCategories, samplingrate=16000, inputLength=16000):
    model = Sequential()
    model.add(Layers.Reshape((1, -1)))
    model.add(Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, inputLength),
                padding='same', sr=samplingrate, n_mels=80,
                fmin=40.0, fmax=samplingrate / 2, power_melgram=1.0,
                return_decibel_melgram=True, trainable_fb=False,
                trainable_kernel=False,
                name='mel_stft'))
    model.add(Normalization2D(int_axis=0))
    model.add(Layers.Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim'))
    model.add(Layers.LSTM(64, return_sequences=True))
    model.add(Layers.LSTM(64, return_sequences=True))
    model.add(Layers.LSTM(64, return_sequences=True))
    model.add(Layers.LSTM(64, return_sequences=True))
    model.add(Layers.Flatten())
    model.add(Layers.Dense(64, activation='relu'))
    model.add(Layers.Dense(32, activation='relu'))
    model.add(Layers.Dense(nCategories, activation='softmax'))
    return model

def my_FilteredStackedLSTMSpeechModel(nCategories, sampligrate=16000, inputLength=16000):
    model = Sequential()
    model.add(Layers.Reshape((1, -1)))
    model.add(Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, inputLength),
                padding='same', sr=sampligrate, n_mels=80,
                fmin=40.0, fmax=sampligrate / 2, power_melgram=1.0,
                return_decibel_melgram=True, trainable_fb=False,
                trainable_kernel=False,
                name='mel_stft'))
    model.add(Normalization2D(int_axis=0))

    model.add(Layers.Conv2D(32, (5, 1), activation='relu', padding='same'))
    model.add(Layers.BatchNormalization())
    model.add(Layers.ReLU())
    model.add(Layers.Dropout(0.2))

    model.add(Layers.Conv2D(16, (5, 1), activation='relu', padding='same'))
    model.add(Layers.BatchNormalization())
    model.add(Layers.ReLU())
    model.add(Layers.Dropout(0.2))

    model.add(Layers.Conv2D(1, (5, 1), activation='relu', padding='same'))
    model.add(Layers.Conv2D(filters=1, kernel_size=(5, 1), activation='relu'))
    model.add(Layers.BatchNormalization())

    model.add(Layers.Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim'))
    model.add(Layers.LSTM(64, return_sequences=True))
    model.add(Layers.LSTM(64, return_sequences=True))
    model.add(Layers.LSTM(64, return_sequences=True))
    model.add(Layers.LSTM(64, return_sequences=True))
    model.add(Layers.Flatten())
    model.add(Layers.Dense(64, activation='relu'))
    model.add(Layers.Dense(32, activation='relu'))
    model.add(Layers.Dense(nCategories, activation='softmax'))

    return model
def AttRNNSpeechModel(nCategories, samplingrate=16000,
                      inputLength=16000, rnn_func=Layers.LSTM):
    # simple LSTM
    sr = samplingrate
    iLen = inputLength

    inputs = Layers.Input((inputLength,), name='input')

    x = Layers.Reshape((1, -1))(inputs)

    m = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, iLen),
                       padding='same', sr=sr, n_mels=80,
                       fmin=40.0, fmax=sr / 2, power_melgram=1.0,
                       return_decibel_melgram=True, trainable_fb=False,
                       trainable_kernel=False,
                       name='mel_stft')
    m.trainable = False

    x = m(x)

    x = Normalization2D(int_axis=0, name='mel_stft_norm')(x)

    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    x = Layers.Permute((2, 1, 3))(x)

    x = Layers.Conv2D(10, (5, 1), activation='relu', padding='same')(x)
    x = Layers.BatchNormalization()(x)
    x = Layers.ReLU()(x)
    x = Layers.Conv2D(1, (5, 1), activation='relu', padding='same')(x)
    x = Layers.BatchNormalization()(x)

    # x = Reshape((125, 80)) (x)
    # keras.backend.squeeze(x, axis)
    x = Layers.Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim')(x)

    x = Layers.Bidirectional(rnn_func(64, return_sequences=True)
                        )(x)  # [b_s, seq_len, vec_dim]
    x = Layers.Bidirectional(rnn_func(64, return_sequences=True)
                        )(x)  # [b_s, seq_len, vec_dim]

    xFirst = Layers.Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
    query = Layers.Dense(128)(xFirst)

    # dot product attention
    attScores = Layers.Dot(axes=[1, 2])([query, x])
    attScores = Layers.Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

    # rescale sequence
    attVector = Layers.Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

    x = Layers.Dense(64, activation='relu')(attVector)
    x = Layers.Dense(32)(x)

    output = Layers.Dense(nCategories, activation='softmax', name='output')(x)

    model = Model(inputs=[inputs], outputs=[output])

    return model
