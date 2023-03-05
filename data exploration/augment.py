import librosa


def augment(data, sr, name):

    result = [(data, name)]

    [result.append((librosa.effects.time_stretch(y=data, rate=x), name + "stretched" + str(x)))
     for x in [0.95, 0.96, 0.97, 0.98, 0.99, 1.01, 1.02, 1.03, 1.04, 1.05]]
    [result.append((librosa.effects.pitch_shift(y=data, sr=sr, n_steps=x), name + "pitched" + str(x)))
     for x in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]]

    return result
