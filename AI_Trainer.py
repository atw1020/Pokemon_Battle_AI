'''

author: arthur wesley

'''

import pokereplay
import parser
import training_data_generator
import random as r

import sys
import os
import random as r
import time as t
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tracemalloc

import numpy as np
import tensorflow as tf
from tensorflow import keras

tracemalloc.start()


def turns_training_data(data):
    '''

    returns a list of turns for the given replay and format

    :param replay_number: replay number to get the data from
    :return: list of turns with all the sub categories concatanated
    '''

    if len(data[1]) == 0:
        return []

    turn_item_sizes = [0] * len(data[1][0])

    for i in range(len(data[1][0])):
        turn_item_sizes[i] = len(data[1][0][i])

    output = [[0] * sum(turn_item_sizes)] * len(data[1])

    for i in range(len(data[1])):

        for j in range(len(data[1][i])):
            output[i][sum(turn_item_sizes[:j]):sum(turn_item_sizes[:j + 1])] = data[1][i][j]

    return output


def collect_turns_for_PCA(format):
    '''

    Collects all of the turns data saved in the replays into one list of turns for PCA

    :param format:
    :return:
    '''

    files = os.listdir("Binary Replays")

    data = []

    # shuffle the files
    # r.shuffle(files)

    i = 0

    for file in files:
        if format in file:
            replay_data = pokereplay.read_replay_file(file)
            data = data + turns_training_data(replay_data)

            if len(data) > 100000:
                break

    return data


def collect_teams_for_PCA(format):
    '''

    collects teams from replays in order to perform principle component analysis on the teams

    :param format: the format to collec the teams from
    :return: data to perform PCA on
    '''

    files = os.listdir("Binary Replays")

    r.shuffle(files)

    data = []

    for file in files:
        if format in file:
            print(file)
            battle_data = pokereplay.read_replay_file("Binary Replays/" + file)

            data = data + battle_data[0]  # battle data [0] refers to the teams data

    return data


def init_model(turn_size, choice_size, team_size, layers=10, breadth=64):
    '''

    creates a keras model

    :param turn_size: the size of each turn
    :param choice_size: the size of each choice
    :param team_size: the size of each team
    :return:
    '''

    turns_layers = layers
    choice_layers = layers
    teams_layers = layers
    end_layers = 6 * layers

    turns_dense_size = breadth
    choices_dense_size = breadth
    teams_dense_size = breadth
    end_dense_size = breadth

    turns_dropout = 0.3
    choices_dropout = 0.3
    teams_dropout = 0.3
    end_dropout = 0.3

    # make the input layers

    RNN_input = keras.layers.Input(shape=(None, turn_size))
    choice_input = keras.layers.Input(shape=(choice_size,))
    team_input = keras.layers.Input(shape=(team_size,))

    # first create the RNN
    RNN = tf.compat.v1.keras.layers.SimpleRNN(turns_dense_size, return_sequences=True)(RNN_input)
    RNN = keras.layers.Dropout(turns_dropout)(RNN)

    for i in range(turns_layers - 2):

        RNN = tf.compat.v1.keras.layers.SimpleRNN(turns_dense_size, return_sequences=True)(RNN)
        RNN = keras.layers.Dropout(turns_dropout)(RNN)

    RNN = tf.compat.v1.keras.layers.SimpleRNN(turns_dense_size, return_sequences=False)(RNN)
    RNN = keras.layers.Dropout(turns_dropout)(RNN)

    # todo: find out if there is any way to manually connect my GPU to tensorflow or if my GPU
    #   just dosen't have the XLA (Accelerated Linear Algebra) Instructions

    # next the choice NN

    CNN = keras.layers.Dense(choices_dense_size, activation="relu")(choice_input)
    CNN = keras.layers.Dropout(choices_dropout)(CNN)

    for i in range(choice_layers - 1):

        CNN = keras.layers.Dense(choices_dense_size, activation="relu")(CNN)
        CNN = keras.layers.Dropout(choices_dropout)(CNN)

    # and the team NN

    TNN = keras.layers.Dense(teams_dense_size, activation="relu")(team_input)
    TNN = keras.layers.Dropout(teams_dropout)(TNN)

    for i in range(teams_layers - 1):

        TNN = keras.layers.Dense(teams_dense_size, activation="relu")(TNN)
        TNN = keras.layers.Dropout(teams_dropout)(TNN)

    # merge the layers

    Merge = keras.layers.concatenate([RNN, CNN, TNN])

    # now go through some more dense layers
    X = keras.layers.Dense(end_dense_size, activation="relu")(Merge)
    X = keras.layers.Dropout(end_dropout)(X)

    for i in range(end_layers - 1):

        X = keras.layers.Dense(end_dense_size, activation="relu")(X)
        X = keras.layers.Dropout(end_dropout)(X)

    out = keras.layers.Dense(1, activation="sigmoid")(X)

    # create the model

    model = keras.models.Model(inputs=[RNN_input, choice_input, team_input], outputs=out)

    # model.summary()

    return model


def get_replay_input_data(path, team_PCA, turns_PCA):
    '''

    takes in a replay number and returns the X and y for that data

    :param path: path to the replay file
    :param team_PCA: the PCA object used to make do the team PCA
    :param turns_PCA: the PCA object used to make the turns PCA
    :return: tuple (X (data about the battle), y (the winner of the battle)
    '''

    t5 = t.time()

    # first load the replay data

    data = pokereplay.read_replay_file(path)

    if len(data[1]) == 0:
        return None

    t6 = t.time()

    # get the team data
    first_team = np.array([data[0][0]] * len(data[1]))
    second_team = np.array([data[0][1]] * len(data[1]))

    # perform PCA

    first_team = team_PCA.transform(first_team)
    second_team = team_PCA.transform(second_team)

    # we want to make it so that the user's team appear's first in the matchup and
    # the opponents team appears second so we re-arrange the teams
    first_player_teams = np.concatenate((first_team, second_team), axis=1)
    second_player_teams = np.concatenate((second_team, first_team), axis=1)

    first_player_teams = list(first_player_teams)
    second_player_teams = list(second_player_teams)

    teams_data = first_player_teams + second_player_teams

    t7 = t.time()

    # get the turns data
    turns_data = turns_training_data(data)

    # there are two kinds of data we get from the turns
    # the choices that were made on each turn and also the
    # data about the battle thus far

    # first we are going to extract the choices data since getting the turns data will be destructive

    # to get the choices we need two lists: one for player 1 and one for player 2

    player_1_choices = [turn[(len(turn) // 2):] for turn in turns_data]
    player_2_choices = [turn[:(len(turn) // 2)] for turn in turns_data]

    # the thing is that the players position in the turns data isn't determined by which player number they are
    # it's determined by which player goes first so we need to go through all the turns and swap the
    # player 1 and player 2 choices if the first player doesn't attack first

    for i in range(len(turns_data)):

        # the last item in the choices tells us which player it is, zero corresponds to player 1 and 1 corresponds
        # to player 2

        # so we can check the last item to see if it is zero and swap based on that
        if player_1_choices[i][-1] == 1:
            temp = player_1_choices[i]
            player_1_choices[i] = player_2_choices[i]
            player_2_choices[i] = temp

    player_1_choices = list(np.array(player_1_choices))
    player_2_choices = list(np.array(player_2_choices))

    choices_data = player_1_choices + player_2_choices

    # now get the data about the turns

    t8 = t.time()

    # start by making a Numpy array of the turns data and performing PCA on it

    turns_data = np.array(turns_data)
    turns_data = turns_PCA.transform(turns_data)

    # convert the outermost layer of the numpy array back to a python list

    turns_data = list(turns_data)

    # we go through the turns in reverse since the desired data is replaced as we go through the list
    for i in reversed(range(len(turns_data))):
        turns_data[i] = np.array(turns_data[:i])

    # now double all the turns data so we can send one copy to each example

    turns_data = turns_data + turns_data

    t9 = t.time()

    # finally we need to look at who won the game

    # the winner can be found in the last item of the data list

    winner = data[2][0]

    # now based on the winner we need to set the output data

    winner_data = [0] * 2 * len(data[1])

    for i in range(len(data[1])):
        # for the first player a win is denoted by the winning player being "zero"
        # and a loss is denoted by the winning player being "one"
        # we can convert the winner player ID to this by subtracting one from the number

        winner_data[i] = 1 - winner

        # we can then set the opposite Y in the output space
        winner_data[len(data[1]) + i] = winner

    winner_data = np.array(winner_data)

    # now amalgamate all this data together to create the training set
    # the data has to be in the format RNN, choice, team

    # the last option is the Y value: the winner

    t10 = t.time()

    times = (t5, t6, t7, t8, t9, t10)

    return turns_data, choices_data, teams_data, winner_data, times


def get_matrix_dimensions(format, Max=-1):
    '''

    finds the dimensions of training data by reading all the replays of a given format

    :param format: the format to get the dimensions for
    :return: empty numpy matrix containing
    '''

    f = open("PCA_teams.bin", "rb")
    PCA_teams = pickle.load(f)
    f.close()

    f = open("PCA_turns.bin", "rb")
    PCA_turns = pickle.load(f)
    f.close()

    team_dimension = PCA_teams.components_.shape
    turns_dimension = PCA_turns.components_.shape

    turn_counts = []

    # count the total number of turns in the replays by going through all the replays and
    # adding up the turns

    replays = os.listdir("Binary Replays/")

    for replay in replays[:Max]:
        if format in replay:

            print("reading", replay)

            data = pokereplay.read_replay_file("Binary Replays/" + replay)
            turn_counts.append(len(data[1]))

            try:
                choice_items = sum(map(len, data[1][0])) // 2
            except IndexError:
                continue

    total_turns = sum(turn_counts)

    # getting the turns data is tricky because the size of the turns data varies from example to
    # example
    turns_data = [0] * total_turns

    index = 0

    for turn_count in turn_counts:
        for turn_no in range(turn_count):
            turns_data[index + turn_no] = np.empty((turn_no, turns_dimension[0],))

        index += turn_count

    # finalize the data and return it

    choice_data = [np.empty((choice_items,))] * 2 * total_turns
    teams_data = [np.empty((team_dimension[0] * 2,))] * total_turns
    winner_data = np.empty((total_turns * 2,))

    return turns_data, choice_data, teams_data, winner_data


def get_training_data(format):
    '''

    loads all replays in the Binary Replays file in a given format and returns training data X and Y

    :param format: format to get the training data from
    :return: Tuple: (X, y) Training data set
    '''

    Max = 1000

    t0 = t.time()

    # start by loading the PCA models from their binary files

    f = open("PCA_teams.bin", "rb")
    PCA_teams = pickle.load(f)
    f.close()

    f = open("PCA_turns.bin", "rb")
    PCA_turns = pickle.load(f)
    f.close()

    # initialize all the numpy arrays

    turns_data, choices_data, teams_data, winner_data = get_matrix_dimensions("gen7ou", Max)

    t2 = 0
    t3 = 0
    t4 = 0
    t5 = 0
    t6 = 0
    t7 = 0
    t8 = 0
    t9 = 0
    t10 = 0

    # go through the replays

    files = os.listdir("Binary Replays/")

    turns_index = 0

    for file in files[:Max]:
        if format in file:

            # get the new data

            print("collecting", file)

            t2 += t.time()

            try:
                new_turns, new_choices, new_teams, new_winner, times = \
                    get_replay_input_data("Binary Replays/" + file, PCA_teams, PCA_turns)
            except TypeError as error:
                # get replay data will return None if there are no turns in the replay
                # if this is the case, just continue
                t3 += t.time()
                t4 += t.time()
                continue

            size_of_new = len(new_turns)

            t3 += t.time()

            # assign the data

            turns_data[turns_index:turns_index + size_of_new] = new_turns
            choices_data[turns_index:turns_index + size_of_new] = new_choices
            teams_data[turns_index:turns_index + size_of_new] = new_teams
            winner_data[turns_index:turns_index + size_of_new] = new_winner

            t4 += t.time()

            (t5_inc, t6_inc, t7_inc, t8_inc, t9_inc, t10_inc) = times

            t5 += t5_inc
            t6 += t6_inc
            t7 += t7_inc
            t8 += t8_inc
            t9 += t9_inc
            t10 += t10_inc

            # objgraph.show_growth()

    # we now need to pad all the turns of the battle so we use the keras preporcessing padding function
    # turns_data = pad_sequences(turns_data, padding="pre")

    # now put everything together

    X = [turns_data, choices_data, teams_data]
    y = winner_data

    # print some summaries

    t1 = t.time()

    print("importing data took", t1 - t0)

    print("delay breakdown:")
    print("formatting took", t3 - t2)
    print("accumulating took", t4 - t3)

    print("formatting breakdown:")
    print("loading the data took", t6 - t5)
    print("formatting the team data took", t7 - t6)
    print("formatting the choices data took", t8 - t7)
    print("formatting the turns data took", t9 - t8)
    print("formatting the winner data took", t10 - t9)

    return X, y


def data_generator(X, y, RNN_dim):
    '''

    generator that yields Training examples from the training set

    :param X: Data
    :param y: labels
    :return: yields 1 training example at a time
    '''

    [turns_data, choices_data, teams_data] = X

    while True:
        # infinitely go through all the items in these lists
        for i in range(len(turns_data)):

            # todo: sort all the turns data by numbers of turns to speed up the algorithm to use
            #       vectorized CPU functions

            if turns_data[i].shape == (0,) or turns_data[i].shape == (1, 0, RNN_dim):
                continue
                turns_data[i] = np.empty((1, 0, RNN_dim))

            else:
                if len(turns_data[i].shape) == 2:
                    turns_data[i] = np.reshape(turns_data[i], (1,) + turns_data[i].shape)

            if len(choices_data[i].shape) == 1:
                choices_data[i] = np.reshape(choices_data[i], (1,) + choices_data[i].shape)

            if len(teams_data[i].shape) == 1:
                teams_data[i] = np.reshape(teams_data[i], (1,) + teams_data[i].shape)

            X_out = [turns_data[i], choices_data[i], teams_data[i]]
            y_out = np.array([y[i]])

            # print()

            # print(turns_data[i].shape)
            # print(choices_data[i].shape)
            # print(teams_data[i].shape)

            # print(y_out)

            yield X_out, y_out


def shuffle_data(X, y):
    '''

    shuffles input data

    :param X: Training data
    :param y: Labels
    :return: Shuffled Tuple of X and Y
    '''

    indices = list(range(len(X[0])))

    r.shuffle(indices)

    for i in range(len(X)):
        X[i] = [X[i][j] for j in indices]

    y = [y[j] for j in indices]

    return X, y


def make_model(format):
    '''

    creates a model for a given format

    :param format: the fromat to create the model for
    :return: trained model
    '''

    # first we need to collect the data

    X, y = get_training_data(format)

    print("shuffling completed")

    # now we can initialize the model

    [turns_data, choices_data, teams_data] = X

    turn_size = turns_data[-1].shape[1]
    choices_size = choices_data[-1].shape[0]
    teams_size = teams_data[-1].shape[0]
    total_turns = len(turns_data)

    model = init_model(turn_size, choices_size, teams_size)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])

    model.fit_generator(data_generator(X, y, turn_size), steps_per_epoch=total_turns, epochs=1, verbose=1) # ,
                        # nb_workers=5, max_queue_size=50)

    return model


def choose_random_params(num_params):

    params = []

    for i in range(num_params):
        rate = 0.1 ** r.uniform(1, 4)
        batch_size = 2 ** r.randint(5, 9)

        params.append((rate, batch_size))

    return params


def make_model_generator(format, depth, bredth, epochs, replay_limit=30):
    '''

    creates and trains a model using the training data generator

    :param format: format to generate the model for
    :return: trained model
    '''

    # replay_limit = 10000

    cb = training_data_generator.CompletionCallback(0.999)

    np.random.seed(0)

    rate = 0.003
    batch_size = 128

    keras.backend.clear_session()

    generator = training_data_generator.DataGenerator(format, batch_size=batch_size, replay_limit=replay_limit)

    # print("learning rate:", rate)
    # print("batch size:", batch_size)
    # print("hidden layer sizes", hidden_layer_sizes)

    adam_clip = keras.optimizers.Adam(lr=rate)

    model = init_model(generator.turns_reduced_dimension, generator.choices_reduced_dimension,
                       generator.team_reduced_dimension, depth, bredth)

    model.compile(optimizer=adam_clip, loss="binary_crossentropy", metrics=["acc"])
    model.summary()

    model.fit_generator(generator, epochs=epochs, verbose=1, max_queue_size=50, callbacks=[cb])

    # print("learning rate:", rate)
    # print("batch size:", batch_size)

    return model


def print_model_weights(model):
    '''

    prints the weights of a given input model

    :param model: Keras model
    :return: None
    '''

    for layer in model.layers:
        print(layer.name)
        print(layer.get_weights())


def print_learning_curves(tier):
    '''



    :return:
    '''

    replay_limits = [100 * 2 ** i for i in range(7)]

    for replay_limit in replay_limits:

        model = make_model_generator(tier, replay_limit)
        model.save("Models/" + tier + ".h5")

        del model

        acc = training_data_generator.get_validation_accuracy(tier)

        print("training accuracy:", acc)


def hyperparameter_search():
    f = open("PCA_turns.bin", "rb")
    turns_PCA = pickle.load(f)
    f.close()

    path = ("Binary Replays/961624567_gen7ou.rply")

    data = pokereplay.read_replay_file(path)

    # get the turns data
    turns_data = turns_training_data(data)

    turns_data = np.array(turns_data)
    turns_data = turns_PCA.transform(turns_data)

    # convert the outermost layer of the numpy array back to a python list

    turns_data = list(turns_data)

    # we go through the turns in reverse since the desired data is replaced as we go through the list
    for i in reversed(range(len(turns_data))):
        turns_data[i] = np.array(turns_data[:i])

    # print(turns_data)
    # and convert back to a numpy array
    turns_data = np.array(turns_data)
    print(turns_data.shape)


def main():
    # training_data_generator.preallocate_training_data("gen7ou")
    #'''
    # print_learning_curves("gen7ou")
    #'''

    print("Control run")
    model = make_model_generator("gen7ou", 10, 64, 10)

    print("Increase Depth")
    model = make_model_generator("gen7ou", 100, 64, 10)

    print("Increase Breadth")
    model = make_model_generator("gen7ou", 10, 640, 10)

    print("Increase Epochs")
    model = make_model_generator("gen7ou", 10, 64, 100)

    # model.save("Models/gen7ou.h5")

    del model
    '''
    acc = training_data_generator.get_validation_accuracy("gen7ou")

    print(acc)
    #'''

    # generator = training_data_generator.DataGenerator()


if __name__ == "__main__":
    main()
