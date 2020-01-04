'''

author: Arthur Wesley

'''

import itertools
import math
import os
import pickle
import random as r
import time as t
from multiprocessing import Pool
import threading
import objgraph

import numpy as np
import pokereplay
from sklearn.decomposition import PCA
from tensorflow import keras
from numba import cuda

import Parser
import Replay_Web_Scrapper

sleeptimes = [0] * 9
sleeptimes[5] = 0

class CompletionCallback(keras.callbacks.Callback):

    def __init__(self, limit=0.999):

        self.limit = limit

    def on_epoch_end(self, epoch, logs={}):

        if logs.get("acc") > self.limit:
            print("\nreached 99% accuracy, canceling training")
            self.model.stop_training = True


class DataGenerator(keras.utils.Sequence):

    def __init__(self, tier, batch_size=32, replay_limit=10000):

        self.do_PCA = True

        self.replay_limit = replay_limit

        self.batch_size = batch_size
        self.tier = tier

        self.replay_nos = os.listdir("Binary Replays")

        i = 0

        while i < len(self.replay_nos):
            if self.tier not in self.replay_nos[i]:
                del self.replay_nos[i]
            else:
                i += 1

        self.PCA_prep()
        self.init_batch_sizes()

        self.t0 = 0
        self.t1 = 0
        self.t2 = 0
        self.t3 = 0
        self.t4 = 0
        self.t5 = 0
        self.t6 = 0
        self.t7 = 0
        self.t8 = t.time()

        # self.epoch_start_time = t.time()

        print("testing", "t" + str(sleeptimes.index(0)))

        self.lock = threading.Lock()

    def on_epoch_end(self):
        '''

        callback that activates upon epoch end

        :return: None
        '''

        # self.epoch_end_time = t.time()

        # self.print_time_summary()
        self.shuffle_training_data()

        # self.epoch_start_time = t.time()

    def __len__(self):
        '''

        returns the number of batches per epoch

        :return: the length of the number of batches
        '''

        return len(self.batches)

    def __getitem__(self, item):
        '''

        gives a batch of data to the algorithm

        :param item: index of the batch to get
        :return: batch of data for backprop
        '''

        self.t0 += t.time()

        t.sleep(sleeptimes[0])

        batch_replays = self.batches[item]
        turn = self.turn_numbers[item]

        # start by initializing the batch

        batch = [np.empty((len(batch_replays) * 2, max(1, turn), self.turns_dimension)),
                 np.empty((len(batch_replays) * 2, self.choices_dimension)),
                 np.empty((len(batch_replays) * 2, self.team_dimension))]

        # batch = [
        #    [[[0] * self.turns_dimension for i in range(max(1, turns))] for j in range(len(batch_replays) * 2)],
        #    [[0] * self.choices_dimension for i in range(len(batch_replays) * 2)],
        #    [[0] * self.team_dimension for i in range(len(batch_replays) * 2)]
        # ]

        self.t1 += t.time()

        t.sleep(sleeptimes[1])

        # and the labels

        labels = np.empty(len(batch_replays) * 2)

        # get training data from the batch
        index = 0
        #'''
        process_replay = lambda replay_no: process_data(turn, replay_no, self.turns_dimension,
                                                        self.choices_dimension, self.team_dimension)

        args = []

        for batch_replay in batch_replays:
            args.append((turn, batch_replay, self.turns_dimension, self.choices_dimension, self.team_dimension))

        x = [(1,), (2,), (3,)]
        '''
        with Pool(8) as p:
            results = p.starmap(process_data, args)

        for result in results:
            turns_data, choices_data, teams_data, labels_data = result

            batch[0][index:index + 2] = turns_data
            batch[1][index:index + 2] = choices_data
            batch[2][index:index + 2] = teams_data

            labels[index:index + 2] = labels_data

            index += 2
        '''
        for replay in batch_replays:
            turns_data, choices_data, teams_data, labels_data = process_replay(replay)

            batch[0][index:index + 2] = turns_data
            batch[1][index:index + 2] = choices_data
            batch[2][index:index + 2] = teams_data

            labels[index:index + 2] = labels_data

            index += 2
        #'''

        t.sleep(sleeptimes[6])

        self.t6 += t.time()

        # batch[0] = np.array(batch[0])

        # do PCA
        if self.do_PCA:

            if turn != 0:
                # player_1_turns = self.PCA_turns.transform(player_1_turns)
                # player_2_turns = self.PCA_turns.transform(player_2_turns)

                # x = len(batch[0])
                # y = len(batch[0][0])
                # z = len(batch[0][0][0])

                x, y, z = batch[0].shape

                batch[0] = batch[0].reshape((x * y, z))
                batch[0] = self.PCA_turns.transform(batch[0])
                batch[0] = batch[0].reshape((x, y, self.turns_reduced_dimension))
            else:
                batch[0] = np.zeros((len(batch_replays) * 2, 1, self.turns_reduced_dimension))

            # player_1_choices = self.PCA_choices.transform(player_1_choices)
            # player_2_choices = self.PCA_choices.transform(player_2_choices)
            batch[1] = self.PCA_choices.transform(batch[1])

            # player_1_teams = self.PCA_teams.transform(player_1_teams)
            # player_2_teams = self.PCA_teams.transform(player_2_teams)
            batch[2] = self.PCA_teams.transform(batch[2])

        t.sleep(sleeptimes[7])
        self.t7 += t.time()

        # self.print_time_summary()

        t.sleep(sleeptimes[8])
        self.t8 += t.time()

        return batch, labels

    def print_time_summary(self):
        '''

        prints a summary of what takes the longest in the importing process

        :return: None
        '''

        print("\ntime summary")
        print("computing gradiants", self.t0 - self.t8)
        print("initializing data", self.t1 - self.t0)
        print("importing the data", self.t3 - self.t2)
        print("formatting the data", self.t4 - self.t3)
        print("putting data into batches", self.t5 - self.t4)
        print("PCA", self.t7 - self.t6)

        print("epoch took", self.epoch_end_time - self.epoch_start_time)
        print("with", "t" + str(sleeptimes.index(0)), "speed up")

    def init_batch_sizes(self):
        '''

        Initializes self.batch_sizes

        :return: None
        '''

        # first thing we need to do is get the number of turns in each replay
        # ---------------------------------------------------------------------------------

        replays = os.listdir("Binary Replays")

        turn_sizes = []
        counter = 0

        t0 = t.time()

        for replay in replays:
            if self.tier in replay:

                counter += 1

                if counter >= self.replay_limit:
                    break

                data = pokereplay.read_replay_file("Binary Replays/" + replay)

                turn_sizes.append((replay, len(data[1])))

        num_replays = len(turn_sizes)

        t1 = t.time()

        print("on average it took", (t1 - t0) / num_replays, "seconds to parse a replay")

        # now we need to find the shapes each team and each turn will have after they go through
        # Principle Component Analysis
        # ---------------------------------------------------------------------------------

        f = open("PCA files/" + self.tier + "_turns_PCA.bin", "rb")
        self.PCA_turns = pickle.load(f)
        f.close()

        f = open("PCA files/" + self.tier + "_choices_PCA.bin", "rb")
        self.PCA_choices = pickle.load(f)
        f.close()

        f = open("PCA files/" + self.tier + "_teams_PCA.bin", "rb")
        self.PCA_teams = pickle.load(f)
        f.close()

        # set the dimensions of the arrays
        # ---------------------------------------------------------------------------------

        self.team_dimension = self.PCA_teams.components_.shape[1]
        self.turns_dimension = self.PCA_turns.components_.shape[1]
        self.choices_dimension = self.PCA_choices.components_.shape[1]

        self.team_reduced_dimension = self.PCA_teams.components_.shape[0]
        self.turns_reduced_dimension = self.PCA_turns.components_.shape[0]
        self.choices_reduced_dimension = self.PCA_choices.components_.shape[0]

        '''
        if self.do_PCA:
            self.team_dimension = self.PCA_teams.components_.shape[0]
            self.turns_dimension = self.PCA_turns.components_.shape[0]
            self.choices_dimension = self.PCA_choices.components_.shape[0]
        else:
            self.team_dimension = self.PCA_teams.components_.shape[1]
            self.turns_dimension = self.PCA_turns.components_.shape[1]
            self.choices_dimension = self.PCA_choices.components_.shape[1]
            
        '''

        # ---------------------------------------------------------------------------------
        # in order to quickly reference data we store the replay numbers that contain the information for each batch
        # and reference these lists when we get data

        # to do this each batch is stored as a tuple of the replay numbers in that batch.

        self.turn_instances = [[] for j in range(max([turn_sizes[i][1] for i in range(len(turn_sizes))]))]

        for size in turn_sizes:

            for i in range(size[1]):
                self.turn_instances[i].append(size[0])

        # finally shuffle all the items in the turn sizes

        self.shuffle_training_data()

        # ---------------------------------------------------------------------------------
        # now we have all the dimensions we need so we can actually initialize the batches and labels

        self.batches = []
        self.turn_numbers = []

        for turn_no, replay_nos in enumerate(self.turn_instances):
            # for each instance of a turn we get two training examples so we only need half as many replays
            # per training example. this means we have to cut the batch size in half

            batch_size = int(self.batch_size / 2)

            self.batches = self.batches + [replay_nos[inc:inc + batch_size]
                                           for inc in range(0, len(replay_nos), batch_size)]

            self.turn_numbers = self.turn_numbers + [turn_no] * (math.ceil(len(replay_nos) / batch_size))

        assert len(self.batches) == len(self.turn_numbers)

    def init_batches(self):
        '''

        Allocates the memory for the batches

        :return: None, sets self.batches
        '''

        # first thing we need to do is get the number of turns in each replay
        # ---------------------------------------------------------------------------------

        replays = os.listdir("Binary Replays")

        turn_sizes = []
        counter = 0

        t0 = t.time()

        for replay in replays:
            if self.tier in replay:

                counter += 1

                if counter >= self.replay_limit:
                    break

                data = pokereplay.read_replay_file("Binary Replays/" + replay)

                turn_sizes.append(len(data[1]))

        num_replays = len(turn_sizes)

        t1 = t.time()

        print("on average it took", (t1 - t0) / num_replays, "seconds to parse a replay")

        # now we need to find the shapes each team and each turn will have after they go through
        # Principle Component Analysis
        # ---------------------------------------------------------------------------------

        f = open("PCA files/" + self.tier + "_turns_PCA.bin", "rb")
        PCA_turns = pickle.load(f)
        f.close()

        f = open("PCA files/" + self.tier + "_choices_PCA.bin", "rb")
        PCA_choices = pickle.load(f)
        f.close()

        f = open("PCA files/" + self.tier + "_teams_PCA.bin", "rb")
        PCA_teams = pickle.load(f)
        f.close()

        # set the dimensions of the arrays
        # ---------------------------------------------------------------------------------

        if self.do_PCA:
            self.team_dimension = PCA_teams.components_.shape[0]
            self.turns_dimension = PCA_turns.components_.shape[0]
            self.choices_dimension = PCA_choices.components_.shape[0]
        else:
            self.team_dimension = PCA_teams.components_.shape[1]
            self.turns_dimension = PCA_turns.components_.shape[1]
            self.choices_dimension = PCA_choices.components_.shape[1]

        # ---------------------------------------------------------------------------------

        # the data is stored in the following format: a list of batches which have up to self.batch_size
        # training examples in them

        # each batch contains data for which the number of turns is constant. ie: all turns in the batch
        # will be turn 27

        # in order to determine the number and shape of these batches we need to know the number of instances
        # of each turn number (ie turn 27) in the data set

        self.turn_instances = [0] * max(turn_sizes)

        for size in turn_sizes:

            for i in range(size):
                self.turn_instances[i] += 1

        # ---------------------------------------------------------------------------------
        # now we have all the dimensions we need so we can actually initialize the batches and labels

        self.batch_sizes = []

        for turn_count, instance_count in enumerate(self.turn_instances):

            # for each instance of a turn we get two training examples so we double the instance count
            # to account for this
            instance_count *= 2

            if instance_count % self.batch_size == 0:
                self.batch_sizes = self.batch_sizes + [[self.batch_size, turn_count]] * \
                                   (instance_count // self.batch_size)
            else:
                self.batch_sizes = self.batch_sizes + [[self.batch_size, turn_count]] * \
                                   (instance_count // self.batch_size) + \
                                   [(instance_count % self.batch_size, turn_count)]

        i = 0
        '''
        for batch_size, num_turns in self.batch_sizes:
            i += 1
            print("batch:", i)
            print("size:", batch_size)
            print("number of turns:", num_turns)
            
        #'''

        self.batches = [[np.empty((batch_size, num_turns, self.turns_dimension), dtype=np.half),
                         np.empty((batch_size, self.choices_dimension), dtype=np.half),
                         np.empty((batch_size, self.team_dimension), dtype=np.half)]
                        for batch_size, num_turns in self.batch_sizes]

        mem = sum([batch_size * num_turns * self.turns_dimension +
                   batch_size * self.choices_dimension + batch_size * self.team_dimension
                   for batch_size, num_turns in self.batch_sizes])

        mem_gb = mem / (1024 ** 3) * 8
        n_replays = len(replays)

        mem_per_replay = mem_gb / n_replays

        max_replays = 8 / mem_per_replay

        print(max_replays)

        self.labels = [np.empty(batch_size) for batch_size, num_instances in self.batch_sizes]

    def set_zero_size_batches(self):
        '''

        sets all of the batches with zero size to all zeros

        :return: None
        '''

        i = 0

        while self.batch_sizes[i][0] == self.batch_size:
            self.batches[i][0] = np.zeros((self.batch_sizes[i][0], 1, self.turns_dimension))

            i += 1
        # since we miss the very last item we need to set it as well
        if self.turn_instances[0] % 128 != 0:
            self.batches[i][0] = np.zeros((self.batch_sizes[i][0], 1, self.turns_dimension))

    def PCA_prep(self):
        '''

        helper function for running PCA

        checks to see if PCA is necessary and runs it if it is

        :return: None
        '''

        PCAs = os.listdir("PCA files")

        if (self.tier + "_turns_PCA.bin" not in PCAs or \
            self.tier + "_choices_PCA.bin" not in PCAs or \
            self.tier + "_teams_PCA.bin" not in PCAs) \
                and self.do_PCA:
            self.run_PCA()

    def run_PCA(self):
        '''

        runs PCA on the given input tier and pickles the results

        :return:
        '''

        replays = os.listdir("Binary Replays")
        r.shuffle(replays)

        turns_data = []
        choices_data = []
        teams_data = []

        counter = 0

        for replay in replays:
            if self.tier in replay:

                t0 = t.time()

                data = pokereplay.read_replay_file("Binary Replays/" + replay)

                if len(data[1]) == 0:
                    continue

                # assign some variables
                choice_size = sum(map(len, data[1][0][:5]))

                # unroll the turns data
                for i in range(len(data[1])):
                    data[1][i] = [data[1][i][j][k] for j in range(len(data[1][i]))
                                  for k in range(len(data[1][i][j]))]

                # append the different parts of the data

                turns_data = turns_data + [data[1][i] for i in range(len(data[1]))]

                choices_data = choices_data + [data[1][i][:choice_size]
                                               for i in range(len(data[1]))] + \
                               [data[1][i][choice_size:choice_size * 2]
                                for i in range(len(data[1]))]

                teams_data = teams_data + [[data[0][i][j]
                                            for i in range(len(data[0])) for j in range(len(data[0][i]))],
                                           [data[0][1 - i][j]
                                            for i in range(len(data[0])) for j in range(len(data[0][i]))]]

                counter += 1
                t1 = t.time()

                print("replay", counter, "took", t1 - t0)

                if counter >= self.replay_limit:
                    break

        print("got data")

        turns_data = np.array(turns_data)
        choices_data = np.array(choices_data)
        teams_data = np.array(teams_data)

        print(turns_data.shape)
        print(choices_data.shape)
        print(teams_data.shape)

        turns_PCA = PCA(n_components=0.99, svd_solver="full")
        turns_PCA.fit(turns_data)

        print("did PCA for turns")

        choices_PCA = PCA(n_components=0.99, svd_solver="full")
        choices_PCA.fit(choices_data)

        print("did PCA for choices")

        teams_PCA = PCA(n_components=0.99, svd_solver="full")
        teams_PCA.fit(teams_data)

        print("did PCA for teams")

        # now pickle the PCA

        f = open("PCA files/" + self.tier + "_turns_PCA.bin", "wb+")
        pickle.dump(turns_PCA, f)
        f.close()

        f = open("PCA files/" + self.tier + "_choices_PCA.bin", "wb+")
        pickle.dump(choices_PCA, f)
        f.close()

        f = open("PCA files/" + self.tier + "_teams_PCA.bin", "wb+")
        pickle.dump(teams_PCA, f)
        f.close()

        print(turns_PCA.components_.shape)
        print(choices_PCA.components_.shape)
        print(teams_PCA.components_.shape)

    def cull_batches_of_size_1(self):
        '''

        deletes all batches which have exactly one turn in them

        :return: None
        '''

        i = 0

        while i < len(self.batches):

            if self.batches[i][0].shape[1] == 1:
                # delete the entry
                del self.batches[i]
                del self.labels[i]
            else:
                i += 1

    def shuffle_training_data(self):
        '''

        shuffles the replays in the training data

        :return:
        '''

        for turn_set in self.turn_instances:
            r.shuffle(turn_set)

    def create_random_normalized_data(self):
        '''

        sets the data to be random data

        :return:
        '''

        self.batches = [[np.random.uniform(0, 1, size=(batch_size, num_turns, self.turns_dimension)),
                         np.random.uniform(0, 1, size=(batch_size, self.choices_dimension)),
                         np.random.uniform(0, 1, size=(batch_size, self.team_dimension))]
                        for batch_size, num_turns in self.batch_sizes]

        self.labels = [np.random.randint(2, size=batch_size) for batch_size, num_instances in self.batch_sizes]

        self.set_zero_size_batches()

class ThreadSaveItter:

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()

def threadsafe_generator(f):
    '''

    a threadsafe generator

    :param f: generator function
    :return:
    '''

    def g(*a, **kwargs):
        return ThreadSaveItter(f(*a, **kwargs))

    return g

@threadsafe_generator
def helper_generator(tier, batch_size):
    '''

    helper function for

    :param tier:
    :param batch_size:
    :return:
    '''

    Generator = DataGenerator(tier, batch_size)

    yield Generator.turns_reduced_dimension
    yield Generator.choices_reduced_dimension
    yield Generator.team_reduced_dimension

    while True:

        for i in range(len(Generator)):
            yield Generator[i]

        Generator.on_epoch_end()


def reformat_data(data):
    '''

    takes in data directly from data parser and outputs the three inputs to the neural network. these inputs are

    turns data: data about the turns of the battle leading up to the current turn
    choices data: data about the choice that we are evaluating
    teams data: data about the teams

    :param data: data about the replay
    :return: turns, choices and teams data for each player
    '''

    choice_size = sum(map(len, data[1][0][:5]))

    # unroll the turns data
    # -------------------------------------------------------------------------------------
    data[1] = [list(itertools.chain.from_iterable(data[1][i])) for i in range(len(data[1]))]

    # the turns data is identical since it is arranged by which player goes first
    player_1_turns = data[1]

    # for the player 2 turns, go through and reverse the Player IDs

    for i in range(len(data[1])):
        data[1][i][1 * choice_size - 1] = 1 - data[1][i][1 * choice_size - 1]
        data[1][i][2 * choice_size - 1] = 1 - data[1][i][2 * choice_size - 1]

    player_2_turns = data[1]

    player_1_choices = [data[1][i][:choice_size] for i in range(len(data[1]))]
    player_2_choices = [data[1][i][choice_size:2 * choice_size] for i in range(len(data[1]))]

    player_1_teams = [data[0][i][j] for i in range(len(data[0])) for j in range(len(data[0][i]))]
    player_2_teams = [data[0][1 - i][j] for i in range(len(data[0])) for j in range(len(data[0][i]))]

    return player_1_turns, player_1_choices, player_1_teams, player_2_turns, player_2_choices, player_2_teams


def load_NN(tier):
    '''

    loads a keras neural network from the given layer

    :param tier: tier to load the network for
    :return: keras nural network
    '''

    return keras.models.load_model("Models/" + tier + ".h5")


def battle_data_to_test_data(data):
    '''

    converts battle data to test data

    :param data: data about a particular battle
    :return: neural network inputs
    '''

    player_1_turns, player_1_choices, player_1_teams, \
        player_2_turns, player_2_choices, player_2_teams = reformat_data(data)

    return np.array(player_1_turns), np.array(player_1_choices), np.array(player_1_teams)


def test_model(tier, replay_no, NN, PCA):

    t0 = t.time()

    replay_text = Replay_Web_Scrapper.get_replay(replay_no, tier, "smogtours")
    data = Parser.parse_replay(replay_text)

    t1 = t.time()

    turns_data, choices_data, teams_data = battle_data_to_test_data(data)
    PCA_turns, PCA_choices, PCA_teams = PCA

    t2 = t.time()

    turns_data = PCA_turns.transform(turns_data)
    choices_data = PCA_choices.transform(choices_data)
    teams_data = PCA_teams.transform(teams_data.reshape(1, -1))

    print("time 2")
    objgraph.show_growth(5)

    t3 = t.time()

    predictions = [0] * turns_data.shape[0]

    t10 = 0
    t11 = 0
    t12 = 0
    t13 = 0
    t14 = 0

    #'''
    for i in range(1, turns_data.shape[0]):

        t10 += t.time()

        a = np.array([turns_data[:i], ])

        t11 += t.time()

        b = choices_data[i].reshape(1, -1)

        t12 += t.time()

        c = teams_data.reshape(1, -1)

        t13 += t.time()

        predictions[i] = NN.predict([a, b, c])[0][0]

        t14 += t.time()

        cuda.select_device(0)
        cuda.close()

    #'''

    winner = data[2][0]

    del turns_data
    del choices_data
    del teams_data

    print("time4")
    objgraph.show_growth(5)

    t4 = t.time()

    #print("requesting data", t1 - t0)
    #print("getting data out of tuples", t2 - t1)
    #print("PCA", t3 - t2)
    #print("making predictions", t4 - t3)

    #print("a", t11 - t10)
    #print("b", t12 - t11)
    #print("c", t13 - t12)
    #print("d", t14 - t13)

    return predictions, winner


def get_validation_accuracy(tier):

    f = open("replay texts/" + tier + " validation replays.txt")
    replays = [line.strip() for line in f]
    f.close()

    NN = load_NN(tier)

    f = open("PCA files/" + tier + "_turns_PCA.bin", "rb")
    PCA_turns = pickle.load(f)
    f.close()

    f = open("PCA files/" + tier + "_choices_PCA.bin", "rb")
    PCA_choices = pickle.load(f)
    f.close()

    f = open("PCA files/" + tier + "_teams_PCA.bin", "rb")
    PCA_teams = pickle.load(f)
    f.close()

    PCA = PCA_turns, PCA_choices, PCA_teams

    results = []

    objgraph.show_growth(5)

    for replay in replays:

        print("next replay")
        objgraph.show_growth(5)

        # print("processing", replay)

        if len(replay) != 6:
            continue

        predictions, winner = test_model(tier, replay, NN, PCA)

        for prediction in predictions:
            predicted_outcome = int(2 * prediction)

            results.append(int(predicted_outcome == winner))

    acc = sum(results) / len(results)

    # todo: don't do PCA on bits that tell the network which player is which

    return acc


def process_data(turn, replay, turns_dimension, choices_dimension, team_dimension):
    '''

    gets training data from a given turn in a replay number

    :param turn: the turn number of the batch we are processing
    :param replay: replay we are processing
    :return: tuple of sub-batch and it's labels
    '''

    turns_data = np.empty((2, max(1, turn), turns_dimension))
    choices_data = np.empty((2, choices_dimension))
    teams_data = np.empty((2, team_dimension))

    labels = np.empty(2)

    t.sleep(sleeptimes[2])

    # generator.t2 += t.time()

    # get the team data
    # -------------------------------------------------------------------------------------
    data = pokereplay.read_replay_file("Binary Replays/" + replay)

    # split the data between player 1 and player 2
    # -------------------------------------------------------------------------------------

    t.sleep(sleeptimes[3])

    # generator.t3 += t.time()

    player_1_turns, player_1_choices, player_1_teams, \
    player_2_turns, player_2_choices, player_2_teams = reformat_data(data)

    t.sleep(sleeptimes[4])

    # generator.t4 += t.time()

    # put the data into the batch
    # -------------------------------------------------------------------------------------

    if turn == 0:
        turns_data[0] = np.ones((1, turns_dimension))
        turns_data[1] = np.ones((1, turns_dimension))
    else:
        turns_data[0] = player_1_turns[:turn]
        turns_data[1] = player_2_turns[:turn]

    choices_data[0] = player_1_choices[turn]
    choices_data[1] = player_2_choices[turn]

    teams_data[0] = player_1_teams
    teams_data[1] = player_2_teams

    labels[0] = 1 - data[2][0]
    labels[1] = data[2][0]

    # t5 sleep
    t.sleep(sleeptimes[5])

    # generator.t5 += t.time()

    return turns_data, choices_data, teams_data, labels