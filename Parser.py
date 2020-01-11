'''

Author Arthur wesley

'''

import Replay_Web_Scrapper
import pokereplay

import os

def parse_replay(replay_text):
    '''

    parses through the text of a replay and gets the crucial details

    :param replay_text: text of the replay
    :return: features of the replay
    '''

    # first things first, split the replay text into individual turns
    turns = split_on_turn_end(replay_text)

    # now go through the turns and add their data to the output

    turns_data = []
    nicknames = dict()

    moves = text_file_to_list("data/moves.txt")
    pokemon = text_file_to_list("data/pokemon.txt")
    other_attrs = text_file_to_list("data/pokemon.txt")

    # first, let's get the input teams
    (teams_data, player1) = get_teams(turns[0])

    for turn in turns[:-1]: # don't include the last turn since the last turn contains no actions
        turns_data.append(get_turn_features(turn, nicknames, moves, pokemon, other_attrs))

    # next let's get the winner
    winner = get_winner(turns[-1], player1)

    assert len(turns_data) > 0

    return [teams_data, turns_data, winner]


def split_on_turn_end(replay_text):
    '''

    makes a single string of replay text into a list of strings where each item in the list corresponds to a turn
    of the battle

    # note a "turn" is defined as any opportunity the AI has to make a decision. when a pokemon is KOed and
    a new pokemon must be sent out this is considered a "turn"

    :param replay_text: single string of entire battle
    :return: list of strings
    '''

    turns = replay_text.split("\n|turn|")

    # now go through the turns line by line to see if any pokemon fainted

    i = 0

    while i < len(turns):

        lines = turns[i].split("\n")

        for j in range(len(lines)):
            if lines[j][:7] == "|faint|":

                # split the turn in two

                new_turn_1 = "\n".join(lines[:j])
                new_turn_2 = "\n".join(lines[j:])

                # delete the old turn

                del turns[i]

                turns.insert(i, new_turn_1)
                turns.insert(i + 1, new_turn_2)

                i += 1

                break

        i += 1

    return turns


def get_turn_features(turn, nicknames_dict, moves_list, pokemon_list, other_attrs):
    '''

    gets data about the turn that took place

    :param turn: text of the turn to analyze
    :param nicknames_dict: dictionary mapping nicknames to pokemon names
    :return: turn features
    '''

    # first do some pre-processing
    turn = pre_processing(turn, nicknames_dict)

    # the turn is made up of 12 sections
    # Note: the 'first player' refers to the player whose action is seen first rather than player 1/player 2

    # section 1: first player switch pokemon
    # section 2: first player move attacking pokemon
    # section 3: first player move attack name
    # section 4: first player can't attack
    # section 6: first player player ID (p1/p2)

    # section 6: second player switch pokemon
    # section 7: second player move attacking pokemon
    # section 8: second player move attack name
    # section 9: second player can't attack
    # section 10: second player player ID (p1/p2)

    # section 11: other attributes (note: attribute are not associated with a pokemon

    # initialize some variables
    data = [[0] * len(pokemon_list)] + [[0] * len(pokemon_list)] + [[0] * len(moves_list)] + [[0]] + [[0]] + \
           [[0] * len(pokemon_list)] + [[0] * len(pokemon_list)] + [[0] * len(moves_list)] + [[0]] + [[0]] + \
           [[0] * len(other_attrs)]

    # now go through the text line by line

    lines = turn.split("\n")
    i = 0

    # go forward to the next switch or move
    while lines[i][:8] != "|switch|" and lines[i][:6] != "|move|" and lines[i][:6] != "|cant|" and \
            lines[i][:11] != "|-activate|":
        i += 1

    data[:5] = get_action_data(lines[i], moves_list, pokemon_list, other_attrs)
    i += 1

    if i >= len(lines):
        return data
        data[10] = get_other_turn_attributes(turn, other_attrs)

    while lines[i][:8] != "|switch|" and lines[i][:6] != "|move|" and lines[i][:6] != "|cant|" and \
                lines[i][:11] != "|-activate|":
        i += 1
        if i >= len(lines):
            return data
            data[10] = get_other_turn_attributes(turn, other_attrs)

    data[5:10] = get_action_data(lines[i], moves_list, pokemon_list, other_attrs)

    data[10] = get_other_turn_attributes(turn, other_attrs)

    return data


def get_action_data(action_text, moves_list, pokemon_list, other_attrs):
    '''

    takes in the text of an action and returns details about the action

    an action is some choice which a battler makes. an action can either by a switch or the use of a move

    :param action_text: text that describes the action that took place
    :return: list representing the action that took place
    '''

    # declare some variables

    data = [[0] * len(pokemon_list)] + [[0] * len(pokemon_list)] + [[0] * len(moves_list)] + [[0]] * 2
    action_text = action_text.split("|")
    action_text = action_text[1:] # the first item in a split like this is actually an empty string so slice it off

    # first assign the first item to be the player

    if "p1a" in action_text[1]:
        data[4][0] = 1

    if action_text[0] == "switch":
        # assign the switch pokemon
        switched_pokemon = action_text[1][5:]
        try:
            data[0][pokemon_list.index(switched_pokemon)] = 1
        except:
            print(action_text)

    if action_text[0] == "move":
        # assign the attacking pokemon

        attacking_pokemon = action_text[1][5:]
        data[1][pokemon_list.index(attacking_pokemon)] = 1

        # assign the attacking move
        attacking_move = action_text[2]
        data[2][moves_list.index(attacking_move)]

    if action_text[0] == "cant" or action_text == "-activate":
        # cant means that we fail to attack because we are asleep or paralyzed or something
        # we set a flag in the second to last position to whenever we can't attack

        data[3][0] = 1

    return data


def get_other_turn_attributes(turn, other_attrs):
    '''

    takes in a turn and returns the other attributes associated with that turn

    unfortunately I'm too lazy to check which player the other attributes apply to. hopefully the
    neural network can work out some of that for itself? not really a good choice and probably one I will regret later

    fixme: asosiate items with pokemon

    :param turn: text of the turn
    :param other_attrs: list of other turn attributes
    :return:
    '''

    # iterate over the attributes

    data = [0] * len(other_attrs)

    for attribute in other_attrs:

        if attribute in turn:
            index = other_attrs.index(attribute)

            data[index] = 1

    return data


def update_nicknames_dict(turn, nicknames_dict):
    '''

    takes in a turn and updates the nicknames dictionary with any new nicknames

    :param turn: text of the turn to use to update the nicknames
    :param nicknames_dict: dictionary mapping nicknames to pokemon
    :return: None
    '''

    # go through the turn line by line and find any line that begins with "switch"
    lines = turn.split("\n")

    for line in lines:
        if line[:8] == "|switch|" or line[:6] == "|drag|":
            line = line.split("|")
            line = line[1:]

            nickname = line[1]
            pokemon = nickname[:5] + line[2].split(", ")[0]

            nicknames_dict[nickname] = pokemon


def replace_nicknames(turn, nicknames_dict):
    '''

    replaces all the nicknames in the turn with the actual pokemon's name

    :param turn: the text of the turn to be updated
    :param nicknames_dict: dictionary mapping nicknames to pokemon names
    :return: turn text with the nicknames replaced
    '''

    for nickname in nicknames_dict.keys():
        turn = turn.replace(nickname + "|", nicknames_dict[nickname] + "|")

    return turn


def pre_processing(turn, nicknames_dict):
    '''

    pre-processes the turn, setting everything to lower case and replacing the nicknames

    :param turn: text of the turn being inputted
    :param nicknames_dict: dictionary mapping nicknames to pokemon names
    :return: pre-processed text
    '''

    turn = turn.lower()

    update_nicknames_dict(turn, nicknames_dict)
    turn = replace_nicknames(turn, nicknames_dict)

    return turn


def text_file_to_list(path):
    '''

    takes in the name of a text file and converts it to a list of strings broken by newlines

    :param path: filepath
    :return: list of text
    '''

    return [line.strip() for line in open(path)]


def get_teams(first_turn):
    '''

    takes in the text of the first turn of the battle and outputs a vector that details the pokemon on each player's
    team

    :param first_turn: text of the first turn of the battle
    :return: vector containing the pokemon in the team
    '''

    first_turn = first_turn.lower()

    pokemon_list = text_file_to_list("data/pokemon.txt")

    # first split the first turn into lines
    lines = first_turn.split("\n")

    output = [[0] * len(pokemon_list) for i in range(2)]

    for line in lines:
        # split the line on separators
        line = line.split("|")

        if len(line) == 1:
            # ignore any blank lines
            continue

        # assign the player (another thing this method returns)

        if line[1] == "player" and line[2] == "p1":
            player_1 = line[3]

        if line[1] != "poke":  # indexing starts at 1 because the first item in the split is an empty string
            continue

        # the line we are looking at contains a pokemon
        # player p1 has priority for the first half of the list while p2 has priority on the second half of the list

        if line[2] == "p1":
            sublist = 0
        else:
            sublist = 1

        pokemon = line[3].split(", ")[0]

        # set the value of the output to 1
        output[sublist][pokemon_list.index(pokemon)] = 1

    return (output, player_1)

def get_winner(last_turn, p1_name):
    '''

    gets the winning player (1 for p1 or 0 for p2)

    :param last_turn: text of the last turn of the battle
    :p1_name: name of player 1
    :return: winning player
    '''

    # go through the last turn line by line

    lines = last_turn.split("\n")

    for line in lines:

        line = line.split("|")

        if len(line) < 2:
            continue

        if line[1] == "win":

            if line[2] == p1_name:
                return [1]
            else:
                return [0]

    # if we don't find a win then the game was a tie
    raise LookupError("tie in replay")


def save_replay(replay_number, tier, server=""):
    '''

    saves a given replay number in the binary replays folder

    :param replay_number: the replay number to look up
    :param tier: tier of the replay
    :return: None
    '''

    replay_text = Replay_Web_Scrapper.get_replay(replay_number, tier, server)
    data = parse_replay(replay_text)

    pokereplay.make_replay_file("Binary Replays/" + replay_number + "_" + tier + ".rply", data)


def save_replays_binary(tier):
    '''

    saves all the replays in the high ladder replay numbers file into the binary replays file

    :param tier: tier to save the replays for
    :return: None
    '''

    replay_nos = text_file_to_list("replay texts/" + tier + " high ladder replay numbers.txt")

    if not os.path.exists("Binary Replays/"):
        os.mkdir("Binary Replays/")

    replays_already_saved = os.listdir("Binary Replays")

    for replay_no in replay_nos:
        if replay_no + "_" + tier + ".rply" in replays_already_saved:
            continue
        print("saving replay", replay_no)
        try:
            save_replay(replay_no, tier)
        except LookupError:
            print("tie in replay")
        except AssertionError:
            print("replay had zero turns")



def main():
    '''

    main function

    :return: None
    '''

    save_replays_binary("gen7ou")

if __name__ == "__main__":

    main()

    #print(len("second player move attacking pokemon"))

    pass

    #print(bytes_to_list_of_binary_values_3(Bytes))




