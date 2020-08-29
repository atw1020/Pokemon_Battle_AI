'''

this program is used to test c programs

problem replays: 955867461_gen7ou.rply, 957467770_gen7ou.rply, 954973242_gen7ou.rply, 955660160_gen7ou.rply,
957286288_gen7ou.rply, 957082629_gen7ou.rply, 959132126_gen7ou.rply, 959132126_gen7ou.rply, 958517786_gen7ou.rply
954806132_gen7ou.rply, 956207513_gen7ou.rply, 956207513_gen7ou.rply

'''

import Replay_Web_Scrapper
import Parser

import sys
import os
import time as t
import gc

import objgraph

import pokereplay

replays = [line.strip() for line in open("replay texts/gen7ou high ladder replay numbers.txt")]

i = 0

for replay_no in replays:
    try:

        replay_text = Replay_Web_Scrapper.get_replay(replay_no, "gen7ou", "")

        data = Parser.parse_replay(replay_text)

        try:
            pokereplay.make_replay_file("test.rply", data)
        except Exception as E:
            print(replay_no)
            raise E

        output_data = pokereplay.read_replay_file("test.rply")

        if output_data != data:
            print(replay_no, "was not properly saved")
        else:
            if (i % (200 // 10)) == 0:
                print(str((i * 100) // 200) + "% complete")

        i += 1

        # if we reach 200 replays, break out of the program
        if i > 200:
            break;

    except LookupError:
        print("tie in replay")
    except AssertionError:
        print("replay had zero turns")