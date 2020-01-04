'''

author: arthur wesley

'''

import time as t

import requests
from bs4 import BeautifulSoup
from selenium import webdriver

import Parser

def get_replay(replay_number, tier, server):
    '''

    gets the text of a replay off of a replay number

    :param replay_number:
    :return:
    '''

    if server != "":
        server = server + "-"

    request = requests.get("https://replay.pokemonshowdown.com/" + server + tier + "-" + str(replay_number) + ".log")

    text = request.text

    del request

    return text


def get_recent_replays(tier):
    '''

    scrapes the latest replays off of the gen 7 replays page

    :return: list of replay numbers
    '''

    url = "https://replay.pokemonshowdown.com/search/?format=" + tier + "&recent"
    
    print(url)

    request = request_multi_page_data(url)

    soup = BeautifulSoup(request, "html.parser")
    list_items = soup.find_all("li")
    list_items = list_items[8:-1]

    replay_numbers = []

    for list_item in list_items:
        link = (list_item.find("a")["href"])
        link = link.split("-")

        link = link[-1]

        replay_numbers.append(link)

    return replay_numbers

def get_replay_numbers_high_ladder(tier):
    '''

    scrapes the latest 2000 elo+ replays off of the gen 7 replays page

    :return: list of replay numbers
    '''

    url = "https://replay.pokemonshowdown.com/search/?format=" + tier + "&rating"

    print(url)

    request = request_multi_page_data(url)

    soup = BeautifulSoup(request, "html.parser")
    list_items = soup.find_all("li")
    list_items = list_items[8:-1]  # strip off the first 8 list items (not replays) and the last item (not replay)

    replay_numbers = []

    for list_item in list_items:

        link = (list_item.find("a")["href"])
        link = link.split("-")

        link = link[-1]

        replay_numbers.append(link)


    return replay_numbers


def request_multi_page_data(url):
    '''

    repeatedly requests data from the pokemon showdown replays page a specified number of times

    :param url: replays url to refresh
    :return: request for page with all all the wanted data loaded
    '''


    driver = webdriver.Safari()

    driver.get(url)

    count = 0
    max = 100

    while driver.find_elements_by_name("moreResults") and count < max:
        driver.find_element_by_name("moreResults").click()
        t.sleep(0.1)

        print("got results", count)

        count += 1

    html = driver.page_source.encode('utf-8')

    return html


def scrape_replays_and_save(tier):
    '''

    scrapes the replay numbers off of pokemon showdown and puts them into the "replay nubmers.txt" file

    :return: None
    '''

    while True:

        try:
            replay_numbers = get_recent_replays(tier)  # + get_replay_numbers_high_ladder(tier)
            break
        except:
            continue

    print(replay_numbers)

    output = open("replay texts/" + tier + " new replay numbers.txt", "w+")

    for replay_number in replay_numbers:
        output.write(replay_number + "\n")


def filter_replay_numbers(tier, threshold):
    '''

    filters replays rated below a given threshold out of the replay numbers.txt files and also screens out duplicates

    :param threshold: the threshold below which replays will be filtered out
    :return: None
    '''

    lines = [line for line in open("replay texts/" + tier + " new replay numbers.txt")]
    already_used_lines = [line for line in open("replay texts/" + tier + " replay numbers.txt")]

    # go through all the lines and delete them if have already been processed

    i = 0

    used_replay_numbers = open("replay texts/" + tier + " replay numbers.txt", "a+")

    while i < len(lines):
        if lines[i] in already_used_lines:
            del lines[i]
        else:

            # put the line into the replay numbers file so that we never reference it again
            used_replay_numbers.write(lines[i])

            # increment the counter
            i += 1

    used_replay_numbers.close()

    lines = list(set(lines))  # easy to remove duplicates

    output = open("replay texts/" + tier + " high ladder replay numbers.txt", "a+")

    t0 = t.time()
    j = 0

    print("filtering", len(lines), "replays")

    for i in range(len(lines)):

        try:
            check_replay_soup(lines[i], tier, threshold, output)
            j += 1
        except:
            continue

    t1 = t.time()

    print("filtered", len(lines), "replays")
    print("processing the replays took", t1 - t0, "seconds averaging", (t1 - t0) / len(lines), "seconds per replay")
    print("got", j, "new high ladder replays")


def check_replay_soup(replay_no, tier, threshold, output):
    ''''''

    request = requests.get("https://replay.pokemonshowdown.com/" + tier + "-" + str(replay_no.strip()))

    print("checking replay number", replay_no.strip())

    soup = BeautifulSoup(request.text, "html.parser")
    # find the small which contains the rating
    upload_data = soup.find_all("small", attrs={"class": "uploaddate"})

    possible_battle_rating = upload_data[0].contents[-1]
    possible_battle_rating = possible_battle_rating.strip()

    # now we check to see if the possible battle rating is neumeric
    # (if the battle isn't rated it should be a date(

    if possible_battle_rating.isnumeric():

        # we also screen out all replays below the threshold

        print(int(possible_battle_rating))

        if int(possible_battle_rating) > threshold:
            # if the replay was above 1700, write the replay back to the replay numbers.txt file
            output.write(replay_no)


def check_replay_log(replay_no, tier, threshold, output):

    request = requests.get("https://replay.pokemonshowdown.com/" + tier + "-" + str(replay_no.strip()) + ".log")
    print("checking replay", replay_no.strip())

    lines = request.text.split("\n|")

    ratings = []

    for line in lines:
        if line[:3] == "raw":

            line = line.split("rating:")
            ratings_data = line[-1]

            ratings_data = ratings_data.split("strong>")
            ratings_data[1] = ratings_data.split("</")

            ratings.append(ratings_data[1])

    rating = min(map(int, ratings))

    if rating > threshold:
        output.write(replay_no)


def scrape_replay_numbers(tier):
    '''

    scrapes the replays pages of the pokemon showdown OU page and saves the results in the replay numbers.txt file

    :return: None
    '''

    scrape_replays_and_save(tier)
    filter_replay_numbers(tier, 1500)


def scrape_replays(tier):
    '''

    scrapes turn data about replays off of pokemon showdown

    :return: replay data
    '''

    replay_numbers = Parser.text_file_to_list("replay texts/" + tier + " high ladder replay numbers.txt")

    #replay_numbers = [955013148]

    input_data = []

    for replay_number in replay_numbers:

        print("formatting data for replay", replay_number)

        text = get_replay(replay_number, tier)
        #input_data.append(Parser.parse_replay(text))

def main():

    ou = "gen7ou"

    scrape_replay_numbers(ou)
    #scrape_replays(ou)

if __name__ == "__main__":
    main()