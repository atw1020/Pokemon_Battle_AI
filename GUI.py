

import pokepy

client = pokepy.V2Client()

output = open("output.txt", "w+")

for i in range(1, 1000):

    item = client.get_ability(i)

    output.write(item.name + "\n")

    if i % 20 == 0:
        print("#", end="")

output.close()