import nltk

v = """
ferrari => {}
citroen => {}
kia => {}
honda => {}
ford => {}
garage1 => g1
garage2 => g2
garage3 => g3
garage4 => g4
be_in => {}
"""
fol_val = nltk.Valuation.fromstring(v)
grammar_file = 'input/simple-sem-edit.fcfg'
car_counter = 0

if len(fol_val["be_in"]) == 1:  # clean up if necessary
    if ('',) in fol_val["be_in"]:
        fol_val["be_in"].clear()


def toy_world_helper():
    print("You can ask me to park a car by saying \"Park [car] in [garage]\".")
    print("There is also two ways to ask where a car is, you can either;")
    print("ask \"Is [car] in [garage]\" to ask if a specific car is somewhere,")
    print("or ask \"what is in [garage]\" to just find out what car is in a garage.")
    print("Finally, you can set and get the numberplates of your cars using;")
    print("\"Set [car] numberplate [numberplate]\" and \"Get [car] numberplate\" respectively")

def park_car(input_arr, counter):
    # park car
    input_arr = remove_spaces(input_arr)
    car = 'car' + str(counter)
    counter += 1
    # insert information and clean if needed
    fol_val['obj' + car] = car
    if input_arr[2][:len(input_arr[2])-1] == "garage":
        try:
            clean_up(input_arr[1])
            fol_val[input_arr[1]].add((car,))
            fol_val["be_in"].add((car, fol_val[input_arr[2]]))
            # confirmation response
            print("I have parked your", input_arr[1], "in", input_arr[2])
        except:
            print("Sorry, I don't know what", input_arr[1], "is, did you misspell it?")
    else:
        print("I don't know where you want me to park it, did you misspell it?")


# Function seems inconsistent, for an unknown reason, sometimes answer will come back as an
# empty list of lists. I have not found a reason or a fix for this, but sometimes it works
# it worked for my chat log, but didn't work when I tried to take a new screenshot of the updated
# introduction text
def check_for_car(input_arr):
    # check car locations
    input_arr = remove_spaces(input_arr)
    g = nltk.Assignment(fol_val.domain)
    m = nltk.Model(fol_val.domain, fol_val)
    sent = 'some ' + input_arr[1] + ' are_in ' + input_arr[2]
    answer = nltk.evaluate_sents([sent], grammar_file, m, g)[0][0]
    if answer[2]:
        print("Your", input_arr[1], "is in", input_arr[2])
    else:
        print("Your", input_arr[1], "is not in", input_arr[2])


def check_garage(input_arr):
    # check garage
    input_arr = remove_spaces(input_arr)
    g = nltk.Assignment(fol_val.domain)
    m = nltk.Model(fol_val.domain, fol_val)
    e = nltk.Expression.fromstring("be_in(x," + input_arr[1] + ")")
    sat = m.satisfiers(e, "x", g)
    if len(sat) == 0:
        print("There is nothing parked here.")
    else:
        # find satisfying objects in the valuation dictionary,
        # and print their type names
        sol = fol_val.values()
        for so in sat:
            for k, v in fol_val.items():
                if len(v) > 0:
                    vl = list(v)
                    if len(vl[0]) == 1:
                        for i in vl:
                            if i[0] == so:
                                print("There is a", k, "parked here")
                                break


def set_plate(input_arr):
    try:
        input_arr = remove_spaces(input_arr)

        fol_val[input_arr[1]].add(('plate', input_arr[2]))  # insert type of plant information
        print("set", input_arr[1], "number plate to", input_arr[2])
    except:
        print(input_arr[1], "doesn't exist, did you misspell it?")


def get_plate(input_arr):
    answered = False
    input_arr = remove_spaces(input_arr)
    car_info = list(fol_val[input_arr[1]])
    for item in car_info:
        if item[0] == 'plate':
            print("The number plate is:", item[1])
            answered = True
    if not answered:
        print("I cannot find a number plate for that car, sorry")


# removes unwanted spaces and capital letters
# to reduce number of errors
# eg turns "garage 1" into "garage1"
def remove_spaces(array):
    counter = 0
    temp_array = [None] * len(array)
    while counter < len(array):
        temp_array[counter] = array[counter].replace(' ', '').lower()
        counter += 1
    return temp_array


# takes empty values out of fol_val to reduce
# confusion in the bot
def clean_up(string):
    if len(fol_val[string]) == 1:  # clean up if necessary
        if ('',) in fol_val[string]:
            fol_val[string].clear()
