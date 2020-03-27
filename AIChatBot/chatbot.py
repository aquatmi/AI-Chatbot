import re
import os
import string
import warnings
import aiml

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from chatbot_functions import *  # my custom functions, I like to have them separate for tidiness
from toy_world_functions import *
from transformer_chatbot import trans_load, trans_predict
from cartpole import CartPoleAgent

warnings.filterwarnings('ignore')

# GLOBALS
IS_CHATTING = True
RESEARCH_MODE = False
BYE_MESSAGE = 'Bye! Remember to wear your seat belt!'
ERROR_MESSAGE = "I think I misunderstood you, did you misspell something?"
INTRO_MESSAGES = [
    "Hello, I'm Willow, the Road Safety Chat-Bot!",
    'If you want to know something, ask me!',
    'Or if you want me to identify a sign say: "identify"',
    'Remember, you can always say "Help" if you need it']
PUNCTUATION = dict((ord(punctuation), None) for punctuation in string.punctuation)
STOP_WORDS = set(stopwords.words('english'))
MODEL = load_model()
CART_POLE_AGENT = CartPoleAgent()
CART_POLE_AGENT.prep()
AGENT = "cartpole-deepq.h5"
AGENT_RISKY = "cartpole-deepq-risky.h5"
# prep function loads and closes the window, this is because I found from testing that the window opens behind the IDE
# the first time it is launched, but subsequent launches are in front, so I launch it once on start so it will always
# open in the foreground.
TRANSFORMER = trans_load()


def normalized_lemmatization(document):
    return lemmatize(word_tokenize(document.lower().translate(PUNCTUATION)))


# KERNEL
kernel = aiml.Kernel()
kernel.setTextEncoding(None)
kernel.bootstrap(learnFiles="input/chatbot.xml")

# DOCUMENT
documents = [get_document('traffic sign'), get_document('highway code'), get_document('fingerpost'),
             get_document('road signs in the united kingdom'), get_document('road safety'), get_document('seat belt'),
             get_document('vehicle safety'), get_document('warning sign'),
             get_document('vienna convention on road signs and signals')]

overall_document = ''
for doc in documents:
    overall_document += doc

overall_document = re.sub(r'\[[0-9]*\]', ' ', overall_document)
overall_document = re.sub(r'\([^)]*\)', '', overall_document)
overall_document = re.sub(r'\s+', ' ', overall_document)
overall_document = overall_document.lower()

overall_document_sentences = sent_tokenize(overall_document)

# INTRO
for message in INTRO_MESSAGES:
    print(message)

# LOOP
while IS_CHATTING:
    try:
        user_input = input(">>> ")
        response = ''
        agent = 'aiml'
        if agent == 'aiml':
            response = kernel.respond(user_input)

        # TRANSFORMER TOGGLE
        if response == "%£togglemode":
            if RESEARCH_MODE:
                RESEARCH_MODE = False
                print("Ok, I wont look up answers.")
            else:
                RESEARCH_MODE = True
                print("I'll let you know what I can find!")

        # OPEN AI
        elif response[:2] == "$$":
            if response == "$$help":
                print("say 'play CartPole' if you want me to play a game!")
                print("sat 'play CartPole Risky' if you want me to play a little different")
            elif response == "$$cartpole":
                print("Easy! I've got this in the bag!")
                CART_POLE_AGENT.play(show=True, episodes=1, n=AGENT)
            elif response == "$$cartpolerisky":
                print("Alright, I'll play risky!")
                CART_POLE_AGENT.play(show=True, episodes=1, n=AGENT_RISKY)

        # TOY WORLD
        elif response[:1] == '$':
            if response == '$toy_world_help':
                toy_world_helper()
            else:
                response = response[1:].split('%')
                ########################################
                # FUNCTIONS FOR THIS SECTION ARE FOUND #
                #  IN THE TOY_WORLD_FUNCTIONS.PY FILE  #
                ########################################
                try:
                    if response[0] == '0':  # PARK * IN *
                        park_car(response, car_counter)
                    if response[0] == '1':  # IS * IN *
                        check_for_car(response)
                    if response[0] == '2':  # WHAT IS IN *
                        check_garage(response)
                    if response[0] == '3':  # SET * NUMBERPLATE *
                        set_plate(response)
                    if response[0] == '4':  # GET * NUMBERPLATE
                        get_plate(response)
                except:
                    print(ERROR_MESSAGE)

        # GENERIC RESPONSE
        elif response == user_input.translate(PUNCTUATION):
            if RESEARCH_MODE:  # IF IN SEARCH MODE, CHAT-BOT WILL LOOK THROUGH CORPUS OF INFORMATION FOR A RESPONSE
                # Process User Input
                overall_document_sentences.append(user_input)

                word_vectors = TfidfVectorizer(tokenizer=normalized_lemmatization, stop_words=STOP_WORDS)
                all_vectors = word_vectors.fit_transform(overall_document_sentences)
                similar_vectors = cosine_similarity(all_vectors[-1], all_vectors)
                similar_sentences = similar_vectors.argsort()[0][-2]

                matching_vectors = similar_vectors.flatten()
                matching_vectors.sort()
                answer_vector = matching_vectors[-2]
                overall_document_sentences.remove(user_input)

                if answer_vector == 0:
                    print('I don\'t know, sorry.')
                else:
                    print(overall_document_sentences[similar_sentences])
            else:  # IF IN NON RESEARCH MODE (OR CHAT MODE), IT WILL DECIDE AN ANSWER FROM ITS TRANSFORMER NETWORK
                answer = trans_predict(response, TRANSFORMER)
                print(answer)

        # IMAGE PREDICTION
        elif response == "Ok! Let me see!":
            print(response)
            image = get_image()
            if image is None:
                print('Please give me an image file.')
            else:
                predict(MODEL, image)

        # CATCH ALL
        elif response != BYE_MESSAGE:
            print(response)

        # END CHAT
        else:
            print(response)
            IS_CHATTING = False

    except(KeyboardInterrupt, EOFError) as exception:
        break
