import wikipediaapi
import numpy as np
import re
import string
import warnings
import aiml
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

#if you get a nltk error, uncomment this and install
#nltk.download()

        # Globals
IS_CHATTING = True
BYE_MESSAGE = 'Bye! Remember to wear your seatbelt!'
PUNCTUATION = dict((ord(punct), None) for punct in string.punctuation)
stopWords = set(stopwords.words('english'))

        # Lemmatization
lem = WordNetLemmatizer()

        # Functions
def get_document(page):
    doc = wiki.page(page)
    if doc.exists():
        return str(doc.summary)
    else:
        return 'null'

def lemmatize(tokens):
    return [lem.lemmatize(token) for token in tokens]

def normalized_lemmatization(doc):
    return lemmatize(word_tokenize(doc.lower().translate(PUNCTUATION)))

        # API
wiki = wikipediaapi.Wikipedia('en') #set up variable and language of wikipedia
wikipediaapi.log.setLevel(level=wikipediaapi.logging.ERROR)

        # Set Up Kernel
kernel = aiml.Kernel() 
kernel.setTextEncoding(None)
kernel.bootstrap(learnFiles="Chatbot.xml")

        # Build up Document
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

overall_document_sents = sent_tokenize(overall_document)

        #Intro Text
print("Hello, I'm Willow, the Road Safety Chat-Bot! What would you like to know?")

        #Loop
while IS_CHATTING:
    try:
        user_input = input(">>> ")
        response = ''
        agent = 'aiml'
        if agent == 'aiml':
            response = kernel.respond(user_input)

        if response == user_input.translate(PUNCTUATION):    
            # Process User Input
            overall_document_sents.append(user_input)
            
            word_vects = TfidfVectorizer(tokenizer=normalized_lemmatization, stop_words = stopWords)
            all_vects = word_vects.fit_transform(overall_document_sents)
            similar_vects = cosine_similarity(all_vects[-1], all_vects)
            similar_sents = similar_vects.argsort()[0][-2]

            matching_vects = similar_vects.flatten()
            matching_vects.sort()
            answer_vect = matching_vects[-2]
            overall_document_sents.remove(user_input)
            
            if answer_vect == 0:
                 print('I don\'t know, sorry.')
            else:
                print(overall_document_sents[similar_sents])
        
                    
                
        elif response != BYE_MESSAGE:
            print(response)
               
        else:
            print(response)
            IS_CHATTING = False
    except(KeyboardInterrupt, EOFError) as exception:
        break
