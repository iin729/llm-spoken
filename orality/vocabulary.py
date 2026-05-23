# and
FILLER_WORDS = [
    # fillers (Fillers - none)
    "um", "uh", "er", "ah", "hmm", "mm", "mmhmm", "uhm", "mhm", "huh", "hem","yeah", "ya", "yep", "yup", "oh","okay", "ok","hey"
]

# row
INTERACTION_MARKERS = [
    # // (Starters & Linkers)
    "well", "so", "anyway", "anyhow", "now", "right", "alright",  "all right", "and then", "any thoughts",
    "anything else", "by the way",
    "listen", "look",  "speaking of", "frankly", "honestly",
    "basically", "literally", "actually", "in fact",

    # (Checkers & Seekers)
    "you know", "you see", "know what i mean", "you get me",
    "do you agree", "don't you think", "am i right", "is that right", "right?",
    "does that make sense", "do you follow", "guess what","do you think",
    "hear me out", "are you with me", "hey listen",

    # (Response & Feedback)
    "got it", "i got you", "i see", "so yeah", "really","exactly", "sure", "carry on", "come on", "go on", "please continue",
    "tell me","tell me more","what do you call it", "what’s it called", "what happened",

    "could you explain", "how to say", "how about you", "how does that sound",
    "what i'm saying", "can you hear me", "would you agree", "would you like to add",

    # (Delay & Interruptions)
    "wait", "hang on", "hold on", "hold up", "just a sec", "one second", "hold on a second",
    "excuse me", "pardon", "sorry to interrupt", "can i just say", "let me jump in",
    "let me add", "let’s say","let’s see","let me cut","let me explain","let me finish",
    "let me stop","let me see", "let's see", "let's say", "let me think",
    "can i cut","can i jump","can i just",
    "can i say", "i’d like","i’d say",
    "can i say something",
    "can i stop", "if i can just add",
    "if i could add",
    "i'd like to add", "before i forget",
    "before you go on", "i'm sorry, but", "like i said",
    "look here",
    "may i add something","may i interrupt","may i just say","stop right there",

    "care to comment", "do you mind if i", "just a minute", "just a moment", "just to clarify",
    "just wait", "wait a minute", "wait a moment", "wait a sec", "wait a second",
    "pardon me", "shall we", "time out"
]

#
STANCE_AND_HEDGES = [

    # (Cognitive & Subjective - )
    "i think", "i guess", "i suppose", "i mean", "i believe", "i reckon", "i bet",
    "i'd say", "i would say", "if i may", "in my opinion", "i believe", "i bet","i wonder",
    "i guess", "i like","mind you", "it presumably","it reckon","it seemed","it suppose",

    # (Intensifiers - but)
    "very", "totally", "definitely", "absolutely", "obviously",
    "seriously", "simply", "just", "quite", "pretty", "surely", "clearly", "essentially", "a little bit",
    "certainly",

    # (Hedges - )
    "kind of", "sort of", "kinda", "sorta", "a bit", "a little", "maybe", "perhaps",
    "probably", "possibly", "like", "likely", "seems", "seemingly", "apparently",
    "or something", "or whatever", "and stuff", "and things"

    # (Vague Nouns)
    "thing", "things", "stuff", "bit", "lot",
    "something", "anything", "everything", "nothing", "whatever",
    "guy", "guys", "someone", "anyone", "people",
]

# “”and
# and
LEXICAL_SIMPLIFICATION = [

    # / (Contractions)
    "i'm", "you're", "he's", "she's", "it's", "we're", "they're",
    "i've", "you've", "we've", "they've", "i'd", "you'd", "he'd",
    "i'll", "you'll", "won't", "can't", "don't", "doesn't", "didn't",
    "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't",
    "ain't", "innit",

    # (Oral Morphology)
    "gonna", "wanna", "gotta", "dunno", "lemme", "gimme",
    "shoulda", "woulda", "coulda", "'em", "cause"

    # / (Light/Basic Verbs - )
    "get", "got", "getting",
    "go", "went", "gone", "going",
    "make", "made", "making",
    "do", "did", "done", "doing",
    "take", "took", "taking",
    "put", "putting",
    "keep", "keeping",
    "let"
]

ALL_VOCAB = FILLER_WORDS + INTERACTION_MARKERS + STANCE_AND_HEDGES + LEXICAL_SIMPLIFICATION
#
# for i in ALL_VOCAB:
#     if i.lower() != i:
#         print(i)
#     if ALL_VOCAB.count(i) > 1:
#         print(i)

# 288
# print(len(ALL_VOCAB))

all_s = ""
# with open(f"./../../dataset/test_train_stat.json","r",encoding='utf-8') as f:
#     all_s = f.read()
#     all_s.strip()
# for one in ALL_VOCAB:
#     if one not in all_s:
#         print(one)
