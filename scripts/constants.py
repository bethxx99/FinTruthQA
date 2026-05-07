TASK_NUM_LABELS = {
    "IS_QUESTION": 1,
    "QUES_RELEVANCE": 1,
    "ANS_RELEVANCE": 3,
    "ANS_READABILITY": 4,
}

SINGLE_SENTENCE_TASKS = {"IS_QUESTION", "QUES_RELEVANCE"}
SENTENCE_PAIR_TASKS = {"ANS_RELEVANCE", "ANS_READABILITY"}
ALL_TASKS = tuple(TASK_NUM_LABELS.keys())

ID_TO_LABEL = {
    "IS_QUESTION": {0: "Negative", 1: "Positive"},
    "QUES_RELEVANCE": {0: "Negative", 1: "Positive"},
    "ANS_RELEVANCE": {0: "1", 1: "2", 2: "3"},
    "ANS_READABILITY": {0: "1", 1: "2", 2: "3", 3: "4"},
}

