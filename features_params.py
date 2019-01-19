# features dictionaries

# baseline
filter_dict = {
    'parent_word_pos': 0,
    'parent_word': 0,
    'parent_pos': 0,
    'child_word_pos': 0,
    'child_word': 0,
    'child_pos': 0,
    'parent_pos_child_word_pos': 0,
    'parent_word_pos_child_pos': 0,
    'parent_child_pos': 0,
}

# competition model
filter_dict_model = {
    # unigram
    'parent_word_pos': 0,
    'parent_word': 0,
    'parent_pos': 0,
    'child_word_pos': 0,
    'child_word': 0,
    'child_pos': 0,
    # bigram
    'parent_child_word_pos': 0,
    'parent_pos_child_word_pos': 0,
    'parent_word_child_word_pos': 0,
    'parent_word_pos_child_pos': 0,
    'parent_word_pos_child_word': 0,
    'parent_child_word': 0,
    'parent_child_pos': 0,
    # extra
    'pre_child_pos': 0,
    'next_child_pos': 0,
    'next_parent_pos': 0,
    'pre_parent_pos': 0
}