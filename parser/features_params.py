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
    'parent_child_pos': 0
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
    'pos_next_parent_previous_child': 0,
    'pos_previous_parent_previous_child': 0,
    'pos_next_parent_next_child': 0,
    'pos_previous_parent_next_child': 0,
    'pos_nn_parent_pp_child': 0,
    'pos_pp_parent_pp_child': 0,
    'pos_nn_parent_nn_child': 0,
    'pos_pp_parent_nn_child': 0
}
