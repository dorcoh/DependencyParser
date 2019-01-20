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
    'pos_parent_child_sibling': 0,
    'word_parent_child_sibling': 0
}

# # competition model
# filter_dict_model = {
#     # unigram
#     'parent_word_pos': 0,
#     'parent_word': 0,
#     'parent_pos': 0,
#     'child_word_pos': 0,
#     'child_word': 0,
#     'child_pos': 0,
#     # bigram
#     'parent_child_word_pos': 1,
#     'parent_pos_child_word_pos': 0, #
#     'parent_word_child_word_pos': 1,
#     'parent_word_pos_child_pos': 0, #
#     'parent_word_pos_child_word': 1,
#     'parent_child_word': 1,
#     'parent_child_pos': 0, #
#     # extra
#     'pos_next_parent_previous_child': 1,
#     'pos_previous_parent_previous_child': 1,
#     'pos_next_parent_next_child': 1,
#     'pos_previous_parent_next_child': 1,
#     'pos_parent_child_sibling': 1,
#     'word_parent_child_sibling': 1
# }
#
