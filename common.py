import csv


def get_text_file(file_path):
    rows = []

    with open(file_path, 'r') as data_tsv:
        d_tsv = csv.reader(data_tsv, delimiter='\t')
        for row in d_tsv:
            rows.append(row)

    return rows
