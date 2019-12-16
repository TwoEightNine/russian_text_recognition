import os
import pandas as pd
import numpy as np

METADATA = 'metadata.csv'
FILE_TEMPLATE = 'img%d.png'
DIR = 'dataset'

COLUMN_ID = 'id'
COLUMN_LETTER = 'letter'


class Dataset(object):

    def __init__(self):
        if not os.path.isdir(DIR):
            os.mkdir(DIR)
        self.metadata_file_name = DIR + '/' + METADATA
        if not os.path.isfile(self.metadata_file_name):
            pd.DataFrame(columns=[COLUMN_ID, COLUMN_LETTER]).to_csv(self.metadata_file_name, index=False)
        self.metadata = pd.read_csv(self.metadata_file_name)

    def add_letter(self, image, letter):
        images_total = self.metadata.shape[0]
        new_id = images_total + 1
        image.save(DIR + '/' + FILE_TEMPLATE % new_id)
        self.metadata.loc[new_id] = [new_id, letter]
        self.metadata.to_csv(self.metadata_file_name, index=False)

    def get_data(self):
        images_total = self.metadata.shape[0]
        file_names = [DIR + '/' + FILE_TEMPLATE % (i + 1) for i in range(images_total)]
        file_names = np.array(file_names)
        letters = self.metadata[COLUMN_LETTER].to_numpy().reshape((-1, 1))
        return file_names, letters
