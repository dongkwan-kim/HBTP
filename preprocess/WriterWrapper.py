# -*- coding: utf-8 -*-

import csv
import datetime


class WriterWrapper:

    def __init__(self, _filename: str, _fieldnames: list):
        file_name = (_filename + '_{0}.csv').format(datetime.datetime.now())
        self.f = open(file_name, 'w', encoding='utf-8')
        self.wr = csv.DictWriter(self.f, fieldnames=_fieldnames)
        self.wr.writeheader()

    def write_row(self, dct):
        self.wr.writerow(dct)

    def close(self):
        self.f.close()

