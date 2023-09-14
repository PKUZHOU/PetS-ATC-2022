
import re
import os 

from torch.random import seed

class AlphaModel:
    def __init__(self, db_path):
        self.db_path = db_path
        self.db = {}

        if os.path.exists(self.db_path):
            self.build_db()

    def build_db(self):
        with open(self.db_path,'r') as f:
            lines = f.readlines()        
        
        for line in lines:
            _, batch_size, seq_len = [int(x) for x in line.split(" ")[:3]] 
            avg_time = float(line.split(" ")[-1])
            if batch_size not in self.db:
                self.db[batch_size] = {}
                    
            self.db[batch_size][seq_len] = avg_time

    def build_model(self):
        # TODO: 
        self.model = None
        
    def query(self, batch_size, seq_len):
        if batch_size in self.db:
            if seq_len in self.db[batch_size]:
                return self.db[batch_size][seq_len]

        # FIXME: simulate interpolation.
        recorded_batches = sorted(list(self.db.keys()))

        # If the batch_size is larger than the max recorded batch, set the latency to INF
        # if batch_size > recorded_batches[-1]:
            # return 1e10
        if batch_size > 128:
            return 1e10
        searched_batch = recorded_batches[-1]
        for b in recorded_batches:
            if b >= batch_size:
                searched_batch = b
                break

        recorded_seq_len = sorted(list(self.db[searched_batch].keys()))

        searched_seq_len = recorded_seq_len[-1]
        for s in recorded_seq_len:
            if s >= seq_len:
                searched_seq_len = s
                break

        return self.db[searched_batch][searched_seq_len]


class BetaModel:
    def __init__(self, db_path):
        self.db_path = db_path
        self.db = {}
        if os.path.exists(self.db_path):
            self.build_db()

        # self.build_db()

    def build_db(self):
        with open(self.db_path, 'r') as f:
            lines = f.readlines()
    
        for line in lines:
            pet_type, batch_size, seq_len = [int(x) for x in line.split(" ")[:3]]
            op_time = float(line.split(" ")[-1])
            
            # this is not necessary
            # op_time = op_time / 10  

            if pet_type not in self.db:
                self.db[pet_type] = {}
            if batch_size not in self.db[pet_type]:
                self.db[pet_type][batch_size] = {}
                    
            self.db[pet_type][batch_size][seq_len] = op_time

    def build_model(self):
        # TODO: 
        self.model = None
        
    def query(self, pet_type, batch_size, seq_len):
        assert pet_type in self.db
        # if pet_type not in self.db:
        #     return 0

        if batch_size in self.db[pet_type]:
            if seq_len in self.db[pet_type][batch_size]:
                return self.db[pet_type][batch_size][seq_len]

        recorded_batches = sorted(list(self.db[pet_type].keys()))
        
        if batch_size > recorded_batches[-1]:
            return 1e10

        # If the batch_size is larger than the max recorded batch, set the latency to INF
        searched_batch = recorded_batches[-1]
        for b in recorded_batches:
            if b >= batch_size:
                searched_batch = b
                break

        recorded_seq_len = sorted(list(self.db[pet_type][searched_batch].keys()))

        searched_seq_len = recorded_seq_len[-1]
        for s in recorded_seq_len:
            if s >= seq_len:
                searched_seq_len = s
                break

        return self.db[pet_type][searched_batch][searched_seq_len]
