
from numpy import Inf
import torch
from perf_model.pet_perf_model import AlphaModel, BetaModel

class PET_Scheduler:
    def __init__(self, query_pool,
                 fifo=None,
                 vocab_size=None,
                 sort_queries=False,
                 test_device=torch.device('cuda:0'),
                 alpha_table_path = "perf_model/alpha_table_1080ti.dat",
                 beta_table_path = "perf_model/beta_table_1080ti.dat",
                 ) -> None:
        self.query_pool = query_pool
        self.fifo = fifo
        self.vocab_size = vocab_size
        self.sort_queries = sort_queries
        self.test_device = test_device
        self.batch_size = 32
        self.alpha_model = AlphaModel(alpha_table_path)
        self.beta_model = BetaModel(beta_table_path)
        self.pet_type_map = {}
        self.get_pet_type_map()
    def get_pet_type_map(self):
        for query in self.query_pool:
            task_id, _, pet_type = query
            if task_id not in self.pet_type_map.keys():
                self.pet_type_map[task_id] = pet_type
            else:
                continue

    def take_second(self, elem):
        return elem[1]

    def alpha(self, batch_size, seq_len):
        return self.alpha_model.query(batch_size, seq_len)

    def beta(self, pet_type, batch_size, seq_len):
        return self.beta_model.query(pet_type, batch_size, seq_len)

    def create_scheduled_batch(self, task_ids, max_len):
        # FIXME: fetch input_ids from fifo. 

        assert(len(task_ids) >= 1)
        assert(max_len >= 1)

        input_ids = torch.randint(low=0,
                                  high=self.vocab_size - 1,
                                  size=(len(task_ids), max_len),
                                  dtype=torch.long,
                                  device=self.test_device)
        
        # Merge the same task.
        different_tasks = set(task_ids)
        n_samples = []
        for task in different_tasks:
            n_samples.append(task_ids.count(task))
        
        task_ids = torch.LongTensor(list(different_tasks))
        n_samples = torch.LongTensor(n_samples)

        batch = [input_ids, task_ids, n_samples]
        return batch
    
    def create_scheduled_mini_batch(self, macro_batch, max_len):
        # FIXME: fetch input_ids from fifo. 
        total_samples = 0
        n_samples = []
        task_ids = []
        mini_batch_lens = []
        for mini_batch, mini_batch_len in macro_batch:
            n_samples.append(len(mini_batch))
            assert(len(set(mini_batch)) == 1)
            task_ids.append(mini_batch[0])
            total_samples += len(mini_batch)
            mini_batch_lens.append(mini_batch_len)

        assert(max_len >= 1)
        assert(total_samples >= 1)

        input_ids = torch.randint(low=0,
                                  high=self.vocab_size - 1,
                                  size=(total_samples, max_len),
                                  dtype=torch.long,
                                  device=self.test_device)
        
 
        task_ids = torch.LongTensor(list(task_ids))
        n_samples = torch.LongTensor(n_samples)
        mini_batch_lens = torch.LongTensor(mini_batch_lens)
        batch = [input_ids, task_ids, n_samples, mini_batch_lens]
        return batch
    

    def batch_schedule(self, bs = 32):
        scheduled_batches = []
        
        # Fix sized batch baseline.
        max_batch = bs 

        # Sort the queries.
        if self.sort_queries:
            self.query_pool = sorted(self.query_pool, key=self.take_second) 

        for i in range(len(self.query_pool) // max_batch):
            task_ids = []
            max_len = 0
            for j in range(max_batch):
                task_id, seq_len, pet_type = self.query_pool[i * max_batch + j]
                max_len = max(seq_len, max_len)
                task_ids.append(task_id)

            scheduled_batches.append(self.create_scheduled_batch(task_ids, max_len))
            
        return scheduled_batches

    # Intra-task batching.
    def intra_task_batching(self, query_pool):
        ## Preprocessing: gather the queries with the same task_id.
        clustered_queries_by_task_id = {}
        for query in query_pool:
            task_id, seq_len, pet_type = query
            if task_id in clustered_queries_by_task_id:
                clustered_queries_by_task_id[task_id].append(query)
            else:
                clustered_queries_by_task_id[task_id] = [query]

        ## DP
        mini_batches = []
        for task_id, queries in clustered_queries_by_task_id.items():
            state_1st_stage = []
            split_idx_list = []

            ### Sort queries according to the sequence length in ascending order.
            queries = sorted(queries, key=self.take_second)
            queries.insert(0, None)  # Sentinel.

            ### Initialize.
            state_1st_stage.append(0)
            split_idx_list.append(0)
            for j in range(1, len(queries)):
                min_cost = Inf  # INF
                split_idx = 0
                for k in range(1, j+1):
                    tmp = state_1st_stage[k-1] + self.beta(queries[j][2], j-k+1, queries[j][1])
                    if tmp < min_cost:
                        min_cost = tmp
                        split_idx = k-1
                split_idx_list.append(split_idx)
                state_1st_stage.append(min_cost)
                
            ### Split queries into mini-batches according to split_idx_list.
            
            end_idx = len(queries) - 1

            while(end_idx > 0):
                start_idx = split_idx_list[end_idx] + 1
                mini_batch = []
                max_len = queries[end_idx][1]
                for j in range(start_idx, end_idx + 1):
                    mini_batch.append(queries[j][0])               
                mini_batches.append((mini_batch, max_len))
                end_idx = split_idx_list[end_idx]        
        return mini_batches

    # Inter-task batching.
    def inter_task_batching(self, mini_batches):
        ## Sort mini_batches according to the max sequence length.
        mini_batches = sorted(mini_batches, key=self.take_second)
        mini_batches.insert(0, None)  # Sentinel.

        tmp = 0
        mini_batch_sum = [0]
        for mini_batch in mini_batches[1:]:
            tmp += len(mini_batch[0])
            mini_batch_sum.append(tmp)

        ## DP.
        state_2nd_stage = []
        split_idx_list = []
        state_2nd_stage.append(0)
        split_idx_list.append(0)

        for i in range(1, len(mini_batches)):
            min_cost = Inf  # INF
            split_idx = 0
            for j in range(1, i+1):
                total_samples = mini_batch_sum[i] - mini_batch_sum[j-1]
                tmp = state_2nd_stage[j-1] + self.alpha(total_samples, mini_batches[i][1])
                if  tmp < min_cost:
                    min_cost = tmp
                    split_idx = j - 1
            split_idx_list.append(split_idx)
            state_2nd_stage.append(min_cost)

        ## Split mini_batches into final scheduled_batches.
        ### Split mini_batches into macro_batches.

        end_idx = len(mini_batches) - 1
        macro_batches = []

        while(end_idx > 0):
            start_idx = split_idx_list[end_idx] + 1
            macro_batch = []
            max_len = mini_batches[end_idx][1]
            for j in range(start_idx, end_idx + 1):
                macro_batch.append(mini_batches[j])               
            macro_batches.append((macro_batch, max_len))
            end_idx = split_idx_list[end_idx]        

        total_samples = 0
        for macro_batch in macro_batches:
             for mini_batch in macro_batch[0]:
                 total_samples += len(mini_batch[0])
        # print(total_samples)

        return macro_batches

    def coordinate_schedule(self, stage = 3):
        scheduled_batches = []
        
        if 1 == stage:
            mini_batches = self.intra_task_batching(self.query_pool)
            
            for mini_batch, max_len in mini_batches:
                task_ids = []
                for task_id in mini_batch:
                    task_ids.append(task_id)
                    
                scheduled_batches.append(self.create_scheduled_batch(task_ids, max_len))
                
        elif 2 == stage:
            mini_batches = []
            # A mini_batch contains a single query.
            for task_id, seq_len, pet_type in self.query_pool:
                mini_batch = []
                mini_batch.append(task_id)
                mini_batches.append((mini_batch, seq_len))

            macro_batches = self.inter_task_batching(mini_batches)

            for macro_batch, max_len in macro_batches:
                task_ids = []
                for mini_batch, mini_batch_len in macro_batch:
                    for task_id in mini_batch:
                        task_ids.append(task_id)
                        
                scheduled_batches.append(self.create_scheduled_batch(task_ids, max_len))
        elif  3 == stage:
            mini_batches = self.intra_task_batching(self.query_pool)
            total_samples = 0

            for mini_batch in mini_batches:
                total_samples += len(mini_batch[0])
            # print(total_samples)

            macro_batches = self.inter_task_batching(mini_batches)

            # FIXME: the following code snippet to create scheduled_batches is
            # exactly same as that in elif.
            for macro_batch, max_len in macro_batches:
                scheduled_batches.append(self.create_scheduled_mini_batch(macro_batch, max_len))

        elif 4 == stage:

            mini_batches = []

            for task_id, seq_len, pet_type in self.query_pool:
                mini_batch = []
                mini_batch.append(task_id)
                mini_batches.append((mini_batch, seq_len))

            macro_batches = self.inter_task_batching(mini_batches)
            for macro_batch, max_len in macro_batches:
                queries = []
                
                for mini_batch, mini_batch_len in macro_batch:
                    for task_id in mini_batch:
                        pet_type = self.pet_type_map[task_id]
                        queries.append((task_id, mini_batch_len, pet_type))
                mini_batches = self.intra_task_batching(queries)
                # macro_batch = (mini_batches, max_len)
                scheduled_batches.append(self.create_scheduled_mini_batch(mini_batches, max_len))
        return scheduled_batches
        
