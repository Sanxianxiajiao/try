import pickle
import numpy as np
import copy
import time
import torch
from sklearn.ensemble import RandomForestClassifier

import xnas.algorithms.RMINAS.sampler.sampling as sampling

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

class RF_suggest():
    def __init__(self, space, logger, api=None, thres_rate=0.05, batch=1000, seed=10):
        np.random.seed(seed)
        self.sampled_history = []  # list[arch_index] / list[arch.ravel()]
        self.trained_arch = []  # list[dict{'arch':arch, 'loss':loss}]
        self.trained_arch_index = []
        self.thres_rate = thres_rate
        self.loss_thres = 0.
        self.batch = batch
        self.space = space
        self.logger = logger
        self.times_suggest = 0  # without warmup
        if self.space == 'nasbench201':
            self.api = api
            self.max_space = 15625
            self.num_estimator = 30
        elif self.space == 'darts':
            self.num_estimator = 98
        elif self.space == 'mb':
            self.num_estimator = 140
        elif self.space == 'nasbenchmacro':
            self.api = api
            self.max_space = int(3**8)
            self.num_estimator = 30
            self.spaces = list(api.keys())
        elif self.space == 'proxyless':
            self.num_estimator = 100
        elif self.space == 'mobilenetV3':
            self.num_estimator = 100
        self.model = RandomForestClassifier(n_estimators=self.num_estimator,random_state=seed)
    
    def _update_lossthres(self):
        losses = [i['loss'] for i in self.trained_arch]
#         losses_wo_inf = []
#         for i in losses:
#             if not np.isinf(i):
#                 losses_wo_inf.append(i)
        self.loss_thres = np.quantile(losses, self.thres_rate) + 1e-9
        self.logger.info("CKA loss_thres: {}".format(self.loss_thres))
        good_arch = (np.array(losses) < self.loss_thres).tolist()
        assert np.sum(good_arch) > 1, "no enough good architectures"
    
    def _index2arch_nb201(self, index):
        assert self.space == 'nasbench201', 'api dismatch'
        _arch_str = self.api.arch(index)
        _arch_arr = sampling.nb201genostr2array(_arch_str)
        return _arch_arr
    
    def _trainedarch2xy(self):
        features = []
        labels = []
        for i in self.trained_arch:
            features.append(i['arch'].ravel())
            labels.append(i['loss'] < self.loss_thres if self.loss_thres else False)
        return features, labels
            
    def warmup_samples(self, num_warmup):
        if self.space == 'nasbench201':
            sampled = list(np.random.choice(self.max_space, size=num_warmup, replace=False))
            self.sampled_history = copy.deepcopy(sampled)
            return sampled
        elif self.space == 'darts':
            return [self._single_sample() for _ in range(num_warmup)]
        elif self.space == 'mb':
            return [self._single_sample() for _ in range(num_warmup)]
        elif self.space == 'nasbenchmacro':
            return [self._single_sample() for _ in range(num_warmup)]
        elif self.space == 'proxyless':
            return [self._single_sample() for _ in range(num_warmup)]
        elif self.space == 'mobilenetV3':
            return [self._single_sample() for _ in range(num_warmup)]
    
    def _single_sample(self, unique=True):
        if self.space == 'nasbench201':
            assert len(self.sampled_history) < self.max_space, "error: oversampled"
            while True:
                sample = np.random.randint(self.max_space)
                if sample not in self.sampled_history:
                    self.sampled_history.append(sample)
                    return sample
        elif self.space == 'darts':
            if unique:
                while True:
                    sample = np.zeros((14, 7)) # 14边，7op
                    node_ids = np.asarray([np.random.choice(range(x,x+i+2), size=2, replace=False) for i, x in enumerate((0,2,5,9))]).ravel() # 选择哪8个边
                    op = np.random.multinomial(1,[1/7.]*7, size=8) # 8条选择的边、7个有意义op
                    sample[node_ids] = op
                    if str(sample) not in self.sampled_history:
                        self.sampled_history.append(str(sample))
                        return sample
            else:
                sample = np.zeros((14, 7)) # 14边，7op
                node_ids = np.asarray([np.random.choice(range(x,x+i+2), size=2, replace=False) for i, x in enumerate((0,2,5,9))]).ravel() # 选择哪8个边
                op = np.random.multinomial(1,[1/7.]*7, size=8) # 8条选择的边、7个有意义op
                sample[node_ids] = op
                return sample
        elif self.space == 'mb':
            if unique:
                while True:
                    c = np.zeros((20, 7))
                    for i in range(20):
                        j = np.random.randint(7)
                        c[i, j] = True
                    if str(c) not in self.sampled_history:
                        self.sampled_history.append(str(c))
                        return c
            else:
                c = np.zeros((20, 7))
                for i in range(20):
                    j = np.random.randint(7)
                    c[i, j] = True
                return c
        elif self.space == 'nasbenchmacro':
            if unique:
                while True:
                    numeric_choice = np.random.randint(3,size=8)
                    if str(numeric_choice) not in self.sampled_history:
                        self.sampled_history.append(str(numeric_choice))
                        return numeric_choice
            else:
                numeric_choice = np.random.randint(3,size=8)
                return numeric_choice
        elif self.space == 'proxyless':
            def gen_sample():
                depth = np.array(np.random.randint(1, 4+1, size=5).tolist() + [1])
                anchors = depth+[0,4,8,12,16,20]
                ks = np.random.choice([3,5,7], size=21)
                expand_ratios = np.random.choice([3,6], size=21)
                ed = 4
                for anchor in anchors:
                    ks[anchor:ed] = 0
                    expand_ratios[anchor:ed] = 0
                    ed += 4
                sample = np.concatenate([depth, ks, expand_ratios])
                return sample
            if unique:
                while True:
                    sample = gen_sample()
                    if sample.tobytes() not in self.sampled_history:
                        self.sampled_history.append(sample.tobytes())
                        return sample
        elif self.space == 'mobilenetV3':
            def gen_sample():
                block_num = 5
                max_depth = 4
                depth = np.array(np.random.choice([2,3,max_depth], size=block_num).tolist())
                anchors = depth+np.arange(0, max_depth*block_num, max_depth)
                ks = np.random.choice([3,5,7], size=max_depth*block_num)
                expand_ratios = np.random.choice([3,4,6], size=max_depth*block_num)
                ed = max_depth
                for anchor in anchors:
                    ks[anchor:ed] = 0
                    expand_ratios[anchor:ed] = 0
                    ed += max_depth
                sample = np.concatenate([depth, ks, expand_ratios])
                return sample
            if unique:
                while True:
                    sample = gen_sample()
                    if sample.tobytes() not in self.sampled_history:
                        self.sampled_history.append(sample.tobytes())
                        return sample
            else:
                sample = gen_sample()
                return sample
                

    def Warmup(self):
        self._update_lossthres()
        features, labels = self._trainedarch2xy()
        self.model.fit(np.asarray(features, dtype='float'), np.asarray(labels, dtype='float'))
    
    def fitting_samples(self):
        self.times_suggest += 1
        start_time = time.time()
        if self.space == 'nasbench201':
            _sample_indexes = np.random.choice(self.max_space, size=self.batch, replace=False)
            _sample_archs = []
            _sample_archs_idx = []
            for i in _sample_indexes:
                if i not in self.trained_arch_index:
                    _sample_archs.append(self._index2arch_nb201(i).ravel())
                    _sample_archs_idx.append(i)
#             print("sample {} archs/batch, cost time: {}".format(len(_sample_archs), time.time()-start_time))
            _sample_archs = np.array(_sample_archs)
            best_id = np.argmax(self.model.predict_proba(_sample_archs)[:,1])
            best_arch_id = _sample_archs_idx[best_id]
            return best_arch_id
        elif self.space == 'darts':
            _sample_batch = np.array([self._single_sample(unique=False).ravel() for _ in range(self.batch)])
            _tmp_trained_arch = [str(i['arch'].ravel()) for i in self.trained_arch]
            _sample_archs = []
            for i in _sample_batch:
                if str(i) not in _tmp_trained_arch:
                    _sample_archs.append(i)
#             print("sample {} archs/batch, cost time: {}".format(len(_sample_archs), time.time()-start_time))
            best_id = np.argmax(self.model.predict_proba(_sample_archs)[:,1])
            best_arch = _sample_archs[best_id].reshape((14, 7))
            return best_arch
        elif self.space == 'mb':
            _sample_batch = np.array([self._single_sample(unique=False).ravel() for _ in range(self.batch)])
            _tmp_trained_arch = [str(i['arch'].ravel()) for i in self.trained_arch]
            _sample_archs = []
            for i in _sample_batch:
                if str(i) not in _tmp_trained_arch:
                    _sample_archs.append(i)
#             print("sample {} archs/batch, cost time: {}".format(len(_sample_archs), time.time()-start_time))
            best_id = np.argmax(self.model.predict_proba(_sample_archs)[:,1])
            best_arch = _sample_archs[best_id].reshape((20, 7))
            return best_arch
        elif self.space == 'nasbenchmacro':
            _sample_indexes = np.random.choice(self.max_space, size=self.batch, replace=False)
            chace_table = set(str(i['arch'].ravel()) for i in self.trained_arch)
            _sample_archs = []
            for i in _sample_indexes:
                if self.spaces[i] not in chace_table:
                    _sample_archs.append(np.array(list(self.spaces[i])).astype(int))
        elif self.space == 'proxyless':
            _sample_batch = np.array([self._single_sample(unique=False).ravel() for _ in range(self.batch)])
            _tmp_trained_arch = [(i['arch'].tobytes()) for i in self.trained_arch]
            _sample_archs = []
            for i in _sample_batch:
                if (i).tobytes() not in _tmp_trained_arch:
                    _sample_archs.append(i)
#             print("sample {} archs/batch, cost time: {}".format(len(_sample_archs), time.time()-start_time))
            best_id = np.argmax(self.model.predict_proba(_sample_archs)[:,1])
            best_arch = _sample_archs[best_id]
            return best_arch
        elif self.space == 'mobilenetV3':
            _sample_batch = np.array([self._single_sample(unique=False).ravel() for _ in range(self.batch)])
            _tmp_trained_arch = [(i['arch'].tobytes()) for i in self.trained_arch]
            _sample_archs = []
            for i in _sample_batch:
                if (i).tobytes() not in _tmp_trained_arch:
                    _sample_archs.append(i)
#             print("sample {} archs/batch, cost time: {}".format(len(_sample_archs), time.time()-start_time))
            best_id = np.argmax(self.model.predict_proba(_sample_archs)[:,1])
            best_arch = _sample_archs[best_id]
            return best_arch

            
    def Fitting(self):
        # Called after adding data into trained_arch list.
        loss = self.trained_arch[-1]['loss']
        features, labels = self._trainedarch2xy()
        self.model.fit(np.asarray(features, dtype='float'), np.asarray(labels, dtype='float'))
        return loss < self.loss_thres if self.loss_thres else False
        
    def optimal_arch(self, method, top=300, use_softmax=True):
        assert method in ['sum', 'greedy'], 'method error.'
#         with open('RF_sampling.pkl', 'wb') as f:
#             pickle.dump((self.loss_thres, self.trained_arch, self.sampled_history), f)
            
        self.logger.info("#times suggest: {}".format(self.times_suggest))
        _tmp_trained_arch = [i['arch'].ravel() for i in self.trained_arch]
#         self.logger.info("Unique archs {} in total archs {}".format(len(np.unique(_tmp_trained_arch, axis=0)), len(self.trained_arch)))
        estimate_archs_tmp = []
        for i in self.trained_arch:
            if (i['loss'] < self.loss_thres if self.loss_thres else False):
                estimate_archs_tmp.append(i)
        
        self.logger.info("#arch < CKA loss_thres: {}".format(len(estimate_archs_tmp)))

        _est_archs_sort = sorted(estimate_archs_tmp, key=lambda d: d['loss']) 
        
        estimate_archs = []
        if top>len(_est_archs_sort):
            self.logger.info('top>all, using all archs.')
        for i in range(min(top, len(_est_archs_sort))):
            estimate_archs.append(_est_archs_sort[i]['arch'])
        
        if self.space == 'nasbench201':
            result = []
            if method == 'sum':
                all_sum = estimate_archs[0]
                for i in estimate_archs[1:]:
                    all_sum = np.add(all_sum, i)
                # print(all_sum)
                sum_max = list(np.argmax(all_sum, axis=1))
                result = copy.deepcopy(sum_max)
            
            elif method == 'greedy':
                path_info =[[[0 for _ in range(5)] for _ in range(5)] for _ in range(6)]
                
                for i in estimate_archs:
                    for j in range(1, 6):
                        path_info[j][np.argmax(i[j-1])][np.argmax(i[j])] += 1
                
                _esti_arch_0 = [0]*5
                for i in estimate_archs:
                    _esti_arch_0 = np.add(i[0], _esti_arch_0)

                startindex = np.argmax(_esti_arch_0)
                path_max = [startindex]
                for i in range(1, 6):
                #     path_max.append(np.argmax(path_info[i][path_max[i-1]]))
                    # one more step
                    max_op_sum = np.max(path_info[i][path_max[i-1]])
                    _tmp_max_idx = []
                    for j in range(5):
                        if path_info[i][path_max[i-1]][j] == max_op_sum:
                            _tmp_max_idx.append(j)
                    if len(_tmp_max_idx) == 1 or i==5:
                        path_max.append(np.argmax(path_info[i][path_max[i-1]]))
                    else:
                        _next_step = np.array([np.sum(path_info[i+1][j]) for j in _tmp_max_idx])
                        _chosen_op = _tmp_max_idx[np.argmax(_next_step)]
                        path_max.append(_chosen_op)
                self.logger.info("path info:\n{}".format(str(path_info)))
                result = copy.deepcopy(path_max)
            _tmp_np = np.array(result)
            op_arr = np.zeros((_tmp_np.size, 5))
            op_arr[np.arange(_tmp_np.size),_tmp_np] = 1
            return op_arr
        elif self.space == 'darts':
            assert method == 'sum', 'only sum is supported in darts.'
            all_sum = estimate_archs[0]
            for i in estimate_archs[1:]:
                all_sum = np.add(all_sum, i)
            if use_softmax:
                all_sum = softmax(all_sum)
            sum_max = np.argmax(all_sum, axis=1)
            start_index = 0
            end_index = 0
            for i in range(2, 6):
                end_index += i
                _, top_index = torch.topk(torch.from_numpy(sum_max[start_index:end_index]), 2)
                mask = list(set(range(i)) - set(list(top_index.numpy())))
                for j in mask:
                    sum_max[start_index+j] = 7
                start_index = end_index
#             print(sum_max)
            _tmp_np = np.array(sum_max)
            op_arr = np.zeros((_tmp_np.size, 8))
            op_arr[np.arange(_tmp_np.size),_tmp_np] = 1
            return op_arr
        elif self.space == 'mb':
            assert method == 'sum', 'only sum is supported in mb.'
            all_sum = estimate_archs[0]
            for i in estimate_archs[1:]:
                all_sum = np.add(all_sum, i)
            print(all_sum)
            if use_softmax:
                all_sum = softmax(all_sum)
            sum_max = np.argmax(all_sum, axis=1)
            print(sum_max)
            _tmp_np = np.array(sum_max)
            op_arr = np.zeros((_tmp_np.size, 7))
            op_arr[np.arange(_tmp_np.size),_tmp_np] = 1
            return op_arr
        elif self.space == 'nasbenchmacro':
            assert method == 'sum', 'only sum is supported in mb.'
            estimate_archs = np.eye(3)[estimate_archs]
            all_sum = estimate_archs.sum(0)
            
            print(all_sum)
            if use_softmax:
                all_sum = softmax(all_sum)
            sum_max = np.argmax(all_sum, axis=1)
            print(sum_max)
            _tmp_np = np.array(sum_max)
            op_arr = np.zeros((_tmp_np.size, 3))
            op_arr[np.arange(_tmp_np.size),_tmp_np] = 1
            return op_arr.argmax(-1)
        elif self.space == 'proxyless':
            assert method == 'sum', 'only sum is supported in mb.'
            estimate_archs = np.array(estimate_archs)
            depth = estimate_archs[:, :6] # 1,2,3,4
            best_depth = np.eye(4)[depth-1].sum(0).argmax(-1)+1
            ks = estimate_archs[:, 6:27]//2 # {3, 5, 7}
            best_ks = np.eye(3)[ks-1].sum(0).argmax(-1) * 2 + 3
            er = estimate_archs[:, 27:]//3 # {3, 6}
            best_er = np.eye(2)[er-1].sum(0).agrmax(-1) * 3 + 3
            ed = 4
            anchors = best_depth+[0,4,8,12,16,20]
            for anchor in anchors:
                best_ks[anchor:ed] = 0
                best_er[anchor:ed] = 0
                ed += 4
            sample = np.concatenate([best_depth, best_ks, best_er])
            return sample
        elif self.space == 'mobilenetV3':
            with open('./tmp.pkl', 'wb') as f:
                pickle.dump(estimate_archs, f)
            estimate_archs = np.array(estimate_archs)
            assert method == 'sum', 'only sum is supported in mb.'
            depth = estimate_archs[:, :5] # 2,3,4
            best_depth = np.eye(3)[depth-2].sum(0).argmax(-1) + 2
            ks = estimate_archs[:, 5:25]//2 # {3, 5, 7}
            best_ks = np.eye(3)[ks-1].sum(0).argmax(-1) * 2 + 3
            er = estimate_archs[:, 25:]-3 # {3, 4, 6}
            best_er = np.eye(4)[er].sum(0).argmax(-1) + 3
            max_depth = 4
            block_num = 5
            anchors = best_depth+np.arange(0, max_depth*block_num, max_depth)
            ed = max_depth
            for anchor in anchors:
                best_ks[anchor:ed] = 0
                best_er[anchor:ed] = 0
                ed += max_depth
            sample = np.concatenate([best_depth, best_ks, best_er])
            return sample
print(1)