import torch
import numpy as np
from torch.utils.data import Dataset
import anndata as ad
import mudata as md
from sklearn import preprocessing

from omicsdgd.functions._data_manipulation import get_column_name_from_unique_values

class omicsDataset(Dataset):
    '''
    General Dataset class for single cell data.
    Applicable for sinlge modalities and multi-modal data.

    Attributes
    ----------
    data: torch.Tensor
        tensor with raw counts of shape (n_samples, n_features)
    scaling_type: string
        variable defining how to calculate scaling factors
    n_sample: int
        number of samples in dataset
    n_features: int
        number of total features in dataset
    library: torch.Tensor
        tensor of per-sample and -modality scaling factors

    Methods
    ----------
    get_labels(idx=None)
        Return sample-specific values of monitored clustering feature (if available)
    get_correction_labels(corr_id, idx=None)
        Return values of numerically transformed correction features per sample
    '''
    def __init__(self,
            data,
            switch=None, 
            scaling_type='sum', 
            meta_label=None, 
            correction=None, 
            modalities=None,
            label='train'
        ):
        '''
        Args: 
            data: mudata or anndata object
            switch: None or integer 
                position at which modalities change in total feature list
            scaling_type: string 
                for different scaling options (currently only supporting 'sum')
            meta_label: None or string
                observable (column name) of which to monitor clustering performance
            correction: None, string or list of strings
                observable (column name(s)) of what features to correct for or disentangle
            modalities: None, string or list of strings
                input depends on data type. Not necessary for mudata objects or unimodal anndata objects.
                For multi-modal anndata objects should be the column name of the var dataframe describing the modalities of each feature.       
        '''

        # first make sure that the raw counts are used
        self._assert_raw_counts(data)

        # the scaling type determines what makes the scaling factors of each sample
        # it is accessed by the loss function
        self.scaling_type = scaling_type

        # get modality name(s), position(s) at which modalities switch in full tensor, and number of features in each modality
        self.modalities, self.modality_switch, self.modality_features = self._get_modality_names(data,modalities,switch)
        print("check: modality names are ", self.modalities)

        # added support for mosaic data
        # if there is no data.obs['modality'] column, then we assume that the data is unimodal or only paired
        self.mosaic = self._check_if_mosaic(data)
        self.mosaic_mask = None
        self.mosaic_train_idx = None
        self.modality_mask = None
        if self.mosaic and label == 'train':
            # if the train set is mosaic, we use 10% of the paired data to minimize distances between modalities
            # for this we need to copy those samples, add them with each modality option (paired, GEX, ATAC)
            # and structure them in such a way that we can use the representation distances in the loss
            #data, data_triangle, self.modality_mask, self.modality_mask_triangle, self.mosaic_train_idx = self._make_mosaic_train_set(data)
            self.modality_mask = self._get_mosaic_mask(data)
            #self.data_triangle = torch.Tensor(data_triangle.X.todense())
        elif self.mosaic and label == 'test':
            self.modality_mask = self._get_mosaic_mask(data)

        # make 1 tensor out of all modalities with shape (n_obs,n_featues)
        #self.data = self._data_to_tensor(data)
        
        # make shape attributes
        #self.n_sample = self.data.shape[0]
        #self.n_features = self.data.shape[1]
        self.n_sample = data.shape[0]
        self.n_features = data.shape[1]

        # get meta data (feature for clustering) and correction factors (if applicable)
        self.meta, self.correction, self.correction_classes = self._init_meta_and_correction(data, meta_label, correction)

        self.correction_labels = None
        if self.correction is not None:
            self.correction_labels = self._init_correction_labels_numerical()

        # compute the scaling factors for each sample based on scaling type
        #self.library = self._get_library()
        self.data = data
        self.library = None

    def __len__(self):
        '''Return number of samples in dataset'''
        return(self.data.shape[0])

    def __getitem__(self, idx):
        '''Return data, scaling factors and index per sample (evoked by dataloader)'''
        expression = self.data[idx]
        lib = self.library[idx]
        return expression, lib, idx
    
    def __str__(self):
        return f"""
        omicsDataset:
            Number of samples: {self.n_sample}
            Modalities: {self.modalities}
            Features per modality: {self.modality_features}
            Total number of features: {self.n_features}
            Scaling of values: {self.scaling_type}
        """
    
    def __eq__(self, other):
        '''Check if two instances are the same'''
        is_eq = False
        if (self.data == other.data) and (self.library == other.library):
            is_eq = True
        return is_eq
    
    def get_labels(self, idx=None):
        '''Return sample-specific values of monitored clustering feature (if available)'''
        if self.meta is not None:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            if idx is None:
                return np.asarray(np.array(self.meta))
            else:
                return np.asarray(np.array(self.meta)[idx])
        else:
            print('tyring to access meta labels, but none was given in data initialization.')
            return None
    
    def _get_library(self):
        '''Create tensor of scaling factors of shape (n_samples,n_modalities)'''
        if self.modality_switch is not None:
            library = torch.cat((torch.sum(self.data[:,:self.modality_switch], dim=-1).unsqueeze(1),torch.sum(self.data[:,self.modality_switch:], dim=-1).unsqueeze(1)),dim=1)
        else:
            library = torch.sum(self.data, dim=-1).unsqueeze(1)
        return library
    
    def _get_modality_names(self, data, modalities, switch):
        '''
        Get the types of modalities of the data.
        In the future this can be important if e.g. 
        including protein abundance for which we need different output distributions.

        This also returns the positions in the data tensor where modalities switch
        and the number of features per modality.
        '''
        if isinstance(data, md.MuData):
            modalities = list(data.mod.keys())
            return modalities, int(data[modalities[0]].shape[1]), [int(data[mod].shape[1]) for mod in modalities]
        elif isinstance(data, ad.AnnData):
            # let's make the rule that if people want to use a multi-modal anndata object, they have to provide the modalities column name as modalities
            # otherwhise I treat the data as unimodal
            if modalities is not None:
                if type(modalities) == list:
                    if len(modalities) > 1: # means that it is not the column name but the names of the modalities
                        modality_name = get_column_name_from_unique_values(data.var, modalities)
                else:
                    modality_name = modalities
                    modalities = list(data.var[modality_name].unique())
                if len(modalities) > 1:
                    switch = np.where(data.var[modality_name] == modalities[1])[0][0]
                    modality_features = [int(np.where(data.var[modality_name] == modalities[1])[0][0]), int((data.shape[1]-np.where(data.var[modality_name] == modalities[1])[0][0]))]
                else:
                    modality_features = [int(data.shape[1])]
            else:
                modality_features = [int(data.shape[1])]
            return modalities, switch, modality_features
        else:
            # can I just remove this?
            if modalities is not None:
                if isinstance(modalities, list):
                    if switch is not None:
                        return modalities, switch, [switch, data.shape[1]-switch]
                    else:
                        raise ValueError('supplied multi-modal data but no modality switch')
                elif isinstance(modalities, str):
                    return [modalities], None, data.shape
            else:
                error_message = 'combination of data format and modality not compatible\n'
                error_message +=  'either provide a moality name or provide data in AnnData or MuData format'
                raise ValueError(error_message)
    
    #def _data_to_tensor(self, data):
    def data_to_tensor(self):
        '''
        Make a tensor out of data. In multi-modal cases, modalities are concatenated.
        This only works with mudata and anndata objects.
        '''
        if isinstance(self.data, md.MuData):
            self.data = torch.cat(tuple([torch.Tensor(self.data[x].X.todense()) for x in self.modalities]), dim=1)
            #return torch.cat(tuple([torch.Tensor(data[x].X.todense()) for x in self.modalities]), dim=1)
        elif isinstance(self.data, ad.AnnData):
            self.data = torch.Tensor(self.data.X.todense())
            #return torch.Tensor(data.X.todense())
        else:
            raise ValueError('unsupported data type supported. please check documentation for further information.')
        self.library = self._get_library()
    
    def _init_meta_and_correction(self, data, meta_label, correction):
        '''
        Depending on the user's input, the model may need to disentangle certain
        correction factors (``correction``) in the representation and monitor the clustering performance
        given a specific feature (``meta_label``). 
        These features are given as the data objects observation column names.
        
        This method initializes corresponding attributes accordingly.
        The correction feature can also be a list of column names to accomodate multiple correction factors.
        '''
        # get sample-wise values of the clustering feature
        try:
            if isinstance(data, md.MuData) or isinstance(data, ad.AnnData):
                meta = data.obs[meta_label].values
            else:
                meta = data[meta_label].values
        except:
            meta = None
            print('no (or incorrect) meta label provided. Monitoring clustering will not be possible.')
        
        # get sample-wise values of the correction features and the number of classes per feature
        correction_features = None
        n_correction_classes = None
        if correction is not None:
            if isinstance(data, md.MuData) or isinstance(data, ad.AnnData):
                correction_features = data.obs[correction].values
            else:
                correction_features = data[correction].values
            #n_correction_classes = [len(correction_features[correction[corr_id]].unique()) for corr_id in range(len(correction))]
            #n_correction_classes = len(correction_features.unique())
            if type(correction_features) is np.ndarray:
                if len(correction_features.shape) > 1:
                    correction_features = correction_features.flatten()
                n_correction_classes = len(list(np.unique(correction_features)))
            else:
                n_correction_classes = len(correction_features.unique())
        
        return meta, correction_features, n_correction_classes
    
    def _init_correction_labels_numerical(self):
        '''
        Transforms correction features into numerical variables for supervised training (clustering performance).
        '''
        #correction_numerical = torch.zeros((self.n_sample, len(self.correction_classes)))
        #for corr_id in range(len(self.correction_classes)):
        #    le = preprocessing.LabelEncoder()
        #    le.fit(self.correction.iloc[:,corr_id].values)
        #    correction_numerical[:,corr_id] = torch.tensor(le.transform(self.correction.iloc[:,corr_id].values))
        le = preprocessing.LabelEncoder()
        le.fit(self.correction)
        correction_numerical = torch.tensor(le.transform(self.correction))
        return correction_numerical
    
    #def get_correction_labels(self, corr_id, idx=None):
    def get_correction_labels(self, idx=None):
        '''
        Return values of numerically transformed correction features per sample
        '''
        if idx is None:
            return self.correction_labels.tolist()
        else:
            return self.correction_labels[idx].tolist()
    
    @staticmethod
    def _assert_raw_counts(data):
        '''
        Rough check whether the counts look like raw counts.
        Looking for zero values since log transformation would not have any.
        But needs to be expanded to cover other normalizations.
        '''
        is_raw = True
        if isinstance(data, md.MuData):
            for mod_id in range(len(list(data.mod.keys()))):
                if data[list(data.mod.keys())[mod_id]].X.todense().min() > 0:
                    is_raw = False
        elif isinstance(data, ad.AnnData):
            if data.X.todense().min() > 0:
                is_raw = False
        else:
            if data.min() > 0:
                is_raw = False
        if not is_raw:
            error_message = 'trying to use data in .X as counts, but they dont seem to be the raw counts. please ensure that they are'
            raise ValueError(error_message)
    
    def _check_if_mosaic(self, data):
        '''Check if data is mosaic data
        that means whether the data has unpaired modalities'''
        mosaic = False
        #if isinstance(data, md.MuData) or # lets worry about mudata later, since there it can be different
        if isinstance(data, ad.AnnData):
            if 'modality' in data.obs.columns:
                if data.obs['modality'].nunique() > 1:
                    mosaic = True
                    print('mosaic data detected')
        return mosaic
    
    """
    def _get_mosaic_mask(self, data):
        '''Return a list of tensors that indicate which samples belong to which modality'''
        if self.mosaic:
            modality_list = data.obs['modality']
            modality_mask = [torch.zeros((data.shape[0])).bool(), torch.zeros((data.shape[0])).bool()]
            mod_name_1 = [x for x in modality_list.unique() if x in ['rna', 'RNA', 'GEX', 'expression']][0]
            mod_name_2 = [x for x in modality_list.unique() if x in ['atac', 'ATAC', 'accessibility']][0]
            modality_mask[0][modality_list == mod_name_1] = True
            modality_mask[1][modality_list == mod_name_2] = True
        return modality_mask
    
    def _make_mosaic_train_set(self, data, split=10):
        '''For mosaic train sets, take 10% of the paired data,
        artificially unpair it and keep all three sets (paired, unpaired, unpaired)
        structured so that we can take the first representation segments and compute triangle
        sizes for the loss'''
        idx_paired = np.where(data.obs["modality"].values == "paired")[0]
        if len(idx_paired) >= 10: # in case I have a 100% unpaired dataset
            # choose every 10th sample (for reproducibility)
            idx_split = idx_paired[::split]
            n_split = len(idx_split)
            # take selected samples and store them in new data object for each modality
            # then append all the other data to the new data object
            data_1 = data.copy()[idx_split,:]
            data_1_rna = data_1.copy()
            data_1_atac = data_1.copy()
            data_1_rna.obs["modality"] = "GEX"
            data_1_atac.obs["modality"] = "ATAC"
            data_1 = data_1.concatenate(data_1_rna)
            data_1 = data_1.concatenate(data_1_atac)
            #mask = np.ones(data.shape[0], dtype=bool)
            #mask[idx_split] = False
            #data_2 = data.copy()[mask,:]
            idx = torch.Tensor(np.arange(n_split)).byte()
            return data, data_1, self._get_mosaic_mask(data), self._get_mosaic_mask(data_1), idx
        else:
            return data, None, self._get_mosaic_mask(data), None, None
    """

    def _get_mosaic_mask(self, data):
        '''Return a list of tensors that indicate which samples belong to which modality'''
        if self.mosaic:
            print("prepping the mosaic mask")
            modality_list = data.obs['modality']
            modality_mask = [torch.ones((data.shape[0])).bool(), torch.ones((data.shape[0])).bool()]
            # return an error if the modalities are not in the order we expect (later)
            if self.modalities[0] not in ['rna', 'RNA', 'GEX', 'expression']:
                raise ValueError('first modality is not RNA. please make RNA the first and ATAC the second modality')
            mod_name_1 = self.modalities[0]
            mod_name_2 = self.modalities[1]
            modality_mask[0][modality_list == mod_name_2] = False
            modality_mask[1][modality_list == mod_name_1] = False
            # make a test dataframe to check correctness
            import pandas as pd
            df_test = pd.DataFrame({
                'modality': modality_list,
                'rna': modality_mask[0].numpy(),
                'atac': modality_mask[1].numpy()
            })
            print(df_test)
        return modality_mask
    
    def get_mask(self, indices):
        if self.modality_mask is None:
            return None
        else:
            return [x[indices] for x in self.modality_mask]

"""
def sparse_coo_to_tensor(mtrx):
    return torch.FloatTensor(mtrx.todense())

from itertools import chain
def collate_sparse_batches(batch):
    start_time = time.time()
    data_batch, library_batch, idx_batch = zip(*batch)
    #data_batch = torch.stack(list(data_batch), dim=0)
    #data_batch = data_batch.to_dense()
    data_batch = scipy.sparse.vstack(list(data_batch))
    data_batch = sparse_coo_to_tensor(data_batch)
    library_batch = torch.stack(list(library_batch), dim=0)
    idx_batch = list(idx_batch)
    print(data_batch.shape, library_batch.shape, len(idx_batch), (time.time() - start_time), " seconds")
    return data_batch, library_batch, idx_batch
"""