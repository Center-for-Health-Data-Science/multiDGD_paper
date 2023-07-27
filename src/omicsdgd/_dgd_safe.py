import torch
import torch.nn as nn
import numpy as np
import anndata as ad
import mudata as md
import wandb
import yaml

from omicsdgd.dataset import omicsDataset
from omicsdgd.latent import RepresentationLayer
from omicsdgd.latent import GaussianMixture, GaussianMixtureSupervised
from omicsdgd.nn import Decoder
from omicsdgd.functions import train_dgd, set_random_seed, count_parameters, sc_feature_selection

# define device (gpu or cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''Central Module of the package'''

class DGD(nn.Module):
    '''
    This is the main class for the Deep Generative Decoder.
    Given a mudata or anndata object, it creates an instance of the DGD
    with all necessary classes from the data.

    Attributes (non-optional ones)
    ----------
    train_set: omicsDataset
        Dataset object derived from only obligatory input `data`.
        it's properties (shape, modality type, observable feature classes) 
        are used to build remaining instances
    param_dict: dict
        dictionary containing hyperparameters for building model instances and training.
        Initialized with default parameters and updated with optional user input.
    decoder: Decoder
        decoder instance initialized based on desired latent dimensionality
        and data features.
    representation: RepresentationLayer
        learnable representation vectors for the training set
    gmm: GMM
        Gaussian mixture model instance for the distribution over latent space.
        If no other information is given (but received a clustering observable),
        number of components is automatically set to the number of classes.

    Methods
    ----------
    train(n_epochs=500, stop_with='loss', stop_after=10, train_minimum=50)
        training of the model instances (decoder, representation, gmm) for a given
        number of epochs with early stopping. 
        Early stopping can be based on the training (or validation, if applicable) loss
        or on the clustering performance on a desired observable (e.g. cell type).
    save()
        saving the model parameters
    load()
        loading trained model parameters to initialized model
    get_representation()
        access training representations
    predict_new()
        learn representations for new data
    differential_expression()
        perform differential expression analysis on selected groupings of data
    '''

    def __init__(
            self, 
            data, 
            parameter_dictionary=None,
            train_validation_split=None, 
            modalities=['rna'],
            scaling='sum', 
            meta_label='celltype',
            modality_switch=None,
            correction=None,
            developer_mode=False,
            feature_selection=None,
            save_dir='./',
            random_seed = 0,
            model_name='dgd'
        ):
        super().__init__()

        # setting the random seed at the beginning for reproducibility
        set_random_seed(random_seed)

        # setting internal helper attributes
        self._init_internal_attribs(model_name, save_dir, scaling, meta_label, developer_mode)

        # if there is a train-val(-test) split, check that it is in accepted format
        self._check_train_split(train_validation_split)

        if self._developer_mode:
            self._init_wandb_logging(parameter_dictionary)

        ###
        # initializing data
        ###
        # this also checks whether the data is in an acceptable format and initialize
        self._init_data(
            data, 
            modality_switch, 
            scaling, 
            meta_label, 
            correction, 
            modalities, 
            feature_selection)
        
        # update parameter dictionary with optional input and info from data
        self.param_dict = self._init_parameter_dictionary(parameter_dictionary)
        if self._developer_mode:
            wandb.config.update(self.param_dict)

        ###
        # initialize GMM as latent distribution and representations
        ###
        self._init_gmm()
        self._init_representations()
        
        # initialize supervised mixture models and representations for correction factors
        # these could be batch effect or medical features of interest (disease state etc.)
        self._init_correction_models(correction)
        
        # initialize decoder
        self._init_decoder()
        self._get_train_status()

        # save param_dict
        self.save_param_dict_to_yaml()
    
    def _init_internal_attribs(self, model_name, save_dir, scaling, meta_label, developer_mode):
        # define what to name the model and where to save it
        self._model_name = model_name
        self._save_dir = save_dir
        # scaling relationship between model output and data
        self._scaling = scaling
        # optional: features to observe and to correct for
        self._clustering_variable = meta_label
        # developer mode changes from notebook use with progress bar to monitoring jobs with wandb
        self._developer_mode = developer_mode
        # important for checking whether a model is trained or has loaded learned parameters
        self.trained_status = nn.Parameter(torch.zeros(1, dtype=torch.bool), requires_grad=False)
    
    def _check_train_split(self, split):
        '''
        Checks that train-validation split is in correct format
        [[train_indices], [validation_indices]]

        Currently it transforms lists of characters into desired format,
        but could also remove support since most non-ML users won't have a validation set and are capable of providing the correct format.
        '''
        if split is not None:
            if isinstance(split, list):
                if len(split) != 2:
                    if isinstance(split[0], str):
                        import difflib
                        train_string = difflib.get_close_matches('train', list(set(split)))[0]
                        val_string = difflib.get_close_matches('val', list(set(split)))[0]
                        if 'val' not in val_string:
                            val_string = difflib.get_close_matches('test', list(set(split)))[0]
                        print('assuming the following train-validation split: train-',train_string,', validation-',val_string)
                        self.train_val_split = [list(np.where(split.iloc[:,0].values == train_string)[0]), list(np.where(split.iloc[:,0].values == val_string)[0])]
                    raise ValueError('wrong format of train_validation_split submitted. should be a list of length 2: [[train],[val]]')
                else:
                    if len(split[0]) < len(split[1]):
                        # switch order in list because validation must be the smaller set
                        self.train_val_split = [split[1], split[0]]
                    else:
                        self.train_val_split = split
                return
            else:
                error_message = 'unknown type provided for train_validation_split.\n'
                error_message += 'valid types are list or None.\n'
                error_message += 'the list can be in 2 forms:\n'
                error_message += '   [[train_indices],[validation_indices]] or\n'
                error_message += '   1D list of categories ("train", "val"/"validation"/"test")'
                raise ValueError(error_message)
        else:
            self.train_val_split = None
    
    def _init_wandb_logging(self, parameter_dictionary): 
        '''start run if in developer mode (otherwise assumes running in notebook)'''
        try:
            wandb.init(
                project=parameter_dictionary['log_wandb'][1], 
                entity=parameter_dictionary['log_wandb'][0], 
                config=parameter_dictionary)
        except:
            raise ValueError('You are trying to run in developer mode, but seem not to have given the parameter dictionary the `log_wandb` statement.')
        wandb.run.name = self._model_name
    
    def _init_data(self, data, modality_switch, scaling_type, meta_label, correction, modalities, feature_selection):
        '''Internal function to derive datasets and data-dependent parameters from input data'''
        
        print('#######################')
        print('Initialized data')
        print('#######################')
        
        # here there is the option to perform feature selection beforehand,
        # i.e. reducing the number of features depending on the `feature_selection` option
        if feature_selection is not None:
            selection_method, data, modality_switch = self._feature_selection(data, modalities, feature_selection)
            print('selected features based on ',selection_method)
        
        if self.train_val_split is not None:
            self.train_set = omicsDataset(data[self.train_val_split[0]], modality_switch, scaling_type, meta_label, correction, modalities)
        else:
            self.train_set = omicsDataset(data, modality_switch, scaling_type, meta_label, correction, modalities)
        if self.train_val_split is not None:
            self.val_set = omicsDataset(data[self.train_val_split[1]], modality_switch, scaling_type, meta_label, correction, modalities)
        else:
            self.val_set = None
        
        print(self.train_set)
    
    @staticmethod
    def _feature_selection(data, modalities, feature_selection):
        '''returns the selection mode (for logging), the reduced data and the new modality swith position (if applicable)'''
        return sc_feature_selection(data, modalities, feature_selection) 

    def _init_parameter_dictionary(self, init_dict=None):
        '''initialize the parameter dictionary from default and update with optional input'''
        # init parameter dictionary based on defaults
        out_dict = {
            'latent_dimension': 20,
            'n_components': 1,
            'n_hidden': 1,
            'n_hidden_modality': 1,
            'n_units': 100,
            'value_init': 'zero', # options are zero or handed values
            'softball_scale': 2,
            'softball_hardness': 5,
            'sd_sd': 1,
            'dirichlet_a': 1,
            'batch_size': 128,
            'learning_rates': [1e-4, 1e-2, 1e-2],
            'betas': [0.5,0.7],
            'weight_decay': 1e-4,
            'log_wandb': ['username', 'projectname']
        }
        out_dict['sd_mean'] = round((2*out_dict['softball_scale'])/(10*out_dict['n_components']),2)
        # update from optional class input dictionary
        if init_dict is not None:
            for key in init_dict.keys():
                out_dict[key] = init_dict[key]
        # add information gained in data initialization
        out_dict['n_features'] = self.train_set.n_features # number of total features
        out_dict['n_features_per_modality'] = self.train_set.modality_features # list of output features generated depending on number of modalities
        out_dict['modalities'] = self.train_set.modalities
        out_dict['modality_switch'] = self.train_set.modality_switch # updating the modality switch if multi-modal data in AnnData object and switch was not given
        
        # automate selection of number of components
        if self.train_set.meta is not None:
            select_n_components = False
            if init_dict is not None:
                if 'n_components' not in list(init_dict.keys()):
                    select_n_components = True
            else:
                select_n_components = True
            if select_n_components:
                out_dict['n_components'] = int(len(list(set(self.train_set.meta))))
                print('selected ',out_dict['n_components'],' number of Gaussian mixture components based on ',self._clustering_variable)
        
        # overwrite hyperparameters with passed parameter_dictionary (if applicable)
        # it is crucial to update the initial value for the GMM component std if number of component changes from default
        # this will be applied at the end of data initialization
        override_sd_init = True
        if init_dict is not None:
            if 'sd_mean' in list(init_dict.keys()):
                override_sd_init = False
        if override_sd_init:
            out_dict['sd_mean'] = round((2*out_dict['softball_scale'])/(10*out_dict['n_components']),2)
        
        return out_dict
    
    def _init_gmm(self):
        '''create GMM instance as distribution over latent space'''
        print('#######################')
        print('Initializing model parts')
        print('#######################')
        self.gmm = GaussianMixture(n_mix_comp=self.param_dict['n_components'],
            dim=self.param_dict['latent_dimension'],
            mean_init=(self.param_dict['softball_scale'],self.param_dict['softball_hardness']),
            sd_init=(self.param_dict['sd_mean'],self.param_dict['sd_sd']),
            weight_alpha=self.param_dict['dirichlet_a']).to(device)
        print(self.gmm)
    
    def _init_representations(self):
        '''create representation instances. If validation split is given, creates 2 representations'''
        # initialize representation(s)
        self.representation = RepresentationLayer(n_rep=self.param_dict['latent_dimension'],
            n_sample=self.train_set.n_sample,
            value_init=self.param_dict['value_init']).to(device)
        print(self.representation)
        self.validation_rep = None
        if self.val_set is not None:
            self.validation_rep = RepresentationLayer(n_rep=self.param_dict['latent_dimension'],
                n_sample=self.val_set.n_sample,
                value_init=self.param_dict['value_init']).to(device)
        self.test_rep = None
    
    def _init_correction_models(self, correction):
        '''create correction models (additional, disentangled representation + gmm instances)'''
        self.correction_models = None
        if correction is not None:
            # correction_models will be nested list. each entry will be a [gmm, rep, val_rep]
            if isinstance(correction, str):
                correction = [correction]
            if isinstance(correction, list):
                self.correction_models = []
                for corr_id in range(len(correction)):
                    n_correction_classes = self.train_set.correction_classes[corr_id]
                    inner_list = []
                    inner_list.append(
                        GaussianMixtureSupervised(Nclass=n_correction_classes,Ncompperclass=1,dim=2,
                            mean_init=(self.param_dict['softball_scale'],self.param_dict['softball_hardness']),
                            sd_init=(round((2*self.param_dict['softball_scale'])/(10*n_correction_classes),2),self.param_dict['sd_sd']),
                            alpha=2).to(device)
                        )
                    inner_list.append(
                        RepresentationLayer(n_rep=inner_list[0].dim,n_sample=self.train_set.n_sample,
                            value_init=inner_list[0].supervised_sampling(self.train_set.get_correction_labels(corr_id),sample_type='origin')).to(device)
                        )
                    if self.validation_rep is not None:
                        inner_list.append(
                            RepresentationLayer(n_rep=inner_list[0].dim,n_sample=self.val_set.n_sample,
                                value_init=inner_list[0].supervised_sampling(self.val_set.get_correction_labels(corr_id),sample_type='origin')).to(device)
                            )
                    self.correction_models.append(inner_list)
            else:
                raise ValueError('correction argument passed with unknown type. provide list of strings of observation names in data')
            print('Correction models initialized (',correction,')')
    
    def _init_decoder(self):
        '''create decoder instance'''
        if self.correction_models is not None:
            updated_latent_dim = self.param_dict['latent_dimension']+len(self.correction_models)*2
        else:
            updated_latent_dim = self.param_dict['latent_dimension']
        self.decoder = Decoder(in_features=updated_latent_dim,parameter_dictionary=self.param_dict).to(device)
        if self._developer_mode:
            wandb.run.summary["N_parameters"] = count_parameters(self.decoder)
        print(self.decoder)
    
    def _get_train_status(self):
        '''print the training status of the model'''
        print('#######################')
        print('Training status')
        print('#######################')
        print(self.trained_status.item())
    
    def train(self, n_epochs=500, stop_with='loss', stop_after=10, train_minimum=50):
        '''
        train model
        options for stopping are 'loss' and 'clustering' (which requires meta_label in DGD init)
        '''
        
        # prepare data loaders
        train_loader = torch.utils.data.DataLoader(
            self.train_set, batch_size=self.param_dict['batch_size'],
            shuffle=True, num_workers=0
            )
        validation_loader = None
        if self.validation_rep is not None:
            validation_loader = torch.utils.data.DataLoader(
                self.val_set, batch_size=self.param_dict['batch_size'],
                shuffle=True, num_workers=0
                )
        
        self.decoder, self.gmm, self.representation, self.validation_rep, self.correction_models = train_dgd(
            self.decoder, self.gmm, self.representation, 
            self.validation_rep, train_loader, validation_loader,
            self.correction_models, n_epochs,
            learning_rates=self.param_dict['learning_rates'],
            adam_betas=self.param_dict['betas'],wd=self.param_dict['weight_decay'],
            stop_method=stop_with,stop_len=stop_after,train_minimum=train_minimum,
            save_dir=self._save_dir,
            developer_mode=self._developer_mode
            )
        
        self.trained_status[0] = 1

        self._get_train_status()
    
    def save(self):
        '''save trained parameters'''
        torch.save(self.state_dict(), self._save_dir+self._model_name+'.pt')
    
    def load_state_dict(self):
        '''load model'''
        checkpoint = torch.load(self._save_dir+self._model_name+'.pt',map_location=torch.device('cpu'))
        
        # dirty hack because I switched naming for one model and dataset
        print(list(checkpoint.keys()))
        if '_trained_status' in list(checkpoint.keys()):
            checkpoint['trained_status'] = checkpoint['_trained_status']
            self._trained_status = self.trained_status
        else:
            self._trained_status = None
        
        self.load_state_dict(checkpoint)
        
        print('#######################')
        print('Training status')
        print('#######################')
        print(self.trained_status.item())
    
    def save_param_dict_to_yaml(self):
        with open(self._save_dir+'hyperparameters.yml', 'w') as outfile:
            yaml.dump(self.param_dict, outfile, default_flow_style=False)
    
    @classmethod
    def load(cls, model_dir, model_name, dataset):
        with open(model_dir+'hyperparameters.yml', 'r') as stream:
            param_dict = yaml.safe_load(stream)
        checkpoint = torch.load(model_dir+model_name+'.pt',map_location=torch.device('cpu'))
        return cls()
    
    def get_representation(self):
        '''returning all representations of training data'''
        return self.representation.z.detach().cpu().numpy()
    
    def predict_new(self):
        '''learn the embedding for new datapoints'''
        return
    
    def differential_expression(self):
        '''doing differential expression analysis'''
        return
    
    def perturbation_experiment(self):
        '''perform a perturbation of a given feature and examine downstream effects'''
        return
    
    def get_normalizes_expression(self, dataset, indices):
        '''
        given a representation for the dataset is learned, returns the normalized (unscaled) model output
        currently only implemented for multiome
        '''
        # get the dimensions of the dataset
        shape = dataset.shape[0]
        # select the fitting representation
        if self.representation.shape[0] == shape:
            return self.decoder(self.representation(indices))[0]
        elif self.validation_rep is not None:
            if self.validation_rep.shape[0] == shape:
                return self.decoder(self.validation_rep(indices))[0]
        elif self.test_rep is not None:
            if self.test_rep.shape[0] == shape:
                return self.decoder(self.test_rep(indices))[0]
        else:
            raise AttributeError('Number of target samples does not match any of the learned representations.')
    
    def get_accessibility_estimates(self, dataset, indices):
        '''
        given a representation for the dataset is learned, returns the normalized (unscaled) model output
        currently only implemented for multiome
        '''
        # get the dimensions of the dataset
        shape = dataset.shape[0]
        # select the fitting representation
        if self.representation.shape[0] == shape:
            return self.decoder(self.representation(indices))[1]
        elif self.validation_rep is not None:
            if self.validation_rep.shape[0] == shape:
                return self.decoder(self.validation_rep(indices))[1]
        elif self.test_rep is not None:
            if self.test_rep.shape[0] == shape:
                return self.decoder(self.test_rep(indices))[1]
        else:
            raise AttributeError('Number of target samples does not match any of the learned representations.')