import torch
import torch.nn as nn
import numpy as np
import anndata as ad
import mudata as md
import wandb
import json
import sys
import os
import matplotlib.pyplot as plt

from omicsdgd.dataset import omicsDataset
from omicsdgd.latent import RepresentationLayer
from omicsdgd.latent import GaussianMixture, GaussianMixtureSupervised
from omicsdgd.nn import Decoder
from omicsdgd.functions import (
    train_dgd,
    set_random_seed,
    count_parameters,
    sc_feature_selection,
    learn_new_representations,
)
from omicsdgd.functions._gene_to_peak_linkage import predict_perturbations

"""Central Module of the package"""


class DGD(nn.Module):
    """
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
    """

    def __init__(
        self,
        data,
        parameter_dictionary=None,
        train_validation_split=None,
        modalities=["rna"],
        scaling="sum",
        meta_label="celltype",
        modality_switch=None,
        correction=None,
        feature_selection=None,
        save_dir="./",
        random_seed=0,
        model_name="dgd",
        print_outputs=True,
        num_workers=None,
    ):
        super().__init__()

        print("CUDA Available:", torch.cuda.is_available())
        print("Number of GPUs:", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("GPU Name:", torch.cuda.get_device_name(0))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("using device: ", self.device)

        # setting the random seed at the beginning for reproducibility
        set_random_seed(random_seed)

        # setting internal helper attributes
        self._init_internal_attribs(
            model_name,
            save_dir,
            scaling,
            meta_label,
            correction,
            print_outputs,
            num_workers,
        )  # needs to go? meta_label to data, dev_mode to train
        # self.trained_status = nn.Parameter(torch.zeros(1, dtype=torch.bool), requires_grad=False)

        # if there is a train-val(-test) split, check that it is in accepted format
        self._check_train_split(train_validation_split)  # needs to go to data

        ###
        # initializing data
        ###
        # this also checks whether the data is in an acceptable format and initialize
        self._init_data(  # for init it should not yet transform to tensors
            data,
            modality_switch,
            scaling,
            meta_label,
            correction,
            modalities,
            feature_selection,
        )

        # update parameter dictionary with optional input and info from data
        self.param_dict = self._init_parameter_dictionary(parameter_dictionary)
        # if self._developer_mode:
        #    wandb.config.update(self.param_dict)

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
        # self._get_train_status()

    def _init_internal_attribs(
        self,
        model_name,
        save_dir,
        scaling,
        meta_label,
        correction,
        print_outputs,
        num_workers,
    ):
        # define what to name the model and where to save it
        self._model_name = model_name
        self._save_dir = save_dir
        # scaling relationship between model output and data
        self._scaling = scaling
        # optional: features to observe and to correct for
        self._clustering_variable = meta_label
        self._correction_variables = correction
        # developer mode changes from notebook use with progress bar to monitoring jobs with wandb
        # self._developer_mode = developer_mode
        # important for checking whether a model is trained or has loaded learned parameters
        self.trained_status = nn.Parameter(
            torch.zeros(1, dtype=torch.bool), requires_grad=False
        )
        self._print_output = print_outputs
        if num_workers is None:
            if self.device != "cpu":
                self._num_workers = 4
            else:
                self._num_workers = 0
        else:
            self._num_workers = num_workers

    def _check_train_split(self, split):
        """
        Checks that train-validation split is in correct format
        [[train_indices], [validation_indices]]

        Currently it transforms lists of characters into desired format,
        but could also remove support since most non-ML users won't have a validation set and are capable of providing the correct format.
        """
        if split is not None:
            if isinstance(split, list):
                if len(split) != 2:
                    if isinstance(split[0], str):
                        import difflib

                        train_string = difflib.get_close_matches(
                            "train", list(set(split))
                        )[0]
                        val_string = difflib.get_close_matches("val", list(set(split)))[
                            0
                        ]
                        if "val" not in val_string:
                            val_string = difflib.get_close_matches(
                                "test", list(set(split))
                            )[0]
                        print(
                            "assuming the following train-validation split: train-",
                            train_string,
                            ", validation-",
                            val_string,
                        )
                        self.train_val_split = [
                            list(np.where(split.iloc[:, 0].values == train_string)[0]),
                            list(np.where(split.iloc[:, 0].values == val_string)[0]),
                        ]
                    raise ValueError(
                        "wrong format of train_validation_split submitted. should be a list of length 2: [[train],[val]]"
                    )
                else:
                    # if len(split[0]) < len(split[1]):
                    #    # switch order in list because validation must be the smaller set
                    #    self.train_val_split = [split[1], split[0]]
                    # else:
                    #    self.train_val_split = split
                    self.train_val_split = split
                return
            else:
                error_message = "unknown type provided for train_validation_split.\n"
                error_message += "valid types are list or None.\n"
                error_message += "the list can be in 2 forms:\n"
                error_message += "   [[train_indices],[validation_indices]] or\n"
                error_message += (
                    '   1D list of categories ("train", "val"/"validation"/"test")'
                )
                raise ValueError(error_message)
        else:
            self.train_val_split = None

    def _init_wandb_logging(self, parameter_dictionary):
        """start run if in developer mode (otherwise assumes running in notebook)"""
        try:
            wandb.init(
                project=parameter_dictionary["log_wandb"][1],
                entity=parameter_dictionary["log_wandb"][0],
                config=parameter_dictionary,
            )
        except:
            raise ValueError(
                "You are trying to run in developer mode, but seem not to have given the parameter dictionary the `log_wandb` statement."
            )
        wandb.run.name = self._model_name

    def _init_data(
        self,
        data,
        modality_switch,
        scaling_type,
        meta_label,
        correction,
        modalities,
        feature_selection,
    ):
        """Internal function to derive datasets and data-dependent parameters from input data"""

        if self._print_output:
            print("#######################")
            print("Initialized data")
            print("#######################")

        # here there is the option to perform feature selection beforehand,
        # i.e. reducing the number of features depending on the `feature_selection` option
        if feature_selection is not None:
            selection_method, data, modality_switch = self._feature_selection(
                data, modalities, feature_selection
            )
            print("selected features based on ", selection_method)

        if self.train_val_split is not None:
            self.train_set = omicsDataset(
                data[self.train_val_split[0]],
                modality_switch,
                scaling_type,
                meta_label,
                correction,
                modalities,
            )
        else:
            self.train_set = omicsDataset(
                data, modality_switch, scaling_type, meta_label, correction, modalities
            )
        if self.train_val_split is not None:
            self.val_set = omicsDataset(
                data[self.train_val_split[1]],
                modality_switch,
                scaling_type,
                meta_label,
                correction,
                modalities,
                label="test",
            )
        else:
            self.val_set = None

        if self._print_output:
            print(self.train_set)

    @staticmethod
    def _feature_selection(data, modalities, feature_selection):
        """returns the selection mode (for logging), the reduced data and the new modality swith position (if applicable)"""
        return sc_feature_selection(data, modalities, feature_selection)

    def _init_parameter_dictionary(self, init_dict=None):
        """initialize the parameter dictionary from default and update with optional input"""
        # init parameter dictionary based on defaults
        # Opening JSON file
        """
        print(os.getcwd())
        with open('./dgd_setup.json') as json_file:
            out_dict = json.load(json_file)
        """
        out_dict = {
            "latent_dimension": 20,
            "n_components": 1,
            "n_hidden": 1,
            "n_hidden_modality": 1,
            "n_units": 100,
            "value_init": "zero",  # options are zero or handed values
            "softball_scale": 2,
            "softball_hardness": 5,
            "sd_sd": 1,
            "softball_scale_corr": 2,
            "softball_hardness_corr": 5,
            "sd_sd_corr": 1,
            "dirichlet_a": 1,
            "batch_size": 128,
            "learning_rates": [1e-4, 1e-2, 1e-2],
            "betas": [0.5, 0.7],
            "weight_decay": 1e-4,
            "decoder_width": 1,
            "log_wandb": ["username", "projectname"],
        }
        # """
        out_dict["sd_mean"] = round(
            (2 * out_dict["softball_scale"]) / (10 * out_dict["n_components"]), 2
        )
        # update from optional class input dictionary
        if init_dict is not None:
            for key in init_dict.keys():
                out_dict[key] = init_dict[key]
        # add information gained in data initialization
        out_dict["n_features"] = self.train_set.n_features  # number of total features
        out_dict[
            "n_features_per_modality"
        ] = (
            self.train_set.modality_features
        )  # list of output features generated depending on number of modalities
        out_dict["modalities"] = self.train_set.modalities
        if self.train_set.modality_switch is not None:
            out_dict["modality_switch"] = int(
                self.train_set.modality_switch
            )  # updating the modality switch if multi-modal data in AnnData object and switch was not given
        else:
            out_dict["modality_switch"] = None
        out_dict["scaling"] = self._scaling
        out_dict["clustering_variable(meta)"] = self._clustering_variable
        out_dict["correction_variable"] = self._correction_variables

        # automate selection of number of components
        if self.train_set.meta is not None:
            select_n_components = False
            if init_dict is not None:
                if "n_components" not in list(init_dict.keys()):
                    select_n_components = True
            else:
                select_n_components = True
            if select_n_components:
                out_dict["n_components"] = int(len(list(set(self.train_set.meta))))
                print(
                    "selected ",
                    out_dict["n_components"],
                    " number of Gaussian mixture components based on ",
                    self._clustering_variable,
                )

        # overwrite hyperparameters with passed parameter_dictionary (if applicable)
        # it is crucial to update the initial value for the GMM component std if number of component changes from default
        # this will be applied at the end of data initialization
        override_sd_init = True
        if init_dict is not None:
            if "sd_mean" in list(init_dict.keys()):
                override_sd_init = False
        if override_sd_init:
            out_dict["sd_mean"] = round(
                (2 * out_dict["softball_scale"]) / (10 * out_dict["n_components"]), 2
            )

        return out_dict

    def _init_gmm(self):
        """create GMM instance as distribution over latent space"""
        if self._print_output:
            print("#######################")
            print("Initializing model parts")
            print("#######################")
        self.gmm = GaussianMixture(
            n_mix_comp=self.param_dict["n_components"],
            dim=self.param_dict["latent_dimension"],
            mean_init=(
                self.param_dict["softball_scale"],
                self.param_dict["softball_hardness"],
            ),
            sd_init=(self.param_dict["sd_mean"], self.param_dict["sd_sd"]),
            weight_alpha=self.param_dict["dirichlet_a"],
        ).to(self.device)
        if self._print_output:
            print(self.gmm)

    def _init_representations(self):
        """create representation instances. If validation split is given, creates 2 representations"""
        # initialize representation(s)
        self.representation = RepresentationLayer(
            n_rep=self.param_dict["latent_dimension"],
            n_sample=self.train_set.n_sample,
            value_init=self.param_dict["value_init"],
        ).to(self.device)
        if self._print_output:
            print(self.representation)
        self.validation_rep = None
        if self.val_set is not None:
            self.validation_rep = RepresentationLayer(
                n_rep=self.param_dict["latent_dimension"],
                n_sample=self.val_set.n_sample,
                value_init=self.param_dict["value_init"],
            ).to(self.device)
        self.test_rep = None

    # as a list, the correction models were not saved in the state_dict
    """
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
                            alpha=2).to(self.device)
                        )
                    inner_list.append(
                        RepresentationLayer(n_rep=inner_list[0].dim,n_sample=self.train_set.n_sample,
                            value_init=inner_list[0].supervised_sampling(self.train_set.get_correction_labels(corr_id),sample_type='origin')).to(self.device)
                        )
                    if self.validation_rep is not None:
                        inner_list.append(
                            RepresentationLayer(n_rep=inner_list[0].dim,n_sample=self.val_set.n_sample,
                                value_init=inner_list[0].supervised_sampling(self.val_set.get_correction_labels(corr_id),sample_type='origin')).to(self.device)
                            )
                    self.correction_models.append(inner_list)
            else:
                raise ValueError('correction argument passed with unknown type. provide list of strings of observation names in data')
            if self._print_output:
                print('Correction models initialized (',correction,')')
    """

    def _init_correction_models(self, correction, dim=2):
        """
        create correction models (additional, disentangled representation + gmm instances)
        the correction input is None or a string as the feature name
        """
        if correction is not None:
            n_correction_classes = self.train_set.correction_classes
            self.correction_gmm = GaussianMixtureSupervised(
                Nclass=n_correction_classes,
                Ncompperclass=1,
                dim=dim,
                mean_init=(
                    self.param_dict["softball_scale_corr"],
                    self.param_dict["softball_hardness_corr"],
                ),
                sd_init=(
                    round(
                        (2 * self.param_dict["softball_scale_corr"])
                        / (10 * n_correction_classes),
                        2,
                    ),
                    self.param_dict["sd_sd_corr"],
                ),
                alpha=2,
            ).to(self.device)
            self.correction_rep = RepresentationLayer(
                n_rep=dim,
                n_sample=self.train_set.n_sample,
                value_init=self.correction_gmm.supervised_sampling(
                    self.train_set.get_correction_labels(), sample_type="origin"
                ),
            ).to(self.device)
            if self.validation_rep is not None:
                self.correction_val_rep = RepresentationLayer(
                    n_rep=dim, n_sample=self.val_set.n_sample, value_init="zero"
                ).to(self.device)
            print(self.correction_gmm)
        else:
            self.correction_gmm = None
            self.correction_rep = None
            self.correction_val_rep = None

    def _init_decoder(self):
        """create decoder instance"""
        if self.correction_gmm is not None:
            updated_latent_dim = (
                self.param_dict["latent_dimension"] + self.correction_gmm.dim
            )
        else:
            updated_latent_dim = self.param_dict["latent_dimension"]
        self.decoder = Decoder(
            in_features=updated_latent_dim, parameter_dictionary=self.param_dict
        ).to(self.device)
        # if self._developer_mode:
        #    wandb.run.summary["N_parameters"] = count_parameters(self.decoder)
        self.param_dict["n_decoder_parameters"] = count_parameters(self.decoder)
        if self._print_output:
            print(self.decoder)

    def _get_train_status(self):
        """print the training status of the model"""
        print("#######################")
        print("Training status")
        print("#######################")
        print(self.trained_status.item())

    def train(
        self,
        n_epochs=500,
        stop_with="loss",
        stop_after=10,
        train_minimum=50,
        developer_mode=False,
        update_covariance_prior=False
    ):
        """
        train model for n_epochs

        options for early stopping are 'loss' and 'clustering' (which requires meta_label in DGD init)

        ealry stopping observation interval is stop_after, and a minimum number of epochs to be trained
        can be specified with train_minimum

        developer_mode defines how progress is displayed.
        If false, it is assumed that a jupyter notebook is used. Then, progress will be shown in a progress bar.
        If true, a wandb run will be initialized, logging many training metrics.
        """

        if developer_mode:
            self._init_wandb_logging(self.param_dict)  # go to train

        # prepare data loaders
        # first step is to transform the data into tensors
        # set the number of workers if you are not using a notebook or if gpu is not available
        print("Preparing data loaders")
        print("   num_workers: ", self._num_workers)
        self.train_set.data_to_tensor()
        train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.param_dict["batch_size"],
            shuffle=True,
            num_workers=self._num_workers,
        )
        validation_loader = None
        if self.validation_rep is not None:
            self.val_set.data_to_tensor()
            validation_loader = torch.utils.data.DataLoader(
                self.val_set,
                batch_size=self.param_dict["batch_size"],
                shuffle=True,
                num_workers=self._num_workers,
            )

        print("Now training")
        (
            self.decoder,
            self.gmm,
            self.representation,
            self.validation_rep,
            self.correction_gmm,
            self.correction_rep,
            self.correction_val_rep,
        ) = train_dgd(
            self.decoder,
            self.gmm,
            self.representation,
            self.validation_rep,
            train_loader,
            validation_loader,
            self.correction_gmm,
            self.correction_rep,
            self.correction_val_rep,
            n_epochs,
            learning_rates=self.param_dict["learning_rates"],
            adam_betas=self.param_dict["betas"],
            wd=self.param_dict["weight_decay"],
            stop_method=stop_with,
            stop_len=stop_after,
            train_minimum=train_minimum,
            device=self.device,
            save_dir=self._save_dir,
            developer_mode=developer_mode,
            update_covariance_prior=update_covariance_prior
        )

        self.trained_status[0] = 1

        self._get_train_status()

    def save(self, save_dir=None, model_name=None):
        """save trained parameters"""
        if save_dir is None:
            save_dir = self._save_dir
        if model_name is None:
            model_name = self._model_name
        torch.save(self.state_dict(), save_dir + model_name + ".pt")
        self.save_param_dict_to_file(save_dir + model_name)

    def load_parameters(self):
        """load model"""
        # load state dict
        checkpoint = torch.load(
            self._save_dir + self._model_name + ".pt", map_location=torch.device("cpu")
        )
        # based on what people may provide for loading vs. training, there might be discrepancies in dataset numbers and sizes
        # e.g. there might have been a train-val split but now they provide no split list
        # so check sizes of representations
        if self.representation.z.shape != checkpoint["representation.z"].shape:
            print(
                "data(split) is not the same as provided for training. Correcting representation size."
            )
            self.representation = RepresentationLayer(
                n_rep=self.param_dict["latent_dimension"],
                n_sample=checkpoint["representation.z"].shape[0],
                value_init=self.param_dict["value_init"],
            ).to(self.device)
        if "validation_rep.z" in list(checkpoint.keys()):
            if self.validation_rep is None:
                self.validation_rep = RepresentationLayer(
                    n_rep=self.param_dict["latent_dimension"],
                    n_sample=checkpoint["validation_rep.z"].shape[0],
                    value_init=self.param_dict["value_init"],
                ).to(self.device)
            elif self.validation_rep.z.shape != checkpoint["validation_rep.z"].shape:
                self.validation_rep = RepresentationLayer(
                    n_rep=self.param_dict["latent_dimension"],
                    n_sample=checkpoint["validation_rep.z"].shape[0],
                    value_init=self.param_dict["value_init"],
                ).to(self.device)
            if self.correction_gmm is not None:
                # if self.correction_val_rep is not None:
                if (not hasattr(self, "correction_val_rep")) and (
                    self.validation_rep is not None
                ):
                    self.correction_val_rep = RepresentationLayer(
                        n_rep=self.correction_gmm.dim,
                        n_sample=checkpoint["validation_rep.z"].shape[0],
                        value_init=self.param_dict["value_init"],
                    ).to(self.device)
            # else:
            #    print('careful! no validation representation has been initialized')
        if "test_rep.z" in list(checkpoint.keys()):
            self.test_rep = RepresentationLayer(
                n_rep=self.param_dict["latent_dimension"],
                n_sample=checkpoint["test_rep.z"].shape[0],
                value_init=self.param_dict["value_init"],
            ).to(self.device)
            if self.correction_gmm is not None:
                self.correction_test_rep = RepresentationLayer(
                    n_rep=self.correction_gmm.dim,
                    n_sample=self.test_rep.z.shape[0],
                    value_init="zero",
                ).to(self.device)
                # if 'new_correction_model' in list(checkpoint.keys()):

        if self.correction_gmm is not None:
            # if the shapes of the means don't match, the correction model has been changed
            if (
                self.correction_gmm.mean.shape
                != checkpoint["correction_gmm.mean"].shape
            ):
                self.correction_gmm = GaussianMixtureSupervised(
                    Nclass=checkpoint["correction_gmm.mean"].shape[0],
                    Ncompperclass=1,
                    dim=checkpoint["correction_gmm.mean"].shape[1],
                    mean_init=(
                        self.param_dict["softball_scale_corr"],
                        self.param_dict["softball_hardness_corr"],
                    ),
                    sd_init=(
                        round(
                            (2 * self.param_dict["softball_scale_corr"])
                            / (10 * checkpoint["correction_gmm.mean"].shape[0]),
                            2,
                        ),
                        self.param_dict["sd_sd_corr"],
                    ),
                    alpha=2,
                ).to(self.device)

        # dirty hack because I switched naming for one model and dataset
        if "_trained_status" in list(checkpoint.keys()):
            checkpoint["trained_status"] = checkpoint["_trained_status"]
            self._trained_status = self.trained_status
        else:
            self._trained_status = None

        self.load_state_dict(checkpoint)

        print("#######################")
        print("Training status")
        print("#######################")
        print(self.trained_status.item())

    def save_param_dict_to_file(self, save_dir):
        # for x in self.param_dict.keys():
        #    if type(self.param_dict[x]) == list:
        #        print((x, [(type(self.param_dict[x][i]), sys.getsizeof(self.param_dict[x][i])) for i in range(len(self.param_dict[x]))]))
        #    else:
        #        print((x, type(self.param_dict[x]), sys.getsizeof(self.param_dict[x])))
        with open(save_dir + "_hyperparameters.json", "w") as fp:
            json.dump(self.param_dict, fp)
        # with open(save_dir+'_hyperparameters.yml', 'w') as outfile:
        #    yaml.dump(self.param_dict, outfile, default_flow_style=False)

    @classmethod
    def load(
        cls,
        data,
        train_validation_split=None,
        feature_selection=None,
        save_dir="./",
        random_seed=0,
        model_name="dgd",
    ):
        # get saved hyper-parameters
        with open(save_dir + model_name + "_hyperparameters.json", "r") as fp:
            param_dict = json.load(fp)
        scaling = param_dict["scaling"]
        meta_label = param_dict["clustering_variable(meta)"]
        correction = param_dict["correction_variable"]
        modality_switch = param_dict["modality_switch"]
        modalities = param_dict["modalities"]

        # init model
        model = cls(
            data=data,
            parameter_dictionary=param_dict,
            train_validation_split=train_validation_split,
            modalities=modalities,
            scaling=scaling,
            meta_label=meta_label,
            modality_switch=modality_switch,
            correction=correction,
            feature_selection=feature_selection,
            save_dir=save_dir,
            random_seed=random_seed,
            model_name=model_name,
            print_outputs=False,
        )

        """
        # load state dict
        checkpoint = torch.load(save_dir+model_name+'.pt',map_location=torch.device('cpu'))
        # based on what people may provide for loading vs. training, there might be discrepancies in dataset numbers and sizes
        # e.g. there might have been a train-val split but now they provide no split list
        # so check sizes of representations
        if model.representation.z.shape != checkpoint['representation.z'].shape:
            print('data(split) is not the same as provided for training. Correcting representation size.')
            model.representation = RepresentationLayer(n_rep=model.param_dict['latent_dimension'],
                n_sample=checkpoint['representation.z'].shape[0],
                value_init=model.param_dict['value_init']).to(self.device)
        if 'validation_rep.z' in list(checkpoint.keys()):
            if model.validation_rep is None:
                model.validation_rep = RepresentationLayer(n_rep=model.param_dict['latent_dimension'],
                    n_sample=checkpoint['validation_rep.z'].shape[0],
                    value_init=model.param_dict['value_init']).to(self.device)
            else:
                if model.validation_rep.z.shape != checkpoint['validation_rep.z'].shape:
                    model.validation_rep = RepresentationLayer(n_rep=model.param_dict['latent_dimension'],
                        n_sample=checkpoint['validation_rep.z'].shape[0],
                        value_init=model.param_dict['value_init']).to(self.device)
        if 'test_rep.z' in list(checkpoint.keys()):
            model.test_rep = RepresentationLayer(n_rep=model.param_dict['latent_dimension'],
                n_sample=checkpoint['test_rep.z'].shape[0],
                value_init=model.param_dict['value_init']).to(self.device)
        
        model.load_state_dict(checkpoint)
        """
        model.load_parameters()
        return model

    def get_latent_representation(self, data=None):
        """returning all representations of training data"""
        if data is not None:
            shape = data.shape
            if self.representation.z.shape[0] == shape:
                return self.representation.z.detach().cpu().numpy()
            elif self.validation_rep is not None:
                if self.validation_rep.z.shape[0] == shape:
                    return self.validation_rep.z.detach().cpu().numpy()
            elif self.test_rep is not None:
                if self.test_rep.z.shape[0] == shape:
                    return self.test_rep.z.detach().cpu().numpy()
        else:
            return self.representation.z.detach().cpu().numpy()

    def init_test_set(self, testdata):
        self.test_set = omicsDataset(
            testdata,
            self.param_dict["modality_switch"],
            self.param_dict["scaling"],
            self.param_dict["clustering_variable(meta)"],
            self.param_dict["correction_variable"],
            self.param_dict["modalities"],
        )
        self.test_set.data_to_tensor()

    def predict_new(
        self,
        testdata,
        n_epochs=10,
        include_correction_error=True,
        indices_of_new_distribution=None,
        start_from_zero=[False, False],
        lrs=[0.01, 0.01],
        init_covariate_supervised=None,
        train_supervised=False,
        cov_beta=1
    ):
        """learn the embedding for new datapoints"""

        # prepare test set and loader
        self.test_set = omicsDataset(
            testdata,
            self.param_dict["modality_switch"],
            self.param_dict["scaling"],
            self.param_dict["clustering_variable(meta)"],
            self.param_dict["correction_variable"],
            self.param_dict["modalities"],
        )
        self.test_set.data_to_tensor()
        test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=8, shuffle=True, num_workers=0
        )

        # init new representation
        # self.test_rep = RepresentationLayer(n_rep=self.param_dict['latent_dimension'],
        #    n_sample=self.test_set.n_sample,
        #    value_init=self.param_dict['value_init']).to(self.device)

        # train that representation
        (
            new_decoder,
            self.test_rep,
            self.correction_test_rep,
            new_correction_model,
        ) = learn_new_representations(
            self.gmm,
            self.decoder,
            test_loader,
            self.test_set.n_sample,
            self.correction_gmm,
            n_epochs=n_epochs,
            lrs=lrs,
            include_correction_error=include_correction_error,
            indices_of_new_distribution=indices_of_new_distribution,
            start_from_zero=start_from_zero,
            init_covariate_supervised=init_covariate_supervised,
            supervised=train_supervised,
            cov_beta=cov_beta
        )
        self.param_dict["test_representation"] = True
        # make the new_correction_model an attribute so it is learned
        if new_correction_model is not None:
            self.correction_gmm = new_correction_model
            if new_decoder is not None:
                self.decoder = new_decoder
        self.save()

    def decoder_forward(self, rep_shape, i=None):
        if self.representation.z.shape[0] == rep_shape:
            if self.correction_gmm is not None:
                if i is not None:
                    z = self.representation(i)
                    z_correction = self.correction_rep(i)
                else:
                    z = self.representation.z
                    z_correction = self.correction_rep.z
                # z_correction = torch.cat(tuple([self.correction_models[corr_id][1](i) for corr_id in range(len(self.correction_models))]), dim=1)
                y = self.decoder(torch.cat((z, z_correction), dim=1))
            else:
                if i is not None:
                    z = self.representation(i)
                else:
                    z = self.representation.z
                y = self.decoder(z)
        elif self.test_rep is not None:  # always preferrably use test rep
            if self.test_rep.z.shape[0] == rep_shape:
                if self.correction_gmm is not None:
                    if i is not None:
                        z = self.test_rep(i)
                        z_correction = self.correction_test_rep(i)
                    else:
                        z = self.test_rep.z
                        z_correction = self.correction_test_rep.z
                    # z_correction = torch.cat(tuple([self.test_correction[corr_id](i) for corr_id in range(len(self.correction_models))]), dim=1)
                    y = self.decoder(torch.cat((z, z_correction), dim=1))
                else:
                    if i is not None:
                        z = self.test_rep(i)
                    else:
                        z = self.test_rep.z
                    y = self.decoder(z)
        elif self.validation_rep is not None:
            if self.validation_rep.z.shape[0] == rep_shape:
                if self.correction_gmm is not None:
                    if i is not None:
                        z = self.validation_rep(i)
                        z_correction = self.correction_val_rep(i)
                    else:
                        z = self.validation_rep.z
                        z_correction = self.correction_val_rep.z
                    # z_correction = torch.cat(tuple([self.correction_models[corr_id][2](i) for corr_id in range(len(self.correction_models))]), dim=1)
                    y = self.decoder(torch.cat((z, z_correction), dim=1))
                else:
                    if i is not None:
                        z = self.validation_rep(i)
                    else:
                        z = self.validation_rep.z
                    y = self.decoder(z)
        else:
            raise ValueError(
                "something is wrong with the shape of the dataset you supplied. There is no representation for it."
            )
        return y

    def predict_from_representation(self, rep, correction_rep=None):
        if self.correction_gmm is not None:
            z = rep.z
            z_correction = correction_rep.z
            y = self.decoder(torch.cat((z, z_correction), dim=1))
        else:
            z = rep.z
            y = self.decoder(z)
        return y

    def differential_expression(self):
        """doing differential expression analysis"""
        return

    def perturbation_experiment(self):
        """perform a perturbation of a given feature and examine downstream effects"""
        return

    def get_prediction_errors(self, predictions, dataset, reduction="sum"):
        """returns errors of given predictions
        reduction: `sum`, `sample` or `none` (defines error shape)"""
        if reduction == "sum":
            error = torch.zeros((1))
        elif reduction == "sample":
            error = torch.zeros((predictions[0].shape[0]))
        elif reduction == "gene":
            reduction = "none"
            error = torch.zeros(
                (
                    predictions[0].shape[0],
                    predictions[0].shape[1] + predictions[1].shape[1],
                )
            )
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=True, num_workers=0
        )
        # x has to be broght into correct shape depending on number of modalities
        for x, lib, i in data_loader:
            # print('mini batch')
            if self.decoder.n_out_groups == 2:
                y = [predictions[0][i, :], predictions[1][i, :]]
                x = [
                    x[:, : self.param_dict["modality_switch"]].to(self.device),
                    x[:, self.param_dict["modality_switch"] :].to(self.device),
                ]
                library = [lib[:, 0].unsqueeze(1).to(self.device), lib[:, 1].unsqueeze(1).to(self.device)]
            elif self.decoder.n_out_groups == 1:
                y = [predictions[0][i, :]]
                x = [x.to(self.device)]
                library = [lib.to(self.device)]
            else:
                raise ValueError("number of modalities currently not supported")
            if reduction != "none":
                temp_error = self.decoder.loss(y, x, scale=library, reduction=reduction).detach().cpu()
                if reduction == "sum":
                    error += temp_error
                elif reduction == "sample":
                    error[i] = temp_error
            else:
                temp_error_1 = self.decoder.loss(
                    y[0], x[0], scale=library[0], reduction=reduction, mod_id=0
                ).detach().cpu()
                if self.decoder.n_out_groups == 2:
                    temp_error_2 = self.decoder.loss(
                        y[1], x[1], scale=library[1], reduction=reduction, mod_id=1
                    ).detach().cpu()
                    error[i, :] = torch.cat((temp_error_1, temp_error_2), dim=1)
                else:
                    error[i, :] = temp_error_1
        return error

    def get_normalized_expression(self, dataset, indices):
        """
        given a representation for the dataset is learned, returns the normalized (unscaled) model output
        currently only implemented for multiome
        """
        # get the dimensions of the dataset
        shape = dataset.X.shape[0]
        return self.decoder_forward(shape, indices)[0]
        """
        # select the fitting representation
        if self.representation.z.shape[0] == shape:
            return self.decoder(self.representation(indices))[0]
        elif self.validation_rep is not None:
            if self.validation_rep.z.shape[0] == shape:
                return self.decoder(self.validation_rep(indices))[0]
        elif self.test_rep is not None:
            if self.test_rep.z.shape[0] == shape:
                return self.decoder(self.test_rep(indices))[0]
        else:
            raise AttributeError('Number of target samples does not match any of the learned representations.')
        """

    def get_accessibility_estimates(self, dataset, indices):
        """
        given a representation for the dataset is learned, returns the normalized (unscaled) model output
        currently only implemented for multiome
        """
        # get the dimensions of the dataset
        shape = dataset.X.shape[0]
        return self.decoder_forward(shape, indices)[1]
        """
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
        """

    def get_modality_reconstruction(self, dataset, mod_id):
        shape = dataset.X.shape[0]
        predictions = self.decoder_forward(shape, torch.arange(shape))[mod_id]
        return predictions * self.library[mod_id].unsqueeze(0)

    def get_reconstruction(self, dataset):
        shape = dataset.X.shape[0]
        predictions = self.decoder_forward(shape, torch.arange(shape))
        return predictions

    def gene2peak(self, gene_name, testset, gene_ref=None, plot=False, quantile=None):
        if gene_name not in testset.var_names:
            if gene_ref is not None:
                if isinstance(gene_ref, str):
                    gene_ref = testset.var[gene_ref]
                else:
                    raise ValueError("gene_ref must be a string")
            else:
                raise ValueError(
                    "Gene was not found in data. It might be spelled wrong or you need to specify the var column name that contains the gene name."
                )
        else:
            gene_ref = testset.var_names
        gene_id = np.where(gene_ref == gene_name)[0][0]
        (
            predictions_original_gene,
            predictions_upregulated_gene,
            indices_of_interest_gene,
        ) = predict_perturbations(self, testset, gene_id)

        # plot the results
        if plot:
            if quantile is None:
                fig, ax = plt.subplots(figsize=(5, 5))
                plt.plot(
                    predictions_original_gene[0][:, gene_id],
                    predictions_original_gene[0][:, gene_id],
                    color="black",
                )
                plt.scatter(
                    predictions_original_gene[0][:, gene_id],
                    predictions_upregulated_gene[0][:, gene_id],
                )
                plt.xlabel("original prediction")
                plt.ylabel("perturbed prediction")
                plt.title("Perturbation results on gene: " + gene_name)
                plt.show()
            else:
                fig, ax = plt.subplots(figsize=(5, 5))
                plt.plot(
                    predictions_original_gene[0][:, gene_id],
                    predictions_original_gene[0][:, gene_id],
                    color="black",
                )
                quantile_val = np.quantile(
                    predictions_original_gene[0][:, gene_id], quantile
                )
                samples_in_quantile = np.where(
                    predictions_original_gene[0][:, gene_id] < quantile_val
                )[0]
                samples_outside_quantile = np.where(
                    predictions_original_gene[0][:, gene_id] >= quantile_val
                )[0]
                plt.scatter(
                    predictions_original_gene[0][samples_in_quantile, gene_id],
                    predictions_upregulated_gene[0][samples_in_quantile, gene_id],
                    color="red",
                )
                plt.scatter(
                    predictions_original_gene[0][samples_outside_quantile, gene_id],
                    predictions_upregulated_gene[0][samples_outside_quantile, gene_id],
                    color="blue",
                )
                plt.xlabel("original prediction")
                plt.ylabel("perturbed prediction")
                plt.title("Perturbation results on gene: " + gene_name)
        predicted_changes = [
            (predictions_upregulated_gene[i] - predictions_original_gene[i])
            for i in range(len(self.train_set.modalities))
        ]
        if quantile is not None:
            predicted_changes = [
                predicted_changes[i][samples_outside_quantile, :]
                for i in range(len(self.train_set.modalities))
            ]
            indices_of_interest_gene = indices_of_interest_gene[
                samples_outside_quantile
            ]
        return predicted_changes, indices_of_interest_gene
