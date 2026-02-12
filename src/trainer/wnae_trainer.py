import time
from pathlib import Path
import time

import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from wnae import WNAE
from wnae._logger import log

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainerWassersteinNormalizedAutoEncoder():
    
    def __init__(
            self,
            config,
            loader,
            encoder,
            decoder,
            device,
            output_path,
            loss_function="wnae",
        ):
        """
        Constructor of the specialized Trainer class.
        """
        
        self.epoch = 0
        
        self.config = config
        self.loader = loader
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.output_path = output_path
        self.loss_function = loss_function
        self.metrics_tracker = {
            "epoch": [],
            "training_loss": [],
            "validation_loss": [],
            "auc": [],
        }

        Path(f"{self.output_path}/sample_feature_1D_hist").mkdir(parents=True, exist_ok=True)
        
        self.hyper_parameters = {}
        self.model = self.__get_model()
    
    def __make_sample_feature_1D_plot(self, positive_samples, negative_samples):
        # print(positive_samples.shape)
        # print(negative_samples.shape)
        positive_samples = positive_samples.detach().cpu().numpy()
        negative_samples = negative_samples.detach().cpu().numpy()
       # pick a feature from positive samples(input) and negative samples(mcmc) and plot their value distribution in one plot
        for feature_idx in range(positive_samples.shape[1]):
    
            Path(f"{self.output_path}/sample_feature_1D_hist/feature_{feature_idx}").mkdir(parents=True, exist_ok=True)
            # print(min(positive_samples[:,feature_idx]), max(positive_samples[:,feature_idx]))
            # print(min(negative_samples[:,feature_idx]), max(negative_samples[:,feature_idx]))
            # print(f"generating for feature {feature_idx}")
            plt.figure()
            plt.hist(positive_samples[:,feature_idx], bins=50, edgecolor='black', histtype='step', label="positive samples") 
            plt.hist(negative_samples[:,feature_idx], bins=50, edgecolor='red', histtype='step', label="negative samples") 
            plt.title(f"Feature {feature_idx} Epoch {self.epoch}")
            plt.legend()
            plt.savefig(f"{self.output_path}/sample_feature_1D_hist/feature_{feature_idx}/epoch_{self.epoch}.png")
            plt.close()
            
    def save_train_plot(self):
        fig, ax1 = plt.subplots()

        x_axis = list(range(len(self.metrics_tracker["training_loss"])))
        # Plot the first dataset on the left y-axis
        ax1.plot(x_axis,self.metrics_tracker["training_loss"], label="training loss")
        ax1.plot(x_axis,self.metrics_tracker["validation_loss"], label="validation loss")
        ax1.legend()
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('loss', color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        # Create the second y-axis
        ax2 = ax1.twinx()

        # Plot the second dataset on the right y-axis
        ax2.plot(x_axis, self.metrics_tracker["auc"], 'b--', label='AUC')
        ax2.set_ylabel('AUC', color='b')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.legend()
        fig.savefig(f"{self.output_path}/train_history.png")
        plt.close()
        
    def __train_epoch(self, training_loader, optimizer):

        self.model.train()

        # Can monitore more quantities than the loss, showing loss as example
        monitored_quantities = {
            "loss": 0.,
        }

        n_batches = 0
        bar_format = '{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        for batch in tqdm(training_loader, bar_format=bar_format):
            n_batches += 1
            x = batch[0]  # batch is a list of len 1 with the tensor inside

            optimizer.zero_grad()
            if self.loss_function == "ae":
                loss, training_dict = self.model.train_step_ae(x)
            elif self.loss_function == "nae":
                loss, training_dict = self.model.train_step_nae(x)
            elif self.loss_function == "wnae":
                loss, training_dict = self.model.train_step(x)
            loss.backward()
            optimizer.step()

            monitored_quantities["loss"] += training_dict["loss"]
            #print(training_dict["mcmc_data"]["samples"][-1][:10])

        monitored_quantities["loss"] /= n_batches

        return monitored_quantities

    def __evaluate(self, loader):
        self.model.eval()
        x = next(iter(loader))[0]
        return self.model.evaluate(x)
  
    def __validate_epoch(self, validation_loader):

        self.model.eval()

        # Can monitore more quantities than the loss, showing loss as example
        monitored_quantities = {
            "loss": 0.,
            "reco_errors": None
        }
        
        bar_format = '{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        n_batches = 0
        for batch in tqdm(validation_loader, bar_format=bar_format):
            n_batches += 1
            x = batch[0]  # x is a list of len 1 with the tensor inside

            validation_dict = self.model.validation_step(x)
            monitored_quantities["loss"] += validation_dict["loss"]
            if monitored_quantities["reco_errors"] == None:
                monitored_quantities["reco_errors"] = validation_dict["reco_errors"]
            else:
                monitored_quantities["reco_errors"] = torch.concat((monitored_quantities["reco_errors"], validation_dict["reco_errors"]))

            if n_batches == 1: # only store the MCMC samples for visualization purpose for one batch
                monitored_quantities["mcmc_samples"] = validation_dict["mcmc_data"]["samples"][-1]
                monitored_quantities["positive_samples"] = x.detach()
                
        monitored_quantities["loss"] /= n_batches

        return monitored_quantities

    def __save_model_checkpoint(self, name, state_dict_only=False):
        path = f"{self.output_path}/{name}.pt"
        if state_dict_only:
            torch.save({"model_state_dict": self.model.state_dict()}, path)
        else:
            torch.save(self.model, path)
        log.info(f"Saved model checkpoint {path}")

    def __fit(self,
              n_epochs,
              optimizer,
              lr_scheduler,
              es_patience,
        ):
        """Fit model.

        Args:
            n_epochs (int): Max number of epochs
            optimizer (torch.optim)
            lr_scheduler (torch.optim.lr_scheduler)
            es_patience (int): Number of epochs for early stop        
        """

        training_loader = self.loader.training_loader
        validation_loader = self.loader.validation_loader
        validation_loader_no_batch = self.loader.validation_loader_no_batch
        ood_loader = self.loader.ood_loader

        best_epoch = 0
        lowest_validation_loss = np.inf
        early_stopping_counter = 0
        early_stopped = False
        
        for i_epoch in range(self.epoch, n_epochs):
            
            self.epoch = i_epoch
            
            # Training and evaluation
            log.info("\nEpoch %d/%d Training" % (i_epoch, n_epochs))

            t0 = time.time()
            
            training_monitored_quantities = self.__train_epoch(
                training_loader,
                optimizer,
            )

            training_loss = training_monitored_quantities["loss"]
            
            if lr_scheduler is not None:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(training_loss)
                else:
                    lr_scheduler.step()
                    
            # lr_value = lr_scheduler._last_lr
                
            log.info("\nEpoch %d/%d Background Evaluation" % (i_epoch, n_epochs))
            validation_monitored_quantities = self.__validate_epoch(validation_loader)
            
            self.__make_sample_feature_1D_plot(validation_monitored_quantities["positive_samples"], validation_monitored_quantities["mcmc_samples"])
            
            training_loss = training_monitored_quantities["loss"]
            validation_loss = validation_monitored_quantities["loss"]
            background_reco_errors = validation_monitored_quantities["reco_errors"]
            
            # background_reco_errors = self.__evaluate(validation_loader_no_batch)["reco_errors"]
            # log.info("\nEpoch %d/%d Background Evaluation" % (i_epoch, n_epochs))
            
            log.info("\nEpoch %d/%d OOD Evaluation" % (i_epoch, n_epochs))
            signal_reco_errors = self.__evaluate(ood_loader)["reco_errors"]
            
            y_true = np.concatenate((np.zeros(len(background_reco_errors)), np.ones(len(signal_reco_errors))))
            y_pred = np.concatenate((background_reco_errors, signal_reco_errors))
            auc = roc_auc_score(y_true, y_pred)

            self.metrics_tracker["epoch"].append(i_epoch)
            self.metrics_tracker["training_loss"].append(training_loss)
            self.metrics_tracker["validation_loss"].append(validation_loss)
            self.metrics_tracker["auc"].append(auc)
            
            # LR scheduler step
            if lr_scheduler is not None:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(training_loss)
                else:
                    lr_scheduler.step()

            t1 = time.time()
            elapsed = t1 - t0

            log.info(f"{elapsed:.2f} s/epoch - loss: {training_loss:.3f} "
                     f"- validation loss: {validation_loss:.3f}")

            metrics_file = f"{self.output_path}/training.csv"
            pd.DataFrame(self.metrics_tracker).to_csv(metrics_file, index=False)

            # Early stopping
            if validation_loss < lowest_validation_loss:
                log.info(f"Validation loss improved from {lowest_validation_loss:.3f} to {validation_loss:.3f}. Saving checkpoint.")
                self.__save_model_checkpoint(name="best")
                lowest_validation_loss = validation_loss
                early_stopping_counter = 0
                best_epoch = i_epoch
            else:
                early_stopping_counter += 1

            if early_stopping_counter > es_patience:
                early_stopped = True
                log.info(f"Epoch {i_epoch}: early stopping")
                break

        self.__save_model_checkpoint(name="last_epoch")

        metrics_file = f"{self.output_path}/training.csv"
        log.info(f"Saving metrics file {metrics_file}")
        pd.DataFrame(self.metrics_tracker).to_csv(metrics_file, index=False)
        with open(f"{self.output_path}/info.txt", "w") as file:
            file.write(f"Best epoch: {best_epoch}\n")
            if early_stopped:
                file.write(f"Early stopping at epoch {i_epoch}.\n")

    def train(self):
        """
        @mandatory
        Runs the training of the previously prepared model on the normalized data
        """

        log.info("Starting model fitting")

        optimizer_args = {
            "params": self.model.parameters(),
            "lr": float(self.config['training']["learning_rate"]),
        }
        torch_optimizer = getattr(torch.optim, self.config['training']["optimizer"])
        optimizer = torch_optimizer(**optimizer_args)
        
        if self.config['training']["lr_scheduler"] is not None:
            lr_scheduler = getattr(torch.optim.lr_scheduler, self.config['training']["lr_scheduler"])(
                optimizer,
                **self.config['training']["lr_scheduler_args"]
            )
        else:
            lr_scheduler = None

        self.__fit(
            n_epochs=self.config['training']["n_epochs"],
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            es_patience=self.config['training']["es_patience"],
        )

        log.info("Finished training")

    def __get_model(self):
        """
        Builds an auto-encoder model as specified in object's fields: input_size,
        intermediate_architecture and bottleneck_size
        """

        model = WNAE(
            encoder=self.encoder,
            decoder=self.decoder,
            **self.config["wnae"],
        )
        model.to(self.device)
        return model
