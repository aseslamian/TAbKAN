# ===== ./tabkan/tuner.py =====
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

class OptunaTuner:
    def __init__(self, model_class, dataset, search_space, device='cuda'):
        """
        Initializes the OptunaTuner.

        Args:
            model_class: The model class to be tuned.
            dataset (dict): A dictionary containing the training and testing data.
            search_space (dict): A dictionary defining the hyperparameter search space for Optuna.
            device (str): The device to run the training on.
        """
        self.model_class = model_class
        self.dataset = dataset
        self.search_space = search_space
        self.device = device
        self.input_shape = dataset['train_input'].shape[1]
        self.output_shape = len(torch.unique(dataset['train_label']))

    def _objective(self, trial):
        """
        The objective function for Optuna optimization.
        This function is called by Optuna for each trial. It builds a model
        with a set of hyperparameters suggested by Optuna, trains it,
        and returns a score.

        Args:
            trial (optuna.trial.Trial): A trial object from Optuna.

        Returns:
            float: The f1 score of the model.
        """
        # --- Build Model Architecture Hyperparameters ---
        model_constructor_args = {}

        depth = trial.suggest_int('depth', self.search_space['depth']['low'], self.search_space['depth']['high'])

        layers = [self.input_shape]
        for i in range(depth):
            neurons = trial.suggest_int(f'neurons_layer_{i}',
                                      self.search_space[f'neurons_layer_{i}']['low'],
                                      self.search_space[f'neurons_layer_{i}']['high'])
            layers.append(neurons)
        layers.append(self.output_shape)

        if 'SplineKAN' in self.model_class.__name__:
            model_constructor_args['width'] = layers
        else:
            model_constructor_args['layers'] = layers

        # Dynamically find other architectural params like 'orders' or 'gridsizes'
        # This allows the tuner to be flexible and work with different KAN models
        # without having to hardcode the hyperparameter names.
        other_arch_params = {}
        for key in self.search_space:
            if "_layer_" in key and key not in ['depth'] and not key.startswith('neurons_'):
                # E.g., key is 'orders1_layer_0'. param_base_name becomes 'orders1'.
                param_base_name = key.split('_layer_')[0]
                if param_base_name not in other_arch_params:
                    other_arch_params[param_base_name] = []

        # Now, build the lists for each found parameter (e.g., one for 'orders1', one for 'orders2')
        for param_base_name in other_arch_params:
            param_list = []
            for i in range(depth):
                param_key = f'{param_base_name}_layer_{i}'
                param_config = self.search_space[param_key]

                # Suggest the hyperparameter value from Optuna
                value = trial.suggest_int(param_key, param_config['low'], param_config['high'])
                param_list.append(value)

            # Add the complete list to the constructor arguments
            model_constructor_args[param_base_name] = param_list


        if 'grid' in self.search_space:
            grid_config = self.search_space['grid']
            model_constructor_args['grid'] = trial.suggest_int('grid', grid_config['low'], grid_config['high'])

        if 'k' in self.search_space:
            k_config = self.search_space['k']
            model_constructor_args['k'] = trial.suggest_int('k', k_config['low'], k_config['high'])
        # --- Get Training Hyperparameters ---
        fit_kwargs = {}

        # Handle learning rate (lr)
        if 'lr' in self.search_space:
            lr_config = self.search_space['lr'].copy() # Make a copy
            lr_config.pop('type', None) # Remove the 'type' key
            fit_kwargs['lr'] = trial.suggest_float('lr', **lr_config)

        # Handle steps
        if 'steps' in self.search_space:
            steps_config = self.search_space['steps'].copy()
            steps_config.pop('type', None)
            fit_kwargs['steps'] = trial.suggest_categorical('steps', **steps_config)

        # --- Instantiate, Train, and Evaluate ---
        model = self.model_class(**model_constructor_args).to(self.device)

        model.fit(self.dataset, loss_fn=nn.CrossEntropyLoss(), **fit_kwargs)

        model.eval()
        with torch.no_grad():
            y_score = model(self.dataset['test_input'].to(self.device)).cpu()
            y_pred = torch.argmax(y_score, dim=1)
            y_true = self.dataset['test_label'].cpu() # Ensure label is on CPU for sklearn

        return f1_score(y_true, y_pred, average="macro")

    def tune(self, n_trials=50, direction="maximize"):
        """
        Tunes the hyperparameters of the model using Optuna.

        Args:
            n_trials (int): The number of trials to run.
            direction (str): The direction of optimization ('maximize' or 'minimize').

        Returns:
            dict: A dictionary containing the best hyperparameters.
        """
        study = optuna.create_study(direction=direction)
        study.optimize(self._objective, n_trials=n_trials, show_progress_bar=True)
        best_params = study.best_params
        return best_params

