from pathlib import Path
from pydantic import BaseModel,model_validator,Field
import yaml
import pickle
from typing import Union,List,Dict
import math
import numpy as np
import regex as re
import subprocess
import time
from tqdm import tqdm
import argparse
from scipy.optimize import minimize
from downstream_parser import DownstreamParser
yaml_file_path = Path("/cra-614/workdirs/16072025_data_mix_downstream_testing/configs/downst_config.yaml")
run_command = f"bash scripts/gpt2_test.sh"
downstream_mapping = {
    'arc_challenge': {
        'NA': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.0, 0.0]
    }, 
    'arc_easy': {
        'NA': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.0, 0.0]
    }, 
    'mmlu': {
        'public_relations': [0.0, 0.0, 0.0, 1.0, 0.0], 
        'high_school_geography': [0.0, 1.0, 0.0, 0.0, 0.0], 
        'high_school_microeconomics': [0.0, 0.0, 0.0, 0.0, 1.0], 
        'us_foreign_policy': [0.0, 0.0, 0.0, 0.0, 1.0], 
        'college_biology': [1.0, 0.0, 0.0, 0.0, 0.0], 
        'nutrition': [0.0, 0.0, 1.0, 0.0, 0.0], 
        'college_medicine': [0.0, 0.0, 1.0, 0.0, 0.0], 
        'astronomy': [0.5, 0.5, 0.0, 0.0, 0.0], 
        'econometrics': [0.0, 0.0, 0.0, 0.0, 1.0], 
        'high_school_biology': [1.0, 0.0, 0.0, 0.0, 0.0], 
        'high_school_macroeconomics': [0.0, 0.0, 0.0, 0.0, 1.0], 
        'high_school_physics': [0.0, 0.0, 1.0, 0.0, 0.0], 
        'management': [0.0, 0.0, 1.0, 0.0, 0.0], 
        'high_school_government_and_politics': [0.0, 0.0, 0.0, 1.0, 0.0], 
        'medical_genetics': [1.0, 0.0, 0.0, 0.0, 0.0], 
        'prehistory': [0.0, 0.0, 0.0, 0.0, 1.0], 
        'human_aging': [0.0, 0.0, 1.0, 0.0, 0.0], 
        'professional_medicine': [0.0, 0.0, 1.0, 0.0, 0.0], 
        'high_school_european_history': [0.0, 0.0, 0.0, 0.0, 1.0], 
        'sociology': [0.0, 0.0, 0.0, 0.0, 1.0], 
        'high_school_us_history': [0.0, 0.0, 0.0, 0.0, 1.0], 
        'global_facts': [0.5, 0.5, 0.0, 0.0, 0.0], 
        'high_school_world_history': [0.0, 0.0, 0.0, 0.0, 1.0], 
        'high_school_chemistry': [0.0, 0.0, 1.0, 0.0, 0.0]
    }
}  # A mapping of downstream tasks to training datasets. Essentially telling the contribution of datasets to downstream performance.
# Command line arguments
parser = argparse.ArgumentParser(description="Orchestrator for dynamic sampling with exploitation or exploration.")
parser.add_argument("--save_dir",type=str,required=True ,help="Name of the saving directory. In case the directory exists with a saved state resume will be done from the saved state.")
parser.add_argument("--resume",action='store_true',default=False,help="If set, the script will resume from the saved state.Other flag except save_dir are ignored. .If not set, the script will start from scratch.")
parser.add_argument("--exploitation",action='store_true',default=False,help="If set, the exploitation strategy is used. If not set, exploration strategy is used.")
parser.add_argument("--prevent_uniform",action='store_true',default=False,help="If set, the uniform sampling is prevented. If not set, the uniform sampling is allowed.") #Useful for the case of exploration.
parser.add_argument("--prevent_oversampling",action='store_true',default=False,help="If set, the oversampling is prevented. If not set, the oversampling is allowed.")
parser.add_argument("--use_data_subset",action='store_true',default=False,help="If set, the data subset feature is used. If not the data  subset is kept at [0,1].")
parser.add_argument("--oversampling_factor",type=float,default=None,help="Oversampling factor to be used if oversampling is allowed. sampling weight<=oversampling_factor*weight_empirical")
parser.add_argument("--downstream_importance",type=float,default=0.5,help="Importance of downstream tasks in the overall reward. Default is 0.5.")
parser.add_argument("--use_accuracy",action='store_true',default=False,help="If set, the 1 - accuracy is used as reward signal instead of the negative log likelihood for downstream tasks. Default is False.")
args = parser.parse_args()
# End command line arguments
 # A mapping of downstream tasks to training datasets. Eseetntially telling the contribution of datasets to downstream performance.
use_data_subset = args.use_data_subset #If true, the data subset feature is used. If false, the data subset is kept at [0,1].
resume = args.resume
use_accuracy = args.use_accuracy #If true, the accuracy is used instead of the log likelihood for downstream tasks. Default is False.
prevent_uniform = args.prevent_uniform #If true, the uniform sampling is prevented. If false, the uniform sampling is allowed.
exploitation_flag = args.exploitation #If true we use 1/(num_datasets)**2 *sqrt(iteration) else we use log10 based exploration.
prevent_oversampling = args.prevent_oversampling #If true, the oversampling is prevented. If false, the oversampling is allowed.
downstream_importance = args.downstream_importance #Importance of downstream tasks in the overall reward. Default is 0.5.
oversampling_factor = args.oversampling_factor # Oversampling factor to be used if oversampling is allowed. sampling weight<=oversampling_factor*weight_empirical 
if prevent_oversampling and oversampling_factor is None:
    raise ValueError("Oversampling factor must be provided if prevent_oversampling is set to True.")
save_path = Path.cwd() / args.save_dir / "save_state.pkl"
current_trainer_log_path = Path.cwd() / args.save_dir / "current_trainer.log"
token_counts = [3345063936,2634899456,24480260096,6350389248,3362258944]
w_emp = [token_cnt/sum(token_counts) for token_cnt in token_counts]
# Sanity checks
if not yaml_file_path.is_file():
    raise FileNotFoundError(f"Yaml file {yaml_file_path} does not exist. Please provide a valid yaml file path.")
if not resume and save_path.is_file():
    raise FileExistsError(f"Save path {save_path} already exists. Please provide a different save path or use the --resume flag to resume from the saved state.")
if resume and not save_path.is_file():
    raise FileNotFoundError(f"Save path {save_path} does not exist. Please provide a valid save path to resume from.")

def fix_oversampling(w_emp:List[float],w_proposed:List[float],oversampling_factor:float=1.5) -> List[float]:
    """
    Adjusts the proposed weights to satisfy the oversampling constraint using optimization.
    It minimizes the L2 distance to the proposed weights subject to:
    1. w_i <= oversampling_factor * w_emp_i
    2. sum(w) = 1
    3. w_i >= 0
    """
    w_emp_np = np.array(w_emp)
    w_proposed_np = np.array(w_proposed)

    # Objective function: ||w - w_proposed||^2_2
    def objective_func(w, w_prop):
        return np.linalg.norm(w - w_prop, ord=2)**2

    # Inequality constraint: w_i <= oversampling_factor * empirical_i
    ineq_constraint = {
        'type': 'ineq',
        'fun': lambda w: oversampling_factor * w_emp_np - w
    }
    # Equality constraint: sum(w) = 1
    eq_constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    constraints = [ineq_constraint, eq_constraint]

    # Bounds for each element of w (w_i >= 0)
    bounds = [(0, 1) for _ in range(len(w_proposed_np))]

    # Initial guess for w
    initial_w = np.copy(w_proposed_np)

    # Solve the optimization problem
    result = minimize(
        fun=objective_func,
        x0=initial_w,
        args=(w_proposed_np,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if result.success:
        return result.x.tolist()
    else:
        print(f"Warning: Oversampling fix optimization failed: {result.message}")
        return w_proposed

def update_output_paths_downstream_tasks(config: Dict,checkpoint_number: int)-> Dict:
    """
    Updates the output_path for EleutherEvalHarness callbacks in the config to the format
    model_dir/eval_dir/task_name/checkpoint_{checkpoint_number}.
    """


    model_dir = Path(config['trainer']['init']['model_dir'])
    callbacks = config['trainer']['init']['callbacks']

    for callback in callbacks:
        if 'EleutherEvalHarness' in callback:
            harness_config = callback['EleutherEvalHarness']
            task_name = harness_config['eeh_args']['tasks']
            new_path = model_dir / 'eval_dir' / task_name / f'checkpoint_{checkpoint_number}'
            harness_config['eeh_args']['output_path'] = str(new_path)
    
    return config

def count_eleuther_callbacks(config: Dict) -> int:
    """
    Counts the number of EleutherEvalHarness callbacks in the config.
    """

    callbacks = config['trainer']['init']['callbacks']
    count = 0
    for callback in callbacks:
        if 'EleutherEvalHarness' in callback:
            count += 1
    return count

def save_pkl_obj(save_obj,file_path: Path) -> None:
    '''
    Save a pkl object to a file.
    '''
    with file_path.open('wb') as f:
        pickle.dump(save_obj,f)

class YamlReader(BaseModel):
    file_path:Path
    @model_validator(mode='after')
    def validate_model(self):
        file_path = self.file_path
        if not file_path.is_file():
            raise ValueError(f"{file_path} does not have a file")
        if file_path.suffix.lower() not in ['.yaml','.yml']:
            raise ValueError("Different extension than yaml")
        return self
    def read_yaml(self) -> dict:
        with self.file_path.open('r',encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data

    def update_yaml(self,new_weights,data_subset_low:dict):
        '''
        New weights is a dict of the format 
        data_dir:weight
        Updates the and overwrites the old yaml file with the new set of weights.
        '''
        if not isinstance(new_weights,dict):
            raise TypeError("New weights must be a dict")
        yaml_file = self.read_yaml()
        for storage_path_dict in yaml_file['trainer']['fit']['train_dataloader']['mixture']:
            cur_path = storage_path_dict['data_dir']
            if cur_path not in new_weights.keys():
                raise KeyError(f"Missing key in new weights:{cur_path}")
            storage_path_dict['weight'] = new_weights[cur_path]
            storage_path_dict['data_subset'] = f"{data_subset_low[cur_path]:.16f}-1.0"
        YamlReader.save_yaml(yaml_file,self.file_path)
        
    @classmethod
    def save_yaml(cls,yaml_file:Union[dict, list],save_path:Path):
        '''
        Saves the yaml file at the path save_path
        '''
        with save_path.open('w',encoding='utf-8') as f:
            yaml.safe_dump(yaml_file,f,default_flow_style=False,sort_keys=False)


class SmoothedMeanWeightUpdater:
    def __init__(self,dataset_names,weights,smoothing_factor=0.9,exploitation_flag=False, prevent_uniform=False,prevent_oversampling=False,oversampling_factor=1.5):
        '''
        dataset names is a list of datasets 
        weights is the starting set of weights.
        '''
        self.prevent_oversampling = prevent_oversampling
        self.oversampling_factor = oversampling_factor
        self.exploitation_flag = exploitation_flag
        self.dataset_names = dataset_names
        self.dataset_map = {name: i for i, name in enumerate(dataset_names)}
        self.num_datasets = len(dataset_names)
        self.weights = weights if weights is not None else [1/len(dataset_names)]
        self.prevent_uniform = prevent_uniform
        total_weights = sum(weights)
        self._probabilities = {name:weight/total_weights for name,weight in zip(self.dataset_names,self.weights)}
        self._estimated_reward = {name:0.0 for name in self.dataset_names}
        self.prev_eps = None
        self.eps = 1/self.num_datasets
        self.smoothing_factor=smoothing_factor
        self.iter_count=1
        self.weight_log_list = [{dataset_name:weight for dataset_name,weight in zip(dataset_names,weights)}]
        self.reward_log_list = [{name:reward for name,reward in self._estimated_reward.items()}]
    def update(self, dataset_name: str, reward: float, iteration: int) -> List[float]:
        """
        Updates the weights based on the provided reward.
        """

        # update cumulative estimated reward
        self._estimated_reward[dataset_name] = self.smoothing_factor*self._estimated_reward[dataset_name] + (1-self.smoothing_factor)*math.exp(reward)

        # calculate epsilons
        self.prev_eps = self.eps
        if self.exploitation_flag:
            self.eps = 1/((self.num_datasets**2) * math.sqrt(iteration))
        else:
            self.eps = min(1/self.num_datasets, math.sqrt(math.log10(self.num_datasets)/(self.num_datasets*iteration)))
        # calculate scaling factor
        total_estimated_rewards = sum([math.exp(r*self.prev_eps) for r in self._estimated_reward.values()])
        scaling_factor = (1-self.num_datasets*self.eps)/total_estimated_rewards

        # update weights
        if self.eps!= 1/self.num_datasets or not self.prevent_uniform:
            for name in self.dataset_names:
                self.weights[self.dataset_map[name]] = math.exp(self._estimated_reward[name]*self.prev_eps)*scaling_factor + self.eps
        else:
            print("Since prevent_uniform is set the weights will not be made uniform. We will continue with the current weights.")
            for name in self.dataset_names:
                self.weights[self.dataset_map[name]] = self._probabilities[name]

        # update probabilities
        total_weights = sum(self.weights)
        for name in self.dataset_names:
            self._probabilities[name] = self.weights[self.dataset_map[name]]/total_weights

        return list(self._probabilities.values())
    
    def group_update(self, dataset_names: List[str], rewards: List, iteration: int) -> List[float]:
        # calculate epsilons
        print("Printing current weights")
        print(self._probabilities)
        self.prev_eps = self.eps
        if self.exploitation_flag:
            self.eps = 1/((self.num_datasets**2) * math.sqrt(iteration))
        else:
            self.eps = min(1/self.num_datasets, math.sqrt(math.log10(self.num_datasets)/(self.num_datasets*iteration)))

        # update cumulative estimated reward
        for name, reward in zip(dataset_names, rewards):
            # smoothed mean
            # self._estimated_reward[name] = self.smoothing_factor*self._estimated_reward[name] + (1-self.smoothing_factor)*reward
            # smoothed exponentiated mean
            self._estimated_reward[name] = self.smoothing_factor*self._estimated_reward[name] + (1-self.smoothing_factor)*math.exp(reward)
        # print(f"Rank: {torch.distributed.get_rank()} -- estimated_reward {self._estimated_reward}")
        # calculate normalized scaling factor
        total_estimated_rewards = sum((r*self.prev_eps) for r in self._estimated_reward.values())
        scaling_factor = (1-self.num_datasets*self.eps)/total_estimated_rewards

        # update weights
        if self.eps!= 1/self.num_datasets or not self.prevent_uniform:
            for name in self.dataset_names:
                # self.weights[self.dataset_map[name]] = math.exp(self._estimated_reward[name]*self.prev_eps)*scaling_factor + self.eps
                self.weights[self.dataset_map[name]] = self._estimated_reward[name]*self.prev_eps*scaling_factor + self.eps
        else:
            print("Since prevent_uniform is set the weights will not be made uniform. We will continue with the current weights.")
            for name in self.dataset_names:
                self.weights[self.dataset_map[name]] = self._probabilities[name]
        # update probabilities
        total_weights = sum(self.weights)
        for name in self.dataset_names:
            self._probabilities[name] = self.weights[self.dataset_map[name]]/total_weights
        if self.prevent_oversampling:
            print("Prvent oversampling is set.")
            check_oversampling = False
            for w_dist,w_proposed in zip(w_emp,self._probabilities.values()):
                if w_proposed > self.oversampling_factor * w_dist:
                    check_oversampling = True
            if check_oversampling:
                print("Oversampling detected with",self._probabilities)
                new_probs = fix_oversampling(w_emp=w_emp,w_proposed=list(self._probabilities.values()),oversampling_factor=self.oversampling_factor)
                for name, new_prob in zip(self.dataset_names, new_probs):
                    self._probabilities[name] = new_prob
        print("Printing new weights")
        print(self._probabilities)
        return list(self._probabilities.values())

class Read_Reward(BaseModel):
    '''
    Uses regex to read the latest avg eval loss from a path of a log file.
    The log file path can be created from the path of the model directory.
    '''
    reward_path: Path
    avg_loss_list: List[float] = Field(default_factory=list)
    use_accuracy_flag: bool = False
    @model_validator(mode='after')
    def validate_model(self):
        if not self.reward_path.is_file():
            raise FileNotFoundError(f"Reward file {self.reward_path} does not exist.")
    def read_rewards(self):
        self.avg_loss_list = []
        with self.reward_path.open('r') as f:
            lines = f.readlines()
        if self.use_accuracy_flag:
            pattern = re.compile(r"eval/accuracy = ([\d\.]+)")
        else:
            pattern = re.compile(r"Avg Eval Loss: ([\d\.]+)")
        for line in lines:
            match = pattern.search(line)
            if match:
                if self.use_accuracy_flag:
                    self.avg_loss_list.append(1 - float(match.group(1)))
                else:
                    self.avg_loss_list.append(float(match.group(1)))



class Orchestrator:
    def __init__(self,yaml_file_path:Path,save_path:Path,total_train_steps:int, run_command:str, exploitation_flag:bool, current_trainer_log_path:Path, prevent_uniform:bool=False,use_data_subset:bool=False,downstream_importance:float=0.5,use_accuracy_flag:bool=False):
        '''
        Initializes the orchestrator with the yaml file path and the save path.
        A object to handle yaml file reading and writing is created.
        A object to read the reward log file is created.
        A object to update the weights is created.
        The update_weight_obj is saved as a pickle file at the save path. The object stores the weights and rewards as dictionaries nested in seperate lists.
        If the save path already exists, the object is loaded from the pickle file and the current object's attributes are updated with the loaded object's attributes.
        '''
        if save_path.is_file():
            print(f"\033[1;34mSaved state detected at {save_path}. Loading the state.\n"
                  "If you want to start from scratch, please delete the file and run the script again.\033[0m")
            loaded_obj = self.load_state(file_path=save_path)
            #Update the current object's attributes with the loaded object's attributes.
            self.__dict__.update(loaded_obj.__dict__) 
            self.get_restart_info()
        else:
            self.downstream_importance = downstream_importance
            self.use_data_subset = use_data_subset
            self.run_command = run_command
            self.exploitation_flag = exploitation_flag
            self.current_trainer_log_path = current_trainer_log_path
            self.total_train_steps = total_train_steps
            self.save_path = save_path
            print(f"\033[1;34mNo saved state detected at {self.save_path}. Starting from scratch.\033[0m")
            self.yaml_file_path = yaml_file_path
            self.prevent_uniform = prevent_uniform
            self.use_accuracy_flag = use_accuracy_flag
            if self.use_accuracy_flag:
                print("Using 1-accuracy as a signal instead of log likelihood.")
            if not self.yaml_file_path.is_file():
                raise FileNotFoundError(f"Yaml file {self.yaml_file_path} does not exist.")
            try:
                self.save_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Error creating directory: {e}")

            self.yaml_reader_obj = YamlReader(file_path=self.yaml_file_path)

            self.update_weight_obj = SmoothedMeanWeightUpdater(dataset_names=self.get_dataset_dirs(),weights=self.get_initial_weights(),exploitation_flag=self.exploitation_flag,prevent_uniform=self.prevent_uniform,
                                                               prevent_oversampling=prevent_oversampling,oversampling_factor=oversampling_factor)

            self.checkpoint_yaml_list = []

            self.model_save_path = Path(self.yaml_reader_obj.read_yaml()['trainer']['init']['model_dir'])
            self.data_subset_low = [{name: 0.0 for name in self.update_weight_obj.dataset_names}]
            self.yaml_reader_obj.update_yaml(new_weights=self.update_weight_obj._probabilities,data_subset_low=self.data_subset_low[0])
            self._downstream_log_list = [] 
            self.num_downstream_tasks = count_eleuther_callbacks(self.yaml_reader_obj.read_yaml())
            self.downstream_task_names = self.get_downstream_task_names(config=self.yaml_reader_obj.read_yaml())
            print(f"Total downstream tasks: {self.num_downstream_tasks}")
            print(f"Downstream task names: {self.downstream_task_names}")
            print(f"!!!Please ensure that model is saved at {self.model_save_path}!!!")
            if not self.model_save_path.is_absolute():
                raise ValueError(f"Provided model_dir path in config is not absolute: {self.model_save_path}")
            self.save_state(file_path=self.save_path)
            print("\033[1;35mSuccessfully initialized the orchestrator.\n \033[0m")            
    def get_restart_info(self):
        print(f"*** Begin print of state information ***")
        print("Restarting from the saved state.")
        print(f"Current trainer log path: {self.current_trainer_log_path}")
        print(f"Run command: {self.run_command}")
        print(f"Total train steps: {self.total_train_steps}")
        print(f"Save path: {self.save_path}")
        print(f"Yaml file path: {self.yaml_file_path}")
        print(f"Model save path: {self.model_save_path}")
        print(f"Exploitation flag: {self.exploitation_flag}")
        print(f"Prevent uniform sampling: {self.prevent_uniform}")
        print(f"Prevent oversampling: {self.update_weight_obj.prevent_oversampling}")
        print(f"Use data subset: {self.use_data_subset}")
        print(f"Use accuracy flag: {self.use_accuracy_flag}")
        if self.update_weight_obj.prevent_oversampling:
            print(f"Oversampling factor: {self.update_weight_obj.oversampling_factor}")
        print(f"*** End print of state information \n \n***")
    def get_dataset_dirs(self):
        '''
        Reads the yaml file and returns the dataset directories from the mixture.
        '''
        yaml_file = self.yaml_reader_obj.read_yaml()
        dataset_dirs = [x['data_dir'] for x in yaml_file['trainer']['fit']['train_dataloader']['mixture']]
        return dataset_dirs

    def get_initial_weights(self):
        '''
        Reads the yaml file and returns the current weights from the mixture.
        '''
        yaml_file = self.yaml_reader_obj.read_yaml()
        dataset_weights = [x['weight'] for x in yaml_file['trainer']['fit']['train_dataloader']['mixture']]
        return dataset_weights

    def get_reward_log_path(self):
        '''
        Reads the yaml file and returns the reward log path from the model directory.
        '''
        yaml_file = self.yaml_reader_obj.read_yaml()
        reward_log_path = self.model_save_path / "cerebras_logs" / "latest" / "run.log"
        return reward_log_path

    def get_downstream_task_names(self,config:Dict)->List[str]:
        """
        Extracts the downstream task names from the EleutherEvalHarness callbacks in the config.
        """
        task_names = []
        for callback in config['trainer']['init']['callbacks']:
            if 'EleutherEvalHarness' in callback:
                harness_config = callback['EleutherEvalHarness']
                task_names.append(harness_config['eeh_args']['tasks'])
        return task_names

    def get_latest_rewards(self):
        '''
        Reads the reward log file and returns the latest rewards.
        Takes the latest num_datasets avg eval loss from the log file.
        '''
        self.reward_reader_obj = Read_Reward(reward_path=self.get_reward_log_path(),use_accuracy_flag=self.use_accuracy_flag)
        self.reward_reader_obj.read_rewards()
        latest_rewards = self.reward_reader_obj.avg_loss_list[-self.update_weight_obj.num_datasets:]
        if len(latest_rewards)<self.update_weight_obj.num_datasets:
            raise ValueError(f"Not enough rewards in the log file. Found {len(latest_rewards)} rewards needed {self.update_weight_obj.num_datasets}.")
        return latest_rewards
    
    def save_state(self,file_path:Path):
        '''
        Saves the state of the orchestrator as a pickle file.
        '''
        with file_path.open('wb') as f:
            pickle.dump(self,f)
        print("\033[1;35mSuccessfully saved the object.\n \033[0m")

    def load_state(self,file_path:Path):
        '''
        Load the state of the orchestrator from a pickle file.
        '''
        with file_path.open('rb') as f:
            loaded_obj = pickle.load(f)
        if not isinstance(loaded_obj, Orchestrator):
            raise TypeError(f"Loaded object is not of type Orchestrator. Found {type(loaded_obj)}")
        return loaded_obj

    def update_data_subset_at_eval(self,new_prob:List[float]):
        processed_token_with_new_prob = [tokens_per_step * self.get_eval_frequency() * prob for prob in new_prob]
        processed_subset = [extra_tokens/ token_cnt for extra_tokens, token_cnt in zip(processed_token_with_new_prob, token_counts)]
        new_data_subset_low = {name: low + processed for name, low, processed in zip(self.update_weight_obj.dataset_names, self.data_subset_low[-1].values(), processed_subset)}
        for name in new_data_subset_low.keys():
            if new_data_subset_low[name] > 1.0:
                new_data_subset_low[name] = 0.0 #Doing a reset if we do not have enough tokens for till the next checkpoint.
            else:
                new_data_subset_low[name] = self.data_subset_low[-1][name]
        self.data_subset_low.append(new_data_subset_low)

    def get_downstream_task_output_dirs(self):
        '''
        Reads the output_path fields for the downstream tasks from the yaml file.
        '''
        config = self.yaml_reader_obj.read_yaml()

        task_dirs = []
        for callback in config['trainer']['init']['callbacks']:
            if 'EleutherEvalHarness' in callback:
                harness_config = callback['EleutherEvalHarness']
                output_path = harness_config['eeh_args']['output_path']
                task_dirs.append(Path(output_path))
        return task_dirs

    def get_downstream_rewards(self):
        '''
        Reads the downstream rewards from the downstream directories in the yaml.
        '''
        task_names = self.downstream_task_names
        downstream_task_output_dirs = self.get_downstream_task_output_dirs()
        downstream_log_likelihood={}
        for task_name, output_dir in zip(task_names, downstream_task_output_dirs):
            if not output_dir.is_dir():
                raise NotADirectoryError(f"Output directory for task {task_name} does not exist: {output_dir}")
            task_parser = DownstreamParser(eval_dir=output_dir,task_name=task_name,use_accuracy_flag=self.use_accuracy_flag)
            downstream_log_likelihood[task_name] = task_parser.process_all_files() # For each task we have a mapping of task: {subject:[eval loss on answer]}
        if self.use_accuracy_flag:
            downstream_rewards = {task_name: {subject:1-sum(log)/len(log) for subject, log in subject_logs.items()} for task_name, subject_logs in downstream_log_likelihood.items()}
        else:
            downstream_rewards = {task_name: {subject:-sum(log)/len(log) for subject, log in subject_logs.items()} for task_name, subject_logs in downstream_log_likelihood.items()} # - sum is used to convert log likelihood to negative log likelihood
        print(f"Downstream rewards: {downstream_rewards}")
        return downstream_rewards

    def calculate_downstream_contribution(self,downstream_rewards_dict:Dict[str,Dict[str,float]])->list:
        '''
        Calculates the downstream contribution to upstream reward.
        This is done by taking the average of the downstream rewards for each task.
        The downstream rewards are in the form of a dictionary with task names as keys and a dictionary of subjects and their rewards as values.
        '''
        downstream_contribution = [0.0]*(self.update_weight_obj.num_datasets)
        for task_name, subject_rewards in downstream_rewards_dict.items():
            if task_name not in downstream_mapping:
                print(f"Warning: Task {task_name} not found in downstream mapping. Skipping.")
                continue
            temp_task_map = downstream_mapping[task_name]
            for subject, reward in subject_rewards.items():
                if subject not in temp_task_map:
                    print(f"Skipping subject {subject} not in {task_name}")
                else:
                    print(f"Using subject {subject} found in task mapping for {task_name}.")
                    for i, weight in enumerate(temp_task_map[subject]):
                        downstream_contribution[i] += reward * weight
        return downstream_contribution

    def combine_rewards(self,latest_rewards:List[float],downstream_contribution_to_upstream_reward:List[float]) -> List[float]:
        """
        Combines the latest rewards with the downstream contribution to upstream reward.
        The latest rewards are the rewards from the training datasets and the downstream contribution is the contribution from the downstream tasks.
        """
        if len(latest_rewards) != self.update_weight_obj.num_datasets:
            raise ValueError(f"Latest rewards length {len(latest_rewards)} does not match number of datasets {self.update_weight_obj.num_datasets}.")
        if len(downstream_contribution_to_upstream_reward) != self.update_weight_obj.num_datasets:
            raise ValueError(f"Downstream contribution length {len(downstream_contribution_to_upstream_reward)} does not match number of datasets {self.update_weight_obj.num_datasets}.")
        combined_rewards = [up_reward*(1-self.downstream_importance) + downstream_contribution*self.downstream_importance for up_reward, downstream_contribution in zip(latest_rewards, downstream_contribution_to_upstream_reward)]
        print(f"Upstream rewards: {latest_rewards}")
        print(f"Downstream contribution: {downstream_contribution_to_upstream_reward}")
        print(f"Combined rewards: {combined_rewards}")
        return combined_rewards
        

    def update_weights_and_save_obj(self):
        '''
        Updates the weights using the latest rewards and saves the update_weight_obj as a pickle file.
        '''
        latest_rewards_list = self.get_latest_rewards()
        latest_rewards_dict = {name:reward for name,reward in zip(self.update_weight_obj.dataset_names,latest_rewards_list)}
        downstream_rewards_dict = self.get_downstream_rewards()
        self._downstream_log_list.append(downstream_rewards_dict)
        downstream_contribution_to_upstream_reward = self.calculate_downstream_contribution(downstream_rewards_dict=downstream_rewards_dict)
        # Append the latest rewards to the reward log list
        (self.update_weight_obj).reward_log_list.append(latest_rewards_dict)
        combined_rewards = self.combine_rewards(latest_rewards_list,downstream_contribution_to_upstream_reward)
        # Update the weights using the latest rewards and increse the iteration count
        new_prob_list = self.update_weight_obj.group_update(dataset_names=self.update_weight_obj.dataset_names,rewards=combined_rewards,iteration=self.update_weight_obj.iter_count)
        self.update_weight_obj.iter_count += 1
        new_prob_dict = {name:prob for name,prob in zip(self.update_weight_obj.dataset_names,new_prob_list)}
        # Append the new probabilities to the weight log list
        (self.update_weight_obj).weight_log_list.append(new_prob_dict)
        if self.use_data_subset:
            self.update_data_subset_at_eval(list(new_prob_dict.values()))
        # Save the object as a pickle file
        self.save_state(file_path=self.save_path)
        # Overwrite the yaml file with the new weights
        self.yaml_reader_obj.update_yaml(new_weights=new_prob_dict,data_subset_low=self.data_subset_low[-1])
    
    def get_eval_frequency(self):
        '''
        Returns the eval steps from the yaml file.
        After this many steps the model is evaluated and the avg eval loss is logged.
        So this is the number of steps after which the weights are updated.
        '''
        yaml_file = self.yaml_reader_obj.read_yaml()
        if 'eval_frequency' not in yaml_file['trainer']['init']['loop']:
            raise KeyError("eval_frequency not found in the yaml file. Please add it to the yaml file. eval_steps has no effect kindly only use only eval_frequency.")
        eval_steps = yaml_file['trainer']['init']['loop']['eval_frequency']
        return eval_steps

    def get_checkpoint_steps(self)->int:
        '''
        Returns the checkpoint steps from the yaml file.
        After this many steps the model is checkpointed.
        '''
        yaml_file = self.yaml_reader_obj.read_yaml()
        checkpoint_steps = yaml_file['trainer']['init']['checkpoint']['steps']
        return checkpoint_steps

    def evaluation_completion_criterion(self)->bool:
        '''
        Check if the killing criterion is met.
        '''
        check_str = "Evaluation completed successfully!"

        log_file_path = self.get_reward_log_path()
        ## If the log file is not present, return False
        ## If the log file is present, check if in the latest log all the datasets have been evaluated.
        if not log_file_path.is_file():
            return False
        with log_file_path.open('r') as f:
            content = f.read()
            count = content.count(check_str)
        if count >= (self.update_weight_obj).num_datasets:
            return True
        else:
            return False

    def run_script_parallel(self):
        '''
        Run the run.py script in parallel using subprocess.
        '''
        self.delete_load_ckpt()

        with self.current_trainer_log_path.open(mode='w') as log_file:
            print(f"Running the script: {self.run_command}")
            log_file.write(f"Running the script: {self.run_command}\n")
            process = subprocess.Popen(
                self.run_command,  
                shell=True,
                stdout=log_file,
                stderr=log_file,
                text=True
            )
        print(f"Started the training process with PID: \033[1;31m{process.pid}\033[0m")
        return process

    def wait_for_log_file(self,log_file_path:Path,call_time:float):
        '''
        Wait till log files modified time is greater than the call time.
        '''
        print("Waiting for log file to be modified")
        while True:
            if log_file_path.is_file():
                modified_time = log_file_path.stat().st_mtime
                if modified_time > call_time:
                    break
            time.sleep(20)
        print("Log file has been modified by the latest call.\nWaiting for eval to be done to kill the process")

    def downstream_eval_completion_criterion(self)->bool:
        '''
        Checks if the downstream evals are completed.
        This is done by matching the occurence of check_str in the downstream log files.
        '''
        check_str = "Saving Eleuther Eval Harness result"
        ## If the log file is not present, return False
        ## If the log file is present, check if in the latest log all the datasets have been evaluated.
        log_file_path = self.get_reward_log_path()
        if not log_file_path.is_file():
            return False
        with log_file_path.open('r') as f:
            content = f.read()
            count = content.count(check_str)
        if count >= self.num_downstream_tasks:
            return True
        else:
            return False
        

    def perform_eval_and_update_weights(self):
        '''
        Waits for the evals to be done and then updates the weights.
        This is done by checking the log file for the latest eval loss.
        '''
        print("Wait for the checkpoint at eval to be saved.")
        while len(self.get_completed_checkpoints()) == len(self.checkpoint_yaml_list):
            time.sleep(30)
        print("Checkpoint at eval has been saved. Proceeding to wait for evals to be done.")
        while(not self.evaluation_completion_criterion()):
            time.sleep(30)
        print("Upstream eval completed. Now waiting for downstream evals to be done.")
        while(not self.downstream_eval_completion_criterion()):
            time.sleep(30)
        print("Both upstream and downstream evals completed. Waiting for 5 minutes for the log fiels to be saved")
        time.sleep(5*60) # wait extra 3 minutes to ensure the files are saved
        if(self.use_data_subset):
            self.increment_data_subset_checkpoint()
        self.checkpoint_yaml_list = self.get_completed_checkpoints()
        self.update_weights_and_save_obj()
        print(f"Eval completed. Updated the weights and saved the object at {self.save_path}.")

        

    def get_completed_steps(self):
        '''
        Returns the number of completed steps from the yaml file. which have been checkpointed.
        '''
        if len(self.checkpoint_yaml_list) == 0:
            return 0
        else:
            last_checkpoint_name = self.checkpoint_yaml_list[-1]
            steps = int(last_checkpoint_name.split('_')[-1].split('.')[0])
            return steps
    
    def get_checkpoints_file_path(self):
        '''
        Returns the path to the checkpoints file.
        '''
        yaml_file = self.yaml_reader_obj.read_yaml()
        checkpoints_file_path = self.model_save_path / "checkpoints_index.yaml"
        return checkpoints_file_path
    
    def delete_load_ckpt(self):
        '''
        Deletes the load checkpoint from the yaml file if extra checkpoints are present.
        This is done to ensure that the latest checkpoint is loaded and not the one from the yaml file.
        This is useful when the training is restarted and the checkpoints are already present.
        '''
        print("In delete_load_ckpt function")
        if(len(self.checkpoint_yaml_list) != 0):
            yaml_file = self.yaml_reader_obj.read_yaml()
            if "ckpt_path" in yaml_file['trainer']['fit'].keys():
                print("Deleting the load checkpoint from the yaml file to load the latest checkpoint.")
                del yaml_file['trainer']['fit']['ckpt_path']
                self.yaml_reader_obj.save_yaml(yaml_file, self.yaml_file_path)
            else:
                print("No load checkpoint found in the yaml file. Not deleting anything.")
        else:
            print("No checkpoints found. Not deleting the load checkpoint from the yaml file.")

    def sync_checkpoints_start(self):
        '''
        Syncs the checkpoints with the yaml file.
        This ensures that the restart of the training will not cause a misalignment between evaluation and checkpointing steps.
        '''
        if len(self.checkpoint_yaml_list) !=0:
            print(f"Synced checkpoints with the yaml file at path {self.get_checkpoints_file_path()}. Total checkpoints: {len(self.checkpoint_yaml_list)}")
            self.yaml_reader_obj.update_yaml(new_weights=self.update_weight_obj._probabilities,data_subset_low=self.data_subset_low[-1])
            YamlReader.save_yaml(self.checkpoint_yaml_list, self.get_checkpoints_file_path())
        else:
            print(f"No checkpoints found. Starting from scratch.")


    def update_yaml_file_max_steps(self,steps:int):
        '''
        Updates the max steps in the yaml file.
        This is used to update the total steps in the yaml file.
        '''
        yaml_file = self.yaml_reader_obj.read_yaml()
        yaml_file['trainer']['init']['loop']['max_steps'] = steps
        self.yaml_reader_obj.save_yaml(yaml_file, self.yaml_file_path)
        print(f"Updated max steps in the yaml file to {steps}")
    def get_completed_checkpoints(self):
        '''
        Reads the checkpoints file and returns the list of checkpoints.
        '''
        checkpoints_file_path = self.get_checkpoints_file_path()
        if not checkpoints_file_path.is_file():
            return []
        else:
            with checkpoints_file_path.open('r') as f:
                checkpoints_list = yaml.safe_load(f)
            return checkpoints_list
    def increment_data_subset_checkpoint(self):
        '''
        Increments the data subset low at the finish of a checkpoint.
        '''
        processed_tokens = [tokens_per_step * self.get_checkpoint_steps() * prob for prob in self.update_weight_obj._probabilities.values()]
        processed_subset = [extra_tokens/ token_cnt for extra_tokens, token_cnt in zip(processed_tokens, token_counts)]
        new_data_subset_low = {name: low + processed for name, low, processed in zip(self.update_weight_obj.dataset_names, self.data_subset_low[-1].values(), processed_subset)}
        for name in new_data_subset_low.keys():
            if new_data_subset_low[name] > 1.0:
                new_data_subset_low[name] = 0.0 #Doing a reset if we do not have enough tokens for till the next checkpoint.
        self.data_subset_low.append(new_data_subset_low)

    def run_without_eval(self,num_extra_checkpoints):
        self.run_script_parallel()
        initial_num_checkpoints = len(self.checkpoint_yaml_list)
        num_cur_checkpoints = len(self.checkpoint_yaml_list)
        print(f"Current number of checkpoints: {num_cur_checkpoints}. Waiting for {num_extra_checkpoints} more checkpoints to be created before checkpoint at eval or end.")
        while num_cur_checkpoints < initial_num_checkpoints + num_extra_checkpoints:
            time.sleep(20)
            num_cur_checkpoints = len(self.get_completed_checkpoints())
            if(num_cur_checkpoints > len(self.checkpoint_yaml_list)):
                print("New checkpoints found. Saving the state with the new checkpoints.")
                self.checkpoint_yaml_list = self.get_completed_checkpoints()
                if(self.use_data_subset):
                    self.increment_data_subset_checkpoint()
                self.save_state(file_path=self.save_path)
        print(f"Completed {num_extra_checkpoints} extra checkpoints. Proceeding to eval or end.")

    def update_yaml_for_downstream_tasks(self,checkpoint_number:int):
        '''
        Updates the yaml file for downstream tasks with the checkpoint number.
        This is done to ensure that the evals for downstream tasks are always stored in a separate directory
        '''
        yaml_file = self.yaml_reader_obj.read_yaml()
        updated_yaml = update_output_paths_downstream_tasks(config=yaml_file,checkpoint_number=checkpoint_number)
        self.yaml_reader_obj.save_yaml(updated_yaml, self.yaml_file_path)
        print(f"Updated the yaml file for downstream tasks with the checkpoint number {checkpoint_number}.")
    def main(self):
        '''
        Main function to run the orchestrator. Main entry point for the orchestrator.
        num_eval_steps is the total number of eval steps that will be run assuming that a final eval step is run after the last training step.
        '''
        self.sync_checkpoints_start()
        eval_frequency = self.get_eval_frequency()
        checkpoint_steps = self.get_checkpoint_steps()

        #Error handling for eval frequency is not divisible by checkpoint steps.
        if eval_frequency % checkpoint_steps != 0:
            raise NotImplementedError(
                f"Eval steps {eval_frequency} is not divisible by checkpoint steps {checkpoint_steps}. \n"
                "Restarting will cause a misalignment between evaluation and checkpointing steps.\n"
                "Please adjust the eval_steps and checkpoint_steps in the yaml file.\n"
            )


        completed_steps = self.get_completed_steps()
        pbar = tqdm(total=self.total_train_steps, initial=completed_steps, desc="Training Steps Progress", unit="step")
        while completed_steps < self.total_train_steps:
            print(f"Completed steps: {completed_steps} out of total {self.total_train_steps}")
            if(completed_steps%eval_frequency==0):
                print(f"Completed steps in sync with evals")
                run_steps = min(eval_frequency, self.total_train_steps - completed_steps)
            else:
                next_sync_step = completed_steps + (eval_frequency - (completed_steps % eval_frequency))
                print(f"Next sync step: {next_sync_step} which is {next_sync_step - completed_steps} steps away from the current completed steps")
                run_steps = min(next_sync_step - completed_steps, self.total_train_steps - completed_steps)
            self.update_yaml_file_max_steps(steps=completed_steps + run_steps)
            self.update_yaml_for_downstream_tasks(checkpoint_number=completed_steps + run_steps)
            if completed_steps+run_steps == self.total_train_steps:
                print(f"***Running the final steps without eval at end.***")
                num_steps_without_eval = math.ceil(run_steps / checkpoint_steps)
                weight_update_at_end=0
            else:
                num_steps_without_eval = run_steps/checkpoint_steps - 1
                weight_update_at_end = 1
            self.run_without_eval(num_extra_checkpoints=num_steps_without_eval)
            if weight_update_at_end:
                print(f"Running the eval step after the last training step. Weights will be updated after this step.")
                self.perform_eval_and_update_weights()
            pbar.update(run_steps)
            completed_steps += run_steps
            print("**\n \n**")
        pbar.close()            

if __name__ == "__main__":
    total_train_steps = YamlReader(file_path=yaml_file_path).read_yaml()['trainer']['init']['loop']['max_steps']
    if(not resume):
        print(f"Total train steps: {total_train_steps}")
    max_position_embedding = YamlReader(file_path=yaml_file_path).read_yaml()['trainer']['init']['model']['max_position_embeddings']
    batch_size = YamlReader(file_path=yaml_file_path).read_yaml()['trainer']['fit']['train_dataloader']['batch_size']
    tokens_per_step = batch_size * max_position_embedding #Batch size * max position embedding gives the number of tokens processed in one step.
    print(f"Batch size: {batch_size}, Max position embedding: {max_position_embedding}, Tokens per step: {tokens_per_step}")
    if prevent_oversampling:
        print(f"Oversampling factor: {oversampling_factor}")
    orchestrator_obj = Orchestrator(
        yaml_file_path=yaml_file_path,
        save_path=save_path,
        total_train_steps=total_train_steps,
        run_command=run_command,
        exploitation_flag=exploitation_flag,
        current_trainer_log_path=current_trainer_log_path,
        prevent_uniform=prevent_uniform,
        use_data_subset=use_data_subset,
        downstream_importance=downstream_importance,
        use_accuracy_flag = use_accuracy
    )    
    orchestrator_obj.main()
    # orchestrator.update_weights_and_save_obj()
    # orchestrator.update_weights_and_save_obj()
    # orchestrator.update_weights_and_save_obj()
    # orchestrator.update_weights_and_save_obj()
    # orchestrator.update_weights_and_save_obj()
    # orchestrator.update_weights_and_save_obj()

    

