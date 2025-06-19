from pathlib import Path
from pydantic import BaseModel,model_validator,Field
import yaml
import pickle
from typing import Union,List
import math
import numpy as np
import regex as re
import subprocess
import time
from tqdm import tqdm
import argparse
# Command line arguments
parser = argparse.ArgumentParser(description="Orchestrator for dynamic sampling with exploitation or exploration.")
parser.add_argument("--save_dir",type=str,required=True ,help="Name of the saving directory. In case the directory exists with a saved state resume will be done from the saved state.")
args = parser.parse_args()
# End command line arguments
total_train_steps = 9578 # Total steps to run the training for.
exploitation_flag = False #If true we use 1/(num_datasets)**2 *sqrt(iteration) else we use log10 based exploration. 
yaml_file_path = Path("/cb/home/harshitr/ws/online_mixing/sample_config.yaml")
save_path = Path.cwd() / args.save_dir / "save_state.pkl"
current_trainer_log_path = Path.cwd() / args.save_dir / "current_trainer.log"
run_command = f"bash /cra-614/workdirs/11062025_data_mix_expt/scripts/gpt2_run_odm.sh"

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
            raise ValueError(f"Different extension than yaml")
        return self
    def read_yaml(self) -> dict:
        with self.file_path.open('r',encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data

    def update_yaml(self,new_weights):
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
        YamlReader.save_yaml(yaml_file,self.file_path)
        
    @classmethod
    def save_yaml(cls,yaml_file:Union[dict, list],save_path:Path):
        '''
        Saves the yaml file at the path save_path
        '''
        with save_path.open('w',encoding='utf-8') as f:
            yaml.safe_dump(yaml_file,f,default_flow_style=False,sort_keys=False)


class SmoothedMeanWeightUpdater:
    def __init__(self,dataset_names,weights,smoothing_factor=0.9):
        '''
        dataset names is a list of datasets 
        weights is the starting set of weights.
        '''
        self.dataset_names = dataset_names
        self.dataset_map = {name: i for i, name in enumerate(dataset_names)}
        self.num_datasets = len(dataset_names)
        self.weights = weights if weights is not None else [1/len(dataset_names)]
        total_weights = np.sum(weights)
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
        if exploitation_flag:
            self.eps = 1/((self.num_datasets**2) * math.sqrt(iteration))
        else:
            self.eps = min(1/self.num_datasets, math.sqrt(math.log10(self.num_datasets)/(self.num_datasets*iteration)))
        # calculate scaling factor
        total_estimated_rewards = sum([math.exp(r*self.prev_eps) for r in self._estimated_reward.values()])
        scaling_factor = (1-self.num_datasets*self.eps)/total_estimated_rewards

        # update weights
        for name in self.dataset_names:
            self.weights[self.dataset_map[name]] = math.exp(self._estimated_reward[name]*self.prev_eps)*scaling_factor + self.eps

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
        if exploitation_flag:
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
        for name in self.dataset_names:
            # self.weights[self.dataset_map[name]] = math.exp(self._estimated_reward[name]*self.prev_eps)*scaling_factor + self.eps
            self.weights[self.dataset_map[name]] = self._estimated_reward[name]*self.prev_eps*scaling_factor + self.eps
        # update probabilities
        total_weights = sum(self.weights)
        for name in self.dataset_names:
            self._probabilities[name] = self.weights[self.dataset_map[name]]/total_weights
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
    @model_validator(mode='after')
    def validate_model(self):
        if not self.reward_path.is_file():
            raise FileNotFoundError(f"Reward file {self.reward_path} does not exist.")
    def read_rewards(self):
        self.avg_loss_list = []
        with self.reward_path.open('r') as f:
            lines = f.readlines()
        pattern = re.compile(r"Avg Eval Loss: ([\d\.]+)")
        for line in lines:
            match = pattern.search(line)
            if match:
                self.avg_loss_list.append(float(match.group(1)))


class Orchestrator:
    def __init__(self,yaml_file_path:Path,save_path:Path,total_train_steps:int):
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
        else:
            self.total_train_steps = total_train_steps
            self.save_path = save_path
            print(f"\033[1;34mNo saved state detected at {self.save_path}. Starting from scratch.\033[0m")
            self.yaml_file_path = yaml_file_path
            if not self.yaml_file_path.is_file():
                raise FileNotFoundError(f"Yaml file {self.yaml_file_path} does not exist.")
            try:
                self.save_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Error creating directory: {e}")

            self.yaml_reader_obj = YamlReader(file_path=self.yaml_file_path)

            self.update_weight_obj = SmoothedMeanWeightUpdater(dataset_names=self.get_dataset_dirs(),weights=self.get_initial_weights())

            self.checkpoint_yaml_list = []

            self.model_save_path = Path(self.yaml_reader_obj.read_yaml()['trainer']['init']['model_dir'])

            print(f"!!!Please ensure that model is saved at {self.model_save_path}!!!")
            if not self.model_save_path.is_absolute():
                raise ValueError(f"Provided model_dir path in config is not absolute: {self.model_save_path}")            

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

    
    def get_latest_rewards(self):
        '''
        Reads the reward log file and returns the latest rewards.
        Takes the latest num_datasets avg eval loss from the log file.
        '''
        self.reward_reader_obj = Read_Reward(reward_path=self.get_reward_log_path())
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

    def update_weights_and_save_obj(self):
        '''
        Updates the weights using the latest rewards and saves the update_weight_obj as a pickle file.
        '''
        latest_rewards_list = self.get_latest_rewards()
        latest_rewards_dict = {name:reward for name,reward in zip(self.update_weight_obj.dataset_names,latest_rewards_list)}
        # Append the latest rewards to the reward log list
        (self.update_weight_obj).reward_log_list.append(latest_rewards_dict)

        # Update the weights using the latest rewards and increse the iteration count
        new_prob_list = self.update_weight_obj.group_update(dataset_names=self.update_weight_obj.dataset_names,rewards=latest_rewards_list,iteration=self.update_weight_obj.iter_count)
        self.update_weight_obj.iter_count += 1
        new_prob_dict = {name:prob for name,prob in zip(self.update_weight_obj.dataset_names,new_prob_list)}
        # Append the new probabilities to the weight log list
        (self.update_weight_obj).weight_log_list.append(new_prob_dict)
        # Save the object as a pickle file
        self.save_state(file_path=self.save_path)
        # Overwrite the yaml file with the new weights
        self.yaml_reader_obj.update_yaml(new_weights=new_prob_dict)
    
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

        with current_trainer_log_path.open(mode='w') as log_file:
            print(f"Running the script: {run_command}")
            log_file.write(f"Running the script: {run_command}\n")
            process = subprocess.Popen(
                run_command,  
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
        print("Evals completed.")
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
            self.yaml_reader_obj.update_yaml(new_weights=self.update_weight_obj._probabilities)
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
    def run_without_eval(self,num_extra_checkpoints):
        self.run_script_parallel()
        initial_num_checkpoints = len(self.checkpoint_yaml_list)
        num_cur_checkpoints = len(self.checkpoint_yaml_list)
        print(f"Current number of checkpoints: {num_cur_checkpoints}. Waiting for {num_extra_checkpoints} more checkpoints to be created before checkpoint at eval or end.")
        while num_cur_checkpoints < initial_num_checkpoints + num_extra_checkpoints:
            time.sleep(20)
            num_cur_checkpoints = len(self.get_completed_checkpoints())
            if(num_cur_checkpoints > len(self.checkpoint_yaml_list)):
                print(f"New checkpoints found. Saving the state with the new checkpoints.")
                self.checkpoint_yaml_list = self.get_completed_checkpoints()
                self.save_state(file_path=self.save_path)
        print(f"Completed {num_extra_checkpoints} extra checkpoints. Proceeding to eval or end.")

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
            if completed_steps+run_steps == self.total_train_steps:
                print(f"***Running the final steps without eval at end.***")
                num_steps_without_eval = math.ceil(run_steps / checkpoint_steps)
                weight_update_at_end=0
            else:
                num_steps_without_eval = run_steps/checkpoint_steps - 1
                weight_update_at_end = 1
            self.run_without_eval(num_extra_checkpoints=num_steps_without_eval)
            if weight_update_at_end:
                print(f"Running the eval step after the last training step.")
                self.perform_eval_and_update_weights()
            pbar.update(run_steps)
            completed_steps += run_steps
            print("**\n \n**")
        pbar.close()            

if __name__ == "__main__":
    orchestrator_obj = Orchestrator(yaml_file_path=yaml_file_path,save_path=save_path,total_train_steps=total_train_steps)
    orchestrator_obj.main()

    # orchestrator.update_weights_and_save_obj()
    # orchestrator.update_weights_and_save_obj()
    # orchestrator.update_weights_and_save_obj()
    # orchestrator.update_weights_and_save_obj()
    # orchestrator.update_weights_and_save_obj()
    # orchestrator.update_weights_and_save_obj()

    

