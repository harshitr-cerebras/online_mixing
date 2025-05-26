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
#TODO: Discuss if I need to build checkpointing capabilities in the orchestrator.
#DIS: Checkpointing should be in sync with evaluation steps. Atleast at each eval steps the checkpoint should be saved.
# Only thing that needs to be changed is the path of the yaml file which will be used to run the model. and where the orchestrator logs are saved.
yaml_file_path = Path("configs/params_gpt2_tiny.yaml")
save_path = Path.cwd() / "dynamic_sampling" / "update_obj.pkl" # Path to save the weight update object asa pickle so that it can be used for debugging.
run_command = f"python run.py CPU --params {yaml_file_path} --mode train_and_eval"

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
        self.eps = min(1/self.num_datasets, math.sqrt(math.log(self.num_datasets)/(self.num_datasets*iteration)))

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
        self.prev_eps = self.eps
        self.eps = min(1/self.num_datasets, math.sqrt(math.log(self.num_datasets)/(self.num_datasets*iteration)))

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
    def __init__(self,yaml_file_path:Path,save_path:Path):
        '''
        Initializes the orchestrator with the yaml file path and the save path.
        A object to handle yaml file reading and writing is created.
        A object to read the reward log file is created.
        A object to update the weights is created.
        The update_weight_obj is saved as a pickle file at the save path. The object stores the weights and rewards as dictionaries nested in seperate lists.
        '''
        self.yaml_file_path = yaml_file_path
        if not self.yaml_file_path.is_file():
            raise FileNotFoundError(f"Yaml file {self.yaml_file_path} does not exist.")
        self.save_path = save_path
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory: {e}")

        self.yaml_reader_obj = YamlReader(file_path=self.yaml_file_path)

        self.update_weight_obj = SmoothedMeanWeightUpdater(dataset_names=self.get_dataset_dirs(),weights=self.get_initial_weights())

        #TODO: Initialize it later after 1 eval is run
        # self.reward_reader_obj = Read_Reward(reward_path=self.get_reward_log_path())

    def get_dataset_dirs(self):
        '''
        Reads the yaml file and returns the dataset directories from the mixture.
        '''
        yaml_file = self.yaml_reader_obj.read_yaml()
        dataset_dirs = [x['data_dir'] for x in yaml_file['trainer']['fit']['train_dataloader']['mixture']]
        return dataset_dirs

    def get_initial_weights(self):
        '''
        Reads the yaml file and returns the initial weights from the mixture.
        '''
        yaml_file = self.yaml_reader_obj.read_yaml()
        dataset_weights = [x['weight'] for x in yaml_file['trainer']['fit']['train_dataloader']['mixture']]
        return dataset_weights

    def get_reward_log_path(self):
        #FIXME: This works assuming run.py is run from the same directory as the orchestrator. Does it need to be fixed and if yes how?
        '''
        Reads the yaml file and returns the reward log path from the model directory.
        '''
        yaml_file = self.yaml_reader_obj.read_yaml()
        reward_log_path = Path.cwd() / yaml_file['trainer']['init']['model_dir'] / "cerebras_logs" / "latest" / "run.log"
        return reward_log_path

    
    def get_latest_rewards(self):
        '''
        Reads the reward log file and returns the latest rewards.
        Takes the latest num_datasets avg eval loss from the log file.
        '''
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

    def get_total_steps(self):
        '''
        Returns the total steps from the yaml file.
        '''
        yaml_file = self.yaml_reader_obj.read_yaml()
        total_steps = yaml_file['trainer']['init']['loop']['max_steps']
        return total_steps
    
    def get_eval_steps(self):
        #TODO: Discuss regarding making this compatible with eval frequency. What to do if exactly one is present and not the other.
        '''
        Returns the eval steps from the yaml file.
        After this many steps the model is evaluated and the avg eval loss is logged.
        So this is the number of steps after which the weights are updated.
        '''
        yaml_file = self.yaml_reader_obj.read_yaml()
        eval_steps = yaml_file['trainer']['init']['loop']['eval_steps']
        return eval_steps

    def get_checkpoint_steps(self)->int:
        '''
        Returns the checkpoint steps from the yaml file.
        After this many steps the model is checkpointed.
        '''
        yaml_file = self.yaml_reader_obj.read_yaml()
        checkpoint_steps = yaml_file['trainer']['init']['checkpoint']['steps']
        return checkpoint_steps

    def check_killing_criterion(self)->bool:
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
        process = subprocess.Popen(
            run_command,  
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
            time.sleep(10)
        print("Log file has been modified by the latest call.\nWaiting for eval to be done to kill the process")


    def kill_process(self,process:subprocess.Popen,call_time:float):
        '''
        call_time is the time when the process was called. We need to wait till the modified time of the log file is greater than the call time.
        Kill the process if it is running and check_killing_criterion is met.
        wait_time is the time to wait between killing the process and checking the killing criterion.
        '''

        # While the process is running, check if the killing criterion is met. IF the process is not running, break the loop.
        self.wait_for_log_file(log_file_path=self.get_reward_log_path(),call_time=call_time)
        while(self.check_killing_criterion() is False and process.poll() is None):
            time.sleep(10)
        if process.poll() is None:
            print(f"Terminating process with PID: \033[1;31m{process.pid}\033[0m")
            process.terminate()
            try:
                process.wait(timeout=60) # Wait for the process to terminate for 60 seconds
                print(f"Process with PID: \033[1;31m{process.pid}\033[0m terminated gracefully.")
            except subprocess.TimeoutExpired:
                print(f"Process with PID: \033[1;31m{process.pid}\033[0m did not terminate. Killing it.")
                process.kill()
                process.wait()

    def main(self):
        '''
        Main function to run the orchestrator. Main entry point for the orchestrator.
        num_eval_steps is the total number of eval steps that will be run assuming that a final eval step is run after the last training step.
        '''
        #TODO: For simplicity assumed that total train steps are divisible by total eval steps. DIscuss what if that is not the case.
        total_train_steps = self.get_total_steps()
        eval_steps = self.get_eval_steps()
        checkpoint_steps = self.get_checkpoint_steps()

        if eval_steps % checkpoint_steps != 0:
            raise NotImplementedError(
                f"Eval steps {eval_steps} is not divisible by checkpoint steps {checkpoint_steps}. "
                "Restarting will cause a misalignment between evaluation and checkpointing steps."
            )
        
        if total_train_steps % eval_steps != 0:
            #FIXME: Discuss and confirm if the implementation should be changed to handle this case.
            num_eval_steps = total_train_steps // eval_steps + 1
            print(f"Total train steps: {total_train_steps}, Eval steps: {eval_steps}, Num eval steps: {num_eval_steps}\n")
            raise NotImplementedError(f"Total train steps {total_train_steps} is not divisible by eval steps {eval_steps}.")
        
        else:
            num_eval_steps = total_train_steps // eval_steps
            print(f"Total train steps: {total_train_steps}, Eval steps: {eval_steps}, Num eval steps: {num_eval_steps}\n")
        
        # Total stops is num_eval_steps - 1. The last step is not included in the eval steps as training is finished.
        pbar=tqdm(total=num_eval_steps)
        for counter in range(1,num_eval_steps):
            print(f"Running script for eval step {counter} of {num_eval_steps}")
            process = self.run_script_parallel()
            self.kill_process(process=process,call_time=time.time())
            if counter==1:
                self.reward_reader_obj = Read_Reward(reward_path=self.get_reward_log_path())
            self.update_weights_and_save_obj()
            print(f"\033[1;32m***Step {counter} of {num_eval_steps} completed successfully ***\n\n\033[0m")
            pbar.update(1)
        # Final Run for the last checkpoint this one will terminate automatically
        process = self.run_script_parallel()
        pbar.write(f"Running the last step of the training process with pid: \033[1;31m{process.pid}\033[0m")    
        while process.poll() is None:
            time.sleep(10)
        pbar.write(f"\n\n\033[1;32m***Final step completed successfully ***\n\n\033[0m")
        pbar.update(1)
        pbar.close()

if __name__ == "__main__":
    orchestrator_obj = Orchestrator(yaml_file_path=yaml_file_path,save_path=save_path)
    orchestrator_obj.main()

    # orchestrator.update_weights_and_save_obj()
    # orchestrator.update_weights_and_save_obj()
    # orchestrator.update_weights_and_save_obj()
    # orchestrator.update_weights_and_save_obj()
    # orchestrator.update_weights_and_save_obj()
    # orchestrator.update_weights_and_save_obj()

    

