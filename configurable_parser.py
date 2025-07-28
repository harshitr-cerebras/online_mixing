'''
This parser implements a general class that takes a task name, directory path
and produces the avg of metrics specified in the parser_config.yaml file.
'''
from pathlib import Path
import glob
import json
import yaml
from typing import List, Dict, Tuple, Union
class AccuracyDownstreamParser:
    def __init__(self,path: Union[str, Path],task_name: str):
        '''
        Parses the accuracy information from a given path for a specific task.
        :param path: Path to the directory containing the accuracy files.
        :param task_name: Name of the task for which accuracy is being parsed.
        '''
        self.path = Path(path)
        with open('parser_config.yaml','r') as f:
            self.downstream_config = yaml.safe_load(f)
        if task_name not in self.downstream_config:
            raise ValueError(f"Task {task_name} is not defined in the downstream configuration.")
        if not self.path.is_dir():
            raise ValueError(f"The provided path {self.path} is not a directory.")
        self.task_name = task_name
    def find_results_file(self) ->Path:
        '''
        Finds the results file for the specified task.
        :return: Path to the results file.
        Raises error if no or multiple files are found.
        '''
        pattern = f"result*.json"
        files = list(self.path.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No results file found for task {self.task_name} in {self.path}.")
        if len(files) > 1:
            raise ValueError(f"Multiple results files found for task {self.task_name} in {self.path}: {files}. Please ensure there is only one results file.")
        return files[0]
    
    def parse_metrics(self) -> Dict[str, float]:
        '''
        Parses the metrics from the results file.
        :return: Dictionary containing the parsed metrics.
        '''
        results_file = self.find_results_file()
        with open(results_file, 'r') as f:
            data = json.load(f)
        data = data['results']
        if self.task_name not in data:
            raise KeyError(f"Task {self.task_name} not found in the results file.")
        data_task = data[self.task_name]
        metric_keys = self.downstream_config[self.task_name]
        metric_dict = {key:data_task[key] for key in metric_keys}
        print(f"Parsed metrics for task {self.task_name}: {metric_dict}")
        avg_metric = sum(metric_dict.values()) / len(metric_dict)
        print(f"Average metric for task {self.task_name}: {avg_metric}")
        return avg_metric

        

if __name__ == "__main__":
    # Example usage
    with open('parser_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    parser = AccuracyDownstreamParser("/cra-614/workdirs/11062025_data_mix_expt/artifacts/downstream_0.1_smoothing_0.5_downstream_importance/eval_dir/arc_easy/checkpoint_1198", "arc_easy")
    print(f"Parsed metrics for task {parser.task_name}: {parser.parse_metrics()}")  