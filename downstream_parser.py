from pathlib import Path
import glob
import json
from typing import List, Dict, Tuple, Union


class DownstreamParser:
    """
    A class for processing downstream evaluation directories containing JSONL files
    from evaluations like MMLU, ARC-Challenge, etc.
    """
    
    def __init__(self, eval_dir: Union[str, Path]):
        """
        Initialize the parser with the evaluation directory path.
        
        Args:
            eval_dir: Path to the directory containing JSONL evaluation files
        """
        self.eval_dir = Path(eval_dir)
        if not self.eval_dir.exists():
            raise FileNotFoundError(f"The specified path does not exist: {self.eval_dir}")
        if not self.eval_dir.is_dir():
            raise NotADirectoryError(f"The specified path is not a directory: {self.eval_dir}")
    
    def get_jsonl_files_list(self) -> List[str]:
        """Return a list of all JSONL files in the evaluation directory in sorted order."""
        return sorted(glob.glob(str(self.eval_dir / "*.jsonl")))
    
    def parse_jsonl(self, file_path: Union[str, Path]) -> List[Dict]:
        """
        Parse a JSONL file and return a list of dictionaries.
        
        Args:
            file_path: Path to the JSONL file
        
        Returns:
            List of dictionaries, one for each line in the JSONL file
        """
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if len(line) == 0:  # Skip empty lines
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num} in {file_path}: {e}")
                        continue
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
        
        return data
    
    def get_subject_and_log_likelihood(self, entry: Dict) -> Tuple[str, float]:
        """
        Extracts the subject and log likelihood of the correct answer 
        from a json entry in a MMLU or ARC-Challenge JSONL file.
        Returns "NA" as default subject for datasets without subject field.
        
        Args:
            entry: Dictionary representing a single entry from JSONL file
            
        Returns:
            Tuple of (subject, log_likelihood)
        """
        # Get subject with default fallback
        subject = entry.get("doc", {}).get("subject", "NA")
        
        # Get correct answer index - handle both "answer" and "answerKey" fields
        doc = entry.get("doc", {})
        if "answer" in doc:
            correct_answer_index = doc["answer"]
        elif "answerKey" in doc:
            answer_key = doc["answerKey"]
            # Handle numeric answer keys (e.g., '4') or letter answer keys (e.g., 'A')
            if answer_key.isdigit():
                correct_answer_index = int(answer_key) - 1 # Convert to 0-based index
            else:
                # Convert letter to index (A=0, B=1, C=2, D=3)
                correct_answer_index = ord(answer_key.upper()) - ord('A')
        else:
            raise KeyError("Neither 'answer' nor 'answerKey' found in doc")
        
        log_likelihood = float(entry["filtered_resps"][correct_answer_index][0])
        
        return subject, log_likelihood
    
    def process_one_file(self, file_path: Union[str, Path]) -> List[Tuple[str, float]]:
        """
        Process a single JSONL file and extract subjects and log likelihoods.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            List of tuples (subject, log_likelihood)
        """
        data = self.parse_jsonl(file_path)
        results = []
        
        for entry in data:
            try:
                subject, log_likelihood = self.get_subject_and_log_likelihood(entry)
                results.append((subject, log_likelihood))
            except KeyError as e:
                print(f"Key error in entry {entry}: {e}")
            except Exception as e:
                print(f"Error processing entry {entry}: {e}")
        
        return results
    
    def process_all_files(self) -> Dict[str, List[float]]:
        """
        Process all JSONL files in the evaluation directory and return combined results.
        
        Returns:
            Dictionary with subjects as keys and lists of log likelihoods as values
        """
        files_list = self.get_jsonl_files_list()
        all_results = {}
        
        for file_path in files_list:
            print(f"Processing file: {file_path}")
            results = self.process_one_file(file_path)
            
            for subject, log_likelihood in results:
                if subject not in all_results:
                    all_results[subject] = []
                all_results[subject].append(log_likelihood)
        
        return all_results
    
    def get_file_count(self) -> int:
        """Return the number of JSONL files found in the evaluation directory."""
        return len(self.get_jsonl_files_list())

if __name__ == "__main__":
    ## Testing code for mmlu
    mmlu_path = Path("/cb/cold/riturajj/gpu_setup/results/new_ift/chat/8B_7a_wd0.2_8b_bs512_3epoch/mmlu/__mnt__local__shared__riturajj__ckpts__merge__8B_7a_wd0.2_8b_bs512_3epoch")
    mmlu_parser = DownstreamParser(mmlu_path)
    # mmlu_files_list = mmlu_parser.get_jsonl_files_list()
    # first_file = mmlu_files_list[0]
    # parsed_data = mmlu_parser.parse_jsonl(first_file)
    # print(parsed_data[0])
    # print(mmlu_parser.get_subject_and_log_likelihood(parsed_data[0]))
    # print(f"Total files processed: {mmlu_parser.get_file_count()}")
    mmlu_res_dict = mmlu_parser.process_all_files()
    mmlu_sub_avg = {sub: sum(logs)/len(logs) for sub, logs in mmlu_res_dict.items()}
    print(mmlu_sub_avg)

    # ## Testing code for arc-challenge
    arc_path = Path("/net/nfs1/srv/nfs/scrap-pool/riturajj/gpu_setup/results/new_ift/chat/8B_7a_wd0.2_8b_bs512_3epoch/arc_challenge/__mnt__local__shared__riturajj__ckpts__merge__8B_7a_wd0.2_8b_bs512_3epoch")
    arc_parser = DownstreamParser(arc_path)
    # arc_files_list = arc_parser.get_jsonl_files_list()
    # print(arc_files_list)
    # first_arc_file = arc_files_list[0]
    # parsed_arc_data = arc_parser.parse_jsonl(first_arc_file)
    # print(parsed_arc_data[0])
    # print(arc_parser.get_subject_and_log_likelihood(parsed_arc_data[0]))
    arc_res_dict = arc_parser.process_all_files()
    # print(arc_res_dict.keys())
    arc_sub_avg = {sub: sum(logs)/len(logs) for sub, logs in arc_res_dict.items()}
    print(arc_sub_avg)