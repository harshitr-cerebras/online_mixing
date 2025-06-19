# ODM Script Usage Instructions

## Overview
This guide provides steps to correctly use the updated `odm.py` script. Follow each instruction to ensure the configuration is correct and debugging information is retained.

---
**Run $python new_odm.py --save_dir save_state in a tmux session.**

## Steps

1. **Use the Updated `new_odm.py`**
   - Make sure you are using the latest version of `odm.py`.

2. **Model Save Path Configuration**
   - The model save path **must be read from the config file**.
   -⚠️ **Do not use --model_dir** in the run script. 
   - model_dir in config must be absolute. Otherwise you would get an appropriate error.

4. **Update Training Parameters**
   -⚠️ Update the `total_train_steps` variable in new_odm.py to reflect the correct number of training steps.
   - Set the correct path for the YAML config file.
   - **Note:** The `max_steps` defined in the YAML file will be **overwritten** by the script.

5. **State Saving**
   - The dynamic sampling update object will be saved at:
     ```
     ./dynamic_sampling/save_state.pkl
     ```
   - This can be used to access all the saved weights later.
6. **Restarting**
   - In case of failure just rerun the new_odm.py file command.

---

## Notes
- Make sure all paths used in the config are valid and absolute.
