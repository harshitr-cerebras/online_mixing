# ODM Script Usage Instructions

## Overview

This guide provides steps to correctly use the updated `odm.py` script. Follow each instruction to ensure the configuration is correct and debugging information is retained.

---

## Usage

### Initial Run

Run in a `tmux` session. If the save directory already exists, and contains a saved state it will throw an error. Delete the directory and start again.

```bash
python odm_upstream.py --save_dir save_dir --exploitation --prevent_uniform --use_data_subset --prevent_oversampling --oversampling_factor 3.0
```

**Note:** Default exploitation is taken as False, data-subset is not used and oversampling is allowed.

```bash
python odm_downstream.py --save_dir downstream_save_dir --exploitation --prevent_uniform --use_data_subset --prevent_oversampling --oversampling_factor 3.0 --downstream_importance 0.5 --use_accuracy
```

### Resuming

```bash
python odm_upstream.py --save_dir save_dir --resume
python odm_downstream.py --save_dir save_dir --resume
```

---

## Steps

1. **Use the Updated `odm_upstream.py`**
   - Make sure you are using the latest version of `odm.py`.

2. **Model Save Path Configuration**
   - The model save path **must be read from the config file**.
   - ⚠️ **Do not use --model_dir** in the run script.
   - `model_dir` in config must be absolute. Otherwise you would get an appropriate error.

3. **Update Training Parameters**
   - ⚠️ Update the `total_train_steps` variable in `odm_upstream.py` to reflect the correct number of training steps.
   - Set the correct path for the YAML config file.
   - **Note:** The `max_steps` defined in the YAML file will be **overwritten** by the script.

4. **State Saving**
   - The dynamic sampling update object will be saved at:
     ```
     ./dynamic_sampling/save_state.pkl
     ```
   - This can be used to access all the saved weights later.

5. **Restarting**
   - In case of failure, just rerun the command with the `--restart True` flag.

---

## Notes

- Make sure all paths used in the config are valid and absolute.
