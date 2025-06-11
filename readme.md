# Use new odm.py
!!Please ensure that run script does not override model save path the code expects it to be taken from config.
!!Keep the saved states of the odm process for debugging.
Model dir in conbfig must be an absolute path
Update total_train_steps and yaml file path (Note that max steps in yaml will be overwritten!!!)
The update object contains all the update information is saved at "./dynamic_sampling/save_state.pkl".