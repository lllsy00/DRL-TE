import os
import time as tt
import resource
import subprocess

max_iters = 2000 
episode_iters = 20 


if __name__ == "__main__":

    if not os.path.exists("./Logs"):
        os.makedirs("./Logs")

    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")

    iters = 0
    counter_store_model = 1
    start_time = tt.time()

    dataset_folder_name1 = "NEW_Biznet"

    while iters < max_iters:
        processes = []
 
        subprocess.call(['python train_Enero_3top_script.py -i '+str(iters)+ ' -c '+str(counter_store_model)+' -e '+str(episode_iters)+ ' -f1 '+dataset_folder_name1], shell=True)
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        counter_store_model = counter_store_model + episode_iters
        iters = iters + episode_iters


