'''
Run the main method implementations
'''

import itertools
import pandas as pd
import numpy as np


# %% ---------------------------------------------

# Buisness school MW example
dat = pd.read_csv("output_data/bsb_bldg_mw_example.csv")


# Alg.
self = ScheduleEvolution(reference_schedule=dat)
self.generate_population(mutation_bounds = 150, N_population=500)
self.evolve(n_epochs = 50,fix_pop_size = 500,cross_breed_top_n = 100,
            breeding_prob =.1, n_room_swap = 4,n_siblings = 1,
            mutation_bounds = 10,time_buffer = 15,overlap_penalty=50,
            stop_threhold=.05, verbose=True)






# %% ---------------------------------------------


self.plot_performance()


self.grab_epoch_state().to_csv("output_data/opt_sched_2.csv",index=False)




# %% -----------------------------------------

# generate data set of all iterations
for key,state in self.epoch_states.items():
    tmp = pd.DataFrame(state.tolist(),
                       columns=['index','room','start_time','end_time']).assign(epoch = key + 1)
    if key == 0:
        out = tmp
    else:
        out = pd.concat([out,tmp])


out.to_csv("output_data/epochs_opt_schedule.csv",index=False)


# Export the loss function information
pd.DataFrame(self.epoch_performance,columns=['epoch','loss']).to_csv("output_data/epochs_opt_loss.csv",index=False)
