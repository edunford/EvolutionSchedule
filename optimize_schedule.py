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

pop_size = 300
prec = .15

self = ScheduleEvolution(reference_schedule=dat,start_time = 480, end_time = 1260, schedule_buffer = 15)
self.generate_population(mutation_bounds = 150, N_population=pop_size)
self.evolve(n_epochs = 100,fix_pop_size = pop_size,cross_breed_top_n = int(pop_size*prec),
            n_mates = 50, n_room_swap = 4,
            mutation_bounds = 10,time_buffer = 30,
            overlap_penalty=50,time_between_classes = 20,
            stop_threhold=.01, stop_calls_threshold=5, verbose=True)
self.is_viable(distance_from_end=15) # Check that the optimized schedule is usable.



# %% ---------------------------------------------

self.export_epoch_state("output_data/opt_sched_2.csv")
self.plot_performance()



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
