'''
Run the main method implementations
'''

import itertools
import pandas as pd
import numpy as np


# %% ---------------------------------------------

# Buisness school MW example
dat = pd.read_csv("output_data/bsb_bldg_mw_example.csv")

dat

# Test
# starter_sched = dat.reset_index()[['index','room','start_time','end_time']].values
self = ScheduleEvolution(reference_schedule=dat)
self.generate_population(mutation_bounds = 150, N_population=250)
self.evolve(n_epochs = 500,fix_pop_size = 50, cross_breed_top_n = 15, n_room_swap = 4,
            n_siblings = 1, mutation_bounds = 10,time_between_classes = 15,
            open_penalty=50,verbose=True)






# %% ---------------------------------------------


self.plot_performance()

pd.DataFrame(self.epoch_performance,columns=["epoch","fitness"]).plot(x="epoch",y="fitness",figsize=(15,10))
self.grab_epoch_state().to_csv("output_data/opt_sched.csv",index=False)




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
