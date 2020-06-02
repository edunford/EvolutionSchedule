'''
Run the main method implementations
'''

import itertools
import pandas as pd
import numpy as np
import time
# from EvolutionScheduler import ScheduleEvolution

# %% -----------------------------------------

# Import relevant data on the buildings.
dat = (pd.read_csv("output_data/course_schedule_data.csv").assign(fixed=0))
dat['orig_start_time'] = dat["start_time"]
dat['orig_end_time'] = dat["end_time"]


# Ignore certain building types
ignore_buildings = ['OFFC','OBS',"PRCL","SCSM","ICC","CBN","REI","WAL","WGR"]

d = dat.query("bldg == 'BSB'").copy()

new_d = d


# Initialize object
gs = GenerateSchedules()

# All candidate dates for a specific building (start on the 26th,
# starting with the 23rd can generate initial position issues)
rel_dates = new_d.start_date.sort_values().unique()
rel_dates = rel_dates[rel_dates!="2020-08-23"]

# Iterate through the relevant date windows and build the schedule sequentially.
first = True
for date in rel_dates:

    print(f"\n\n\t\t----------------- {date} --------------------")

    if first: # For the initial schedule, have radical changes
        gs.set_params(initial_mutation_bounds=30,
                      mutation_bounds = 5,
                      overlap_penalty = 50,
                      stop_calls_threshold = 50,
                      stop_threshold = 0,
                      time_between_classes = 10,
                      time_buffer = 10,
                      prop_rooms_swap = .1,
                      prop_top_pop = .1,
                      n_mates = 100,
                      initial_pop_size=1000,pop_size=300)
        first = False

    else: # the subsequent schedules, have more radical periods
        gs.set_params(initial_mutation_bounds=200,
                      mutation_bounds = 5,
                      overlap_penalty = 50,
                      stop_calls_threshold = 50,
                      stop_threshold = 0,
                      time_between_classes = 10,
                      time_buffer = 10,
                      prop_rooms_swap = .1,
                      prop_top_pop = .1,
                      n_mates = 100,
                      initial_pop_size=1000,pop_size=300)

    # Optimize the schdule for the relevant date
    gs.optimize(data=new_d[(date >= new_d.start_date) & (date < new_d.end_date)].copy(),
                ignore_buildings=ignore_buildings)

    new_d = gs.recompose(new_d)



# %% -----------------------------------------


gs.se.plot_performance()




# %% ---------------------------------------------

# test = {}


dat.query("bldg == 'BSB' and room=='130' and days=='MW'")
bldg_dat
test["M"].query("room=='130'")

# %% -----------------------------------------

self.export_epoch_state("output_data/opt_sched_2.csv")
self.plot_performance()
