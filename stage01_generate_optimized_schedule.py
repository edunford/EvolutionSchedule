'''
Run the main method implementations
'''




import itertools
import pandas as pd
import numpy as np
import time



# %% ---------------------------------------------


def grab_day(dat,day):
    '''Extract data for a specific day from target building'''
    return dat.loc[dat.days.str.contains(day),:].copy().reset_index(drop=True)

def detect_double_booked(dat):
    '''
    Is there overlap in the existing schedule (that is, did the registrar already double book a room)
    If not removed, there might not be sufficient time in the day to resolve those booking issues, preventing convesion of an optimal solution.
    '''
    S1 = dat[['room','start_time','end_time']].values.copy()
    out = []
    for r in np.unique(S1[:,0]):
        for t in np.arange(0,24*60,5):
            s = S1[(S1[:,0] == r) & (t >= S1[:,1] ) & (t <= S1[:,2])]
            if s.shape[0] > 1:
                out.append(np.stack([s[:,0],s[:,1],s[:,2],np.array([t]*s.shape[0])],axis=1))
    if len(out) > 0:
        issues = pd.DataFrame(np.concatenate(out),columns=["room","start_time","end_time","i_time"]).drop_duplicates().reset_index(drop=True)
        # return issues
        return dat.merge(issues,on=["room","start_time","end_time"],how="inner").sort_values(['room','i_time']).reset_index(drop=True)

def anti_join(A,B):
    '''Method for an anti_join'''
    m = pd.merge(left=A, right=B, how='left', indicator=True)
    return m.loc[m._merge=="left_only",:].drop(columns="_merge").reset_index(drop=True)



# %% -----------------------------------------
# Focus on a building

# Import relevant data on the buildings.
dat = (pd.read_csv("output_data/course_schedule_data.csv").assign(fixed=0))

# Parameters
pop_size = 200
prop_top_pop = .05
start_time = 480
end_time = 1260
schedule_buffer = 30
n_epochs = 10000
n_mates = 20
prop_rooms_swap = .5
time_buffer = 30
time_between_classes = 20
stop_threhold = .01
stop_calls_threshold = 100

# All the buildings with schedules that need to be optimized
ignore_buildings = ['OFFC','OBS',"PRCL","SCSM"]
buildings = [b for b in dat.bldg.drop_duplicates().tolist() if b not in ignore_buildings]

# All day schedules are based off of.
days = ["M","T","W","R","F","S","U"]

optimized_schedules = {} # Dictionary container for each schedule.
existing_duplicate_schedules = {b:[] for b in buildings}

for target_building in buildings:

    # Check out the building schedules
    bldg_dat = dat[dat.bldg == target_building].reset_index().rename(columns={'index': 'class_id'})
    bldg_dat["class_id"] = bldg_dat["bldg"] + "_" + bldg_dat["class_id"].astype(str)
    bldg_dat['orig_start_time'] = bldg_dat["start_time"]
    bldg_dat['orig_end_time'] = bldg_dat["end_time"]

    # Keep track
    dropped_class = [] # Double-booked classes that were dropped

    # Print
    print(f"\n\n--------\nBuilding: {target_building}\n--------\n\n")

    for day in days:

        print(f"- Optimizing all {day}-based schedules in building '{target_building}'.")

        # Focus on a day
        tmp_dat = grab_day(bldg_dat,day)

        if tmp_dat.shape[0] == 0:
            continue

        # Detect if there are redundancies in the existing schedule
        issues = detect_double_booked(tmp_dat)

        # Keep a record of the mismatched entries.
        existing_duplicate_schedules[target_building].append(issues)

        # Remove those redundancies if there are any (so not all class are represented)
        if issues is not None:
            redundancies = (issues
                            .sort_values(["room","i_time",'course'])
                            .groupby(['room',"i_time"])
                            .head(1)
                            .drop(columns=['i_time'])
                            .drop_duplicates())
            tmp_dat = anti_join(tmp_dat,redundancies)

            # Save the dropped classes data (for inspection later)
            dropped_class.append(redundancies)


        # Number of rooms to swap out when mutating
        n_rooms = tmp_dat.room.drop_duplicates().size
        n_rooms_swap = np.floor(n_rooms*prop_rooms_swap).astype(int)

        # Run Algorithm to optimize schedule
        on, failed, n_tries = True,False,0
        while on:

            t_alg_start = time.time() # Track run time.

            self = ScheduleEvolution(reference_schedule=tmp_dat,
                                     start_time = start_time, end_time = end_time,
                                     schedule_buffer = schedule_buffer)
            self.generate_population(mutation_bounds = 30, N_population=pop_size)
            self.evolve(n_epochs = n_epochs,fix_pop_size = pop_size,
                        cross_breed_top_n = int(pop_size*prop_top_pop),
                        n_mates = n_mates, n_room_swap = n_rooms_swap,
                        mutation_bounds = 5,time_buffer = time_buffer,
                        overlap_penalty=100,time_between_classes = time_between_classes,
                        stop_threhold=stop_threhold,
                        stop_calls_threshold=stop_calls_threshold, verbose=False)


            # Print progress report: how long did the algorithm take to run.
            print(f"\t * Converged in {round(time.time() - t_alg_start,2)} seconds.")

            # Grab the optimized schedule
            opt_state = self.grab_epoch_state()

            # Check that the optimized schedule is usable.
            if self.is_viable(distance_from_end=10):
                on = False
                print(f"\t * Schedule is valid.")

            elif n_tries == 10:
                # raise ValueError(f"An optimal schedule for building {target_building} for {day}-based schedules could not be generated. Please investigate.")
                print(f"\t * An optimal schedule for building {target_building} for {day}-based schedules could not be generated. Please investigate.")
                failed = True
                break

            else: # Try again but this time using the optimal state as a starter template
                print(f"\t * Schedule is not valid, trying again.")
                n_tries += 1
                # n_mates += 100

        if failed:
            continue

        # Establish start/end times on input data
        tmp_dat['start_time'] = opt_state['start_time']
        tmp_dat['end_time'] = opt_state['end_time']
        tmp_dat.loc[:,'fixed'] = 1 # Fix all classroom schedules with opimized times

        # Map back onto data.
        all_but_current = anti_join(bldg_dat,tmp_dat[['class_id']])
        bldg_dat = pd.concat([all_but_current,tmp_dat],sort=True)

    # Flag all classes that were ignored due to existing overlap (i.e. double-booked in the operative schedule)
    if len(dropped_class) > 0:
        dc = pd.concat(dropped_class,sort=True)[['class_id','room','days','max_enrl','times']].drop_duplicates().reset_index(drop=True)
        bldg_dat["ignored"] = bldg_dat.class_id.isin(dc.class_id).astype(int)
    else:
        bldg_dat["ignored"] = 0


    # generate dictionary entry
    entry = {target_building: (bldg_dat[['bldg','room','course','days','n_students',
                                         'max_enrl','start_time','end_time',
                                         'orig_start_time','orig_end_time','ignored']]
                               .sort_values(["room",'start_time'])).reset_index(drop=True)}

    # Store optimized schedule for the building.
    optimized_schedules.update(entry)


# %% -----------------------------------------


# Save the optimized version of the schedule using the date as a unique identifier
save_time = str(pd.datetime.now()).replace(" ","_").replace("-","_").split(".")[0]
out_file = "output_data/optimized_schedules/optimize_course_schedule_"+ save_time + ".csv"
pd.concat(optimized_schedules.values()).to_csv(out_file,index=False)



# %% ---------------------------------------------

# test = {}


dat.query("bldg == 'BSB' and room=='130' and days=='MW'")
bldg_dat
test["M"].query("room=='130'")

# %% -----------------------------------------

self.export_epoch_state("output_data/opt_sched_2.csv")
self.plot_performance()
