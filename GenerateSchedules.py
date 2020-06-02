import itertools
import pandas as pd
import numpy as np
import time
# from EvolutionScheduler import ScheduleEvolution

class GenerateSchedules:
    '''
    Method generates optimized versions of the existing academic schedule for all relevant buildings on campus.
    '''

    def __init__(self):
        self.params_loaded = False

    def set_params(self,initial_pop_size = 200, pop_size = 200,
                   initial_mutation_bounds = 30,
                   mutation_bounds = 5,overlap_penalty=100,
                   prop_top_pop = .05,start_time = 480, end_time = 1320,
                   schedule_buffer = 30, n_epochs = 10000,
                   n_mates = 20, prop_rooms_swap = .5,
                   time_buffer = 30, time_between_classes = 20,
                   stop_threshold = .01, stop_calls_threshold = 100):
        """Set the parameter settings for the optimization method.

        Args:
            initial_pop_size (int):
                the size of the initial population when first generating iterations of the output data. Defaults to 200.
            pop_size (int):
                the size of the population used in the evolution algorithm for subsequent iteration after the initial population is generated. Defaults to 200.
            initial_mutation_bounds (int):
                The bounds on the possible mutations (in 5 minute intervals) when generating the inital populations.
            mutation_bounds (int):
                The bounds on the possible mutations (in 5 minute intervals) when mutating a schedule.
            overlap_penalty (int):
                The additional penality placed classes within the same room that overlap. This can't happen, so the penalty should be strict/high. Defaults to 100.
            prop_top_pop (float):
                The proportion of the population top performers that should be used in the evolution algorithm. Defaults to .05.
            start_time (int):
                Time in minutes when the school day starts. Defaults to 480.
            end_time (int):
                Time in minutes when the school day ends. Defaults to 1320.
            schedule_buffer (int):
                The number of minutes around the start and end time that the schedule is allowed to go over. Defaults to 30.
            n_epochs (int):
                the number of generational cycles to go through to generate results. Default is 10000.
            n_mates (int):
                N number of mate combinations to bread. cross_breed_top_n! number of mating    combinations are generated.
            prop_rooms_swap (float):
                Proportion of rooms to swap when evolving. Defaults to .5.
            time_buffer (int):
                the time buffer used when assessing the congestion rates at sample time points. That is a time point is sampled and then a time_buffer +/- that time point is used to calculate the congestion rate. Defaults to 30.
            time_between_classes (int):
                The required time set in-between classes. Defaults to 20.
            stop_threshold (float):
                a threshold in the optimization method where by the optimizer makes a call to stop due to insufficient change in the fitness of the proposed schedule. Defaults to .01.
            stop_calls_threshold (int):
                Number of stop calls that need to be made before quiting optimization. Defaults to 100.
        """
        self.pop_size = pop_size
        self.initial_pop_size = initial_pop_size
        self.mutation_bounds = mutation_bounds
        self.initial_mutation_bounds = initial_mutation_bounds
        self.prop_top_pop = prop_top_pop
        self.start_time = start_time
        self.end_time = end_time
        self.schedule_buffer = schedule_buffer
        self.n_epochs = n_epochs
        self.n_mates = n_mates
        self.prop_rooms_swap = prop_rooms_swap
        self.time_buffer = time_buffer
        self.time_between_classes = time_between_classes
        self.stop_threshold = stop_threshold
        self.stop_calls_threshold = stop_calls_threshold
        self.overlap_penalty = overlap_penalty
        self.params_loaded = True


    def grab_day(self,data,day):
        '''Extract data for a specific day from target building'''
        return data.loc[data.days.str.contains(day),:].copy().reset_index(drop=True)

    def anti_join(self,A,B,by=None):
        '''Method for an anti_join'''
        m = pd.merge(left=A, right=B, on=by, how='left', indicator=True)
        return m.loc[m._merge=="left_only",:].drop(columns="_merge").reset_index(drop=True)


    def optimize(self,data=None,ignore_buildings=None, verbose=True):
        """Method iterates through all relevant building schedules and optimizes each
        to minimize congestion using the optimization algorithm specified in the
        EvolutionScheduler method.

        Args:
            data (data frame):
                relevant (pre-processed) schedule data to be optimized.
            ignore_buildings (list):
                List of all buildings that should be ignored. Defaults to None.
            verbose (bool):
                Print off progress regarding the optimization method. Defaults to True.

        Returns:
            stores a dictionary of optimized schedules by building.

        """

        if self.params_loaded == False:
            # If parameters weren't set, use default.
            self.set_params()

        self.optimized_schedules = {} # Dictionary container for each schedule.

        # All the buildings with schedules that need to be optimized
        if ignore_buildings is None:
            buildings = data.bldg.drop_duplicates().tolist()
        else:
            buildings = [b for b in data.bldg.drop_duplicates().tolist() if b not in ignore_buildings]

        # All day schedules are based off of.
        days = ["M","T","W","R","F","S","U"]

        for target_building in buildings:

            # Check out the building schedules
            bldg_dat = data[data.bldg == target_building].reset_index()

            # Print
            if verbose:
                print(f"\n\n--------\nBuilding: {target_building}\n--------\n\n")

            for day in days:

                print(f"- Optimizing all {day}-based schedules in building '{target_building}'.")

                # Check out a specific building-day
                tmp_dat = self.grab_day(bldg_dat,day)

                # Ignore conditions
                if tmp_dat.shape[0] <= 1:
                    # If there are no classrooms or only one to optimize for a building-day, continue on
                    continue

                elif tmp_dat.fixed.sum() == tmp_dat.shape[0]:
                    # If all the classrooms have already been optimized (within a specific time period), continue on
                    continue

                # Number of rooms to swap out when mutating
                n_rooms = tmp_dat.room.drop_duplicates().size
                n_rooms_swap = np.floor(n_rooms*self.prop_rooms_swap).astype(int)

                # Run Algorithm to optimize schedule
                on, failed, n_tries = True,False,0
                while on:

                    # After these initial days are set, need to allow for a much larger margin.
                    if day not in ["M","T"]:
                        self.initial_mutate_bounds = 500

                    t_alg_start = time.time() # Track run time.

                    # Retain the optimization method framework.
                    self.se = ScheduleEvolution(reference_schedule=tmp_dat,
                                                start_time = self.start_time,
                                                end_time = self.end_time,
                                                schedule_buffer = self.schedule_buffer)
                    self.se.generate_population(mutation_bounds = self.initial_mutation_bounds,N_population=self.initial_pop_size)
                    self.se.evolve(n_epochs = self.n_epochs,fix_pop_size = self.pop_size,
                                cross_breed_top_n = int(self.pop_size*self.prop_top_pop),
                                n_mates = self.n_mates, n_room_swap = n_rooms_swap,
                                mutation_bounds = self.mutation_bounds,time_buffer = self.time_buffer,
                                overlap_penalty=self.overlap_penalty,time_between_classes = self.time_between_classes,
                                stop_threshold=self.stop_threshold,
                                stop_calls_threshold=self.stop_calls_threshold, verbose=True)


                    # Print progress report: how long did the algorithm take to run.
                    if verbose:
                        print(f"\n\t * Converged in {round(time.time() - t_alg_start,2)} seconds. Fitness = {self.se.epoch_performance[len(self.se.epoch_performance)-1][1]}")

                    # Grab the optimized schedule
                    opt_state = self.se.grab_epoch_state()

                    # Check that the optimized schedule is usable.
                    # if self.se.is_viable(distance_from_end=5):
                    if self.se.is_valid(time_between_classes=self.time_between_classes):
                        on = False
                        if verbose:
                            print(f"\t * Schedule is valid.")

                    elif n_tries == 10:
                        # If no success after 10 tries, quit.
                        if verbose:
                            print(f"\t * An optimal schedule for building {target_building} for {day}-based schedules could not be generated. Please investigate.")
                        failed = True
                        break

                    else: # Try again but this time using the optimal state as a starter template
                        if verbose:
                            print(f"\t * Schedule is not valid, trying again.")
                        n_tries += 1

                if failed:
                    continue

                # Establish start/end times on input data
                tmp_dat['start_time'] = opt_state['start_time']
                tmp_dat['end_time'] = opt_state['end_time']
                tmp_dat.loc[:,'fixed'] = 1 # Fix all classroom schedules with opimized times

                # Map back onto data.
                all_but_current = self.anti_join(bldg_dat,tmp_dat[['course']])
                bldg_dat = pd.concat([all_but_current,tmp_dat],sort=True)


            # generate dictionary entry
            entry = {target_building:(bldg_dat.sort_values(["room",'start_time'])).reset_index(drop=True)}

            # Store optimized schedule for the building.
            self.optimized_schedules.update(entry)

    def compile_schedule(self):
        '''Return a version of the current optimized schedule as data frame'''
        return pd.concat(self.optimized_schedules.values(),sort=False)


    def recompose(self,original_data=None):
        '''
        Return original data with the optimized schedules mapped on.
        '''
        col_ord = ['bldg', 'room', 'course', 'days', 'fixed',
                   'full_term', 'max_enrl',
                   'start_date','end_date','start_time', 'end_time',
                   'orig_start_time', 'orig_end_time', 'times']
        merge_on = ['bldg', 'room', 'course', 'days','start_date']
        return pd.concat([self.anti_join(original_data,
                                         self.compile_schedule()[merge_on],
                                         by=merge_on),
                          self.compile_schedule()],sort=True)[col_ord]


    def export_schedule(self,data):
        '''
        Export the final version of the schedule.
        '''
        # Save the optimized version of the schedule using the date as a unique identifier
        save_time = str(pd.datetime.now()).replace(" ","_").replace("-","_").split(".")[0]
        out_file = "output_data/optimized_schedules/optimize_course_schedule_"+ save_time + ".csv"
        data.to_csv(out_file,index=False)
