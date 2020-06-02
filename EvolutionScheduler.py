'''
Evolutionary algorithm to optimized class schedules to reduce congestion.
'''

import itertools
import pandas as pd
import numpy as np
import sys

class ScheduleEvolution:
    '''
    Implementation of an evolutionary algorithm to locate the optimal class
    schedules to minimize overlap.

        - Define the individuals (as data) -- need to convert the real data into a data structure that can be manipulated.
        - Generate the population by randomly shifting the class order around.
        - Calculate fitness (maximize class overlap, minimize mutual time out)
            + Make sure to add large penalty if classes are scheduled to occur at exactly the same time in the same room (by any margin) --- overlap here is very bad.
        - Cross-breed the best performing schedules + a little random mutation
        - Repeat
    '''

    def __init__(self,reference_schedule=None,start_time = 480, end_time = 1260, schedule_buffer = 15):
        """Short summary.

        Args:
            reference_schedule (type): Description of parameter `reference_schedule`. Defaults to None.
            start_time (int): Time of day the classes can start in raw minutes. Defaults to 480 (8:00am).
            end_time (int): Time of day the classes can end in raw minutes. Defaults to 1260 (9:00pm).
            schedule_buffer (int): Wiggle room around the start and end times

        Returns:
            type: Description of returned object.

        """
        self.raw_data = reference_schedule # Retain the raw reference data
        self.reference_schedule = self.raw_data.reset_index()[['index','room','max_enrl','start_time','end_time','fixed']].values
        self.population = None
        self.epoch_performance = [] # Performance across epochs
        self.epoch_states = {} # save states across epochs to get a sense of performance improvements
        self.fitness_metric = None

        # Add schedule buffer to ensure there isn't congestion at the end
        # or the start of the day
        self.max_time = end_time + schedule_buffer
        self.min_time = start_time - schedule_buffer
        self.n_classes = self.reference_schedule.shape[0] - self.reference_schedule[:,5].sum() # relavant to mutations, so ignore classes that must remain fixed (due to having already been optimized within another schedule)
        self.all_rooms = np.unique(self.reference_schedule[:,1])
        self.time_periods = np.arange(self.min_time,self.max_time,10)


    def random_mutation(self,mutation_bounds,n):
        '''Random step rounded to 5 minute intervals'''
        return (np.random.uniform(-mutation_bounds,mutation_bounds,n)/5).round(0) * 5


    def generate_population(self, mutation_bounds, N_population = 50):
        """Generate initial population from the current pre-covid 19 schedule.

        Args:
            mutation_bounds (int): Max mutation level the schedule can be pushed around by. Default 10 minutes. Mutations can only take on values of 5 minute intervals.
            N_population (int): how many candidate schedules should be generated to make up the population. Default 50.

        Returns:
            list: containing numpy arrays of the mutated versions of the data sets.

        """
        # Generate generic mutations of the starter schedule.
        population = []
        for i in range(N_population):
            mutations = self.random_mutation(mutation_bounds=mutation_bounds,n=self.n_classes)
            new_sched = self.reference_schedule.copy()
            not_fixed = new_sched[:,5] == 0 # All schedules that aren't fixed and can be mutated.
            new_sched[not_fixed,3:5] = new_sched[not_fixed,3:5] + mutations.reshape(self.n_classes,1)

            nudge_mutation_bounds = 30 # This should be small and fixed

            # check new schedule falls within the current bounds of the day (for that building)
            # if over, nudge back into interval + a little noise
            over_time = new_sched[:,4] > self.max_time
            if any(over_time):
                time_over = self.max_time - new_sched[over_time,4]
                noise = self.random_mutation(mutation_bounds=mutation_bounds,n=len(time_over))
                noise = -np.abs(noise) # All strictly negative
                new_sched[over_time,3:5] = new_sched[over_time,3:5] + (time_over+noise).reshape(len(time_over),1)

            # if under, nudge back into interval + a little noise
            under_time = new_sched[:,3] < self.min_time
            if any(under_time):
                time_under = new_sched[under_time,3] - self.min_time
                noise = self.random_mutation(mutation_bounds=mutation_bounds,n=len(time_under))
                noise = np.abs(noise) # All strictly positive
                new_sched[under_time,3:5] = new_sched[under_time,3:5] - (time_under+noise).reshape(len(time_under),1)

            # Add new schedule to the population
            population.append([new_sched,0])

        # Write or overwrite existing population
        self.population = population


    def generate_mutations(self,schedule=None,mutation_bounds = 10):
        """Generate mutated versions of the crossbred schedules.

        Args:
            schedule (numpy array): crossbred schedule to be mutated.
            mutation_bounds (int): Max mutation level the schedule can be pushed around by. Default 10 minutes

        Returns:
            list: containing numpy arrays of the mutated versions of the data sets.

        """
        mutations = self.random_mutation(mutation_bounds=mutation_bounds,n=self.n_classes)
        new_sched = schedule.copy()
        not_fixed = new_sched[:,5] == 0 # All schedules that aren't fixed and can be mutated.
        new_sched[not_fixed,3:5] = new_sched[not_fixed,3:5] + mutations.reshape(self.n_classes,1)

        nudge_mutation_bounds = 30 # This should be small and fixed

        # check new schedule falls within the current bounds of the day (for that building)
        # if over, nudge back into interval + a little noise
        over_time = new_sched[:,4] > self.max_time
        if any(over_time):
            time_over = self.max_time - new_sched[over_time,4]
            noise = self.random_mutation(mutation_bounds=mutation_bounds,n=len(time_over))
            noise = -np.abs(noise) # All strictly negative
            new_sched[over_time,3:5] = new_sched[over_time,3:5] + (time_over+noise).reshape(len(time_over),1)

        # if under, nudge back into interval + a little noise
        under_time = new_sched[:,3] < self.min_time
        if any(under_time):
            time_under = new_sched[under_time,3] - self.min_time
            noise = self.random_mutation(mutation_bounds=mutation_bounds,n=len(time_under))
            noise = np.abs(noise) # All strictly positive
            new_sched[under_time,3:5] = new_sched[under_time,3:5] - (time_under+noise).reshape(len(time_under),1)

        # Return new schedule
        return [new_sched,0]


    def score_fitness(self,schedule=None,
                      time_buffer = 10,time_between_classes = 10,
                      overlap_penalty = 100):

        def overlap_score(start_time_1=None,end_time_1=None,
                          start_time_2=None,end_time_2=None,
                          buffer = 0):
            '''
            '''
            total_time = (end_time_2 - start_time_1)
            time_between = (end_time_1 + buffer - start_time_2)
            time_a = end_time_1 + buffer - start_time_1
            time_b = end_time_2 - start_time_2
            score = (total_time - time_between)/ (time_a + time_b)
            return np.abs(score)

        fitness = 0
        S = schedule
        for t in self.time_periods:
            in_session = (( t >= (S[:,3] - time_between_classes) ) & (t <= (S[:,4] + time_between_classes) ))
            coming_up = (( (t - time_buffer) >= S[:,3]) & ( (t - time_buffer) <= S[:,4])) | (( (t + time_buffer) >= S[:,3]) & ( (t + time_buffer) <= S[:,4]))

            # Within room overlap (penalize more the greater the overlap)
            O = [(overlap_score(start_time_1=a[3],end_time_1 = a[4],
                                start_time_2 = b[3],end_time_2 = b[4],
                                buffer = time_between_classes))
                 for a in S[in_session,:]
                 for b in S[in_session,:]
                 if (a[0] != b[0]) and (a[1] == b[1])]
            w = - sum([overlap_penalty/.001 if i == 0 else overlap_penalty/i for i in O if i < 1])


            # Between room congestion
            b = - S[coming_up & ~in_session,2].sum()/(time_buffer*2)

            fitness += w + b

        return fitness


    def calc_fitness(self,
                     overlap_penalty=50,
                     time_between_classes=15,
                     time_buffer = 10):
        """Short summary.

        Args:
            time_buffer (type): Description of parameter `time_buffer`. Defaults to 10.
            overlap_penalty (type): Description of parameter `overlap_penalty`. Defaults to 50.

        Returns:
            type: Description of returned object.

        """

        fitness_metric = []

        for version, schedule in  enumerate(self.population):

            # Track if current schedule has already been assessed.
            S = schedule[0]
            checked = schedule[1]

            if checked == 0:
                fitness = self.score_fitness(schedule=S,
                                             time_buffer = time_buffer,
                                             overlap_penalty=overlap_penalty,
                                             time_between_classes=time_between_classes)
                # Append the fitness metrics for the specific version
                fitness_metric.append([version,fitness])
                schedule[1] = 1
            else:
                fitness_metric.append(self.fitness_metric[version])

        # write/overwrite fitness metrics for the current generation.
        self.fitness_metric = np.array(fitness_metric)


    def crossbreed(self,cross_breed_top_n = 3,n_mates = 100,
                   n_room_swap = 3, mutation_bounds = 10):
        """Cross breed the top performing schedules by switch a specified number of rooms between
        the top N number of performers. The crossbred schedules are then mutated slightly. More than
        one mutated version can be generated (siblings).

        Args:
            cross_breed_top_n (int): N number of top performing schedules to crossbreed.
            breeding_prob (float): Probability of two pairs being selected to crossbreed. Default is .5.
            n_room_swap (int): Number of rooms to randomly swap in the crossbreeding.
            mutation_bounds (int): The bounding range (in minute) for the mutations.
        """

        # If you ask for too many rooms, will set to the max
        if n_room_swap > len(self.all_rooms):
            n_room_swap = len(self.all_rooms)

        # sorted
        metrics = self.fitness_metric[self.fitness_metric[:,1].argsort()[::-1][:self.fitness_metric.shape[0]]]
        top_performers = metrics[:cross_breed_top_n,:]

        # Generate all mating options only randomly select some pairs.
        # Always breed the top 3 performers, randomly select the rest
        mate_pairs_selected = [(0,1),(1,2),(2,3),(0,2),(0,3),(1,3)]
        mate_pairs = [i for i in itertools.combinations(top_performers[:,0],2) if i not in mate_pairs_selected]
        if n_mates > len(mate_pairs):
            n_mates = len(mate_pairs)
        select_ind = np.random.choice([i for i in range(1,len(mate_pairs))],n_mates,replace=False).tolist()
        mate_pairs_selected.extend([mate_pairs[i] for i in select_ind])

        # Crossbreed the the selected mates
        for i, j in mate_pairs_selected:

            # Mate 1
            mate_1 = self.population[i][0].copy()

            # Mate 2
            mate_2 = self.population[j][0].copy()

            # Randomly swap N_number of room configurations.
            rooms_to_swap = np.random.choice(self.all_rooms,n_room_swap,replace=False)
            is_rel = [True if room in rooms_to_swap else False for room in mate_1[:,1] ]
            m1_room_contributions = mate_1[is_rel,:]
            m2_room_contributions = mate_2[is_rel,:]
            mate_1[is_rel,:] =  m2_room_contributions
            mate_2[is_rel,:] =  m1_room_contributions

            # Generate mutated version(s) of the crossbred mates
            mutate_mate_1 = self.generate_mutations(schedule = mate_1, mutation_bounds = mutation_bounds)
            mutate_mate_2 = self.generate_mutations(schedule = mate_2, mutation_bounds = mutation_bounds)

            # Append a mate to the population, flip a coin to choose which
            coin_flip = np.random.binomial(1,.5,1)
            if coin_flip == 1:
                self.population.append(mutate_mate_1)
            else:
                self.population.append(mutate_mate_2)



    def evolve(self,n_epochs = 5, fix_pop_size=5, cross_breed_top_n = 3,
               n_mates = 1, n_room_swap = 3,
               mutation_bounds = 10, time_buffer = 15,
               overlap_penalty = 5, time_between_classes = 10,
               stop_threshold=.005,stop_calls_threshold=10,verbose=False):
        """Main method for iterating over epochs generating a new population of performers.

        Args:
            n_epochs (int): the number of generational cycles to go through to generate results.
            cross_breed_top_n (int): N number of top performing schedules to crossbreed.
            n_mates (int): N number of mate combinations to bread. cross_breed_top_n! number of mating combinations are generated.
                This argument allows one to select a reasonable number of options.
            n_room_swap (int): Number of rooms to randomly swap in the crossbreeding.
            mutation_bounds (int): The bounding range (in minute) for the mutations.
            time_between_classes (int): the set amount of time that needs to be specified between classes
            open_penalty (int): an added penalty for instances when there is no one in class (i.e. no overlapping class periods)
            stop_threshold (float): minimium precent change allowed in improvement. Default is .005
            verbose (bool): print out which epoch you're on.

        Returns:
            Optimization performance statistics and the state of the best performing schedule at each epoch.

        """

        n_stop_calls = 0 # Number of stop calls. When reaches threhold, loop will cease.

        # Iterate for the specified number of epochs (evolution step)
        for epoch in range(n_epochs):

            if  self.fitness_metric is None:
                self.calc_fitness(time_buffer = time_buffer,
                                  overlap_penalty=overlap_penalty,
                                  time_between_classes=time_between_classes)

            else:

                # crossbreed a new generation from the current top performers
                self.crossbreed(cross_breed_top_n = cross_breed_top_n,
                                n_mates = n_mates,
                                n_room_swap = n_room_swap,
                                mutation_bounds = mutation_bounds)


                # Calculate the fitness
                self.calc_fitness(time_buffer = time_buffer,
                                  overlap_penalty=overlap_penalty,
                                  time_between_classes=time_between_classes)

            # Order metrics and only retain enough performance to maintain the original population level.
            metrics = self.fitness_metric[self.fitness_metric[:,1].argsort()[::-1][:self.fitness_metric.shape[0]]]
            survivors = metrics[:fix_pop_size,:].astype(int)

            # Grab survivors
            new_population = []
            for s in survivors[:,0]:
                new_population.append(self.population[s])

            # Overload the existing population with the next generation
            self.population = new_population

            # Overload performance metrics to hold the current state
            self.fitness_metric = survivors
            self.fitness_metric[:,0] = [i for i in range(fix_pop_size)]

            # Store performance of the best fitness
            self.epoch_performance.append([epoch,survivors[0,1]])

            # Store state of the best performer
            self.epoch_states.update({ epoch: self.population[0][0]})

            if verbose:
                if epoch == 0:
                    max_state = survivors[0,1]
                current = survivors[0,1]
                prec = 1 - (current/max_state)
                sys.stdout.write(f"\r\tepoch {epoch} - fitness: {round(prec*100,2)}%")
                sys.stdout.flush()
                # print(f'''epoch {epoch} - fitness: {survivors[0,1]}''')

            # Check on the amount of change, if insufficient, alter the change rate or stop running
            if epoch > 2:
                last = np.abs(self.epoch_performance[-2][1]).astype(float)
                current = np.abs(self.epoch_performance[-1][1]).astype(float)
                delta = (last - current)/last

                if delta <= stop_threshold:

                    mutation_bounds = np.abs(self.random_mutation(30,1))
                    # prop_rooms_swap = np.random.uniform(.05,.5)
                    # n_room_swap = round(self.n_classes*prop_rooms_swap)
                    n_stop_calls += 1

                    # if n_stop_calls > 50:
                    #     mutation_bounds = np.abs(self.random_mutation(200,1))
                    #     # prop_rooms_swap = np.random.uniform(.1,.5)
                    #     # n_room_swap = round(self.n_classes*prop_rooms_swap)
                    #     n_room_swap = 1


                    # if more than N stop calls have been made, stop loop and start over again.
                    if n_stop_calls == stop_calls_threshold:
                        if verbose:
                            print("\n\tFailed to reach a valid conversion.")
                        break

                else:
                    # Reset if alg escaped from a local minimum.
                    n_stop_calls = 0
                    prop_rooms_swap = .1
                    n_room_swap = round(self.n_classes*prop_rooms_swap)
                    mutation_bounds = 5
                    # n_mates = 10



            if self.is_valid(time_between_classes=time_between_classes):
                break


    def is_viable(self,distance_from_end=10):
        """Check if the converged schedule is valid.

            - ensure there is no overlap in classes.
            - ensure there is no class in a set window of another class starting.

        Arguments:
            distance_from_end (int): number of minute after a class that should exist with no activty. Default is 10.

        Returns:
            type: Description of returned object.
        """

        # Generate a copy of the inputed building schedule
        S1 = self.epoch_states[len(self.epoch_states)-1].copy()

        # Order Schedule
        S1 = S1[np.lexsort((S1[:,3],S1[:,1]))]

        # Compare entering classes to exiting classes
        S2 = np.roll(S1,-1,axis=0)

        # Drop the rolled value and adjust
        S1 = S1[:-1,:]
        S2 = S2[:-1,:]

        # Only compare classes that are in the same room
        rel_ind = np.where(S1[:,1] == S2[:,1])

        # Time between classes -- negative == overlap, positive == time out.
        T = (S2[rel_ind,3] - S1[rel_ind,4]) -distance_from_end
        if (np.sign(T) == -1).sum() == 0:
            return True
        else:
            return False


    def is_valid(self,time_between_classes = 10):

        def overlap_score(start_time_1=None,end_time_1=None,
                          start_time_2=None,end_time_2=None,
                          buffer = 0):
            '''
            '''
            total_time = (end_time_2 - start_time_1)
            time_between = (end_time_1 + buffer - start_time_2)
            time_a = end_time_1 + buffer - start_time_1
            time_b = end_time_2 - start_time_2
            score = (total_time - time_between)/ (time_a + time_b)
            return np.abs(score)

        # Grab optimized schedule
        S = self.epoch_states[len(self.epoch_states)-1].copy()

        return np.all([ (overlap_score(start_time_1=a[3],end_time_1 = a[4],start_time_2 = b[3],end_time_2 = b[4],buffer = time_between_classes) >= 1) for a in S for b in S if (a[0] != b[0]) and (a[1] == b[1])])


    def grab_epoch_state(self,state_ind=None):
        """Return the current state of the schedule."""
        if state_ind is None:
            return pd.DataFrame(self.epoch_states[len(self.epoch_states)-1],
                                columns = ["index","room","max_enrl","start_time",'end_time','fixed'])
        else:
            return pd.DataFrame(self.epoch_states[state_ind],
                                columns = ["index","room","max_enrl","start_time",'end_time','fixed'])

    def export_epoch_state(self,path="",state_ind=None):
        """Return the current state of the schedule."""
        if state_ind is None:
            pd.DataFrame(self.epoch_states[len(self.epoch_states)-1],
                         columns = ["index","room","max_enrl","start_time",'end_time','fixed']).to_csv(path,index=False)
        else:
            return pd.DataFrame(self.epoch_states[state_ind],
                                columns = ["index","room","max_enrl","start_time",'end_time','fixed']).to_csv(path,index=False)

    def plot_performance(self,figsize=(10,5)):
        """Return the current state of the schedule."""
        pd.DataFrame(self.epoch_performance,columns=["epoch","fitness"]).plot(x="epoch",y="fitness",figsize=figsize)
