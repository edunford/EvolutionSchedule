'''
Evolutionary algorithm to optimized class schedules to reduce congestion.
'''

import itertools
import pandas as pd
import numpy as np

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
        self.reference_schedule = self.raw_data.reset_index()[['index','room','max_enrl','start_time','end_time']].values
        self.population = None
        self.epoch_performance = [] # Performance across epochs
        self.epoch_states = {} # save states across epochs to get a sense of performance improvements
        self.fitness_metric = None

        # Add schedule buffer to ensure there isn't congestion at the end
        # or the start of the day
        self.max_time = end_time + schedule_buffer
        self.min_time = start_time - schedule_buffer
        self.n_classes = self.reference_schedule.shape[0]
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
            new_sched[:,3:] = new_sched[:,3:] + mutations.reshape(new_sched.shape[0],1)

            # check new schedule falls within the current bounds of the day (for that building)
            # if over, nudge back into interval + a little noise
            over_time = new_sched[:,4] > self.max_time
            if any(over_time):
                time_over = new_sched[over_time,4] - self.max_time
                noise = self.random_mutation(mutation_bounds=mutation_bounds,n=len(time_over))
                new_sched[over_time,3:] = new_sched[over_time,3:] - (time_over+noise).reshape(len(time_over),1)

            # if under, nudge back into interval + a little noise
            under_time = new_sched[:,3] < self.min_time
            if any(under_time):
                time_under = self.min_time - new_sched[under_time,3]
                noise = self.random_mutation(mutation_bounds=mutation_bounds,n=len(time_under))
                new_sched[under_time,3:] = new_sched[under_time,3:] + (time_under+noise).reshape(len(time_under),1)

            # Add new schedule to the population
            population.append(new_sched)

        # Write or overwrite existing population
        self.population = population


    def generate_mutations(self,schedule=None,N_siblings = 1,mutation_bounds = 10):
        """Generate mutated versions of the crossbred schedules.

        Args:
            schedule (numpy array): crossbred schedule to be mutated.
            N_siblings (int): how many mutant siblings to generate from crossbred schedule. Default 1.
            mutation_bounds (int): Max mutation level the schedule can be pushed around by. Default 10 minutes

        Returns:
            list: containing numpy arrays of the mutated versions of the data sets.

        """

        # Generate generic mutations of the starter schedule.
        population = []
        for i in range(N_siblings):
            mutations = self.random_mutation(mutation_bounds=mutation_bounds,n=self.n_classes)
            new_sched = schedule.copy()
            new_sched[:,3:] = new_sched[:,3:] + mutations.reshape(new_sched.shape[0],1)

            # check new schedule falls within the current bounds of the day (for that building)
            # if over, nudge back into interval + a little noise
            over_time = new_sched[:,4] > self.max_time
            if any(over_time):
                time_over = new_sched[over_time,4] - self.max_time
                noise = self.random_mutation(mutation_bounds=mutation_bounds,n=len(time_over))
                new_sched[over_time,3:] = new_sched[over_time,3:] - (time_over+noise).reshape(len(time_over),1)

            # if under, nudge back into interval + a little noise
            under_time = new_sched[:,3] < self.min_time
            if any(under_time):
                time_under = self.min_time - new_sched[under_time,3]
                noise = self.random_mutation(mutation_bounds=mutation_bounds,n=len(time_under))
                new_sched[under_time,3:] = new_sched[under_time,3:] + (time_under+noise).reshape(len(time_under),1)

            # Add new schedule to the population
            population.append(new_sched)

        # Append the new mutated schedules to the existing population
        return population


    def within_room(self,schedule,overlap_penalty = 50):
        """Short summary.

        Args:
            schedule (type): Description of parameter `schedule`.
            overlap_penalty (type): Description of parameter `overlap_penalty`. Defaults to 50.

        Returns:
            type: Description of returned object.

        """

        # Generate a copy of the inputed building schedule
        S1 = schedule.copy()

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
        T = S2[rel_ind,3] - S1[rel_ind,4]
        T[T==0] = -1 # offset instances when a class starts as one ends

        # calculate the student congestion rate (SCR) -- which is the average number of students
        # in the hall for a specific period of time.
        enter_stu = S2[rel_ind,2]
        exit_stu = S1[rel_ind,2]
        scr = (exit_stu + enter_stu)/T

        # If negative means there is class overlap, penalize these occurences.
        scr[scr<0] = T[T<0]*overlap_penalty

        # Convert all positive instances (SCR) to negative. Aim is to select a schedule where the SCR is
        # is low as possible.
        scr[scr>0] = np.negative(scr[scr>0])

        # Compute the fitness.
        fitness = scr.sum()

        return fitness


    def between_rooms(self,schedule,time_buffer = 10):
        """Short summary.

        Args:
            schedule (type): Description of parameter `schedule`.
            time_buffer (type): Description of parameter `time_buffer`. Defaults to 10.

        Returns:
            type: Description of returned object.

        """

        # Iterate over all sample time periods and calculate the congestion score.
        fitness = 0
        for time in self.time_periods:

            # Generate a copy of the inputed schedule
            S1 = schedule.copy().astype(float)

            # Temporal window of the class to examine
            low_range = (time - time_buffer)
            high_range = (time + time_buffer)

            # What is the average number of students out in the time window.
            S1[:,3] = S1[:,3] - low_range # See all classes that are about to start within the buffer range
            S1[:,4] = high_range - S1[:,4] # See all classes that just ended in the buffer range

            # Only retain those classes that are within the buffer range
            S1 = S1[ (S1[:,3] > 0) & (S1[:,3] <= time_buffer) | (S1[:,4] > 0) & (S1[:,4] <= time_buffer)]

            # calculate the average number of students in the hall for each room.
            S1[S1[:,3] > 0,2] = S1[S1[:,3] > 0,2]/S1[S1[:,3] > 0,3]
            S1[S1[:,4] > 0,2] = S1[S1[:,4] > 0,2]/S1[S1[:,4] > 0,4]

            # Take the total average number students in the hall as a fitness score.
            if S1.shape[0] > 0:
                fitness -= S1[:,2].mean()

        return fitness


    def calc_fitness(self,time_buffer = 10,overlap_penalty=50):
        """Short summary.

        Args:
            time_buffer (type): Description of parameter `time_buffer`. Defaults to 10.
            overlap_penalty (type): Description of parameter `overlap_penalty`. Defaults to 50.

        Returns:
            type: Description of returned object.

        """

        # Record the fitness metrics for the current generation
        fitness_metric = []

        # Iterate through all schedule versions in the population
        for version, schedule in  enumerate(self.population):

            # Fitness
            fitness1 = self.within_room(schedule=schedule,overlap_penalty=overlap_penalty)
            fitness2 = self.between_rooms(schedule=schedule,time_buffer=time_buffer)
            fitness_total = fitness1 + fitness2

            # Append the fitness metrics for the specific version
            fitness_metric.append([version,fitness_total])

        # write/overwrite fitness metrics for the current generation.
        self.fitness_metric = np.array(fitness_metric)



    def crossbreed(self,cross_breed_top_n = 3,breeding_prob = .5,
                   n_room_swap = 3, n_siblings = 1, mutation_bounds = 10):
        """Cross breed the top performing schedules by switch a specified number of rooms between
        the top N number of performers. The crossbred schedules are then mutated slightly. More than
        one mutated version can be generated (siblings).

        Args:
            cross_breed_top_n (int): N number of top performing schedules to crossbreed.
            breeding_prob (float): Probability of two pairs being selected to crossbreed. Default is .5.
            n_room_swap (int): Number of rooms to randomly swap in the crossbreeding.
            n_siblings (int): N number of additional mutations to generate from the crossbred sets.
            mutation_bounds (int): The bounding range (in minute) for the mutations.
        """

        # If you ask for too many rooms, will set to the max
        if n_room_swap > len(self.all_rooms):
            n_room_swap = len(self.all_rooms)

        # sorted
        metrics = self.fitness_metric[self.fitness_metric[:,1].argsort()[::-1][:self.fitness_metric.shape[0]]]
        top_performers = metrics[:cross_breed_top_n,:]

        # Generate all mating options only randomly select some pairs.
        # Always breed the top 2 performers
        mate_pairs = [i for i in itertools.combinations(top_performers[:,0],2) if (np.random.binomial(1,breeding_prob,1) == 1) or (i == (0,1))]

        # Crossbreed the the selected mates
        for i, j in mate_pairs:

            # Mate 1
            mate_1 = self.population[i].copy()

            # Mate 2
            mate_2 = self.population[j].copy()

            # Randomly swap N_number of room configurations.
            rooms_to_swap = np.random.choice(self.all_rooms,n_room_swap,replace=False)
            is_rel = [True if room in rooms_to_swap else False for room in mate_1[:,1] ]
            m1_room_contributions = mate_1[is_rel,:]
            m2_room_contributions = mate_2[is_rel,:]
            mate_1[is_rel,:] =  m2_room_contributions
            mate_2[is_rel,:] =  m1_room_contributions

            # Generate mutated version(s) of the crossbred mates
            mutate_mate_1 = self.generate_mutations(schedule = mate_1, N_siblings = n_siblings, mutation_bounds = mutation_bounds)
            mutate_mate_2 = self.generate_mutations(schedule = mate_2, N_siblings = n_siblings, mutation_bounds = mutation_bounds)

            # Append the mutations and any additional siblings to population
            for s in range(n_siblings):
                self.population.append(mutate_mate_1[s])
                self.population.append(mutate_mate_2[s])



    def evolve(self,n_epochs = 5, fix_pop_size=5, cross_breed_top_n = 3, breeding_prob =.5, n_room_swap = 3,
               n_siblings = 1, mutation_bounds = 10, time_buffer = 15,
               overlap_penalty = 5,stop_threhold=.005,verbose=False):
        """Main method for iterating over epochs generating a new population of performers.

        Args:
            n_epochs (int): the number of generational cycles to go through to generate results.
            cross_breed_top_n (int): N number of top performing schedules to crossbreed.
            n_room_swap (int): Number of rooms to randomly swap in the crossbreeding.
            n_siblings (int): N number of additional mutations to generate from the crossbred sets.
            mutation_bounds (int): The bounding range (in minute) for the mutations.
            time_between_classes (int): the set amount of time that needs to be specified between classes
            open_penalty (int): an added penalty for instances when there is no one in class (i.e. no overlapping class periods)
            stop_threhold (float): minimium precent change allowed in improvement. Default is .005
            verbose (bool): print out which epoch you're on.

        Returns:
            Optimization performance statistics and the state of the best performing schedule at each epoch.

        """

        n_stop_calls = 0 # Number of stop calls. When reaches threhold, loop will cease.

        # Iterate for the specified number of epochs (evolution step)
        for epoch in range(n_epochs):

            if  self.fitness_metric is None:
                self.calc_fitness(time_buffer = time_buffer,
                                  overlap_penalty=overlap_penalty)

            else:

                # crossbreed a new generation from the current top performers
                self.crossbreed(cross_breed_top_n = cross_breed_top_n,breeding_prob = breeding_prob, n_room_swap = n_room_swap,
                                n_siblings = n_siblings, mutation_bounds = mutation_bounds)

                # Calculate the fitness
                self.calc_fitness(time_buffer = time_buffer,
                                  overlap_penalty=overlap_penalty)

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

            # Store state of the vest performer
            self.epoch_states.update({ epoch : self.population[0]})

            if verbose:
                print(f'''epoch {epoch} - fitness: {survivors[0,1]}''')

            # Check on the amount of change, if insufficient stop running
            if epoch > 2:
                last = np.abs(self.epoch_performance[-2][1]).astype(float)
                current = np.abs(self.epoch_performance[-1][1]).astype(float)
                delta = (last - current)/last

                if delta <= stop_threhold:

                    n_stop_calls += 1

                    # if more than 5 stop calls have been made, stop loop
                    if n_stop_calls == 5:
                        if verbose:
                            print("Converged.")
                        break

                elif n_stop_calls > 0:
                    # Reset if alg escaped from a local minimum.
                    n_stop_calls = 0





    def grab_epoch_state(self,state_ind=None):
        """Return the current state of the schedule."""
        if state_ind is None:
            return pd.DataFrame(self.epoch_states[len(self.epoch_states)-1],
                                columns = ["index","room","max_enrl","start_time",'end_time'])
        else:
            return pd.DataFrame(self.epoch_states[state_ind],
                                columns = ["index","room","max_enrl","start_time",'end_time'])

    def plot_performance(self):
        """Return the current state of the schedule."""
        pd.DataFrame(self.epoch_performance,columns=["epoch","fitness"]).plot(x="epoch",y="fitness")
