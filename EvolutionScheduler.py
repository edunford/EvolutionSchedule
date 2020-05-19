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

    def __init__(self,reference_schedule=None,schedule_buffer=10):
        self.raw_data = reference_schedule # Retain the raw reference data
        self.reference_schedule = self.raw_data.reset_index()[['index','room','start_time','end_time']].values
        self.population = None
        self.epoch_performance = [] # Performance across epochs
        self.epoch_states = {} # save states across epochs to get a sense of performance improvements
        self.fitness_metric = None

        # Add schedule buffer to ensure there isn't congestion at the end
        # or the start of the day
        self.max_time = self.reference_schedule[:,3].max() + schedule_buffer
        self.min_time = self.reference_schedule[:,2].min() - schedule_buffer
        self.n_classes = self.reference_schedule.shape[0]
        self.all_rooms = np.unique(self.reference_schedule[:,1])
        self.all_minutes = np.arange(self.min_time,self.max_time,10)


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
            new_sched[:,2:] = new_sched[:,2:] + mutations.reshape(new_sched.shape[0],1)

            # check new schedule falls within the current bounds of the day (for that building)
            # if over, nudge back into interval + a little noise
            over_time = new_sched[:,3] > self.max_time
            if any(over_time):
                time_over = new_sched[over_time,3] - self.max_time
                noise = self.random_mutation(mutation_bounds=mutation_bounds,n=len(time_over))
                new_sched[over_time,2:] = new_sched[over_time,2:] - (time_over+noise).reshape(len(time_over),1)

            # if under, nudge back into interval + a little noise
            under_time = new_sched[:,2] < self.min_time
            if any(under_time):
                time_under = self.min_time - new_sched[under_time,2]
                noise = self.random_mutation(mutation_bounds=mutation_bounds,n=len(time_under))
                new_sched[under_time,2:] = new_sched[under_time,2:] + (time_under+noise).reshape(len(time_under),1)

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
            new_sched[:,2:] = new_sched[:,2:] + mutations.reshape(new_sched.shape[0],1)

            # check new schedule falls within the current bounds of the day (for that building)
            # if over, nudge back into interval + a little noise
            over_time = new_sched[:,3] > self.max_time
            if any(over_time):
                time_over = new_sched[over_time,3] - self.max_time
                noise = self.random_mutation(mutation_bounds=mutation_bounds,n=len(time_over))
                new_sched[over_time,2:] = new_sched[over_time,2:] - (time_over+noise).reshape(len(time_over),1)

            # if under, nudge back into interval + a little noise
            under_time = new_sched[:,2] < self.min_time
            if any(under_time):
                time_under = self.min_time - new_sched[under_time,2]
                noise = self.random_mutation(mutation_bounds=mutation_bounds,n=len(time_under))
                new_sched[under_time,2:] = new_sched[under_time,2:] + (time_under+noise).reshape(len(time_under),1)

            # Add new schedule to the population
            population.append(new_sched)

        # Append the new mutated schedules to the existing population
        return population




    def calc_fitness(self,time_between_classes = 15,within_room_violation_penalty=100,open_penalty=5):
        """Short summary.

        Args:
            time_between_classes (type): Description of parameter `time_between_classes`. Defaults to 15.
            within_room_violation_penalty (type): Description of parameter `within_room_violation_penalty`. Defaults to 100.
            open_penalty (int): the size of the penalty when there is no class

        Returns:
            type: Description of returned object.

        """

        # Internals
        def within_room_penalty(schedule,room,time_between_classes = 15,fitness=0,overlap_penalty=100):
            """Scan for all the classes within a specific room. If the schedule doesn't
            allocated the specified time between the classes,
            penalize (harshly) the algorithms fitness.


            Args:
                schedule (numpy array): schedule in the population.
                room (int): room number in the relevant building.
                time_between_classes (int): Time (in minutes) between classes.
                fitness (int): the fitness score of the candidate schedule
                overlap_penalty (int): The size of the penalty for class transitions violations,
                    that is class class overlap or not enough time between. Defaults to 100.

            Returns:
                type: Description of returned object.

            """


            # Select all classes within a specific room.
            is_rel_room = schedule[:,1]==room
            rel_room = schedule[is_rel_room,:]
            rel_room = rel_room[rel_room[:,2].argsort()]

            for i in range(1,rel_room.shape[0]):
                end_start_diff = (rel_room[i-1,3] - rel_room[i,2])
                if end_start_diff >=  -time_between_classes:
                    if end_start_diff >= 0: # Greater penalty as there is direct overlap
                        fitness -= (np.abs(end_start_diff)+time_between_classes)*50
                    else:
                        fitness -= np.abs(end_start_diff)+time_between_classes

            return fitness



        # Container for the fitness score.
        fitness_metric = []

        # Iterate through all individual schedules in the population
        for version, schedule in  enumerate(self.population):
            fitness = 0

            # Scan through and calculate the within room penalty
            # Ensures there is sufficient time between classes.
            for room in self.all_rooms:
                fitness = within_room_penalty(schedule=schedule,
                                              room = room,
                                              time_between_classes=time_between_classes,
                                              fitness=fitness,
                                              overlap_penalty=within_room_violation_penalty,)

            # Iterate across rooms to sum up the number of minutes with classroom overlap.
            # minutes of overlap are good (as they mean there is less time when students are
            # in the hallway)
            for minute in self.all_minutes:
                minute_sum = sum(( minute >= schedule[:,2]) & (minute <= schedule[:,3]))
                if minute_sum > 0:
                    pass
                    # fitness += minute_sum
                else:
                    fitness -= open_penalty

            # Append on the fitness metrics
            fitness_metric.append([version,fitness])

        # write/overwrite fitness metrics for the current generation.
        self.fitness_metric = np.array(fitness_metric)



    def crossbreed(self,cross_breed_top_n = 3,n_room_swap = 3, n_siblings = 1, mutation_bounds = 10):
        """Cross breed the top performing schedules by switch a specified number of rooms between
        the top N number of performers. The crossbred schedules are then mutated slightly. More than
        one mutated version can be generated (siblings).

        Args:
            cross_breed_top_n (int): N number of top performing schedules to crossbreed.
            n_room_swap (int): Number of rooms to randomly swap in the crossbreeding.
            n_siblings (int): N number of additional mutations to generate from the crossbred sets.
            mutation_bounds (int): The bounding range (in minute) for the mutations.
        """

        # If you ask for too many rooms, will set to the max
        if n_room_swap > len(self.all_rooms):
            n_room_swap = len(self.all_rooms)

        # sorted
        metrics = self.fitness_metric[self.fitness_metric[:,1].argsort()[::-1][:self.fitness_metric.shape[0]]]
        top_performers = metrics[:cross_breed_top_n+1,:]


        # Crossbreed the best performers
        for i in range(cross_breed_top_n):

            # Best performer
            mate_1 = self.population[top_performers[i,0]].copy()


            for j in range(cross_breed_top_n):

                if i > j:

                    # second mate
                    mate_2 = self.population[top_performers[j,0]].copy()

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



    def evolve(self,n_epochs = 5, fix_pop_size=5, cross_breed_top_n = 3, n_room_swap = 3,
               n_siblings = 1, mutation_bounds = 10,time_between_classes = 15,
               open_penalty=5,verbose=False):
        """Main method for iterating over epochs generating a new population of performers.

        Args:
            n_epochs (int): the number of generational cycles to go through to generate results.
            cross_breed_top_n (int): N number of top performing schedules to crossbreed.
            n_room_swap (int): Number of rooms to randomly swap in the crossbreeding.
            n_siblings (int): N number of additional mutations to generate from the crossbred sets.
            mutation_bounds (int): The bounding range (in minute) for the mutations.
            time_between_classes (int): the set amount of time that needs to be specified between classes
            open_penalty (int): an added penalty for instances when there is no one in class (i.e. no overlapping class periods)
            verbose (bool): print out which epoch you're on.

        Returns:
            Optimization performance statistics and the state of the best performing schedule at each epoch.

        """

        # Iterate for the specified number of epochs (evolution step)
        for epoch in range(n_epochs):

            if  self.fitness_metric is None:
                self.calc_fitness(time_between_classes = time_between_classes,
                                  within_room_violation_penalty=within_room_violation_penalty,
                                  open_penalty=open_penalty)

            else:

                # crossbreed a new generation from the current top performers
                self.crossbreed(cross_breed_top_n = cross_breed_top_n, n_room_swap = n_room_swap,
                                n_siblings = n_siblings, mutation_bounds = mutation_bounds)

                # Calculate the fitness
                self.calc_fitness(time_between_classes = time_between_classes,
                                  within_room_violation_penalty=within_room_violation_penalty,
                                  open_penalty=open_penalty)

            # Order metrics and only retain enough performance to maintain the original population level.
            metrics = self.fitness_metric[self.fitness_metric[:,1].argsort()[::-1][:self.fitness_metric.shape[0]]]
            survivors = metrics[:fix_pop_size,:]

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


    def grab_epoch_state(self,state_ind=None):
        """Return the current state of the schedule."""
        if state_ind is None:
            return pd.DataFrame(self.epoch_states[len(self.epoch_states)-1],
                                columns = ["index","room","start_time",'end_time'])
        else:
            return pd.DataFrame(self.epoch_states[state_ind],
                                columns = ["index","room","start_time",'end_time'])

    def plot_performance(self):
        """Return the current state of the schedule."""
        pd.DataFrame(self.epoch_performance,columns=["epoch","fitness"]).plot(x="epoch",y="fitness")
