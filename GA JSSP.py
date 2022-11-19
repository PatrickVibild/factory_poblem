# -*- coding: utf-8 -*-
"""
Reference: https://github.com/wurmen/Genetic-Algorithm-for-Job-Shop-Scheduling-and-NSGA-II/
"""
import random

'''==========Solving job shop scheduling problem by gentic algorithm in python======='''
# importing required modules
import pandas as pd
import numpy as np
import time
import copy
import matplotlib.pyplot as plt


def run():
    re_entry = "JSP_dataset_re_entry.xlsx"
    parallel = "JSP_dataset_parallel.xlsx"
    using = parallel
    ''' ================= initialization setting ======================'''
    pt_tmp = pd.read_excel(using, sheet_name="Processing Time", index_col=[0])
    ms_tmp = pd.read_excel(using, sheet_name="Machines Sequence", index_col=[0])
    dependency_tmp = pd.read_excel(using, sheet_name="Sequence Dependency", index_col=[0])
    machine_parallel = pd.read_excel(using, sheet_name="Machines", index_col=[0])
    m_dep_tmp = pd.read_excel(using, sheet_name="Machine Dependency", index_col=[0])

    dfshape = pt_tmp.shape
    num_tasks = dfshape[0]
    num_jobs = ms_tmp.shape[1]
    # number of jobs

    processing_time = [list(map(int, pt_tmp.iloc[i])) for i in range(num_tasks)]
    machine_sequence = [list(map(int, ms_tmp.iloc[i])) for i in range(num_tasks)]
    dependency_time = [list(map(int, dependency_tmp.iloc[i])) for i in range(num_tasks)]
    machines = [list(map(int, machine_parallel.iloc[i])) for i in range(1)][0]
    num_mc = max(max(machine_sequence))  # number of machines
    num_gene = num_mc * num_jobs  # number of genes in a chromosome

    # load dependencies. 1 is open 2 is close gate.
    machine_dependency = [list(map(int, m_dep_tmp.iloc[i])) for i in range(num_tasks)]
    # verify that dependencies is correct. At least one 1 in each sequence, and length equal to parallel machines.
    for job in machine_dependency:
        for j in range(num_tasks):
            stringfy = str(job[j])
            gates = list(stringfy)
            if not gates.__contains__('1') or len(gates) != machines[j]:
                raise ValueError('All jobs need a 1 as a gate and length equal to number of machines')
            bool_gate = []
            for gate in gates:
                if gate == '1':
                    bool_gate.append(True)
                else:
                    bool_gate.append(False)
            job[j] = bool_gate

    # raw_input is used in python 2
    population_size = int(input('Please input the size of population: ') or 30)  # default value is 30
    crossover_rate = float(input('Please input the size of Crossover Rate: ') or 0.8)  # default value is 0.8
    mutation_rate = float(input('Please input the size of Mutation Rate: ') or 0.2)  # default value is 0.2
    mutation_selection_rate = float(input('Please input the mutation selection rate: ') or 0.2)
    num_mutation_jobs = round(num_gene * mutation_selection_rate)
    num_iteration = int(input('Please input number of iteration: ') or 2000)  # default value is 2000

    shifting = input('Please input if shifting is available:') or 'False'
    if shifting != 'False':
        shifting_rate = float(input('Please input the shifting Mutation rate: ') or 0.8)
        shifting_blocks = int(input('Please input shifting distance (0 to X): ') or 5)

    start_time = time.time()

    '''
    Detect if its a Flex flow shop.
    '''
    is_flow = True
    # compare that all jobs are equal in sequence to job1
    for job in machine_sequence:
        if job != machine_sequence[0]:
            is_flow = False
            break
    '''==================== main code ==============================='''
    '''----- generate initial population -----'''
    Tbest = 9e10
    best_list, best_obj = [], []
    population_list = []
    makespan_record = []

    for i in range(population_size):
        # generate a random permutation of 0 to num_tasks*num_mc-1
        nxm_random_num = list(np.random.permutation(num_gene))
        nxm_random_num = list(np.float_(nxm_random_num))
        # add floats to integers.
        floats = np.random.uniform(low=0, high=1, size=(num_gene,))

        population_list.append(nxm_random_num)  # add to the population_list
        for j in range(num_gene):
            # convert to job number format, every job appears m times
            population_list[i][j] = (population_list[i][j] % num_tasks) + floats[j]

    for n in range(num_iteration):
        Tbest_now = 9e10

        '''-------- two point crossover --------'''
        parent_list = copy.deepcopy(population_list)
        offspring_list = copy.deepcopy(population_list)
        # generate a random sequence to select the parent chromosome to crossover
        S = list(np.random.permutation(population_size))

        for m in range(int(population_size / 2)):
            crossover_prob = np.random.rand()
            if crossover_rate >= crossover_prob:
                parent_1 = population_list[S[2 * m]][:]
                parent_2 = population_list[S[2 * m + 1]][:]
                child_1 = parent_1[:]
                child_2 = parent_2[:]
                cutpoint = list(np.random.choice(num_gene, 2, replace=False))
                cutpoint.sort()

                child_1[cutpoint[0]:cutpoint[1]] = parent_2[cutpoint[0]:cutpoint[1]]
                child_2[cutpoint[0]:cutpoint[1]] = parent_1[cutpoint[0]:cutpoint[1]]
                offspring_list[S[2 * m]] = child_1[:]
                offspring_list[S[2 * m + 1]] = child_2[:]

        '''----------Strip decimals---------'''
        offspring_decimals = np.empty((population_size,num_gene))
        for i in range(population_size):
            for j in range(num_gene):
                offspring_decimals[i][j] = offspring_list[i][j] % 1
                offspring_list[i][j] = int(offspring_list[i][j])
        offspring_decimals = list(offspring_decimals)
        '''----------repairment-------------'''
        for m in range(population_size):

            job_count = {}
            larger, less = [], []  # 'larger' record jobs appear in the chromosome more than m times, and 'less' records less than m times.
            for i in range(num_tasks):
                if i in offspring_list[m]:
                    count = offspring_list[m].count(i)
                    pos = offspring_list[m].index(i)
                    job_count[i] = [count, pos]  # store the above two values to the job_count dictionary
                else:
                    count = 0
                    job_count[i] = [count, 0]
                if count > num_jobs:
                    larger.append(i)
                elif count < num_jobs:
                    less.append(i)

            for k in range(len(larger)):
                chg_job = larger[k]
                while job_count[chg_job][0] > num_jobs:
                    for d in range(len(less)):
                        if job_count[less[d]][0] < num_jobs:
                            # here we repair sequence with new numbers. We also generate new floats for them.
                            offspring_decimals[m][job_count[chg_job][1]] = np.random.uniform(low=0, high=1, size=(1,))
                            offspring_list[m][job_count[chg_job][1]] = less[d]
                            job_count[chg_job][1] = offspring_list[m].index(chg_job)
                            job_count[chg_job][0] = job_count[chg_job][0] - 1
                            job_count[less[d]][0] = job_count[less[d]][0] + 1
                        if job_count[chg_job][0] == num_jobs:
                            break

        '''--------mutation--------'''
        for m in range(len(offspring_list)):
            mutation_prob = np.random.rand()
            if mutation_rate >= mutation_prob:
                # chooses the position to mutation
                m_chg = list(np.random.choice(num_gene, num_mutation_jobs, replace=False))
                # save the value which is on the first mutation position
                t_value_last = offspring_list[m][m_chg[0]]
                t_value_last_decimal = offspring_decimals[m][m_chg[0]]
                for i in range(num_mutation_jobs - 1):
                    # displacement
                    offspring_list[m][m_chg[i]] = offspring_list[m][m_chg[i + 1]]
                    offspring_decimals[m][m_chg[i]] = offspring_decimals[m][m_chg[i + 1]]

                # move the value of the first mutation position to the last mutation position
                offspring_list[m][m_chg[num_mutation_jobs - 1]] = t_value_last
                offspring_decimals[m][m_chg[num_mutation_jobs - 1]] = t_value_last_decimal

        '''------------shifting mutration-----------------'''
        if shifting != 'False':
            total_values = []
            for process in processing_time:
                total_values.append(sum(process))
            max_process = max(total_values)
            idx_max_process = total_values.index(max_process)
            for m in range(population_size):
                probability = np.random.rand()
                if probability > shifting_rate:
                    for i in range(num_gene):
                        if offspring_list[m][i] == idx_max_process:
                            tmp_int = offspring_list[m][i]
                            tmp_dec = offspring_decimals[m][i]
                            new_idx = max(0, i - random.randint(3, shifting_blocks))
                            offspring_list[m][i] = offspring_list[m][new_idx]
                            offspring_decimals[m][i] = offspring_decimals[m][new_idx]
                            offspring_list[m][new_idx] = tmp_int
                            offspring_decimals[m][new_idx] = tmp_dec



        '''--------Concatenate decimals to integers--------------'''
        offspring_list = np.float_(offspring_list)
        offspring_list = list(np.add(offspring_list, offspring_decimals))
        '''--------fitness value(calculate makespan)-------------'''
        # parent and offspring chromosomes combination
        total_chromosome = copy.deepcopy(parent_list) + copy.deepcopy(offspring_list)
        chrom_fitness, chrom_fit = [], []
        total_fitness = 0
        for m in range(population_size * 2):
            # initialization of factory entry.
            job_keys = [j for j in range(num_tasks)]
            key_count = {key: 0 for key in job_keys}
            job_count = {key: 0 for key in job_keys}
            machine_keys = []
            for i in range(len(machines)):
                for j in range(machines[i]):
                    machine_keys.append((i+1)*10 + j+1)

            machine_count = {key: 0 for key in machine_keys}
            machine_last_job_dict = {key: None for key in machine_keys}
            for chromosome in total_chromosome[m]:
                i = int(chromosome)
                decimal = chromosome % 1
                # time of assigning job 'i'. Excel Job = i+1
                gen_t = int(processing_time[i][key_count[i]])
                # Job assigned to machine gen_m.

                # changing the machine depending on the decimal.
                # the chromosome decimal decides what machine is being picked up.
                gen_m = int(machine_sequence[i][key_count[i]])

                suffix = 1

                # loading machine options
                dependency = machine_dependency[i][gen_m-1]
                open_machines = dependency.count(True)
                cut_out = 1 / open_machines
                # here we obtain in which open machine we should allocate the job depending on its decimal.
                while 1:
                    if decimal < cut_out:
                        break
                    suffix += 1
                    cut_out += cut_out

                gated_suffix = 1
                # here we transform the machine to the right one counting that it cna not be allocated to false machine
                for d in dependency:
                    if d is False:
                        gated_suffix += 1
                    else:
                        suffix -= 1
                        if suffix <= 0:
                            break
                        gated_suffix += 1
                gen_m = gen_m*10 + gated_suffix
                # adding sequence dependant times.

                machine_job_shift = job_count[i] - machine_count[gen_m]

                # The machine gen_m has been used previously and checking its time compared to current job.
                if machine_last_job_dict[gen_m] is not None and machine_job_shift < \
                        dependency_time[machine_last_job_dict[gen_m]][i]:
                    if machine_job_shift > 0:
                        changing_time = dependency_time[machine_last_job_dict[gen_m]][i] - machine_job_shift
                    else:
                        changing_time = dependency_time[machine_last_job_dict[gen_m]][i]
                    gen_t += changing_time

                # add time to job count
                job_count[i] = job_count[i] + gen_t
                # add time to mach count
                machine_count[gen_m] = machine_count[gen_m] + gen_t

                if machine_count[gen_m] < job_count[i]:
                    machine_count[gen_m] = job_count[i]
                elif machine_count[gen_m] > job_count[i]:
                    job_count[i] = machine_count[gen_m]

                key_count[i] = key_count[i] + 1
                machine_last_job_dict[gen_m] = i

            makespan = max(job_count.values())
            chrom_fitness.append(1 / makespan)
            chrom_fit.append(makespan)
            total_fitness = total_fitness + chrom_fitness[m]

        '''----------selection(roulette wheel approach)----------'''
        pk, qk = [], []

        for i in range(population_size * 2):
            pk.append(chrom_fitness[i] / total_fitness)
        for i in range(population_size * 2):
            cumulative = 0
            for j in range(0, i + 1):
                cumulative = cumulative + pk[j]
            qk.append(cumulative)

        selection_rand = [np.random.rand() for i in range(population_size)]

        for i in range(population_size):
            if selection_rand[i] <= qk[0]:
                population_list[i] = list(copy.deepcopy(total_chromosome[0]))
            else:
                for j in range(0, population_size * 2 - 1):
                    if selection_rand[i] > qk[j] and selection_rand[i] <= qk[j + 1]:
                        population_list[i] = list(copy.deepcopy(total_chromosome[j + 1]))
                        break
        '''----------comparison----------'''
        for i in range(population_size * 2):
            if chrom_fit[i] < Tbest_now:
                Tbest_now = chrom_fit[i]
                sequence_now = copy.deepcopy(total_chromosome[i])
        if Tbest_now <= Tbest:
            Tbest = Tbest_now
            sequence_best = copy.deepcopy(sequence_now)

        makespan_record.append(Tbest)
        if n % 100 == 0:
            print('Iteration nr: ' + str(n) + ' with value:' + str(Tbest))
    '''----------result----------'''
    print("optimal sequence", sequence_best)
    print("optimal value:%f" % Tbest)
    print('the elapsed time:%s' % (time.time() - start_time))

    plt.title("GA algorithm results")
    plt.plot([i for i in range(len(makespan_record))], makespan_record, 'b')
    plt.ylabel('makespan', fontsize=15)
    plt.xlabel('generation', fontsize=15)
    plt.show()


if __name__ == '__main__':
    run()
