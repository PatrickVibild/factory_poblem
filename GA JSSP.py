# -*- coding: utf-8 -*-
"""
Reference: https://github.com/wurmen/Genetic-Algorithm-for-Job-Shop-Scheduling-and-NSGA-II/
"""

'''==========Solving job shop scheduling problem by gentic algorithm in python======='''
# importing required modules
import pandas as pd
import numpy as np
import time
import copy
import matplotlib.pyplot as plt


def run():
    ''' ================= initialization setting ======================'''
    pt_tmp = pd.read_excel("JSP_dataset.xlsx", sheet_name="Processing Time", index_col=[0])
    ms_tmp = pd.read_excel("JSP_dataset.xlsx", sheet_name="Machines Sequence", index_col=[0])
    dependency_tmp = pd.read_excel("JSP_dataset.xlsx", sheet_name="Sequence Dependency", index_col=[0])

    dfshape = pt_tmp.shape
    num_mc = dfshape[1]  # number of machines
    num_job = dfshape[0]  # number of jobs
    num_gene = num_mc * num_job  # number of genes in a chromosome

    processing_time = [list(map(int, pt_tmp.iloc[i])) for i in range(num_job)]
    machine_sequence = [list(map(int, ms_tmp.iloc[i])) for i in range(num_job)]
    dependency_time = [list(map(int, dependency_tmp.iloc[i])) for i in range(num_job)]
    # raw_input is used in python 2
    population_size = int(input('Please input the size of population: ') or 30)  # default value is 30
    crossover_rate = float(input('Please input the size of Crossover Rate: ') or 0.8)  # default value is 0.8
    mutation_rate = float(input('Please input the size of Mutation Rate: ') or 0.2)  # default value is 0.2
    mutation_selection_rate = float(input('Please input the mutation selection rate: ') or 0.2)
    num_mutation_jobs = round(num_gene * mutation_selection_rate)
    num_iteration = int(input('Please input number of iteration: ') or 2000)  # default value is 2000

    start_time = time.time()

    '''==================== main code ==============================='''
    '''----- generate initial population -----'''
    Tbest = 9e10
    best_list, best_obj = [], []
    population_list = []
    makespan_record = []
    for i in range(population_size):
        nxm_random_num = list(np.random.permutation(num_gene))  # generate a random permutation of 0 to num_job*num_mc-1
        population_list.append(nxm_random_num)  # add to the population_list
        for j in range(num_gene):
            # convert to job number format, every job appears m times
            population_list[i][j] = population_list[i][j] % num_job

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

        '''----------repairment-------------'''
        for m in range(population_size):
            job_count = {}
            larger, less = [], []  # 'larger' record jobs appear in the chromosome more than m times, and 'less' records less than m times.
            for i in range(num_job):
                if i in offspring_list[m]:
                    count = offspring_list[m].count(i)
                    pos = offspring_list[m].index(i)
                    job_count[i] = [count, pos]  # store the above two values to the job_count dictionary
                else:
                    count = 0
                    job_count[i] = [count, 0]
                if count > num_mc:
                    larger.append(i)
                elif count < num_mc:
                    less.append(i)

            for k in range(len(larger)):
                chg_job = larger[k]
                while job_count[chg_job][0] > num_mc:
                    for d in range(len(less)):
                        if job_count[less[d]][0] < num_mc:
                            offspring_list[m][job_count[chg_job][1]] = less[d]
                            job_count[chg_job][1] = offspring_list[m].index(chg_job)
                            job_count[chg_job][0] = job_count[chg_job][0] - 1
                            job_count[less[d]][0] = job_count[less[d]][0] + 1
                        if job_count[chg_job][0] == num_mc:
                            break

        '''--------mutatuon--------'''
        for m in range(len(offspring_list)):
            mutation_prob = np.random.rand()
            if mutation_rate >= mutation_prob:
                m_chg = list(
                    np.random.choice(num_gene, num_mutation_jobs, replace=False))  # chooses the position to mutation
                t_value_last = offspring_list[m][m_chg[0]]  # save the value which is on the first mutation position
                for i in range(num_mutation_jobs - 1):
                    offspring_list[m][m_chg[i]] = offspring_list[m][m_chg[i + 1]]  # displacement

                offspring_list[m][m_chg[
                    num_mutation_jobs - 1]] = t_value_last  # move the value of the first mutation position to the last mutation position

        '''--------fitness value(calculate makespan)-------------'''
        # parent and offspring chromosomes combination
        total_chromosome = copy.deepcopy(parent_list) + copy.deepcopy(offspring_list)
        chrom_fitness, chrom_fit = [], []
        total_fitness = 0
        for m in range(population_size * 2):
            # initialization of factory entry.
            job_keys = [j for j in range(num_job)]
            key_count = {key: 0 for key in job_keys}
            job_count = {key: 0 for key in job_keys}
            machine_keys = [j + 1 for j in range(num_mc)]
            machine_count = {key: 0 for key in machine_keys}
            machine_last_job_dict = {key: None for key in machine_keys}
            for i in total_chromosome[m]:
                # time of assigning job 'i'. Excel Job = i+1
                gen_t = int(processing_time[i][key_count[i]])
                # Job assigned to machine gen_m.
                gen_m = int(machine_sequence[i][key_count[i]])
                # adding sequence dependant times.

                machine_job_shift = job_count[i] - machine_count[gen_m]

                # The machine gen_m has been used previously and checking its time compared to current job.
                if machine_last_job_dict[gen_m] is not None and machine_job_shift < dependency_time[machine_last_job_dict[gen_m]][i]:
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
                population_list[i] = copy.deepcopy(total_chromosome[0])
            else:
                for j in range(0, population_size * 2 - 1):
                    if selection_rand[i] > qk[j] and selection_rand[i] <= qk[j + 1]:
                        population_list[i] = copy.deepcopy(total_chromosome[j + 1])
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
