# import os
# import subprocess
# import time
# import numpy as np
# import matplotlib.pyplot as plt

# # List of h values to test (particle size)
# h_values = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1]

# # Template for the sbatch script content
# sbatch_template = """#!/bin/bash
# #SBATCH -J sph
# #SBATCH -o sph_%j.out
# #SBATCH -e sph_%j.err
# #SBATCH -A m4776
# #SBATCH -C cpu
# #SBATCH -c 32
# #SBATCH --qos=debug
# #SBATCH -t 00:10:00
# #SBATCH -N 1
# #SBATCH -n 1

# export SLURM_CPU_BIND="cores"
# export OMP_NUM_THREADS=8
# export OMP_PROC_BIND=spread
# export OMP_PLACES=threads
# make exe
# srun ./sph.x -s {h_value}
# """

# # Placeholder for job ids and their corresponding h values
# job_h_map = {}

# # Function to estimate the number of particles based on h (assuming a 1x1x1 box)
# def estimate_num_particles(h):
#     volume_of_box = 1.0
#     volume_of_particle = (h**3)
#     num_particles = volume_of_box / volume_of_particle
#     return int(num_particles)

# # Function to check if jobs are running
# def are_jobs_running(job_ids):
#     result = subprocess.run("squeue -u $USER", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
#     running_jobs = set()
#     for line in result.stdout.splitlines()[1:]:  # Skip header line
#         parts = line.split()
#         running_jobs.add(parts[0])  # Add job ID to running_jobs
#     # Return whether any of the provided job IDs are still running
#     return any(job_id in running_jobs for job_id in job_ids)

# # Submit SPH simulations in batches of 2
# h_iter = iter(h_values)
# batch_size = 2

# while True:
#     # Submit up to two jobs at a time
#     job_ids = []
#     for _ in range(batch_size):
#         try:
#             h = next(h_iter)
#         except StopIteration:
#             break

#         # Create a temporary sbatch file with the appropriate h value
#         sbatch_file = f"submit_sph_job_{h}.sub"
#         with open(sbatch_file, 'w') as f:
#             f.write(sbatch_template.format(h_value=h))

#         # Submit the sbatch job
#         print(f"Submitting simulation with h = {h}...")
#         result = subprocess.run(f"sbatch {sbatch_file}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

#         # Capture the job ID from the submission output
#         for line in result.stdout.splitlines():
#             if "Submitted batch job" in line:
#                 job_id = line.split()[-1]
#                 job_h_map[job_id] = h
#                 job_ids.append(job_id)
#                 print(f"Job {job_id} submitted for h = {h}")

#     # Wait for the jobs to complete before submitting more
#     if job_ids:
#         print(f"Waiting for jobs {job_ids} to complete...")
#         while are_jobs_running(job_ids):
#             time.sleep(10)  # Check every 10 seconds

#     # If all h values have been submitted, break the loop
#     if not job_ids:
#         break

# # Placeholder for results: (h, num_particles, execution_time)
# results = []

# # After all jobs are done, parse the output files and extract execution times
# for h in h_values:
#     output_file = f"sph_h_{h}.out"
#     with open(output_file, 'r') as f:
#         for line in f:
#             if "Ran in" in line:
#                 execution_time = float(line.split()[2])  # Assuming format: "Ran in X seconds"
#                 num_particles = estimate_num_particles(h)
#                 results.append((h, num_particles, execution_time))
#                 print(f"h = {h}, N = {num_particles}, time = {execution_time}s")

# # Convert results to numpy arrays for plotting
# h_vals, n_vals, times = zip(*results)
# n_vals = np.array(n_vals)
# times = np.array(times)

# # Plot the results on a log-log scale
# plt.figure()
# plt.loglog(n_vals, times, '-o', label='SPH Runtime')
# plt.xlabel('Number of Particles (N)')
# plt.ylabel('Execution Time (seconds)')
# plt.title('SPH Runtime vs Number of Particles')
# plt.grid(True)
# plt.legend()

# # Save the plot
# plt.savefig('sph_scaling_plot.png')
# plt.show()

import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt

# List of h values to test (particle size)
h_values = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1]

# Template for the sbatch script content
sbatch_template = """#!/bin/bash
#SBATCH -J sph
#SBATCH -o sph_%j.out
#SBATCH -e sph_%j.err
#SBATCH -A m4776
#SBATCH -C cpu
#SBATCH -c 32
#SBATCH --qos=debug
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH -n 1

export SLURM_CPU_BIND="cores"
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
make exe
srun ./sph.x -s {h_value}
"""

# Placeholder for job ids and their corresponding h values
job_h_map = {}

# Function to estimate the number of particles based on h (assuming a 1x1x1 box)
def estimate_num_particles(h):
    volume_of_box = 1.0
    volume_of_particle = (h**3)
    num_particles = volume_of_box / volume_of_particle
    return int(num_particles)

# Function to check if jobs are running
def are_jobs_running(job_ids):
    result = subprocess.run("squeue -u $USER", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    running_jobs = set()
    for line in result.stdout.splitlines()[1:]:  # Skip header line
        parts = line.split()
        running_jobs.add(parts[0])  # Add job ID to running_jobs
    # Return whether any of the provided job IDs are still running
    return any(job_id in running_jobs for job_id in job_ids)

# Submit SPH simulations in batches of 2
h_iter = iter(h_values)
batch_size = 2

while True:
    # Submit up to two jobs at a time
    job_ids = []
    for _ in range(batch_size):
        try:
            h = next(h_iter)
        except StopIteration:
            break

        # Create a temporary sbatch file with the appropriate h value
        sbatch_file = f"submit_sph_job_{h}.sub"
        with open(sbatch_file, 'w') as f:
            f.write(sbatch_template.format(h_value=h))

        # Submit the sbatch job
        print(f"Submitting simulation with h = {h}...")
        result = subprocess.run(f"sbatch {sbatch_file}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        # Capture the job ID from the submission output
        for line in result.stdout.splitlines():
            if "Submitted batch job" in line:
                job_id = line.split()[-1]
                job_h_map[job_id] = h
                job_ids.append(job_id)
                print(f"Job {job_id} submitted for h = {h}")

    # Wait for the jobs to complete before submitting more
    if job_ids:
        print(f"Waiting for jobs {job_ids} to complete...")
        while are_jobs_running(job_ids):
            time.sleep(10)  # Check every 10 seconds

        # Rename output files from sph_<job_id>.out to sph_h_<h>.out
        for job_id in job_ids:
            h = job_h_map[job_id]
            old_output_file = f"sph_{job_id}.out"
            new_output_file = f"sph_h_{h}.out"
            if os.path.exists(old_output_file):
                os.rename(old_output_file, new_output_file)
                print(f"Renamed {old_output_file} to {new_output_file}")

    # If all h values have been submitted, break the loop
    if not job_ids:
        break

# Placeholder for results: (h, num_particles, execution_time)
results = []

# After all jobs are done, parse the output files and extract execution times
for h in h_values:
    output_file = f"sph_h_{h}.out"
    with open(output_file, 'r') as f:
        for line in f:
            if "Ran in" in line:
                execution_time = float(line.split()[2])  # Assuming format: "Ran in X seconds"
                num_particles = estimate_num_particles(h)
                results.append((h, num_particles, execution_time))
                print(f"h = {h}, N = {num_particles}, time = {execution_time}s")

# Convert results to numpy arrays for plotting
h_vals, n_vals, times = zip(*results)
n_vals = np.array(n_vals)
times = np.array(times)

# Plot the results on a log-log scale
plt.figure()
plt.loglog(n_vals, times, '-o', label='SPH Runtime')
plt.xlabel('Number of Particles (N)')
plt.ylabel('Execution Time (seconds)')
plt.title('SPH Runtime vs Number of Particles')
plt.grid(True)
plt.legend()

# Save the plot
plt.savefig('sph_log_time_scaling_plot.png')
plt.show()
