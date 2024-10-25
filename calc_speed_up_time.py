import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt

# Serial execution time
serial_time = 644.447  # seconds

# List of thread counts to test
thread_counts = [1, 2, 4, 8, 16, 32, 64, 128]

# Template for the sbatch script content
sbatch_template = """#!/bin/bash
#SBATCH -J sph
#SBATCH -o sph_%j.out
#SBATCH -e sph_%j.err
#SBATCH -A m4776
#SBATCH -C cpu
#SBATCH -c {n_threads}
#SBATCH --qos=debug
#SBATCH -t 00:20:00
#SBATCH -N 1
#SBATCH -n 1

export SLURM_CPU_BIND="cores"
export OMP_NUM_THREADS={n_threads}
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
make exe
srun ./sph.x
"""

# Placeholder for job ids and their corresponding thread counts
job_thread_map = {}

# Function to check if jobs are running
def are_jobs_running(job_ids):
    result = subprocess.run("squeue -u $USER", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    running_jobs = set()
    for line in result.stdout.splitlines()[1:]:  # Skip header line
        parts = line.split()
        running_jobs.add(parts[0])  # Add job ID to running_jobs
    # Return whether any of the provided job IDs are still running
    return any(job_id in running_jobs for job_id in job_ids)

# Submit SPH simulations in batches
thread_iter = iter(thread_counts)
batch_size = 2

while True:
    # Submit up to `batch_size` jobs at a time
    job_ids = []
    for _ in range(batch_size):
        try:
            n_threads = next(thread_iter)
        except StopIteration:
            break

        # Create a temporary sbatch file with the appropriate thread count
        sbatch_file = f"submit_sph_job_{n_threads}.sub"
        with open(sbatch_file, 'w') as f:
            f.write(sbatch_template.format(n_threads=n_threads))

        # Submit the sbatch job
        print(f"Submitting simulation with {n_threads} threads...")
        result = subprocess.run(f"sbatch {sbatch_file}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        # Capture the job ID from the submission output
        for line in result.stdout.splitlines():
            if "Submitted batch job" in line:
                job_id = line.split()[-1]
                job_thread_map[job_id] = n_threads
                job_ids.append(job_id)
                print(f"Job {job_id} submitted for {n_threads} threads")

    # Wait for the jobs to complete before submitting more
    if job_ids:
        print(f"Waiting for jobs {job_ids} to complete...")
        while are_jobs_running(job_ids):
            time.sleep(10)  # Check every 10 seconds

        # Rename output files from sph_<job_id>.out to sph_h_<h>.out
        for job_id in job_ids:
            h = job_thread_map[job_id]
            old_output_file = f"sph_{job_id}.out"
            new_output_file = f"sph_thread_{h}.out"
            if os.path.exists(old_output_file):
                os.rename(old_output_file, new_output_file)
                print(f"Renamed {old_output_file} to {new_output_file}")

    # If all thread counts have been submitted, break the loop
    if not job_ids:
        break

# Placeholder for results: (n_threads, execution_time)
results = []

# After all jobs are done, parse the output files and extract execution times
for n_threads in thread_counts:
    output_file = f"sph_thread_{n_threads}.out"
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                if "Ran in" in line:
                    execution_time = float(line.split()[2])  # Assuming format: "Ran in X seconds"
                    results.append((n_threads, execution_time))
                    print(f"{n_threads} threads, time = {execution_time}s")

# Convert results to numpy arrays for plotting
if results:  # Ensure results is not empty
    threads, times = zip(*results)
    threads = np.array(threads)
    times = np.array(times)

    # Calculate speedup: Speedup = Serial Time / Parallel Time
    speedup = serial_time / times

    # Plot the speedup
    plt.figure()
    plt.plot(threads, speedup, '-o', label='Speedup')
    plt.xlabel('Number of Threads/CPU cores')
    plt.ylabel('Speedup')
    plt.title('Speedup vs Number of Threads/CPU cores')
    plt.grid(True)
    plt.legend()

    # Save the plot
    plt.savefig('speedup_plot.png')
    plt.show()
else:
    print("No results to plot.")
