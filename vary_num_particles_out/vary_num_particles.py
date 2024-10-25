import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd

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
#SBATCH -t 00:15:00
#SBATCH -N 1
#SBATCH -n 1

module load intel-oneapi/2022.1.0

export SLURM_CPU_BIND="cores"
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

make realclean
make exe

mkdir -p $HOME/CS5220-P2/vtune_results_threading_particles_{h_value}
vtune -collect threading -result-dir $HOME/CS5220-P2/vtune_results_threading_particles_{h_value} -- srun ./sph.x -s {h_value}
"""

# Placeholder for job ids and their corresponding h values
job_h_map = {}

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
#         print(f"submitting simulation with {h} num threads...")
#         result = subprocess.run(f"sbatch {sbatch_file}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

#         # Capture the job ID from the submission output
#         for line in result.stdout.splitlines():
#             if "Submitted batch job" in line:
#                 job_id = line.split()[-1]
#                 job_h_map[job_id] = h
#                 job_ids.append(job_id)
#                 print(f"job {job_id} submitted for {h} num threads")
#                 submitted = True

#     # Wait for the jobs to complete before submitting more
#     if job_ids:
#         print(f"waiting for jobs {job_ids} to complete...")
#         while are_jobs_running(job_ids):
#             time.sleep(10)  # Check every 10 seconds

#         # Rename output files from sph_<job_id>.out to sph_h_<h>.out
#         for job_id in job_ids:
#             h = job_h_map[job_id]
#             old_output_file = f"sph_{job_id}.out"
#             new_output_file = f"sph_{h}_thread.out"
#             if os.path.exists(old_output_file):
#                 os.rename(old_output_file, new_output_file)
#                 print(f"renamed {old_output_file} to {new_output_file}")

#     # If all h values have been submitted, break the loop
#     if not job_ids:
#         print('all values have been submitted')
#         break

# Placeholder for results: (h, num_particles, execution_time)
results = []

def finalize_vtune_results(result_dir, h):
    if not os.path.exists(result_dir):
        print(f"result directory {result_dir} does not exist. Skipping finalization.")

    # Finalize the results
    finalize_command = f"vtune -finalize -r {result_dir}"
    result = subprocess.run(finalize_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if result.returncode != 0:
        print(f"error finalizing results for {h} particles")
        print(result.stderr)
    else:
        print(f"finalized results for {h} particles")

    # Save the summary report to a CSV file
    report_file = f"/global/homes/a/acs378/CS5220-P2/poor_cpu_util_wait_vs_num_particles/summary_{h}_particles.csv"
    report_command = f"vtune -report summary -r {result_dir} -format csv -report-output {report_file}"
    result = subprocess.run(report_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if result.returncode != 0:
        print(f"error saving report summary for {h} threads")
        print(result.stderr)
    else:
        print(f"saved report summary to {report_file}")

# for h in h_values:
#     finalize_vtune_results(f"/global/homes/a/acs378/CS5220-P2/vtune_results_threading_particles_{h}", h)

# Directory containing the summary files
directory = '/global/homes/a/acs378/CS5220-P2/poor_cpu_util_wait_vs_num_particles'  # Replace with the actual path

# Lists to store extracted data
h_values = []
wait_times = []
total_times = []
serial_times = []
cpu_utilizations = []

# Regex patterns to extract metrics from the file content
h_value_pattern = re.compile(r'summary_([\d.]+)_particles\.csv')
wait_time_pattern = re.compile(r'Wait Time with poor CPU Utilization\s+(\d+\.\d+)s')
elapsed_time_pattern = re.compile(r'Elapsed Time\s+(\d+\.\d+)')
serial_time_pattern = re.compile(r'Serial Time \(outside parallel regions\)\s+(\d+\.\d+)s')
cpu_utilization_pattern = re.compile(r'Effective CPU Utilization\s+(\d+\.\d+)%')

# Iterate over the files in the directory
for filename in os.listdir(directory):
    if h_value_pattern.match(filename):
        # Extract h_value from the filename
        h_value_match = h_value_pattern.search(filename)
        if h_value_match:
            h_value = float(h_value_match.group(1))
        
        # Read the content of the file
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Extract the metrics
        wait_time_match = wait_time_pattern.search(content)
        elapsed_time_match = elapsed_time_pattern.search(content)
        serial_time_match = serial_time_pattern.search(content)
        cpu_utilization_match = cpu_utilization_pattern.search(content)
        thread_count_match = thread_count_pattern.search(content)
        
        if wait_time_match and elapsed_time_match:
            wait_time = float(wait_time_match.group(1))
            elapsed_time = float(elapsed_time_match.group(1))
            serial_time = float(serial_time_match.group(1)) if serial_time_match else None
            cpu_utilization = float(cpu_utilization_match.group(1)) if cpu_utilization_match else None

            # Store the extracted data
            h_values.append(h_value)
            wait_times.append(wait_time)
            total_times.append(elapsed_time)
            if serial_time is not None:
                serial_times.append(serial_time)
            if cpu_utilization is not None:
                cpu_utilizations.append(cpu_utilization)

# Create a DataFrame for easier plotting
data = pd.DataFrame({
    'h_value': h_values,
    'Wait Time with poor CPU Utilization': wait_times,
    'Elapsed Time': total_times
})

if serial_times:
    data['Serial Time'] = serial_times

if cpu_utilizations:
    data['CPU Utilization'] = cpu_utilizations

data = data.sort_values(by='h_value')

# Calculate normalized wait times
data['Normalized Wait Time'] = data['Wait Time with poor CPU Utilization'] / data['Elapsed Time']

# Plot 1: Wait Time with Poor CPU Utilization vs Number of Threads
plt.figure(figsize=(10, 6))
plt.plot(data['h_value'], data['Wait Time with poor CPU Utilization'], '-o', label='Wait Time with Poor CPU Utilization')
plt.xlabel('Number of Particles (-s)')
plt.ylabel('Wait Time with Poor CPU Utilization (s)')
plt.title('Wait Time with Poor CPU Utilization vs Number of Particles')
plt.grid(True)
plt.legend()
plt.savefig('poor_cpu_util_wait_vs_num_particles/poor_cpu_util_wait_v_num_particles.png')

# Plot 2: Total Elapsed Time vs Number of Threads
plt.figure(figsize=(10, 6))
plt.plot(data['h_value'], data['Elapsed Time'], '-o', label='Total Elapsed Time')
plt.xlabel('Number of Number of Particles (h_value)')
plt.ylabel('Elapsed Time (s)')
plt.title('Total Elapsed Time vs Number of Particles')
plt.grid(True)
plt.legend()
plt.savefig('poor_cpu_util_wait_vs_num_particles/total_elapsed_time_v_num_particles.png')

# Plot 3: Normalized Wait Time (as a fraction of total time) vs Number of Threads
plt.figure(figsize=(10, 6))
plt.plot(data['h_value'], data['Normalized Wait Time'], '-o', label='Normalized Wait Time (fraction of total time)')
plt.xlabel('Number of Number of Particles (h_value)')
plt.ylabel('Normalized Wait Time (fraction of total time)')
plt.title('Normalized Wait Time vs Number of Particles')
plt.grid(True)
plt.legend()
plt.savefig('poor_cpu_util_wait_vs_num_particles/normalized_wait_time_v_num_particles.png')

# Plot 4: Serial Time vs Number of Threads (if Serial Time is available)
if 'Serial Time' in data.columns:
    plt.figure(figsize=(10, 6))
    plt.plot(data['h_value'], data['Serial Time'], '-o', label='Serial Time (outside parallel regions)')
    plt.xlabel('Number of Number of Particles (h_value)')
    plt.ylabel('Serial Time (s)')
    plt.title('Serial Time vs Number of Particles')
    plt.grid(True)
    plt.legend()
    plt.savefig('poor_cpu_util_wait_vs_num_particles/serial_time_v_num_particles.png')