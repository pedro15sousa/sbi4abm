#!/usr/bin/env python3
"""Run the sample data.

See README.md in this directory for more information.
"""

import argparse
import gzip
import multiprocessing
import os
import shutil
import subprocess
import sys
import pandas as pd

def try_remove(f):
    try:
        os.remove(f)
    except OSError as e:
        pass

try:
    cpu_count = len(os.sched_getaffinity(0))
except AttributeError:
    # os.sched_getaffinity isn't available
    cpu_count = multiprocessing.cpu_count()
if cpu_count is None or cpu_count == 0:
    cpu_count = 2

print(cpu_count)

# Default values
covid_sim_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../report9/covid-sim'))
report9_dir = os.path.join(covid_sim_dir, 'report9')
gb_suppress_dir = os.path.join(report9_dir, 'GB_suppress')
output_dir = os.path.join(gb_suppress_dir, 'output')
src_dir = os.path.join(covid_sim_dir, 'src')
threads = cpu_count
country = "United_Kingdom"
# covidsim = None
covidsim = os.path.join(covid_sim_dir, "build", "src", "CovidSim")

# Some command_line settings
# r = 3.0
r = 2.6
rs = r/2

# Lists of places that need to be handled specially
united_states = [ "United_States" ]
canada = [ "Canada" ]
usa_territories = ["Alaska", "Hawaii", "Guam", "Virgin_Islands_US", "Puerto_Rico", "American_Samoa"]
nigeria = ["Nigeria"]
school_file = None

def check_build():
    # Determine whether we need to build the tool or use a user supplied one:
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
    if covidsim is not None:
        exe = covidsim
    else:
        build_dir = os.path.join(output_dir, "build")

        # Ensure we do a clean build
        shutil.rmtree(build_dir, ignore_errors=True)
        os.makedirs(build_dir, exist_ok=False)
        cwd = os.getcwd()
        os.chdir(build_dir)
        print(os.getcwd())

        # Build
        subprocess.run(['cmake', src_dir], check=True)
        subprocess.run(['cmake', '--build', '.'], check=True)

        # Where the exe ends up depends on the OS.
        if os.name == 'nt':
            exe = os.path.join(build_dir, "src", "Debug", "CovidSim.exe")
        else:
            exe = os.path.join(build_dir, "src", "CovidSim")

        os.chdir(cwd)
    return exe

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def get_population_file():
    # Population density file in gziped form, text file, and binary file as
    # processed by CovidSim

    wpop_file = os.path.join(
            report9_dir,
            "population",
            "GB_pop2018_nhs.txt")
    
    wpop_bin = os.path.join(
            report9_dir,
            "population",
            "GB_pop2018.bin")

    return wpop_file, wpop_bin


def config_pre_parameter():
    # Configure pre-parameter file.  This file doesn't change between runs:
    pp_file = os.path.join(gb_suppress_dir, "preGB_R0=2.0.txt")
    return pp_file


def config_no_intervention():
    # Configure No intervention parameter file.  This is run first
    # and provides a baseline
    no_int_file = os.path.join(gb_suppress_dir, "p_NoInt.txt")
    return no_int_file

def config_intervention():
    # Configure an intervention (controls) parameter file.
    cf = os.path.join(gb_suppress_dir, "p_PC_CI_HQ_SD.txt")
    return cf

def get_network_bin():
    # This is the temporary network that represents initial state of the
    # simulation
    # using 120 threats
    network_bin = os.path.join(
            report9_dir,
            "population",
            "NetworkGB_120T.bin")
    return network_bin

def run_intervention_sim(exe, cf, pp_file, wpop_bin, 
                         network_bin, param_values):
    relative_spatial_contact_rate_given_social_distancing, delay_to_start_case_isolation = param_values
    # update_param_file(cf, param_value)
    print("Param Value: ", str(param_values[0]))
    cmd = [
            exe,
            "/c:120".format(threads)
            ]
    cmd.extend([
            "/NR:1",
            "/PP:" + pp_file,
            "/P:" + cf,
            "/CLP1:" + "400",
            "/CLP2:" + "1000",
            "/CLP3:" + "1000",
            "/CLP4:" + "1000",
            "/CLP5:" + "300",
            "/CLP6:" + str(relative_spatial_contact_rate_given_social_distancing),
            "/CLP7:" + str(delay_to_start_case_isolation),
            "/O:" + os.path.join(output_dir,
                "PC_CI_HQ_SD_400_300_R0=2.6"),
            "/D:" + wpop_bin, # Binary pop density file (speedup)
            "/L:" + network_bin, # Network to load
            "/R:{0}".format(rs),
            "98798150",
            "729101",
            "17389101",
            "4797132"
            ])
    print("Command line: " + " ".join(cmd))
    process = subprocess.run(cmd, check=True)


def outcomes():
    # This is where you would read the output files and calculate outcomes
    # of interest.  For example, the number of infected people at the end of
    # the simulation.

    # For now we just print the name of the output files
    cumulative_deaths = {}
    cumulative_crit = {}
    cumulative_sari = {}
    cumulative_ili = {}
    cumulative_mild = {}
    files_to_check = [f for f in os.listdir(output_dir) if f.endswith(".avNE.severity.xls")]
    scenarios = [f.replace(".avNE.severity.xls", "") for f in files_to_check]
    
    for scenario in scenarios:
        severity_file_name = os.path.join(output_dir, f"{scenario}.avNE.severity.xls")
        print(f"Checking {severity_file_name}")
        if os.path.exists(severity_file_name):
            severity_results = pd.read_csv(severity_file_name, sep="\t")
            severity_results = severity_results.dropna(axis=1, how='all')

            print(severity_results.columns)
            
            if 'cumDeath' in severity_results.columns:
                total_cum_death = severity_results['cumDeath'].iloc[-1]
                cumulative_deaths[scenario] = total_cum_death
            
            if 'cumCritical' in severity_results.columns:
                total_cum_crit = severity_results['cumCritical'].iloc[-1]
                cumulative_crit[scenario] = total_cum_crit

            if 'cumSARI' in severity_results.columns:
                total_cum_sari = severity_results['cumSARI'].iloc[-1]
                cumulative_sari[scenario] = total_cum_sari

            if 'cumILI' in severity_results.columns:
                total_cum_ili = severity_results['cumILI'].iloc[-1]
                cumulative_ili[scenario] = total_cum_ili

            if 'cumMild' in severity_results.columns:
                total_cum_mild = severity_results['cumMild'].iloc[-1]
                cumulative_mild[scenario] = total_cum_mild
            else:
                print(f"Warning: 'cumDeath' column not found in {severity_file_name}")
        else:
            print(f"Warning: File {severity_file_name} does not exist.")
    
    return cumulative_deaths[scenarios[0]], cumulative_crit[scenarios[0]], cumulative_sari[scenarios[0]], cumulative_ili[scenarios[0]], cumulative_mild[scenarios[0]]

class Model:

    def __init__(self):
        self.exe = check_build()
        self.pp_file = config_pre_parameter()
        self.no_int_file = config_no_intervention()
        self.cf = config_intervention()
        self.wpop_file, self.wpop_bin = get_population_file()
        self.network_bin = get_network_bin()
        # self.non_inter = True

    def simulate(self, pars=None, seed=None, T=None):
        if not (pars is None):
            param_values = [float(p) for p in pars]
        else:
            param_values = [0.25, 1]

        run_intervention_sim(self.exe, self.cf, self.pp_file, self.wpop_bin, self.network_bin, param_values)
        cumulative_deaths, cumulative_crit, cumulative_sari, cumulative_ili, cumulative_mild = outcomes()
        print("Param values: ", param_values)
        # cumulative_deaths = 0
        # cumulative_crit = 0
        # cumulative_sari = 0
        # cumulative_ili = 0
        # cumulative_mild = 0
        return cumulative_deaths, cumulative_crit, cumulative_sari, cumulative_ili, cumulative_mild


if __name__ == "__main__":
    # param_values = [0.8, 0.2]
    param_values = [0.85, 1]
    model = Model()
    model.simulate(param_values)
    cumulative_deaths, cumulative_crit, cumulative_sari, cumulative_ili, cumulative_mild = outcomes()
    print(cumulative_deaths)
    print(cumulative_crit)
    print(cumulative_sari)
    print(cumulative_ili)
    print(cumulative_mild)