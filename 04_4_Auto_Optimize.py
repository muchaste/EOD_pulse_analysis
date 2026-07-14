import os
import sys
import glob
import subprocess

# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing all event data (passed to 05_2 as root folder)
ROOT_FOLDER    = r'E:\\'

# Folder where results CSVs, proposals, and plots are saved
OUTPUT_FOLDER  = r'E:\track_tuning\output_combined_greedy'

# Maximum number of optimization iterations to run
MAX_ITERATIONS = 3

# Optional pass1 override for 05_2: '', 'greedy', or 'lap'.
PASS1_MODE_OVERRIDE = 'greedy'

# Python interpreter to use (current interpreter by default)
PYTHON_EXE     = sys.executable

# Paths to the scripts
SCRIPT_04_3    = os.path.join(os.path.dirname(__file__), '04_3_Iterative_Optimization.py')
SCRIPT_04_2    = os.path.join(os.path.dirname(__file__), '04_2_Tracking_Parameter_Tuning.py')

# ============================================================
# SETUP
# ============================================================
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
flag_path = os.path.join(OUTPUT_FOLDER, '_optimization_state.txt')

def read_flag():
    state = {}
    if os.path.isfile(flag_path):
        with open(flag_path) as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    k, v = line.split('=', 1)
                    state[k.strip()] = v.strip()
    return state

def run_script(script_path, extra_env=None):
    env = os.environ.copy()
    env['TUNING_RESULTS_FOLDER'] = OUTPUT_FOLDER
    env['TUNING_ROOT_FOLDER']    = ROOT_FOLDER
    env['TUNING_OUTPUT_FOLDER']  = OUTPUT_FOLDER
    if extra_env:
        env.update(extra_env)
    result = subprocess.run([PYTHON_EXE, script_path], env=env)
    if result.returncode != 0:
        print(f"\nERROR: {os.path.basename(script_path)} exited with code {result.returncode}. Stopping.")
        sys.exit(result.returncode)

# ============================================================
# STARTUP
# ============================================================
n_proposals_done = len(glob.glob(os.path.join(OUTPUT_FOLDER, 'proposals_iter*.csv')))
n_results_done   = len(glob.glob(os.path.join(OUTPUT_FOLDER, 'parameter_tuning_results_iter*.csv')))

print("=" * 70)
print("AUTOMATED ITERATIVE PARAMETER OPTIMIZATION")
print("=" * 70)
print(f"Root folder:    {ROOT_FOLDER}")
print(f"Output folder:  {OUTPUT_FOLDER}")
print(f"Max iterations: {MAX_ITERATIONS}")
print(f"Proposals done: {n_proposals_done}")
print(f"Results done:   {n_results_done}")
print()

# Check if already converged from a previous run
state = read_flag()
if state.get('converged') == '1':
    it = state.get('iteration', '?')
    print(f"Already converged at iteration {it} (from previous run). Exiting.")
    sys.exit(0)

# ============================================================
# AUTOMATION LOOP
# ============================================================
# At the start of each outer loop:
#   - 05_5 generates proposals_iter{N}.csv  (if not already done for this N)
#   - 05_2 evaluates those proposals → parameter_tuning_results_iter{N}.csv
# We detect which step to start from based on what files exist.

for loop_i in range(MAX_ITERATIONS):
    n_results_done = len(glob.glob(os.path.join(OUTPUT_FOLDER, 'parameter_tuning_results_iter*.csv')))
    current_iter   = n_results_done + 1

    print("=" * 70)
    print(f"LOOP {loop_i + 1}  —  optimization iteration {current_iter}")
    print("=" * 70)

    # ---- STEP 1: run 05_5 to generate proposals ----
    proposals_path = os.path.join(OUTPUT_FOLDER, f'proposals_iter{current_iter:02d}.csv')
    if os.path.isfile(proposals_path):
        print(f"[05_5] proposals_iter{current_iter:02d}.csv already exists, skipping.\n")
    else:
        print(f"[05_5] Generating proposals_iter{current_iter:02d}.csv ...\n")
        run_script(SCRIPT_04_3)
        print(f"\n[05_5] Done.")

    state = read_flag()
    proposals_path = state.get('proposals_path', proposals_path)

    if state.get('converged') == '1':
        print(f"\n*** CONVERGED at iteration {state.get('iteration', current_iter)} ***")
        break

    # ---- STEP 2: run 05_2 to evaluate proposals ----
    result_csv = os.path.join(OUTPUT_FOLDER, f'parameter_tuning_results_iter{current_iter:02d}.csv')
    if os.path.isfile(result_csv):
        print(f"[05_2] parameter_tuning_results_iter{current_iter:02d}.csv already exists, skipping.\n")
    else:
        print(f"[05_2] Evaluating {os.path.basename(proposals_path)} ...\n")
        extra_env = {'TUNING_PROPOSALS_CSV': proposals_path}
        if PASS1_MODE_OVERRIDE:
            extra_env['TUNING_PASS1_MODE'] = PASS1_MODE_OVERRIDE
        run_script(SCRIPT_04_2, extra_env=extra_env)
        print(f"\n[04_2] Done.")

    if not os.path.isfile(result_csv):
        print(f"ERROR: expected output {result_csv} was not created. Stopping.")
        sys.exit(1)

    print(f"Completed iterations: {len(glob.glob(os.path.join(OUTPUT_FOLDER, 'parameter_tuning_results_iter*.csv')))}\n")

else:
    print(f"\nReached MAX_ITERATIONS ({MAX_ITERATIONS}) without converging.")
    print("Increase MAX_ITERATIONS or explore the convergence curve.")

print("\nDone.")
