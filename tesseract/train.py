# make training MODEL_NAME=name-of-the-resulting-model DATA_DIR=/data GROUND_TRUTH_DIR=/data/GT
# make training MODEL_NAME=sanity_check START_MODEL=heb_best TESSDATA=../tessdata/ MAX_ITERATIONS=2000 LANG_TYPE=RTL
# GROUND_TRUTH_DIR
# TESSDATA
# MAX_ITERATIONS
# EPOCHS
# DEBUG_INTERVAL
# LEARNING_RATE
# NET_SPEC
# FINETUNE_TYPE
# LANG_TYPE
# PSM
# RANDOM_SEED
# RATIO_TRAIN
# TARGET_ERROR_RATE
# OUTPUT_DIR
# DATA_DIR

import subprocess


def run_make_command(target: str, makefile_path: str, options: dict) -> None:
    # Base command with Makefile path
    command = ["make", "-f", makefile_path]

    # Add options as key=value pairs
    for key, value in options.items():
        command.append(f"{key}={value}")

    # Append the target
    command.append(target)
    command.append("--debug")

    print("Running command:")
    print(" ".join(command))
    # Execute the command
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("Make command output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running make command: {e}")
        print("Make command stderr:")
        print(e.stderr)


target = "training"
makefile_path = r"C:/Users/sgala/Documents/python_projects/tesseract/tesstrain/Makefile"
options = {
    "MODEL_NAME": "sanity_check",
    "START_MODEL": "heb_best",
    "TESSDATA": r"C:/Users/sgala/Documents/python_projects/tesseract/tessdata/",
    "LANG_TYPE": "RTL",
    "GROUND_TRUTH_DIR": r'"C:\Users\sgala\Documents\python_projects\tesseract\tesstrain\data\sanity_check-ground-truth"',
    "DATA_DIR": "/data"
}

run_make_command(target=target,makefile_path=makefile_path,options=options)