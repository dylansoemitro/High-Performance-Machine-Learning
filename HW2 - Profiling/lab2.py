import subprocess

scripts = [
    'lab2-C1.py',
    'lab2-C2.py',
    'lab2-C3.py',
    'lab2-C4.py',
    'lab2-C5.py',
    'lab2-C6.py',
    'lab2-C7.py',
    'lab2-Q3.py',
]

for script in scripts:
    print(f"Running {script} with default parameters...")
    subprocess.run(['python', script], check=True)
    print(f"Finished running {script}\n")