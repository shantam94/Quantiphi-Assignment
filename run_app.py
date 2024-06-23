import subprocess
import sys

# Collect arguments for the Streamlit app
args = sys.argv[1:]
# print(args)
# Construct the command
command = ['streamlit', 'run', 'main.py'] 

# Run the Streamlit app
subprocess.run(command)
