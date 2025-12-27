import os
import subprocess
import logging

# Setup logging to log the terminal output to a file
logging.basicConfig(filename='download_bdf_files.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# Function to run the git-annex command
def run_git_annex_command(directory, file_name):
    try:
        # Change to the directory where the file is located
        os.chdir(directory)
        logging.info(f"Running git annex get for file: {file_name} in {directory}")

        # Run the git-annex get command to download the file
        result = subprocess.run(['git', 'annex', 'get', file_name], capture_output=True, text=True)

        if result.returncode == 0:
            logging.info(f"Successfully downloaded {file_name} in {directory}")
            print(f"Successfully downloaded {file_name} in {directory}")
        else:
            logging.error(f"Error downloading {file_name} in {directory}: {result.stderr}")
            print(f"Error downloading {file_name} in {directory}: {result.stderr}")

    except Exception as e:
        logging.error(f"Exception occurred while running git-annex in {directory}: {str(e)}")
        print(f"Exception occurred while running git-annex in {directory}: {str(e)}")

# Function to iterate through all the subject directories
def download_files_in_directory(base_directory):
    # Iterate through all sub-001 to sub-024
    for sub_num in range(1, 25):
        sub_directory = f"sub-{sub_num:03d}"  # Sub-directory format: sub-001, sub-002, ..., sub-024

        # Iterate through ses-01 and ses-02
        for ses_num in range(1, 4):
            ses_directory = f"ses-{ses_num:02d}"  # ses-01 or ses-02
            eeg_directory = os.path.join(base_directory, sub_directory, ses_directory, "eeg")

            # Check if the EEG directory exists
            if os.path.isdir(eeg_directory):
                # List all .bdf files in the 'eeg' directory
                for file_name in os.listdir(eeg_directory):
                    if file_name.endswith('.bdf'):
                        file_path = os.path.join(eeg_directory, file_name)

                        # Run git-annex to download the .bdf file
                        run_git_annex_command(eeg_directory, file_name)
            else:
                logging.warning(f"EEG directory not found: {eeg_directory}")
                print(f"EEG directory not found: {eeg_directory}")

if __name__ == '__main__':
    # Define the base directory for your dataset (adjust for your local path)
    base_directory = r'C:\projects\eegnets\ds001787'  # Use raw string to handle backslashes in Windows paths

    # Start the process
    download_files_in_directory(base_directory)
