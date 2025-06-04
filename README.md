# Chemical Data Generation
Repository for doctoral research on machine learning for data generation, with applications in chemical data generation and classification.

<!---
Notes on terminology:
Instead of lab-generated spectra - EXPERIMENTAL spectra
Instead of synthetic spectra - IN-SILICO spectra
Instead of machine - INSTRUMENT
--->
<!---
GitHub resources:
https://www.gitkraken.com/learn/git/git-flow
https://nvie.com/posts/a-successful-git-branching-model/
--->

## Steps to run this repo:
1. Set up a virtual environment with Python 3.10. You can use either venv or conda/micromamba:
   - Using venv (Linux/macOS):
     ```bash
     python3.10 -m venv your_env_name
     source your_env_name/bin/activate
     ```
   - Using venv (Windows):
     ```cmd
     python3.10 -m venv your_env_name
     .\your_env_name\Scripts\activate
     ```
   - Using conda:
     ```bash
     conda create -n your_env_name python=3.10
     conda activate your_env_name
     ```
   - Using micromamba:
     ```bash
     micromamba create -n your_env_name python=3.10
     micromamba activate your_env_name
     ```
2. Install packages using:
   ```bash
   pip install -r requirements.txt
   ```
   Verify correct installation with:
   ```bash
   pip list
   ```
3. Add virtual environment to `.gitignore` as it is unnecessary for GitHub to track.
4. Create new kernel using:
   ```bash
   python -m ipykernel install --user --name=your_kernel_name
   ```
5. If necessary, update git config with username and email using:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "youremail@example.com"
   ```

## Steps to run on Turing:
1. ssh into turing account and run commands ```sinteractive``` and ```tmux```.  (```tmux kill-server``` to kill all existing tmux sessions)
2. Activate virtual env using one of the commands above.
3. To run jupyter notebooks and save output to notebook file: jupyter nbconvert --to notebook --execute --inplace notebook_name.ipynb

## Steps to set up ssh:
1. Use ```nano config``` to update config file (likely at ~/.ssh/config) with:
  Host [host_nickname]
    HostName [host address, like: turing.wpi.edu]
    User [username]
2. Eliminate need to enter passphrase by:
* starting ssh agent with ```eval "$(ssh-agent -s)"```.
* adding your ssh key with ```ssh-add ~/.ssh/id_rsa```.
3. ssh into vm with ```ssh host_nickname```.

To save output to a file: ```python your_script.py > output.txt 2>&1```.
## Notation:
Mathematical notation rules applied throughout this project are taken from [this](https://wookai.github.io/paper-tips-and-tricks/math.html) article.
![Notation rules](images/notation_rules.png)

## Naming and Formatting Conventions:
File systems - Preprocessing technique (e.g. spectrum, CARL) -> model (encoder, decoder) + '_results' -> training dataset (e.g. universal, group) + f'_{model}'
  * Example: plots -> CARL -> encoder_results -> group_encoder

Data file column names for n columns - 
  * Col number:    0,    1,      2,   ..., n-len(classes), n-len(classes)-1, ...,     n
  * Name:        Index, p_184, p_185, ...,   e.g. DEB,        e.g. DEM,      ...,   Label 
  * Meaning: sample ID, sample values ...,      class one-hot encodings      ..., class name

  Note: column formatting established after creation of many data files. As a result, some files may have different column formatting.