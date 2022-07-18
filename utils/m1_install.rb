#!/usr/bin/ruby

# build espresso
def build_espresso(download_path)

    # get current directory so you can jump back later.

    git_cmd = "git clone ... " + download_path + "/espressomd"
    system(git_cmd)
    # Check for failure

    build_1 = "cd " + download_path + "/espressomd"
    build_2 = "mkdir build && cd build"
    build_3 = "cmake ../"
    build_4 = "cmake --build ."

    system(build_1)
    system(build_2)
    system(build_3)
    system(build_2)

    # Check that install was successful.
    # Set espresso executable to path.

    jump_path = "cd /home/path"
    system(jump_path)
end

# Install jaxlib version
def install_jaxlib()
    cmd = "wget jax path"
    system(cmd)
    cmd = "pip install jax_path"
    system(cmd)

    # Check python installed correctly.
end

# create_conda_environment
def build_conda_env(conda_path)
    cmd = conda_path " create -n swarmrl python=3.9"
    system(cmd)
end

# Check conda status
def check_conda_status()
    cmd = "which conda"
    system(cmd)

    # Check that it exists and raise error if not.
end

# Install the swarmrl package
def install_swarmrl()
    # TODO: Split this shit.
    cmd = "cd ../ && pip install -r requirements.txt && pip install -r dev-requirements.txt && pip install -e ."
    system(cmd)
    # Check for successful install

    cmd = "cd utils"
    system(cmd)
end

# Run the swarmrl test suite.
def run_swarm_tests()
    cmd_1 = "cd ../CI"
    system(cmd_1)

    cmd_2 = "pytest unit_tests/"
    system(cmd_2)
    cmd_3 = "pytest integration_tests/"
    system(cmd_3)
    cmd_4 = "pypresso run_espresso_test_suite.py"
    system(cmd_4)
end

def main()
    # Preparation for the install.
    conda_path = check_conda_status()  # find the conda executable.
    build_conda_env(conda_path)        # Create a new conda environment.

    build_espresso()                   # Build espresso.
    install_jaxlib()                   # Build jaxlib.

    run_swarm_tests                    # Run the swarmrl tests.
end
