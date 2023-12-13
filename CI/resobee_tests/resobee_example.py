#!/usr/bin/env python
# coding: utf-8

# In[1]:


import swarmrl.engine.resobee as resobee
import subprocess
import multiprocessing
import os
infomsg = "I "


# In[2]:


resobee_root_path = "/tikhome/stovey/work/Repositories/swarm-rc"

build_path = os.path.join(resobee_root_path, "build")
config_dir = os.path.join(resobee_root_path, 'workflow/projects/debug/parameter-combination-0/seed-0')

target = 'many_body_simulation'
resobee_executable = os.path.join(resobee_root_path, 'build/src', target)


# In[6]:


from IPython.utils import io
with io.capture_output() as captured:

    # run this to rebuild the resobee executable
    max_workers = multiprocessing.cpu_count()
    print(infomsg, f"Found {max_workers} cores.")

    # print(infomsg)
    # print(infomsg, "Clean previous build...")
    # cmd = ['cmake', '--build', build_path, '--target', 'clean']
    # print(infomsg, ' '.join(cmd))
    # subprocess.run(cmd)

    # print(infomsg)
    # print(infomsg, "Configuring cmake...")
    # cmd = ['cmake', f'-B {build_path}']
    # print(infomsg, ' '.join(cmd))
    # subprocess.run(cmd)

    print(infomsg)
    print(infomsg, f"Building target {target}...")
    cmd = ["cmake", "--build", build_path, "--target", target, "-j", str(max_workers)]
    print(infomsg, ' '.join(cmd))
    subprocess.run(cmd)


# In[4]:


system_runner = resobee.ResoBee(
    resobee_executable=resobee_executable,
    config_dir=config_dir
)


# In[ ]:


# todo: cannot overwrite h5 output file for some reason

system_runner.integrate(0, None)


# In[ ]:




