import pyrado


if not pyrado.mujoco_available:
    raise ImportError(
        "You are trying to use are MuJoCo-based environment, but the required mujoco_py module can not be imported.\n"
        "Try adding\n"
        "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USER/.mujoco/mujoco200/bin\n"
        "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so\n"
        "to your shell's rc-file.\n"
        "If you are using PyCharm or CLion, also add the environment variables above to your run configurations. "
        "Note that the IDE will not resolve $USER for some reason, so enter the user name directly, "
        "or run it from your terminal."
    )
