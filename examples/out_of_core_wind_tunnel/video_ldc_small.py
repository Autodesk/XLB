# GPU only baseline, Basically the same problem but at the highest possible resolution for a single GH200

import numpy as np
import tabulate
import warp as wp
import mpi4py # TODO: actually learn how mpi works...
mpi4py.rc.thread_level = 'serialized'  # or 'funneled'
import mpi4py.MPI as MPI

wp.init()

from ldc import LDC

if __name__ == '__main__':

    # Parameters
    output_directory = "./output_video_ldc_small"
    #shape = 1920
    #shape = 960
    shape = 256
    nr_frames = 1024
    save_q_criterion_frequency = 256
    nr_steps = nr_frames * save_q_criterion_frequency
    collision = "BGK"
    velocity_set = "D3Q19"
    amr_block_shape = (128, 128, 128)
    amr_ghost_cell_thickness = 16
    nr_streams = 2
    tau = 0.5005
    #tau = 0.51
    #collision = "BGK"
    collision = "SmagorinskyLESBGK"

    # MPI
    comm = MPI.COMM_WORLD

    # Make wind tunnel
    ldc = LDC(
        output_directory=output_directory,
        shape=shape,
        tau=tau,
        nr_steps=nr_steps,
        save_q_criterion_frequency=save_q_criterion_frequency,
        collision=collision,
        velocity_set=velocity_set,
        use_amr=True,
        amr_block_shape=amr_block_shape,
        amr_ghost_cell_thickness=amr_ghost_cell_thickness,
        nr_streams=nr_streams,
        comm=comm,
    )

    # Run MLUPs test
    ldc.run()
