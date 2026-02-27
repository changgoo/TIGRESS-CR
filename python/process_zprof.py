import sys
import gc
from load_sim_tigresspp import LoadSimTIGRESSPP
from mpi4py import MPI

if __name__ == "__main__":
    COMM = MPI.COMM_WORLD
    spp = LoadSimTIGRESSPP(sys.argv[1], verbose=COMM.rank == 0)

    mynums = [spp.nums[i] for i in range(len(spp.nums)) if i % COMM.size == COMM.rank]
    print(COMM.rank, mynums)
    for num in mynums:
        gc.collect()
        zprof = spp.load_zprof_postproc_one(num, force_override=False)
