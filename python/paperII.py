#!/usr/bin/env python3
# basic imports
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cmasher as cmr

# data loader
from cr_zprof import cr_data_load, load_group, load_windpdf
from load_sim_tigresspp import LoadSimTIGRESSPP

# plotting scripts
import plotting_scripts as ps
from plot_slices import plot_snapshot_comp, plot_slices_cr

# set directories (on stellar)
basedir = "/scratch/gpfs/changgoo/tigress_classic/"
outdir = "../paperII_figures"

# slices
tslice = slice(150, 500)
zslice = slice(-50, 50)

# sim group
simgroup = dict()
model_name = dict()
model_color = dict()


def load(model_dict,verbose=True):
    global simgroup, model_name, model_color

    # initialize simgroup
    groups = ["vAi", "vAtot", "nost", "sigma28", "sigma29", "crmhd"]
    for gr in groups:
        simgroup[gr] = dict()

    # categorize models
    mlist = []
    for k, d in model_dict.items():
        sim = LoadSimTIGRESSPP(d, verbose=verbose)
        par = sim.par
        base_split = sim.basename.split("-")
        model = []
        if par["cr"]["self_consistent_flag"] == 0:
            sigma_exp = int(-np.log10(par["cr"]["sigma"]))
            sigma = f"sigma{sigma_exp}"
            model.append(sigma)
            if par["cr"]["valfven_flag"] == 1:
                model.append("vAi")
            elif par["cr"]["valfven_flag"] == 0:
                model.append("vAtot")
            elif par["cr"]["valfven_flag"] == -1:
                model.append("nost")
                if par["cr"]["vs_flag"] == 1:
                    print("no streaming with invalud valfven_flag")
            if base_split[0] != "crmhd":
                model.append(base_split[0].split("_")[1])
        else:
            model.append(base_split[0])
            if par["problem"]["beta0"] != 1:
                model.append(f"b{par['problem']['beta0']}")
            if par["cr"]["vmax"] != 2.0e9:
                model.append(f"Vmax{int(par['cr']['vmax'] / 1.0e9)}")
        if "rst" in base_split[-1]:
            model.append(base_split[-1])
        newkey = "-".join(model)

        if newkey in mlist:
            print(f"{newkey} is already there")
        else:
            for gr in groups:
                if gr in newkey:
                    simgroup[gr][newkey] = sim
        mlist.append(newkey)
        model_name[newkey] = newkey
        if verbose:
            print(f"Renamed {k} --> {newkey}")

    # load data/ assign colors
    for group in simgroup:
        load_group(simgroup, group)
        for i, k in enumerate(simgroup[group]):
            if k not in model_color:
                model_color[k] = f"C{i}"

    # setup for plotting scripts
    ps.setup(outdir, model_name, model_color)


if __name__ == "__main__":
    load()
