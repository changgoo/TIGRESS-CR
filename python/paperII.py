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


def load(verbose=True):
    # find models
    model_dict = cr_data_load(basedir)

    for k,d in model_dict.items():
        a,b = k.replace("crmhd_duale","crmhd-duale").replace("8pc-b1-diode-lngrad_out-","").split("_")
        head = a.replace("sigma","σ").replace("crmhd-","")
        if b == "va1":
            tail = "vAi"
        elif b == "va0":
            tail = "vAtot"
        elif b == "va-1":
            tail = "nost"
        newk = "-".join([head,tail])
        print(newk)
        sim = LoadSimTIGRESSPP(d, verbose=verbose)
        if tail not in simgroup:
            simgroup[tail] = dict()
        simgroup[tail][newk] = sim
        model_name[newk] = newk

    simgroup["σ29"] = {"σ29-vAi":simgroup["vAi"]["σ29-vAi"],
                       "σ29-vAtot":simgroup["vAtot"]["σ29-vAtot"],
                       "σ29-nost":simgroup["nost"]["σ29-nost"],
                       "duale-σ29-vAi":simgroup["vAi"].pop("duale-σ29-vAi")}
    simgroup["σ28"] = {"σ28-vAi":simgroup["vAi"]["σ28-vAi"],
                       "σ28-vAtot":simgroup["vAtot"]["σ28-vAtot"],
                       "σ28-nost":simgroup["nost"]["σ28-nost"],
                       }
    # simgroup["σ27"] = {"σ27-vAi":simgroup["vAi"]["σ27-vAi"],
    #                    "σ27-vAtot":simgroup["vAtot"]["σ27-vAtot"],
    #                    "σ27-nost":simgroup["nost"]["σ27-nost"],
    #                    }

    # load data
    i = 0
    for group in simgroup:
        load_group(simgroup, group)
        for k in simgroup[group]:
            if k not in model_color:
                model_color[k] = f"C{i}"
                i+=1

    # setup for plotting scripts
    ps.setup(outdir, model_name, model_color)


if __name__ == "__main__":
    load()
