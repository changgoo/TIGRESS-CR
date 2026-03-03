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


def load(model_dict, gr, tslice=slice(200, 500), colors=None, verbose=True):
    simgroup = dict()
    model_name = dict()
    model_color = dict()

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
        # if "rst" in base_split[-1]:
        #     model.append(base_split[-1])
        newkey = "-".join(model)

        mlist.append(newkey)
        model_name[newkey] = newkey.replace("sigma", "σ").replace("-vAi", "")

        print(f"Renamed {k} --> {newkey}: {model_name[newkey]}")
        simgroup[gr][newkey] = sim

    # load data/ assign colors
    for group in simgroup:
        load_group(simgroup, group)
        for i, (m, s) in enumerate(simgroup[group].items()):
            if isinstance(tslice, dict):
                s.tslice_Myr = tslice[m]
            else:
                s.tslice_Myr = tslice
            s.tslice = slice(s.tslice_Myr.start / s.u.Myr, s.tslice_Myr.stop / s.u.Myr)

            load_windpdf(s, both=True)
            # zp_pp = s.load_zprof_postproc()

            if colors is None:
                model_color[m] = f"C{i}"
            else:
                model_color[m] = colors[i]

            print(
                f"Loaded {m} in group {group} with name {model_name[m]} and color {model_color[m]}"
            )

    return simgroup, model_name, model_color


if __name__ == "__main__":
    load()
