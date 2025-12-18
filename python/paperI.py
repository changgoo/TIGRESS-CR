#!/usr/bin/env python3
# basic imports
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cmasher as cmr

# data loader
from cr_zprof import cr_data_load, load_group, load_windpdf
from load_sim_tigresspp import LoadSimTIGRESSPPAll

# plotting scripts
import plotting_scripts as ps
from plot_slices import plot_snapshot_comp, plot_slices_cr

# model initialization
model_name = {
    "crmhd_v2-8pc-b1-diode-lngrad_out": "crmhd",
    "mhd_v2-8pc-b1-diode": "mhd",
}

model_color = {
    "crmhd_v2-8pc-b1-diode-lngrad_out": "#E77500",
    "mhd_v2-8pc-b1-diode": "#000000",
}
model_edge_color = {
    "mhd_v2-8pc-b1-diode": "#E77500",
    "crmhd_v2-8pc-b1-diode-lngrad_out": "#000000",
}

model_default = [
    "mhd_v2-8pc-b1-diode",
    "crmhd_v2-8pc-b1-diode-lngrad_out",
]

# set directories
outdir = "../figures"
basedir = "/scratch/gpfs/EOST/changgoo/tigress_classic/"

# set snapshot numbers
snapshot_nums = dict(mhd=[88, 75], crmhd=[78, 65])

# sim group
simgroup = dict()
group = "default"

tslice = slice(150, 500)
zslice = slice(-50, 50)


def load(verbose=True):
    # find models
    model_dict = cr_data_load(basedir)

    # load simulations
    sa = LoadSimTIGRESSPPAll(model_dict)

    global simgroup
    simgroup[group] = {m: sa.set_model(m, verbose=verbose) for m in model_default}
    sims = simgroup[group]

    # load data
    load_group(simgroup, group)
    for m, s in sims.items():
        zp = s.load_zprof_postproc()
        load_windpdf(s, tslice=tslice, both=True)

    # setup for plotting scripts
    ps.setup(outdir, model_name, model_color)


def draw_figures(num="all"):
    sims = simgroup[group]

    if num == "all" or num == 1:
        # time evolution
        fig = ps.plot_history(simgroup, group, tslice=tslice, savefig=False)

        for m, s in simgroup[group].items():
            name = model_name[m]
            color = model_color[m]
            ecolor = model_edge_color[m]
            for marker, n in zip(["^", "v"], snapshot_nums[name]):
                slc = s.get_slice(n, "allslc.z", slc_kwargs=dict(z=0, method="nearest"))
                time = slc.attrs["time"] * s.u.Myr
                sfr = np.interp(time, s.hst["time"], s.hst["sfr40"])
                plt.sca(fig.axes[0])
                plt.plot(time, sfr, marker=marker, color=color, mec=ecolor)
        fig.savefig(osp.join(outdir, f"{group}_history.pdf"))

    if num == "all" or num == 2:
        # SFR
        ps.plot_pressure_t(simgroup, group, zslice=zslice)
        ps.plot_pressure_z(simgroup, group, tslice=tslice)
        ps.plot_vertical_equilibrium_t(simgroup, group, zmax=1000)

    if num == "all" or num == 5:
        # filling factors
        ps.plot_area_mass_fraction_z(simgroup, group, tslice=tslice)

    if num == "all" or num == 6:
        # velocities
        ps.plot_velocity_z(simgroup, group, ph="wc", tslice=tslice)
        ps.plot_velocity_z(simgroup, group, ph="hot", tslice=tslice)

    if num == "all" or num == 7:
        # kappa
        ps.plot_kappa_z(simgroup, group, tslice=tslice)
        ps.plot_kappa_z(
            simgroup,
            group,
            phases=[["CNM", "UNM"], "WNM", ["WHIM", "HIM"]],
            tslice=tslice,
        )

    if num == "all" or num == 8:
        # gain/loss
        ps.plot_gainloss_z(simgroup, group, tslice=tslice)
        ps.plot_gainloss_z(
            simgroup,
            group,
            phases=[["CNM", "UNM", "WNM"], "WHIM", "HIM"],
            tslice=tslice,
        )

    if num == "all" or num == 9:
        # flux space-time
        ps.plot_flux_tz(simgroup, group)

    if num == "all" or num == 10:
        # flux
        ps.plot_flux_z(simgroup, group, vz_dir=1, both=True, tslice=tslice)

    if num == "all" or num == 11:
        # loading
        ps.plot_loading_z(simgroup, group, vz_dir=None, both=True, tslice=tslice)

    if num == "all" or num == 12:
        # momentum transfer
        ps.plot_momentum_transfer_z(simgroup, group, showall=False, tslice=tslice)
        ps.plot_momentum_transfer_z(
            simgroup, group, hot="trb", showall=True, tslice=tslice
        )
        ps.plot_momentum_transfer_z(
            simgroup, group, hot="full", showall=True, tslice=tslice
        )

    if num == "all" or num == 13:
        # joint pdfs
        ps.plot_jointpdf(simgroup, group)

    if num == "all" or num == 14:
        # pdfs
        ps.plot_voutpdf(simgroup, group)

    if num == "all" or num == 3:
        # snapshots
        with plt.style.context({"axes.grid": False}):
            nums = [snapshot_nums["mhd"][0], snapshot_nums["crmhd"][0]]
            fig1, fig2 = plot_snapshot_comp(
                sims,
                model_default,
                nums,
                model_name=model_name,
                model_color=model_color,
                kpc=True,
            )
            fig1.savefig(osp.join(outdir, "snapshot_faceon.png"))
            fig2.savefig(osp.join(outdir, "snapshot_edgeon.png"))

        with plt.style.context({"axes.grid": False}):
            nums = [snapshot_nums["mhd"][1], snapshot_nums["crmhd"][1]]
            fig1, fig2 = plot_snapshot_comp(
                sims,
                model_default,
                nums,
                model_name=model_name,
                model_color=model_color,
                kpc=True,
            )
            fig1.savefig(osp.join(outdir, "snapshot_faceon_quiet.png"))
            fig2.savefig(osp.join(outdir, "snapshot_edgeon_quiet.png"))

    if num == "all" or num == 4:
        sim = sims[model_default[1]]
        # adjust label, norm, etc
        sim.dfi["pok_trbz"]["label_name"] = "$P_{\\rm kin}$"
        sim.dfi["pok"]["label_name"] = "$P_{\\rm th}$"
        sim.dfi["pok_mag"]["label_name"] = "$P_{\\rm B}$"
        sim.dfi["vmag"]["label_name"] = "$|v|$"
        sim.dfi["pok_cr"]["imshow_args"]["norm"] = LogNorm(5.0e1, 5.0e4)
        for pok_field in ["pok", "pok_mag", "pok_trbz", "pok_cr"]:
            norm = sim.dfi["pok_cr"]["imshow_args"]["norm"]
            sim.dfi[pok_field]["imshow_args"]["norm"] = norm
        for f in ["vmag", "Vcr_mag", "VAi_mag"]:
            sim.dfi[f]["imshow_args"]["norm"] = LogNorm(vmin=1, vmax=500)
            sim.dfi[f]["imshow_args"]["cmap"] = cmr.prinsenvlag_r
        with plt.style.context({"axes.grid": False}):
            f = plot_slices_cr(sim, snapshot_nums["crmhd"][0], kpc=False)
            f.savefig(osp.join(outdir, "snapshot.png"))

            f = plot_slices_cr(sim, snapshot_nums["crmhd"][1], kpc=False)
            f.savefig(osp.join(outdir, "snapshot_quiet.png"))


if __name__ == "__main__":
    load()
    draw_figures(num="all")
