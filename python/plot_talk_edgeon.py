"""Edge-on (x-z) talk figures/movies for the LGR4 MHD and CRMHD models.

Three figures per snapshot, all cropped to z >= zmin (default -500 pc):
  - slc: y=0 slices of nH, T, vz
  - prj: y-projections of Sigma, outflow and inflow mass flux (all gas)
  - flx: outflow mass flux for wc (T<=2e4 K), int (2e4-5e5 K), hot (T>5e5 K)
Mass fluxes are y-averaged rho*v_z; outflow/inflow means v_z*z > 0 / < 0.

Usage
-----
    # one snapshot for inspection
    python plot_talk_edgeon.py --num 600
    # all snapshots + movies (parallel)
    mpirun -n <N> python plot_talk_edgeon.py --movie
    # R8 models (caches and figures under <outdir>/R8_{mhd,crmhd}/)
    mpirun -n <N> python plot_talk_edgeon.py --suite R8 --movie
"""

import argparse
import gc
import os
import os.path as osp

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpi4py import MPI
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from load_sim_tigresspp import LoadSimTIGRESSPP
from movie_utils import make_movie
from plot_slices import _imshow_slice

# cache_in_outdir: basedir is not writable, so prj.y caches go to <outdir>/<suite>_<model>
suites = dict(
    LGR4=dict(
        root="/projects/EOSTRIKE/TIGRESSPP-CR/",
        cache_in_outdir=False,
        models=dict(
            mhd="midtall_LGR4_mhd_duale-8pc-b1-mhdbc_diode-fgas_0.90-r200-t2.5-icpx",
            crmhd="6kpc_LGR4_crmhd_duale-8pc-b1-mhdbc_diode-crbc_lngrad_out-fgas_0.90-r200-t2.5-icpx",
        ),
    ),
    R8=dict(
        root="/scratch/gpfs/EOST/ronan/",
        cache_in_outdir=True,
        models=dict(
            mhd="R8_mhd_duale-8pc-b1-mhdbc_diode-fgas_0.90-r200-icpx",
            crmhd="6kpc_R8_crmhd-8pc-b1-mhdbc_diode-crbc_lngrad_out-fgas_0.90-r200-t2.5-icpx",
        ),
    ),
)
model_name = dict(mhd="MHD", crmhd="CRMHD")
outdir_default = "/scratch/gpfs/EOST/changgoo/talk_figures/"
talk_rc = {
    "font.size": 16,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "axes.linewidth": 1.2,
}
cbar_thickness = 0.2  # inches


def _setup_axes(sim, nf, zmin, panel_width=1.8, gap=0.4):
    """Fixed-size layout (in inches) so every movie frame has the same pixel size."""
    Lx = sim.domain["Lx"][0]
    zmax = sim.domain["re"][2]
    panel_height = panel_width * (zmax - zmin) / Lx
    left, right, bottom, top = 0.85, 0.3, 0.8, 1.6
    xsize = left + nf * panel_width + (nf - 1) * gap + right
    ysize = bottom + panel_height + top
    fig = plt.figure(figsize=(xsize, ysize), num=0)
    axes = []
    for i in range(nf):
        x0 = left + i * (panel_width + gap)
        rect = [x0 / xsize, bottom / ysize, panel_width / xsize, panel_height / ysize]
        axes.append(fig.add_axes(rect))
    return fig, axes


def _decorate(sim, fig, axes, im_list, labels, zmin, tMyr, name, ticks=None):
    le = sim.domain["le"] * 1.0e-3
    re = sim.domain["re"] * 1.0e-3
    if ticks is None:
        ticks = [None] * len(axes)
    for ax, im, label, tick in zip(axes, im_list, labels, ticks):
        ax_height = ax.get_position().height * fig.get_figheight()
        ax.set(xlim=(le[0], re[0]), ylim=(zmin * 1.0e-3, re[2]))
        ax.set_xticks([-0.5, 0, 0.5])
        cax = inset_axes(
            ax,
            width="85%",
            height="100%",
            loc="lower center",
            bbox_to_anchor=(0, 1.01, 1, cbar_thickness / ax_height),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=tick)
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
        cbar.set_label(label)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False, labelbottom=False)
    axes[0].set(xlabel=r"$x\,[{\rm kpc}]$", ylabel=r"$z\,[{\rm kpc}]$")
    axes[0].annotate(
        f"{name}\nt={tMyr:.0f} Myr",
        (0.05, 0.99),
        ha="left",
        va="top",
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="k", lw=1, alpha=0.8),
    )


@plt.rc_context(talk_rc)
def plot_talk_slices(sim, num, name, zmin=-500, flist=("nH", "T", "vz"), savdir=None):
    # existing slice caches always live in basedir, even when savdir is elsewhere
    slc = sim.get_slice(
        num,
        "allslc.y",
        savdir=osp.join(sim.basedir, "allslc.y"),
        slc_kwargs=dict(y=0, method="nearest"),
    )
    slc = slc.sel(z=slice(zmin, None))
    fig, axes = _setup_axes(sim, len(flist), zmin)
    im_list, labels = [], []
    for ax, field in zip(axes, flist):
        plt.sca(ax)
        dfi_ = sim.dfi[field]
        data = dfi_["func"](slc, sim.u) if "func" in dfi_ else slc[field]
        im_list.append(_imshow_slice(data, "x", "z", 1.0e-3, dfi_["imshow_args"]))
        labels.append(dfi_["label_name"] + "\n" + dfi_["label_unit"])
    tMyr = slc.attrs["time"] * sim.u.Myr
    _decorate(sim, fig, axes, im_list, labels, zmin, tMyr, name)
    if savdir is not None:
        os.makedirs(savdir, exist_ok=True)
        fig.savefig(osp.join(savdir, f"slc_{num:04d}.png"), dpi=200)
    return fig


flux_unit = r"$[{\rm M_\odot\,kpc^{-2}\,yr^{-1}}]$"
flux_ticks = [1.0e-5, 1.0e-3, 1.0e-1]
# zero flux (masked by LogNorm) and values below vmin take the bottom color
out_kwargs = dict(
    norm=LogNorm(1.0e-5, 1.0e-1),
    cmap=cmr.ember.with_extremes(bad=cmr.ember(0), under=cmr.ember(0)),
)
in_kwargs = dict(
    norm=LogNorm(1.0e-5, 1.0e-1),
    cmap=cmr.freeze.with_extremes(bad=cmr.freeze(0), under=cmr.freeze(0)),
)


def get_prj_split(sim, num):
    """y-projection with phase and outflow/inflow split; rebuild old-format caches."""
    prj = sim.get_prj(num, "y", prefix="prj.y")
    if "mflux_out" not in prj or "int" not in prj.phase:
        prj = sim.get_prj(num, "y", prefix="prj.y", force_override=True)
    return prj


def _plot_prj_panels(sim, num, name, zmin, panels, savname):
    fig, axes = _setup_axes(sim, len(panels), zmin)
    im_list, labels, ticks = [], [], []
    for ax, (data, kwargs, label, tick) in zip(axes, panels):
        plt.sca(ax)
        im_list.append(_imshow_slice(data, "x", "z", 1.0e-3, kwargs))
        labels.append(label)
        ticks.append(tick)
    tMyr = panels[0][0].attrs["time"] * sim.u.Myr
    _decorate(sim, fig, axes, im_list, labels, zmin, tMyr, name, ticks=ticks)
    if savname is not None:
        os.makedirs(osp.dirname(savname), exist_ok=True)
        fig.savefig(savname, dpi=200)
    return fig


@plt.rc_context(talk_rc)
def plot_talk_projections(sim, num, name, zmin=-500, savdir=None):
    """Sigma, outflow and inflow mass flux magnitudes for all gas."""
    prjdata = get_prj_split(sim, num).sel(z=slice(zmin, None)).sel(phase="whole")
    prjkwargs, _ = sim.set_prj_dfi()
    sigma = prjdata["Sigma"].assign_attrs(prjdata.attrs)
    panels = [
        (sigma, prjkwargs["Sigma"],
         r"$\Sigma_{\rm gas}$" + "\n" + r"$[{\rm M_\odot\,pc^{-2}}]$", None),
        (np.abs(prjdata["mflux_out"]), out_kwargs,
         r"$\mathcal{F}_{\rho,{\rm out}}$" + "\n" + flux_unit, flux_ticks),
        (np.abs(prjdata["mflux_in"]), in_kwargs,
         r"$\mathcal{F}_{\rho,{\rm in}}$" + "\n" + flux_unit, flux_ticks),
    ]
    savname = None if savdir is None else osp.join(savdir, f"prj_{num:04d}.png")
    return _plot_prj_panels(sim, num, name, zmin, panels, savname)


@plt.rc_context(talk_rc)
def plot_talk_outflow_phase(sim, num, name, zmin=-500, savdir=None):
    """Outflow mass flux magnitudes for wc (T<=2e4), int (2e4-5e5), hot (T>5e5)."""
    prjdata = get_prj_split(sim, num).sel(z=slice(zmin, None))
    panels = []
    for phase in ["wc", "int", "hot"]:
        data = np.abs(prjdata["mflux_out"].sel(phase=phase)).assign_attrs(prjdata.attrs)
        label = rf"$\mathcal{{F}}_{{\rho,{{\rm out}}}}^{{\rm {phase}}}$"
        panels.append((data, out_kwargs, label + "\n" + flux_unit, flux_ticks))
    savname = None if savdir is None else osp.join(savdir, f"flx_{num:04d}.png")
    return _plot_prj_panels(sim, num, name, zmin, panels, savname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=list(suites), default="LGR4")
    parser.add_argument("--model", choices=["mhd", "crmhd", "all"], default="all")
    parser.add_argument("--num", type=int, default=None, help="single snapshot")
    parser.add_argument("--zmin", type=float, default=-500.0)
    parser.add_argument("--basedir-root", default=None, help="parent of model dirs")
    parser.add_argument("--outdir", default=outdir_default)
    parser.add_argument("--movie", action="store_true", help="make mp4 at the end")
    args = parser.parse_args()

    COMM = MPI.COMM_WORLD
    suite = suites[args.suite]
    models = suite["models"]
    root = suite["root"] if args.basedir_root is None else args.basedir_root
    mlist = list(models) if args.model == "all" else [args.model]
    for m in mlist:
        savdir = osp.join(args.outdir, f"{args.suite}_{m}")
        sim = LoadSimTIGRESSPP(
            osp.join(root, models[m]),
            savdir=savdir if suite["cache_in_outdir"] else None,
            verbose=False,
        )
        nums = [args.num] if args.num is not None else sim.nums
        mynums = nums[COMM.rank :: COMM.size]
        for num in mynums:
            print(COMM.rank, m, num, flush=True)
            for func in [plot_talk_slices, plot_talk_projections, plot_talk_outflow_phase]:
                f = func(sim, num, model_name[m], zmin=args.zmin, savdir=savdir)
                plt.close(f)
            gc.collect()

        COMM.barrier()
        if args.movie and COMM.rank == 0:
            for kind in ["slc", "prj", "flx"]:
                fin = osp.join(savdir, f"{kind}_*.png")
                fout = osp.join(args.outdir, f"{args.suite}_{m}_{kind}.mp4")
                make_movie(fin, fout, fps_in=15, fps_out=15)
