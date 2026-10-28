"""Face-on (x-y) gas surface density talk movies with star clusters, LGR4 or R8.

One moon-theme frame per snapshot and model: Sigma_gas from the z-projection cache
(<basedir>/prj.z, all gas) with young star clusters on top. Clusters are particles
with stellar mass mstar > 0 and age < --agemax (default 40 Myr, the SN-producing lifetime),
projected along z, coloured by age (cool_r, 0-agemax) with symbol area ~ sqrt(mstar).
mstar is used rather than the sink mass (mass), which also counts the gas reservoir and
is up to ~10x mstar for clusters younger than ~5 Myr. Zero-mass particles (runaway OB
stars / SN tracers) are not drawn.

Annotations are kept to the run name, time, a 200 pc scale bar and two colour bars on the
right: Sigma (top 65% of the map height) and cluster age (bottom 25%).
The Sigma colour scale (pink_r, 1-1000 Msun/pc^2) is the same for both suites.
Frames have a fixed pixel size (no tight bbox) so they can be stitched into a movie.

Data: <basedir-root>/<model dir> with model dirs from plot_talk_edgeon.suites; if the
suite root does not exist on this machine, /projects/EOSTRIKE/TIGRESSPP-CR/ is used.

Usage
-----
    python plot_talk_faceon.py --suite R8 --num 300             # one frame per model
    mpirun -n 4 python plot_talk_faceon.py --suite LGR4 --movie  # all frames + mp4

Outputs: <outdir>/<suite>_<model>/fon_NNNN.png and <outdir>/<suite>_<model>_fon.mp4
"""

import argparse
import gc
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from mpi4py import MPI

from load_sim_tigresspp import LoadSimTIGRESSPP
from movie_utils import make_movie
from plot_talk_edgeon import outdir_default, suites

moon_style = osp.join(osp.dirname(osp.abspath(__file__)), "..", "talk-moon.mplstyle")
moon_muted = "#93a1a1"
moon_color = dict(mhd="#93a1a1", crmhd="#cb4b16")
moon_name = dict(mhd="MHD", crmhd="CRMHD")
basedir_fallback = "/projects/EOSTRIKE/TIGRESSPP-CR/"
sigma_norm = (1.0, 1.0e3)  # Msun/pc^2, both suites
sigma_cmap = plt.get_cmap("pink_r")
sigma_cmap = sigma_cmap.with_extremes(under=sigma_cmap(0.0), over=sigma_cmap(1.0))
age_cmap = plt.get_cmap("cool_r")
scalebar_pc = 200.0


def cluster_size(mstar_msun):
    """Marker area [pt^2] from cluster stellar mass; area ~ sqrt(mstar) keeps all sizes readable."""
    return 40.0 * np.sqrt(mstar_msun / 1.0e3)


def young_clusters(sim, num, agemax):
    sp = sim.load_parbin(num)
    sp = sp[sp["mstar"] > 0].copy()
    sp["mstar_msun"] = sp["mstar"] * sim.u.mass.to("Msun").value  # stars only, not sink mass
    sp["age_myr"] = sp["age"] * sim.u.Myr
    sp = sp[(sp["age_myr"] >= 0) & (sp["age_myr"] < agemax)]
    return sp.sort_values("age_myr", ascending=False)  # youngest on top


def plot_faceon(sim, num, suite, model, agemax=40.0, savdir=None):
    prj = sim.get_prj(num, "z", prefix="prj.z", savdir=osp.join(sim.basedir, "prj.z"))
    sigma = prj["Sigma"].sel(phase="whole")
    sp = young_clusters(sim, num, agemax)
    tMyr = float(prj.attrs["time"]) * sim.u.Myr
    le = sim.domain["le"]
    re = sim.domain["re"]
    with plt.style.context(["default", moon_style, {"savefig.bbox": "standard"}]):
        # fixed layout in inches: square map, title above, colorbars right, scale bar below
        fig = plt.figure(figsize=(7.8, 7.0))  # 1170 x 1050 px (even, for h264)
        W, H = fig.get_size_inches()
        x0, y0, w = 0.3, 0.75, 5.6
        ax = fig.add_axes([x0 / W, y0 / H, w / W, w / H])
        cax = fig.add_axes([(x0 + w + 0.2) / W, (y0 + 0.35 * w) / H, 0.25 / W, 0.65 * w / H])
        aax = fig.add_axes([(x0 + w + 0.2) / W, y0 / H, 0.25 / W, 0.25 * w / H])
        im = ax.imshow(
            sigma.values, origin="lower", extent=[le[0], re[0], le[1], re[1]],
            norm=LogNorm(*sigma_norm), cmap=sigma_cmap, interpolation="nearest",
        )  # fmt: skip
        sc = ax.scatter(
            sp["x1"], sp["x2"], s=cluster_size(sp["mstar_msun"]), c=sp["age_myr"],
            cmap=age_cmap, norm=Normalize(0, agemax), edgecolors="w", linewidths=0.6,
        )  # fmt: skip
        ax.set(xlim=(le[0], re[0]), ylim=(le[1], re[1]), aspect="equal", xticks=[], yticks=[])
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(r"$\Sigma_{\rm gas}$ [$M_\odot\,{\rm pc^{-2}}$]")
        cb.outline.set_visible(False)
        ab = fig.colorbar(sc, cax=aax, ticks=[0, agemax / 2, agemax])
        ab.set_label("age [Myr]", fontsize=20)
        ab.outline.set_visible(False)
        aax.tick_params(labelsize=20)
        ytop = (y0 + w + 0.35) / H
        fig.text(x0 / W, ytop, f"{suite}, {moon_name[model]}", color=moon_color[model],
                 ha="left", va="center")  # fmt: skip
        fig.text((x0 + w) / W, ytop, f"{tMyr:.0f} Myr", ha="right", va="center")
        # scale bar below the map, in data units of the map's x axis
        xs = (x0 + w * scalebar_pc / (re[0] - le[0])) / W
        ybar = (y0 - 0.35) / H
        fig.add_artist(plt.Line2D([x0 / W, xs], [ybar, ybar], color="#eee8d5", lw=4,
                                  solid_capstyle="butt", transform=fig.transFigure))  # fmt: skip
        fig.text(xs + 0.15 / W, ybar, f"{scalebar_pc:.0f} pc", va="center", fontsize=20)
        if savdir is not None:
            os.makedirs(savdir, exist_ok=True)
            fig.savefig(osp.join(savdir, f"fon_{num:04d}.png"), dpi=150)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=list(suites), default="LGR4")
    parser.add_argument("--model", choices=["mhd", "crmhd", "all"], default="all")
    parser.add_argument("--num", type=int, default=None, help="single snapshot")
    parser.add_argument("--agemax", type=float, default=40.0, help="cluster age cut [Myr]")
    parser.add_argument("--basedir-root", default=None, help="parent of model dirs")
    parser.add_argument("--outdir", default=outdir_default)
    parser.add_argument("--movie", action="store_true", help="make mp4 at the end")
    args = parser.parse_args()

    COMM = MPI.COMM_WORLD
    suite = suites[args.suite]
    models = suite["models"]
    root = args.basedir_root
    if root is None:
        root = suite["root"] if osp.isdir(suite["root"]) else basedir_fallback
    mlist = list(models) if args.model == "all" else [args.model]
    for m in mlist:
        savdir = osp.join(args.outdir, f"{args.suite}_{m}")
        sim = LoadSimTIGRESSPP(osp.join(root, models[m]), verbose=False)
        nums = [args.num] if args.num is not None else sim.nums
        mynums = nums[COMM.rank :: COMM.size]
        for num in mynums:
            print(COMM.rank, m, num, flush=True)
            f = plot_faceon(sim, num, args.suite, m, agemax=args.agemax, savdir=savdir)
            plt.close(f)
            gc.collect()

        COMM.barrier()
        if args.movie and COMM.rank == 0:
            fin = osp.join(savdir, "fon_*.png")
            fout = osp.join(args.outdir, f"{args.suite}_{m}_fon.mp4")
            make_movie(fin, fout, fps_in=15, fps_out=15)
