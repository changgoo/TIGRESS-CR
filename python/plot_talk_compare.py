"""MHD vs CRMHD talk figures (LGR4 or R8) from zprof, history, SN and wind-PDF data.

Figures (written to <outdir>/<suite>_compare/, file names prefixed by <suite>):
  1. massflux_tz:  outflow mass flux in the t-z plane for wc/int/hot
  2. history:      SFR and SN energy injection rate, one moon-theme figure each
                   (<suite>_sfr_history_moon.png, <suite>_snE_history_moon.png)
  3. fluxpdf:      outflow mass/energy flux PDFs over v_out, split by phase
  4. momentum:     vertical momentum transfer (paper and talk versions)
  5. loading:      outflow mass and energy loading vs |z| for all gas
                   (moon theme, <suite>_loading_z_moon.png)
  6. pressure:     P_MHD (both runs) and P_CR vs |z|, all gas (box-area average)
                   and warm/cold (averaged over the warm/cold area)
                   (<suite>_pressure_z_moon.png, <suite>_pressure_z_wc_moon.png)

Phases: wc (T<2e4 K: CNM+UNM+WNM), int (2e4-5e5 K: WHIM), hot (T>5e5 K: HIM).
The flux PDFs split by sound speed instead (log c_s = 1.2, 1.9), since the wind
PDFs are binned in (v_out, c_s). The momentum figures keep the paper's two
phases (wc and hot = T>2e4 K).

All time averages use one window for both models and every figure: from --tstart
(default 200 Myr) to the last time covered by the history, zprof and wind-PDF
data of both models (wind PDFs are every 10 snapshots, so they set the end).
Wind PDFs are read from windpdf/outpdf.nc plus any per-snapshot
windpdf/windpdf.NNNNN.nc written after it (LGR4 MHD 481-591 were added this way).
Windows with the default --tstart: LGR4 200-577.9 Myr, R8 200-488.9 Myr.

Data: <basedir-root>/<model dir> with model dirs from plot_talk_edgeon.suites.
The default root is the suite's root, or /projects/EOSTRIKE/TIGRESSPP-CR/ if
that does not exist on this machine (both suites are there on stellar).

Usage
-----
    python plot_talk_compare.py             # all LGR4 figures
    python plot_talk_compare.py --suite R8  # all R8 figures
    python plot_talk_compare.py --fig 1 3   # selected figures
"""

import argparse
import glob
import os
import os.path as osp

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

import plotting_scripts as ps
from cr_zprof import update_flux
from load_sim_tigresspp import LoadSimTIGRESSPP
from plot_talk_edgeon import model_name, suites, talk_rc

suite = "LGR4"  # reset from --suite in main
models = suites[suite]["models"]
basedir_fallback = "/projects/EOSTRIKE/TIGRESSPP-CR/"
pressure_ylim = dict(LGR4=(2.0e2, 2.0e5), R8=(5.0e1, 5.0e4))  # 3 dex for both
outdir_default = "/scratch/gpfs/EOST/changgoo/talk_figures/"
model_color = dict(mhd="k", crmhd="#E77500")
phases = dict(wc=["CNM", "UNM", "WNM"], int="WHIM", hot="HIM")
all_phases = ["CNM", "UNM", "WNM", "WHIM", "HIM"]
phase_color = dict(wc="tab:blue", int="tab:green", hot="tab:red")
phase_label = dict(
    wc=r"wc ($T<2\times10^4\,{\rm K}$)",
    int=r"int ($2\times10^4$-$5\times10^5\,{\rm K}$)",
    hot=r"hot ($T>5\times10^5\,{\rm K}$)",
)
# log c_s [km/s] cuts for the wind PDFs, ~ T = 2e4 and 5e5 K for ionized gas
logcs_cuts = [1.2, 1.9]
zprof_fields = [
    "area", "rho", "press", "mom3", "Pturbz", "rhogz",
    "Pmag1", "Pmag2", "Pmag3", "Sz_Bpress", "Sz_Btens",
    "Ekin_flux1", "Ekin_flux2", "Ekin_flux3", "Eth_flux",
    "0-Ec", "0-Fc3",
]  # fmt: skip
flux_unit = r"$[{\rm M_\odot\,kpc^{-2}\,yr^{-1}}]$"


def windpdf_files(s):
    """Merged outpdf.nc and the per-snapshot windpdf files written after it."""
    wdir = osp.join(s.savdir, "windpdf")
    fmerged = osp.join(wdir, "outpdf.nc")
    with xr.open_dataarray(fmerged) as da:
        tmax = float(da.time.max())
    extra = []
    for f in sorted(glob.glob(osp.join(wdir, "windpdf.?????.nc"))):
        with xr.open_dataset(f) as ds:
            t = float(ds.time)
        if t > tmax + 1.0e-3:
            extra.append(f)
            tmax = t
    return fmerged, extra, tmax


def load_sim(m, basedir, zprof=True):
    """Load history, data end times and (if zprof) the needed zprof fields."""
    s = LoadSimTIGRESSPP(osp.join(basedir, models[m]), verbose=False)
    s.hst = s.make_monotonic(s.read_hst())
    zp = s.load_zprof()
    s.tend = dict(
        hst=float(s.hst["time"].max()),
        zprof=float(zp.time.max()) * s.u.Myr,
        windpdf=windpdf_files(s)[2] * s.u.Myr,
    )
    if not zprof:
        zp.close()
        return s
    fields = [f for f in zprof_fields if f in zp]
    s.zprof = zp[fields].drop_sel(phase="whole").load()
    zp.close()
    # two-phase split used by the paper's momentum-transfer plots
    s.zp_ph = xr.concat(
        [
            s.zprof.sel(phase=phases["wc"]).sum(dim="phase").assign_coords(phase="wc"),
            s.zprof.sel(phase=["WHIM", "HIM"])
            .sum(dim="phase")
            .assign_coords(phase="hot"),
        ],
        dim="phase",
    )
    return s


def set_common_window(sims, tstart):
    """Set s.tslice[_Myr] to [tstart, last time covered by all data of all models]."""
    tend = min(min(s.tend.values()) for s in sims.values())
    for m, s in sims.items():
        s.tslice_Myr = slice(tstart, tend)
        s.tslice = slice(tstart / s.u.Myr, tend / s.u.Myr + 1.0e-3)  # keep end snapshot
        ends = ", ".join(f"{k} {v:.1f}" for k, v in s.tend.items())
        print(f"{model_name[m]} data end [Myr]: {ends}")
    print(f"common averaging window: {tstart:g}-{tend:.1f} Myr")
    return tstart, tend


def sel_phase(da, ph):
    da = da.sel(phase=phases[ph])
    return da.sum(dim="phase") if "phase" in da.dims else da


def ref_fluxes(s):
    """SN-driven reference fluxes from the mean sfr40 over the averaging window."""
    h = s.hst
    tsel = (h["time"] >= s.tslice_Myr.start) & (h["time"] <= s.tslice_Myr.stop)
    sfr = h["sfr40"][tsel].mean()
    mstar = 1 / np.sum(s.pop_synth["snrate"] * 0.1)  # stellar mass per SN
    return dict(
        mflux=sfr,
        pflux_MHD=sfr / mstar * 1.25e5,
        pflux_CR=sfr / mstar * 1.25e5,
        eflux_MHD=sfr / mstar * 1.0e51,
        eflux_CR=sfr / mstar * 1.0e51,
    )


def _save(fig, savdir, name):
    os.makedirs(savdir, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(osp.join(savdir, f"{name}.{ext}"), dpi=200)
    print("saved", osp.join(savdir, name))


@plt.rc_context(talk_rc | {"axes.grid": False})
def plot_massflux_tz(sims, savdir=None):
    """Outflow (v_z*z>0) mass flux vs t and z; one row per phase, same t range."""
    tmin = max(float(s.zprof.time.min()) for s in sims.values())
    tmax = min(float(s.zprof.time.max()) for s in sims.values())
    norm = LogNorm(1.0e-5, 1.0e-1)
    cmap = cmr.ember.with_extremes(bad=cmr.ember(0), under=cmr.ember(0))
    fig, axes = plt.subplots(
        3, len(sims), figsize=(6 * len(sims), 11), sharex=True, sharey=True,
        constrained_layout=True, squeeze=False,
    )  # fmt: skip
    for j, (m, s) in enumerate(sims.items()):
        zp = s.zprof.sel(time=slice(tmin, tmax))
        area = np.prod(s.domain["Lx"][:2])
        conv = (s.u.density * s.u.velocity).to("Msun/(kpc2*yr)").value
        mom3 = zp["mom3"]
        mflux_out = (
            xr.where(zp.z > 0, mom3.sel(vz_dir=1), -mom3.sel(vz_dir=-1)) / area * conv
        )
        t = zp.time.values * s.u.Myr
        dt = t[1] - t[0]
        extent = [t[0] - dt / 2, t[-1] + dt / 2,
                  s.domain["le"][2] * 1.0e-3, s.domain["re"][2] * 1.0e-3]  # fmt: skip
        for i, ph in enumerate(phases):
            ax = axes[i, j]
            data = sel_phase(mflux_out, ph).transpose("z", "time")
            im = ax.imshow(
                data.values, extent=extent, origin="lower", aspect="auto",
                norm=norm, cmap=cmap, interpolation="nearest",
            )  # fmt: skip
            ax.annotate(
                ph, (0.03, 0.97), xycoords="axes fraction", ha="left", va="top",
                color="w", fontsize="large",
            )  # fmt: skip
        axes[0, j].set_title(model_name[m], color=model_color[m])
    plt.setp(axes[:, 0], ylabel=r"$z\,[{\rm kpc}]$")
    plt.setp(axes[-1, :], xlabel=r"$t\,[{\rm Myr}]$")
    fig.colorbar(
        im, ax=axes, shrink=0.6, pad=0.02,
        label=r"$\mathcal{F}_{\rho,{\rm out}}$" + " " + flux_unit,
    )  # fmt: skip
    if savdir is not None:
        _save(fig, savdir, f"{suite}_massflux_tz")
    return fig


def sn_energy_rate(s, dt=1.0, window=10):
    """SN energy injection rate [erg kpc^-2 yr^-1] from the SN log.

    SNe are counted in dt-Myr bins and smoothed with a window-bin running mean.
    Each SN carries E_inj * 1e51 erg, whether injected as energy or momentum.
    Rows from run segments abandoned at a restart are dropped (like make_monotonic):
    a row is kept only if no later row has an earlier time.
    """
    tcode = pd.read_csv(s.files["sn"], usecols=["time"])["time"].values
    tmin_later = np.append(np.minimum.accumulate(tcode[::-1])[::-1][1:], np.inf)
    tsn = tcode[tcode <= tmin_later] * s.u.Myr
    edges = np.arange(np.floor(tsn.min()), np.ceil(tsn.max()) + dt, dt)
    nsn, _ = np.histogram(tsn, bins=edges)
    area_kpc2 = np.prod(s.domain["Lx"][:2]) * 1.0e-6
    ESN = s.par["feedback"]["E_inj"] * 1.0e51
    rate = pd.Series(nsn * ESN / (area_kpc2 * dt * 1.0e6))
    rate = rate.rolling(window, center=True, min_periods=1).mean()
    return 0.5 * (edges[1:] + edges[:-1]), rate.values


moon_style = osp.join(osp.dirname(osp.abspath(__file__)), "..", "talk-moon.mplstyle")
moon_color = dict(mhd="#93a1a1", crmhd="#cb4b16")
moon_name = dict(mhd="without CRs", crmhd="with CRs")
moon_muted = "#93a1a1"


def _history_moon(curves, trange, title, ylabel, savname, savdir=None):
    """One moon-theme time-series panel; curves = {m: (t, y, mean)}."""
    with plt.style.context(["default", moon_style]):
        fig, ax = plt.subplots(figsize=(12, 5))
        for m, (t, y, avg) in curves.items():
            ax.plot(t, y, color=moon_color[m], lw=2.5, label=moon_name[m])
            ax.hlines(avg, *trange["avg"], color=moon_color[m], ls="--", lw=2)
        ax.axvspan(*trange["avg"], color="#073642", zorder=0, lw=0)
        ax.spines[["left", "bottom"]].set_position(("outward", 10))
        ax.set_xlim(*trange["plot"])
        ax.set_ylim(bottom=0)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("time $t$ [Myr]")
        t0, t1 = trange["avg"]
        leg = ax.legend(
            loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
            title=f"{suite}\ndashed: mean over\n{t0:.0f}-{t1:.0f} Myr",
        )  # fmt: skip
        leg.get_title().set_color(moon_muted)
        leg._legend_box.align = "left"
        if savdir is not None:
            os.makedirs(savdir, exist_ok=True)
            fig.savefig(osp.join(savdir, f"{savname}.png"))
            print("saved", osp.join(savdir, f"{savname}.png"))
    return fig


def plot_history(sims, savdir=None):
    """SFR (10 Myr) and SN energy injection rate as two moon-theme figures.

    Curves span the common history range of both models; dashed lines are means
    over the common averaging window s.tslice_Myr (shaded).
    """
    tavg = (next(iter(sims.values())).tslice_Myr.start,
            next(iter(sims.values())).tslice_Myr.stop)  # fmt: skip
    trange = dict(
        plot=(max(s.hst["time"].min() for s in sims.values()),
              min(s.hst["time"].max() for s in sims.values())),
        avg=tavg,
    )  # fmt: skip
    sfr, snE = {}, {}
    for m, s in sims.items():
        tsn, Edot = sn_energy_rate(s)
        for out, t, y, f in [
            (sfr, s.hst["time"].values, s.hst["sfr10"].values, "sfr10"),
            (snE, tsn, Edot * 1.0e-47, "Edot_SN"),
        ]:
            avg = np.mean(y[(t >= tavg[0]) & (t <= tavg[1])])
            out[m] = (t, y, avg)
            print(f"{model_name[m]}: <{f}>[{tavg[0]:g},{tavg[1]:.1f}] = {avg:.3e}")
    figs = [
        _history_moon(
            sfr, trange, "star formation rate",
            r"$\Sigma_{\rm SFR}$ " + flux_unit, f"{suite}_sfr_history_moon", savdir,
        ),
        _history_moon(
            snE, trange, "supernova energy injection rate",
            r"$\dot{E}_{\rm SN}/A$" + "\n" + r"$[10^{47}\,{\rm erg\,kpc^{-2}\,yr^{-1}}]$",
            f"{suite}_snE_history_moon", savdir,
        ),
    ]  # fmt: skip
    return figs


def load_outflux(s, zlist, fluxes=("mflux", "eflux")):
    """Time-averaged outflow flux PDFs at +-z, folded and normalized to a mean flux."""
    fmerged, extra, _ = windpdf_files(s)
    zsel = sorted([-z for z in zlist] + list(zlist))
    with xr.open_dataarray(fmerged) as da:
        pdf = [da.sel(flux=list(fluxes), z=zsel).sel(time=s.tslice).load().rename(None)]
    for f in extra:
        with xr.open_dataset(f) as ds:
            if s.tslice.start <= float(ds.time) <= s.tslice.stop:
                da = ds["out"].sel(flux=list(fluxes), z=zsel).expand_dims("time")
                # bin centers differ from the merged file at the 1e-16 level
                for c in ["logcs", "logvz"]:
                    assert np.allclose(da[c], pdf[0][c])
                da = da.assign_coords(logcs=pdf[0].logcs, logvz=pdf[0].logvz)
                pdf.append(da.load().rename(None))
    pdf = xr.concat(pdf, dim="time")
    print(f"  {model_name[s.key]} windpdf: {pdf.time.size} snapshots,"
          f" t = {float(pdf.time.min()):.0f}-{float(pdf.time.max()):.0f} (code)")  # fmt: skip
    pdf = pdf.mean(dim="time")
    # each slab is z0 +- 50 pc; count cells as in cr_zprof.load_windpdf
    zfc = np.linspace(s.domain["le"][2], s.domain["re"][2], s.domain["Nx"][2] + 1)
    zcc = 0.5 * (zfc[1:] + zfc[:-1])
    dnz = len(zcc[(zcc > 950) & (zcc < 1050)])
    pdf = pdf / np.prod(s.domain["Nx"][:-1]) / dnz
    folded = [
        pdf.sel(z=[-z, z]).mean(dim="z").assign_coords(z=z) for z in zlist
    ]
    return xr.concat(folded, dim="z")


def sel_cs_phase(pdf, ph):
    """Sum a wind PDF over the log c_s range of a phase (coords are bin right edges)."""
    logcs = pdf.logcs
    eps = 1.0e-6
    if ph == "wc":
        mask = logcs <= logcs_cuts[0] + eps
    elif ph == "int":
        mask = (logcs > logcs_cuts[0] + eps) & (logcs <= logcs_cuts[1] + eps)
    else:
        mask = logcs > logcs_cuts[1] + eps
    return pdf.where(mask, 0.0).sum(dim="logcs")


@plt.rc_context(talk_rc)
def plot_flux_pdf(sims, zlist=(1000, 2000, 4000), savdir=None):
    """dF/dlog v_out for mass and (MHD) energy flux, split by phase, at several |z|."""
    outflux = {m: load_outflux(s, zlist) for m, s in sims.items()}
    model_ls = dict(mhd="--", crmhd="-")
    fig, axes = plt.subplots(
        2, len(zlist), figsize=(4.5 * len(zlist), 8), sharex=True, sharey="row",
        constrained_layout=True,
    )  # fmt: skip
    for m, pdf in outflux.items():
        dbin = float(pdf.logvz[1] - pdf.logvz[0])
        for j, z in enumerate(zlist):
            for i, flux in enumerate(["mflux", "eflux"]):
                ax = axes[i, j]
                for ph in phases:
                    y = sel_cs_phase(pdf.sel(z=z, flux=flux), ph) / dbin
                    ax.plot(pdf.logvz, y, color=phase_color[ph], ls=model_ls[m])
            axes[0, j].set_title(rf"$|z|={z / 1.0e3:g}\,{{\rm kpc}}$")
    for ax in axes[0]:
        ax.set_yscale("log")
        ax.set_ylim(1.0e-6, 1.0e-1)
    for ax in axes[1]:
        ax.set_yscale("log")
        ax.set_ylim(1.0e41, 1.0e47)
    axes[0, 0].set_xlim(0, 3.5)
    plt.setp(axes[-1, :], xlabel=r"$\log\,v_{\rm out}\,[{\rm km\,s^{-1}}]$")
    axes[0, 0].set_ylabel(
        r"$d\mathcal{F}_{\rho}/d\log v_{\rm out}$" + "\n"
        + r"$[{\rm M_\odot\,kpc^{-2}\,yr^{-1}\,dex^{-1}}]$"
    )  # fmt: skip
    axes[1, 0].set_ylabel(
        r"$d\mathcal{F}_{E,{\rm MHD}}/d\log v_{\rm out}$" + "\n"
        + r"$[{\rm erg\,kpc^{-2}\,yr^{-1}\,dex^{-1}}]$"
    )  # fmt: skip
    ph_handles = [Line2D([], [], color=phase_color[ph], label=ph) for ph in phases]
    m_handles = [
        Line2D([], [], color="k", ls=model_ls[m], label=model_name[m]) for m in sims
    ]
    axes[0, 0].legend(handles=ph_handles, loc="upper left", fontsize="small")
    axes[1, 0].legend(handles=m_handles, loc="upper left", fontsize="small")
    if savdir is not None:
        _save(fig, savdir, f"{suite}_fluxpdf")
    return fig


def plot_momentum(sims, savdir=None):
    """Paper (both hemispheres, z_ref=1 kpc) and talk (upper, from 1 kpc) versions."""
    simgroup = dict(talk=sims)
    figs = []
    with plt.rc_context({"font.size": 13}):
        f = ps.plot_momentum_transfer_z(
            simgroup, "talk", show_option=1, zmin=0, zref=1000, savefig=False
        )
        figs.append(f)
        if savdir is not None:
            _save(f, savdir, f"{suite}_momentum_transfer")
        f = ps.plot_momentum_transfer_z_talk(
            simgroup, "talk", show_option=1, zmin=1000, zref=1000, savefig=False
        )
        figs.append(f)
        if savdir is not None:
            _save(f, savdir, f"{suite}_momentum_transfer_talk")
    return figs


def momentum_transfer_terms(s, zmin, zmax, zref):
    """Upper-half momentum transfer terms between zmin and z, per snapshot.

    Same terms as ps.plot_momentum_transfer_z_talk (show_option=1), for z in
    [zmin, zmax] with z > 0 only. With P = P_turb,z + P_th + Pi_B (per area) and
    W(z) = int_zmin^z (rho g_sg + rho g_ext) dz' (the weight change, < 0 here):
      warm = Delta P^wc - W^wc        (net momentum gained by warm/cold gas)
      hot  = -(Delta P^hot - W^hot)   (momentum handed over by hot gas)
      cr   = -Delta P_c               (all phases)
    Delta X = X(z) - X(zmin). Phases: wc = T < 2e4 K, hot = T > 2e4 K. All are
    divided by the time-mean warm/cold P at zref.
    """
    dset = s.zp_ph.sum(dim="vz_dir").sel(time=s.tslice)
    area = np.prod(s.domain["Lx"][:2])
    dz = s.domain["dx"][2]
    gzext = np.interp(dset.z, s.extgrav["z"], s.extgrav["gz"])
    P = dset["Pturbz"] + dset["press"] + dset["Pmag1"] + dset["Pmag2"] - dset["Pmag3"]
    P = P * s.u.pok / area
    dW = (dset["rhogz"] + dset["rho"] * gzext) * s.u.pok / area
    zsel = dict(z=slice(zmin, zmax))
    Pref = P.sel(phase="wc").sel(z=zref, method="nearest").mean(dim="time")
    dP = P.sel(**zsel) - P.sel(**zsel).isel(z=0)
    W = dW.sel(**zsel).cumsum(dim="z") * dz
    W = W - W.isel(z=0)
    terms = dict(
        warm=(dP - W).sel(phase="wc") / Pref,
        hot=-(dP - W).sel(phase="hot") / Pref,
    )
    if s.options["cosmic_ray"]:
        Pcr = (dset["0-Ec"] / 3.0 * s.u.pok / area).sum(dim="phase").sel(**zsel)
        terms["cr"] = -(Pcr - Pcr.isel(z=0)) / Pref
    return terms


def plot_momentum_moon(sims, zmin, zmax, zref, reflabel, savname, savdir=None):
    """Moon-theme momentum transfer, one panel per run (upper half, zmin < z < zmax)."""
    style = dict(
        warm=dict(color="#268bd2", lw=4, label="net gain by warm/cold gas"),
        hot=dict(color="#dc322f", lw=3, label="from hot gas"),
        cr=dict(color="#859900", lw=3, ls="--", label="from CRs"),
    )
    with plt.style.context(["default", moon_style]):
        fig, axes = plt.subplots(1, len(sims), figsize=(12, 5), sharey=True)
        for ax, (m, s) in zip(axes, sims.items()):
            terms = momentum_transfer_terms(s, zmin, zmax, zref)
            for k, da in terms.items():
                z = da.z * 1.0e-3
                ax.plot(z, da.mean(dim="time"), **style[k])
                if k == "warm":
                    q = da.quantile([0.16, 0.84], dim="time")
                    ax.fill_between(z, q.isel(quantile=0), q.isel(quantile=1),
                                    color=style[k]["color"], alpha=0.25, lw=0)  # fmt: skip
                vals = np.interp([zmax * 1.0e-3], z, da.mean(dim="time"))
                print(f"{model_name[m]:5s} {k:4s} at z={zmax * 1.0e-3:g} kpc: {vals[0]:.3f}")
            ax.axhline(0, color=moon_muted, lw=1, zorder=0)
            ax.set_title(moon_name[m], loc="left", color=moon_color[m])
            ax.set_xlim(zmin * 1.0e-3, zmax * 1.0e-3)
            ax.set_xlabel("height $z$ [kpc]")
        axes[0].set_ylabel("momentum transfer\n" + reflabel)
        t0, t1 = s.tslice_Myr.start, s.tslice_Myr.stop
        handles = [Line2D([], [], **style[k]) for k in style]
        leg = fig.legend(
            handles=handles, loc="upper left", bbox_to_anchor=(1.0, 0.9),
            borderaxespad=0,
            title=f"{suite}, upper half\nmean {t0:.0f}-{t1:.0f} Myr\n"
                  r"warm/cold: $T<2\times10^4$ K" "\n" r"hot: $T>2\times10^4$ K",
        )  # fmt: skip
        leg.get_title().set_color(moon_muted)
        leg._legend_box.align = "left"
        fig.tight_layout()
        if savdir is not None:
            os.makedirs(savdir, exist_ok=True)
            fig.savefig(osp.join(savdir, f"{savname}.png"))
            print("saved", osp.join(savdir, f"{savname}.png"))
    return fig


def plot_momentum_talk(sims, savdir=None):
    """Moon-theme momentum transfer above and below 1 kpc."""
    figs = [
        plot_momentum_moon(
            sims, 1000, 4000, 1000,
            r"$\langle\Delta P\rangle/\langle P_{\rm MHD}^{\rm wc}(1\,{\rm kpc})\rangle$",
            f"{suite}_momentum_transfer_upper_moon", savdir,
        ),
        plot_momentum_moon(
            sims, 0, 1000, 0,
            r"$\langle\Delta P\rangle/\langle P_{\rm MHD}^{\rm wc}(0)\rangle$",
            f"{suite}_momentum_transfer_lower_moon", savdir,
        ),
    ]  # fmt: skip
    return figs


def _fold(da):
    """Average the z>0 and z<0 halves onto |z|."""
    upper = da.sel(z=slice(0, None))
    lower = da.sel(z=slice(None, 0)).isel(z=slice(None, None, -1))
    lower = lower.assign_coords(z=-lower.z)
    return 0.5 * (upper + lower.interp(z=upper.z))


def pressure_profiles(s, phase="all"):
    """P_MHD and P_CR [K cm^-3] vs |z| for a phase, folded over both halves.

    P_MHD = P_turb,z + P_th + (B_x^2 + B_y^2 - B_z^2)/8pi (vertical MHD stress),
    P_CR = E_c/3. Each is summed over the phase and divided by the area that
    phase occupies at each z (the box area for phase="all"). Returns, per
    quantity, the per-snapshot profiles and the time mean sum(P)/sum(area).
    """
    ph = all_phases if phase == "all" else phases[phase]
    dset = s.zprof.sel(time=s.tslice, phase=ph).sum(dim=["phase", "vz_dir"])
    if phase == "all":
        area = xr.full_like(dset["area"], np.prod(s.domain["Lx"][:2]))
    else:
        area = dset["area"]
    P = dict(
        mhd=dset["Pturbz"] + dset["press"] + dset["Pmag1"] + dset["Pmag2"]
        - dset["Pmag3"]
    )  # fmt: skip
    if s.options["cosmic_ray"]:
        P["cr"] = dset["0-Ec"] / 3.0
    farea = _fold(area)
    out = {}
    for k, da in P.items():
        fda = _fold(da * s.u.pok)
        out[k] = (
            fda / farea.where(farea > 0),
            fda.sum(dim="time") / farea.sum(dim="time"),
        )
    return out


def plot_pressure(sims, phase="all", zmax=5, savdir=None):
    """Moon-theme P_MHD (both runs) and P_CR (with CRs) vs |z| for one phase."""
    styles = {
        ("mhd", "mhd"): dict(color=moon_color["mhd"], label=r"$P_{\rm MHD}$, without CRs"),
        ("crmhd", "mhd"): dict(color=moon_color["crmhd"], label=r"$P_{\rm MHD}$, with CRs"),
        ("crmhd", "cr"): dict(color="#859900", ls="--", label=r"$P_{\rm CR}$, with CRs"),
    }  # fmt: skip
    tag = dict(all="all gas", wc="warm/cold gas\n" r"($T<2\times10^4$ K)")[phase]
    zvals = [0, 0.5, 1, 2, 4]
    with plt.style.context(["default", moon_style]):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for m, s in sims.items():
            for k, (da, mean) in pressure_profiles(s, phase).items():
                st = styles[(m, k)]
                z = da.z * 1.0e-3
                q = da.quantile([0.16, 0.84], dim="time")
                ax.fill_between(z, q.isel(quantile=0), q.isel(quantile=1),
                                color=st["color"], alpha=0.2, lw=0)  # fmt: skip
                ax.plot(z, mean, lw=3, **st)
                vals = np.interp(zvals, z, mean)
                print(f"{model_name[m]:5s} {phase} P_{k} at {zvals} kpc:",
                      np.array2string(vals, precision=3))  # fmt: skip
        ax.set_yscale("log")
        ax.set_xlim(0, zmax)
        ax.set_xlabel("height $|z|$ [kpc]")
        ax.set_ylabel(r"pressure $P/k_B$ [${\rm K\,cm^{-3}}$]")
        s = next(iter(sims.values()))
        t0, t1 = s.tslice_Myr.start, s.tslice_Myr.stop
        ax.set_ylim(*pressure_ylim[suite])
        leg = ax.legend(
            loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
            title=f"{suite}, {tag}\nmean {t0:.0f}-{t1:.0f} Myr",
        )  # fmt: skip
        leg.get_title().set_color(moon_muted)
        leg._legend_box.align = "left"
        if savdir is not None:
            name = f"{suite}_pressure_z_moon" if phase == "all" else f"{suite}_pressure_z_{phase}_moon"
            os.makedirs(savdir, exist_ok=True)
            fig.savefig(osp.join(savdir, f"{name}.png"))
            print("saved", osp.join(savdir, f"{name}.png"))
    return fig


def plot_loading(sims, zmax=5, savdir=None):
    """Outflow mass and energy loading vs |z| for all gas (both hemispheres), moon theme.

    eta = time-mean outflow (v_z*z>0) flux over the common window, normalized by
    the SN-driven reference flux from the mean sfr40 (see ref_fluxes). Solid:
    MHD (gas) flux; dashed: CR flux. Bands are the 16-84th percentiles in time.
    """
    panels = [
        ("mflux", None, "mass loading", r"$\eta_M$"),
        ("eflux_MHD", "eflux_CR", "energy loading", r"$\eta_E$"),
    ]
    zvals = [1, 2, 3, 4]
    with plt.style.context(["default", moon_style]):
        fig, axes = plt.subplots(1, len(panels), figsize=(10, 5), sharex=True)
        for m, s in sims.items():
            ref = ref_fluxes(s)
            dset = update_flux(s, s.zprof.sel(time=s.tslice), vz_dir=1, both=True)
            c = moon_color[m]
            for ax, (f, fcr, _, _) in zip(axes, panels):
                plt.sca(ax)
                for ff, ls in [(f, "-"), (fcr, "--")]:
                    if ff is None or ff not in dset:
                        continue
                    ps.plot_zprof(
                        dset, ff, all_phases, norm=ref[ff], line="mean", color=c,
                        ls=ls, lw=3, quantile=(ls == "-"),
                    )  # fmt: skip
                    # loading factors at a few heights, for quoting in the talk
                    eta = dset[ff].sel(phase=all_phases).sum(dim="phase")
                    eta = eta.mean(dim="time") / ref[ff]
                    vals = np.interp(zvals, dset.z * 1.0e-3, eta)
                    print(f"{model_name[m]:5s} eta[{ff}] at {zvals} kpc:",
                          np.array2string(vals, precision=3))  # fmt: skip
        for ax, (_, _, title, label) in zip(axes, panels):
            ax.set_title(title, loc="left")
            ax.set_ylabel(label)
            ax.set_yscale("log")
            ax.set_xlabel("height $|z|$ [kpc]")
        axes[0].set_xlim(0, zmax)
        axes[0].set_xticks(np.arange(0, zmax + 0.1, 2))
        t0, t1 = next(iter(sims.values())).tslice_Myr.start, next(
            iter(sims.values())
        ).tslice_Myr.stop
        run_handles = [
            Line2D([], [], color=moon_color[m], lw=3, label=moon_name[m]) for m in sims
        ]
        flux_handles = [
            Line2D([], [], color="#eee8d5", lw=3, label="gas flux"),
            Line2D([], [], color="#eee8d5", lw=3, ls="--", label="CR flux"),
        ]
        leg = fig.legend(
            handles=run_handles + flux_handles, loc="upper left",
            bbox_to_anchor=(1.0, 0.92), borderaxespad=0,
            title=f"{suite}, all gas, outflow\nmean {t0:.0f}-{t1:.0f} Myr",
        )  # fmt: skip
        leg.get_title().set_color(moon_muted)
        leg._legend_box.align = "left"
        fig.tight_layout()
        if savdir is not None:
            os.makedirs(savdir, exist_ok=True)
            fig.savefig(osp.join(savdir, f"{suite}_loading_z_moon.png"))
            print("saved", osp.join(savdir, f"{suite}_loading_z_moon.png"))
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=list(suites), default="LGR4")
    parser.add_argument("--fig", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--tstart", type=float, default=200.0,
                        help="start of the common averaging window [Myr]")  # fmt: skip
    parser.add_argument("--zmax", type=float, default=5.0, help="|z| limit [kpc]")
    parser.add_argument("--basedir-root", default=None, help="parent of model dirs")
    parser.add_argument("--outdir", default=outdir_default)
    args = parser.parse_args()

    suite = args.suite
    models = suites[suite]["models"]
    root = args.basedir_root
    if root is None:
        root = suites[suite]["root"]
        root = root if osp.isdir(root) else basedir_fallback
    savdir = osp.join(args.outdir, f"{suite}_compare")
    need_zprof = bool(set(args.fig) - {2})
    sims = {m: load_sim(m, root, need_zprof) for m in models}
    for m, s in sims.items():
        s.key = m
    set_common_window(sims, args.tstart)
    ps.setup(savdir, model_name, model_color, zlim_=args.zmax)

    if 1 in args.fig:
        plot_massflux_tz(sims, savdir=savdir)
    if 2 in args.fig:
        plot_history(sims, savdir=savdir)
    if 3 in args.fig:
        plot_flux_pdf(sims, zlist=(500, 1000, 3000), savdir=savdir)
    if 4 in args.fig:
        plot_momentum(sims, savdir=savdir)
        plot_momentum_talk(sims, savdir=savdir)
    if 5 in args.fig:
        plot_loading(sims, zmax=args.zmax, savdir=savdir)
    if 6 in args.fig:
        for phase in ["all", "wc"]:
            plot_pressure(sims, phase=phase, zmax=args.zmax, savdir=savdir)
    plt.close("all")
