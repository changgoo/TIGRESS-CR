import os
import os.path as osp
import glob
import sys

filepath = os.path.dirname(__file__)

import xarray as xr
import astropy.units as au
import astropy.constants as ac
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize, LogNorm, SymLogNorm
import cmasher as cmr

from cr_zprof import (
    update_flux,
    update_stress,
    fit_exponential_profile,
)

plt.style.use(osp.join(filepath, "paper.mplstyle"))

fig_outdir = None
model_name = None
model_color = None


def setup(outdir, names, colors):
    """Setup output directory and plotting styles.

    Creates output directory if it does not exist and
    sets up color and label dictionaries for models.

    Parameters
    ----------
    outdir : str
        Output directory path
    names : dict
        Dictionary mapping model keys to display names
    colors : dict
        Dictionary mapping model keys to colors
    """
    if not osp.exists(outdir):
        os.makedirs(outdir)
    global fig_outdir, model_name, model_color
    fig_outdir = outdir
    model_name = names
    model_color = colors


def plot_injection(s, tmin=150, tmax=500, kpc=True, **kwargs):
    """Plot CR injection rate as a function of z from supernova positions.

    Extracts SN positions from the SN catalog, histograms them in z,
    and computes the injection rate based on simulation parameters
    (energy per SN, particle mass, and time interval).

    Parameters
    ----------
    s : LoadSim
        Simulation object with domain and parameters
    tmin : float
        Minimum simulation time (Myr) to include SNe (default=150)
    tmax : float
        Maximum simulation time (Myr) to include SNe (default=500)
    kpc : bool
        If True, convert z coordinates to kpc (default=True)
    **kwargs
        Additional arguments passed to plt.plot()
    """
    if "sn" in s.files:
        if osp.exists(s.files["sn"]):
            sn = pd.read_csv(s.files["sn"])
    snsel = sn[sn["time"] > tmin]
    nsn, zfc = np.histogram(
        snsel["x3_inj"],
        range=[s.domain["le"][2], s.domain["re"][2]],
        bins=s.domain["Nx"][2],
    )
    zcc = 0.5 * (zfc[1:] + zfc[:-1])
    if kpc:
        zcc = zcc / 1.0e3
    dt = (tmax - tmin) * s.u.time
    area = s.domain["Lx"][0] * s.domain["Lx"][1] * s.u.length**2
    dz = s.domain["dx"][2] * s.u.length
    ESN = s.par["feedback"]["E_inj"] * 1.0e51 * au.erg
    fcr = s.par["feedback"]["fe_CR"]
    crinj = nsn * ESN * fcr / area / dt / dz

    plt.plot(zcc, crinj.to("erg/(s*cm3)"), **kwargs)


# ----------------------------------------
# definition of base plotting functions
# ----------------------------------------
def plot_zprof_mean_quantile(ydata, kpc=True, quantile=True, **kwargs):
    """Plot z-profile as mean line with quantile shading.

    Plots the time-averaged profile as a line and optionally shades
    the 16th-84th percentile range (1-sigma equivalent).

    Parameters
    ----------
    ydata : xr.DataArray
        Profile data with time dimension (z-profile over time)
    kpc : bool
        If True, convert z coordinates to kpc (default=True)
    quantile : bool
        If True, overlay shaded quantile band (default=True)
    **kwargs
        Additional arguments passed to plt.plot() and plt.fill_between()
    """
    q = ydata.quantile([0.16, 0.5, 0.84], dim="time")
    qmean = ydata.mean(dim="time")
    z = q.z / 1.0e3 if kpc else q.z

    plt.plot(z, qmean, **kwargs)
    if quantile:
        plt.fill_between(
            z,
            q.sel(quantile=0.16),
            q.sel(quantile=0.84),
            color=kwargs["color"],
            alpha=0.2,
            linewidth=0,
        )


def plot_zprof_quantile(ydata, kpc=True, quantile=True, **kwargs):
    """Plot z-profile median line with quantile shading.

    Plots the time-median profile as a line and optionally shades
    the 16th-84th percentile range.

    Parameters
    ----------
    ydata : xr.DataArray
        Profile data with time dimension
    kpc : bool
        If True, convert z coordinates to kpc (default=True)
    quantile : bool
        If True, overlay shaded quantile band (default=True)
    **kwargs
        Additional arguments passed to plt.plot() and plt.fill_between()
    """
    q = ydata.quantile([0.16, 0.5, 0.84], dim="time")
    z = q.z / 1.0e3 if kpc else q.z

    plt.plot(z, q.sel(quantile=0.5), **kwargs)
    if quantile:
        plt.fill_between(
            z,
            q.sel(quantile=0.16),
            q.sel(quantile=0.84),
            color=kwargs["color"],
            alpha=0.2,
            linewidth=0,
        )


def plot_zprof(
    zprof, field, ph, norm=1.0, line="median", quantile=True, kpc=True, **kwargs
):
    """Plot a z-profile field for a given phase with optional normalization.

    Extracts and normalizes data for a specific field and phase,
    then plots using median or mean line style with optional quantiles.

    Parameters
    ----------
    zprof : xr.Dataset
        Z-profile dataset with field variables
    field : str
        Field name to plot
    ph : str
        Phase name (e.g., 'wc', 'hot', 'CNM')
    norm : float
        Normalization factor (default=1.0)
    line : str
        Line style: 'median' or 'mean' (default='median')
    quantile : bool
        If True, overlay shaded quantile band (default=True)
    **kwargs
        Additional arguments for plotting
    """
    if field not in zprof:
        return
    ydata = zprof[field].sel(phase=ph) / norm
    if ydata.phase.size > 1:
        ydata = ydata.sum(dim=["phase"])
    if line == "median":
        plot_zprof_quantile(ydata, kpc=kpc, quantile=quantile, **kwargs)
    elif line == "mean":
        plot_zprof_mean_quantile(ydata, kpc=kpc, quantile=quantile, **kwargs)


def plot_zprof_field(
    zprof, field, ph, line="median", quantile=True, kpc=True, **kwargs
):
    """Plot area-weighted z-profile field.

    Computes area-weighted average of a field for a given phase
    and plots with quantile shading.

    Parameters
    ----------
    zprof : xr.Dataset
        Z-profile dataset
    field : str
        Field name to plot
    ph : str
        Phase name
    line : str
        Line style: 'median' or 'mean'
    quantile : bool
        If True, overlay quantile band
    **kwargs
        Additional arguments for plotting
    """
    if field not in zprof:
        return
    ydata = zprof[field].sel(phase=ph)
    area = zprof["area"].sel(phase=ph)
    if ydata.phase.size > 1:
        ydata = ydata.sum(dim=["phase"]) / area.sum(dim=["phase"])
    else:
        ydata = ydata / area
    if line == "median":
        plot_zprof_quantile(ydata, kpc=kpc, quantile=quantile, **kwargs)
    elif line == "mean":
        plot_zprof_mean_quantile(ydata, kpc=kpc, quantile=quantile, **kwargs)


def plot_zprof_frac(
    zprof,
    field,
    ph,
    denominator="area",
    line="median",
    quantile=True,
    kpc=True,
    **kwargs,
):
    """Plot fractional z-profile (e.g., volume fraction by phase).

    Computes the fraction of a field relative to a denominator
    (typically area for volume fraction) and plots with quantiles.

    Parameters
    ----------
    zprof : xr.Dataset
        Z-profile dataset
    field : str
        Numerator field name
    ph : str or list
        Phase name(s) to include in numerator
    denominator : str
        Denominator field (default='area')
    line : str
        Line style: 'median' or 'mean'
    quantile : bool
        If True, overlay quantile band
    **kwargs
        Additional arguments for plotting
    """
    if field not in zprof:
        return
    if "whole" in zprof.phase:
        area = zprof[denominator].sel(phase="whole")
    else:
        area = zprof[denominator].sum(dim="phase")
    ydata = zprof[field].sel(phase=ph)
    if ydata.phase.size > 1:
        ydata = ydata.sum(dim=["phase"])
    if line == "median":
        plot_zprof_quantile(ydata / area, kpc=kpc, quantile=quantile, **kwargs)
    elif line == "mean":
        plot_zprof_mean_quantile(ydata / area, kpc=kpc, quantile=quantile, **kwargs)


# ----------------------------------------
# plotting functions space-time diagrams
# ----------------------------------------
def plot_massflux_tz(simgroup, gr, kpc=True, ph="wc", savefig=True):
    """Create space-time diagram of mass flux for a simulation group.

    Plots outflow (top row) and inflow (bottom row) mass flux as
    2D color maps (z vs time) for each simulation in the group.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    kpc : bool
        If True, convert z coordinates to kpc (default=True)
    ph : str
        Phase selection (default='wc' for warm+cold+neutral)
    savefig : bool
        If True, save figure (default=True)
    """
    with plt.style.context({"axes.grid": False}):
        sims = simgroup[gr]
        models = list(sims.keys())
        nmodels = len(models)
        fig, axes = plt.subplots(
            2,
            nmodels,
            figsize=(3 * nmodels, 5),
            sharey=True,
            sharex=True,
            constrained_layout=True,
        )
        for m, axs in zip(models, axes.T):
            s = sims[m]

            area = np.prod(s.domain["Lx"][:-1])
            units = s.u.Msun / s.u.pc**2 / s.u.Myr
            mflux_out = s.zp_ph["mom3"].sel(vz_dir=1).sel(phase=ph).T * units / area
            mflux_in = -s.zp_ph["mom3"].sel(vz_dir=-1).sel(phase=ph).T * units / area
            zsel_up = s.zp_ph.z.sel(z=slice(0, s.zp_ph.z.max()))
            zsel_lo = s.zp_ph.z.sel(z=slice(s.zp_ph.z.min(), 0))
            z_up = zsel_up / 1.0e3 if kpc else zsel_up
            z_lo = zsel_lo / 1.0e3 if kpc else zsel_lo

            plt.sca(axs[0])
            im_out = plt.pcolormesh(
                s.zp_ph.time * s.u.Myr,
                z_up,
                mflux_out.sel(z=slice(0, s.zp_ph.z.max())),
                cmap=cmr.ember,
                norm=LogNorm(1.0e-5, 1.0e-1),
            )
            im_out = plt.pcolormesh(
                s.zp_ph.time * s.u.Myr,
                z_lo,
                mflux_in.sel(z=slice(s.zp_ph.z.min(), 0)),
                cmap=cmr.ember,
                norm=LogNorm(1.0e-5, 1.0e-1),
            )
            plt.title(model_name[m], color=model_color[m])
            plt.sca(axs[1])
            im_in = plt.pcolormesh(
                s.zp_ph.time * s.u.Myr,
                z_up,
                mflux_in.sel(z=slice(0, s.zp_ph.z.max())),
                cmap=cmr.cosmic,
                norm=LogNorm(1.0e-5, 1.0e-1),
            )
            im_in = plt.pcolormesh(
                s.zp_ph.time * s.u.Myr,
                z_lo,
                mflux_out.sel(z=slice(s.zp_ph.z.min(), 0)),
                cmap=cmr.cosmic,
                norm=LogNorm(1.0e-5, 1.0e-1),
            )
        plt.setp(axes[:, 0], ylabel=r"$z\, [{\rm kpc}]$")
        plt.setp(axes[1, :], xlabel=r"$t\, [{\rm Myr}]$")
        # plt.ylim(bottom=0)
        cbar_out = plt.colorbar(
            im_out,
            shrink=0.8,
            ax=axes[0, :],
            pad=0.02,
            label=f"$\\mathcal{{F}}_{{M}}^{{\\rm {ph},out}}$"
            r"$\,[{\rm M_\odot\,kpc^{-2}\,yr}]$",
        )
        cbar_in = plt.colorbar(
            im_in,
            shrink=0.8,
            ax=axes[1, :],
            pad=0.02,
            label=f"$\\mathcal{{F}}_{{M}}^{{\\rm {ph},in}}$"
            r"$\,[{\rm M_\odot\,kpc^{-2}\,yr}]$",
        )
        if savefig:
            plt.savefig(osp.join(fig_outdir, f"{gr}_massflux_tz.png"))
    return fig


def plot_flux_tz(simgroup, gr, kpc=True, savefig=True):
    """Create space-time diagrams of various fluxes for a simulation group.

    Plots mass flux, MHD pressure flux, and MHD energy flux as 2D color maps
    (z vs time) for outflows, with separate panels for different flux types.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    kpc : bool
        If True, convert z coordinates to kpc (default=True)
    savefig : bool
        If True, save figure (default=True)
    """
    with plt.style.context({"axes.grid": False}):
        sims = simgroup[gr]
        models = list(sims.keys())
        nmodels = len(models)
        fig, axes = plt.subplots(
            4,
            nmodels,
            figsize=(4 * nmodels, 8),
            sharey=True,
            sharex=True,
            constrained_layout=True,
        )
        norm = dict(
            mflux=LogNorm(1.0e-5, 1.0e-1),
            pflux_MHD=LogNorm(1.0e-3, 10),
            eflux_MHD=LogNorm(1.0e42, 1.0e47),
            pflux_CR=LogNorm(1.0e-3, 10),
            eflux_CR=LogNorm(1.0e42, 1.0e47),
        )
        label_unit = dict(
            mflux=r"$\,[{\rm M_\odot\,kpc^{-2}\,yr}]$",
            pflux_MHD=r"$\,[{\rm M_\odot\,km/s\,kpc^{-2}\,yr}]$",
            eflux_MHD=r"$\,[{\rm erg\,kpc^{-2}\,yr}]$",
        )
        cmap_outin = [cmr.ember, cmr.cosmic]
        for m, axs in zip(models, axes.T):
            s = sims[m]
            dset_outin = []
            for vz_dir in [1, -1]:
                dset_outin.append(update_flux(s, s.zp_ph, vz_dir=vz_dir, both=False))
            flux_field = "mflux"
            ph = "wc"
            im_outin = []
            for ax, flux, cmap in zip(axs, dset_outin, cmap_outin):
                z = flux.z / 1.0e3 if kpc else flux.z
                plt.sca(ax)
                if flux_field not in flux:
                    continue
                im = plt.pcolormesh(
                    flux.time * s.u.Myr,
                    z,
                    flux[flux_field].sel(phase=ph).T,
                    cmap=cmap,
                    norm=norm[flux_field],
                )
                im_outin.append(im)
            flux_field = "eflux_MHD"
            flux = dset_outin[0]
            z = flux.z / 1.0e3 if kpc else flux.z
            cmap = cmr.sunburst
            for ax, ph in zip(axs[2:], ["wc", "hot"]):
                plt.sca(ax)
                if flux_field not in flux:
                    continue
                im = plt.pcolormesh(
                    flux.time * s.u.Myr,
                    z,
                    flux[flux_field].sel(phase=ph).T,
                    cmap=cmap,
                    norm=norm[flux_field],
                )
                im_outin.append(im)
            plt.sca(axs[0])
            plt.title(model_name[m], color=model_color[m])
        plt.setp(axes[:, 0], ylabel=r"$z\, [{\rm kpc}]$")
        plt.setp(axes[-1, :], xlabel=r"$t\, [{\rm Myr}]$")
        for axs, im, lab, flux_field, ph in zip(
            axes,
            im_outin,
            ["out", "in", "out", "out"],
            ["mflux", "mflux", "eflux_MHD", "eflux_MHD"],
            ["wc", "wc", "wc", "hot"],
        ):
            flux_name = flux_field.replace("_", ",").replace("flux", "").upper()
            plt.colorbar(
                im,
                shrink=0.8,
                ax=axs,
                pad=0.02,
                label=f"$\\mathcal{{F}}_{{{flux_name[0]}{{\\rm {flux_name[1:]}}}}}^{{\\;\\rm {ph},{lab}}}$"
                + label_unit[flux_field],
            )
        if savefig:
            plt.savefig(osp.join(fig_outdir, f"{gr}_flux_tz.png"))
    return fig


def plot_pressure_tz(simgroup, gr, ph="wc", kpc=True, savefig=True):
    """Create space-time diagrams of pressure components for a simulation group.

    Plots CR, thermal, kinetic, and magnetic pressure as 2D color maps
    (z vs time) for each simulation in the group.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    ph : str
        Phase selection (default='wc')
    kpc : bool
        If True, convert z coordinates to kpc (default=True)
    savefig : bool
        If True, save figure (default=True)
    """
    with plt.style.context({"axes.grid": False}):
        sims = simgroup[gr]
        models = list(sims.keys())
        nmodels = len(models)
        fig, axes = plt.subplots(
            4,
            nmodels,
            figsize=(3 * nmodels, 10),
            sharey=True,
            sharex=True,
            constrained_layout=True,
        )
        imlist = []
        for m, axs in zip(models, axes.T):
            s = sims[m]
            area = np.prod(s.domain["Lx"][:-1])
            dset = s.zp_ph.sum(dim="vz_dir").sel(phase=ph)
            area = dset["area"]
            if "Pi_B" not in dset:
                dset["Pi_B"] = (dset["Pmag1"] + dset["Pmag2"] - dset["Pmag3"]) * s.u.pok
            if "Pok_B" not in dset:
                dset["Pok_B"] = (
                    dset["Pmag1"] + dset["Pmag2"] + dset["Pmag3"]
                ) * s.u.pok
            if "Pok_cr" not in dset and s.options["cosmic_ray"]:
                dset["Pok_cr"] = dset["0-Ec"] / 3.0 * s.u.pok
            if "Pok_th" not in dset:
                dset["Pok_th"] = dset["press"] * s.u.pok
            if "Pok_kin" not in dset:
                dset["Pok_kin"] = dset["Pturbz"] * s.u.pok
            z = dset.z / 1.0e3 if kpc else dset.z
            for ax, pfield in zip(axs, ["Pok_cr", "Pok_th", "Pok_kin", "Pok_B"]):
                plt.sca(ax)
                if pfield in dset:
                    im = plt.pcolormesh(
                        dset.time * s.u.Myr,
                        z,
                        dset[pfield].T / area,
                        cmap=plt.cm.plasma,
                        norm=LogNorm(1.0e1, 5.0e4),
                    )
                    imlist.append(im)
                    if pfield == "Pok_cr":
                        plt.title(model_name[m])
                else:
                    plt.axis("off")
                    imlist.append(None)

        plt.setp(axes[:, 0], ylabel=r"$z\, [{\rm kpc}]$")
        plt.setp(axes[-1, :], xlabel=r"$t\, [{\rm Myr}]$")
        for im, axs, pfield in zip(
            imlist, axes, ["Pok_cr", "Pok_th", "Pok_kin", "Pok_B"]
        ):
            if im is None:
                continue
            lab = pfield.split("_")[-1]
            cbar = plt.colorbar(
                im,
                shrink=0.8,
                ax=axs,
                pad=0.02,
                label=f"$P_{{\\rm {lab} }}/k_B$" + r"$\,[{\rm cm^{-3}\,K}]$",
            )
        if savefig:
            plt.savefig(osp.join(fig_outdir, f"{gr}_pressures_tz.png"))
    return fig


# ----------------------------------------
# plotting functions time-averaged z-profiles
# ----------------------------------------
def plot_pressure_z(
    simgroup, gr, tslice=slice(150, 500), ph="wc", kpc=True, savefig=True
):
    """Plot pressure components vs. height with exponential fits.

    Creates 4-panel plot showing CR, thermal, kinetic, and magnetic pressure
    profiles for all simulations in a group. Fits exponential profiles to
    pressure data and overlays fitted curves.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    ph : str
        Phase selection (default='wc')
    """
    sims = simgroup[gr]
    models = list(sims.keys())
    fig, axes = plt.subplots(1, 4, figsize=(8, 3), sharey=True, constrained_layout=True)
    for i, m in enumerate(models):
        s = sims[m]
        c = model_color[m]
        dset = s.zp_ph.sel(time=tslice).sum(dim="vz_dir")
        dset = update_stress(s, dset)
        rho = (dset["rho"].sel(phase=ph) / dset["area"].sel(phase=ph)).mean(dim="time")
        print(
            m,
            np.sqrt((rho * rho.z**2).sum(dim="z") / rho.sum(dim="z")).data,
            ((rho * s.domain["dx"][-1]).sum(dim="z") / (2 * rho.max(dim="z"))).data,
        )
        z = rho.z.values / 1.0e3 if kpc else rho.z.values
        fit_params = fit_exponential_profile(
            z, rho.values, return_all=True, zmin=0.0, zmax=1.0
        )
        if fit_params:
            P0, H = fit_params["P0"], fit_params["H"]
            print(f"{m} rho: P0={P0:.2e}, H={H:.3f} kpc")
        for ax, pfield in zip(axes, ["Pok_cr", "Pok_th", "Pok_kin", "Pi_B"]):
            plt.sca(ax)
            if pfield in dset:
                plot_zprof_field(
                    dset, pfield, ph, kpc=kpc, color=c, label=model_name[m]
                )
                # fitting with an exponential profile

                Pz = dset[pfield].sel(phase=ph).mean(dim="time") / dset["area"].sel(
                    phase=ph
                ).mean(dim="time")

                P = Pz.values

                # fitting range
                zmin = 1.0 if pfield == "Pok_cr" else 0.0
                zmax = 2.0 if pfield == "Pok_cr" else 1.0

                # Fit exponential profile
                fit_params = fit_exponential_profile(
                    z, P, return_all=True, zmin=zmin, zmax=zmax
                )
                if fit_params:
                    P0, H = fit_params["P0"], fit_params["H"]
                    z_fit = np.linspace(z.min(), z.max(), 100)
                    P_fit = P0 * np.exp(-np.abs(z_fit) / H)
                    plt.plot(z_fit, P_fit, "--", color=c, alpha=0.7, linewidth=0.5)
                    print(f"{m} {pfield}: P0={P0:.2e}, H={H:.3f} kpc")

            lab = pfield.split("_")[-1]
            if pfield.startswith("Pok"):
                plt.title(f"$P_{{\\rm {lab}}}$")
            else:
                plt.title(f"$\\Pi_{{\\rm {lab}}}$")
    plt.sca(axes[0])
    plt.ylabel(r"$\overline{P}^{\rm \; wc}(z)/k_B\,[{\rm cm^{-3}\,K}]$")
    plt.setp(axes, "yscale", "log")
    plt.setp(axes, "xlabel", r"$z\,[{\rm kpc}]$")
    plt.setp(axes, "ylim", (10, 1.0e5))
    plt.setp(axes, "xlim", (-4, 4))
    for ax in axes:
        plt.sca(ax)
        plt.axvline(1, ls="--", color="k", lw=1)
        plt.axvline(-1, ls="--", color="k", lw=1)
    plt.sca(axes[1])
    plt.legend(fontsize="small")
    if savefig:
        plt.savefig(osp.join(fig_outdir, f"{gr}_pressures_z.pdf"))
    return fig


def plot_volume_fraction_z(
    simgroup, gr, tslice=slice(150, 500), kpc=True, savefig=True
):
    """Plot phase volume fractions as a function of height.

    Shows the contribution of each phase (CNM+UNM, WNM, WHIM, HIM)
    to the total volume as a function of altitude.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    """
    sims = simgroup[gr]
    models = list(sims.keys())
    nmodels = len(models)
    fig, axes = plt.subplots(
        1,
        nmodels,
        figsize=(4 * nmodels, 3),
        sharey=True,
        sharex=True,
        constrained_layout=True,
    )
    for i, m in enumerate(models):
        s = sims[m]

        dset = s.zprof.sum(dim="vz_dir").sel(time=tslice)
        plt.sca(axes[i])
        for ph, color, label in zip(
            [["CNM", "UNM"], "WNM", "WHIM", "HIM"],
            ["C0", "limegreen", "gold", "C3"],
            ["CNM+UNM", "WNM", "WHIM", "HIM"],
        ):
            plot_zprof_frac(dset, "area", ph, kpc=kpc, color=color, label=label)
        plt.title(model_name[m])
    plt.sca(axes[0])
    plt.legend()
    plt.ylim(0, 1)
    plt.xlabel(r"$z\,[{\rm kpc}]$")
    plt.ylabel(r"Volume Fraction")
    if savefig:
        plt.savefig(osp.join(fig_outdir, f"{gr}_volume_fraction.pdf"))

    return fig


def plot_profile_frac_z(
    simgroup,
    gr,
    vz_dir=None,
    field="rho",
    line="median",
    tslice=slice(150, 500),
    kpc=True,
    savefig=True,
):
    """Plot fractional profiles (e.g., density) by phase as a function of height.

    Optionally filters by vertical velocity direction to separate
    inflows and outflows. Shows fractional contribution of each phase.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    vz_dir : int or None
        Vertical velocity direction filter (1 for outflow, -1 for inflow, None for both)
    field : str
        Field to plot (default='rho' for density)
    line : str
        Line style (default='median')
    savefig : bool
        If True, save figure (default=True)
    """
    sims = simgroup[gr]
    models = list(sims.keys())
    nmodels = len(models)
    fig, axes = plt.subplots(
        1,
        nmodels,
        figsize=(4 * nmodels, 3),
        sharey=True,
        sharex=True,
        constrained_layout=True,
    )
    for i, m in enumerate(models):
        s = sims[m]
        if vz_dir is None:
            dset = s.zprof.sum(dim="vz_dir").sel(time=tslice)
        else:
            dset_upper = (
                s.zprof.sel(vz_dir=vz_dir)
                .sel(time=tslice)
                .sel(z=slice(0, s.domain["re"][2]))
            )
            dset_lower = (
                s.zprof.sel(vz_dir=-vz_dir)
                .sel(time=tslice)
                .sel(z=slice(s.domain["le"][2], 0))
            )
            dset = xr.concat([dset_lower, dset_upper], dim="z")
        plt.sca(axes[i])
        for ph, color, label in zip(
            [["CNM", "UNM"], "WNM", "WHIM", "HIM"],
            ["C0", "limegreen", "gold", "C3"],
            ["CNM+UNM", "WNM", "WHIM", "HIM"],
        ):
            plot_zprof_frac(
                dset, field, ph, kpc=kpc, line=line, color=color, label=label
            )

        plt.title(model_name[m])
    plt.sca(axes[0])
    plt.legend()
    # plt.ylim(0, 1)
    plt.xlim(-4, 4)
    plt.yscale("log")
    plt.ylabel(r"$\langle q \rangle$")
    plt.setp(axes, "xlabel", r"$z\,[{\rm kpc}]$")
    if savefig:
        plt.savefig(osp.join(fig_outdir, f"{gr}_{field}_profile_frac_z.pdf"))
    return fig


def plot_profile_z(
    simgroup, gr, field="rho", tslice=slice(150, 500), kpc=True, savefig=True
):
    """Plot vertical profiles of a field by phase for a simulation group.

    Creates side-by-side plots showing profiles of a specified field
    (density, pressure, etc.) as a function of height for each simulation,
    with lines for each phase (CNM+UNM, WNM, WHIM, HIM).

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    field : str
        Field to plot (default='rho' for density)
    savefig : bool
        If True, save figure (default=True)
    """
    sims = simgroup[gr]
    models = list(sims.keys())
    nmodels = len(models)
    fig, axes = plt.subplots(
        1,
        nmodels,
        figsize=(4 * nmodels, 3),
        sharey=True,
        sharex=True,
        constrained_layout=True,
    )
    for i, m in enumerate(models):
        s = sims[m]

        dset = s.zprof.sum(dim="vz_dir").sel(time=tslice)
        plt.sca(axes[i])
        for ph, color, label in zip(
            [["CNM", "UNM"], "WNM", "WHIM", "HIM"],
            ["C0", "limegreen", "gold", "C3"],
            ["CNM+UNM", "WNM", "WHIM", "HIM"],
        ):
            plot_zprof_field(dset, field, ph, kpc=kpc, color=color, label=label)

        plt.title(model_name[m])
    plt.sca(axes[0])
    plt.legend()
    # plt.ylim(0, 1)
    plt.xlim(-4, 4)
    plt.yscale("log")
    plt.ylabel(r"$\overline{q}$")
    plt.setp(axes, "xlabel", r"$z\,[{\rm kpc}]$")
    if savefig:
        plt.savefig(osp.join(fig_outdir, f"{gr}_{field}_profile_z.pdf"))
    return fig


def plot_fraction_z(
    simgroup, gr, field="rho", tslice=slice(150, 500), kpc=True, savefig=True
):
    """Plot fractional profiles of a field by phase for a simulation group.

    Creates side-by-side plots showing fractional contribution of each phase
    to a field as a function of height, with values normalized to 1.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    field : str
        Field to plot (default='rho' for density)
    """
    sims = simgroup[gr]
    models = list(sims.keys())
    nmodels = len(models)
    fig, axes = plt.subplots(
        1,
        nmodels,
        figsize=(4 * nmodels, 3),
        sharey=True,
        sharex=True,
        constrained_layout=True,
    )
    for i, m in enumerate(models):
        s = sims[m]

        dset = s.zprof.sum(dim="vz_dir").sel(time=tslice)
        plt.sca(axes[i])
        for ph, color, label in zip(
            [["CNM", "UNM"], "WNM", "WHIM", "HIM"],
            ["C0", "limegreen", "gold", "C3"],
            ["CNM+UNM", "WNM", "WHIM", "HIM"],
        ):
            plot_zprof_frac(
                dset, field, ph, kpc=kpc, denominator=field, color=color, label=label
            )

        plt.title(model_name[m])
    plt.sca(axes[0])
    plt.legend()
    plt.ylim(0, 1)
    plt.xlabel(r"$z\,[{\rm kpc}]$")
    plt.ylabel("Fraction")
    if savefig:
        plt.savefig(osp.join(fig_outdir, f"{gr}_{field}_fraction.pdf"))
    return fig


def plot_fraction_ph_z(
    simgroup, gr, field="rho", tslice=slice(150, 500), kpc=True, savefig=True
):
    """Plot fractional profiles separated by aggregated phase categories.

    Creates a 2-panel plot comparing warm cloud (wc) and hot phases,
    showing fractional profiles of a field for all simulations.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    field : str
        Field to plot (default='rho' for density)
    """
    sims = simgroup[gr]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8, 3),
        sharey=True,
        sharex=True,
        constrained_layout=True,
    )
    for m, s in sims.items():
        s = sims[m]
        name = model_name[m]
        color = model_color[m]
        dset = s.zp_ph.sum(dim="vz_dir").sel(time=tslice)
        for ax, ph in zip(axes, ["wc", "hot"]):
            plt.sca(ax)
            plot_zprof_frac(
                dset, field, ph, kpc=kpc, denominator=field, color=color, label=name
            )

            plt.title(ph)
            plt.xlabel(r"$z\,[{\rm kpc}]$")
            plt.xlim(-4, 4)
    plt.sca(axes[0])
    plt.legend()
    plt.ylim(0, 1)
    plt.ylabel("Fraction")
    if savefig:
        plt.savefig(osp.join(fig_outdir, f"{gr}_{field}_fraction.pdf"))
    return fig


def plot_rho_z(simgroup, gr, tslice=slice(150, 500), kpc=True, savefig=True):
    """Plot density and pressure/energy profiles for warm and hot phases.

    Creates a 4x2 grid of plots showing density, kinetic, thermal, and
    magnetic pressure for warm cloud and hot phases, organized by simulation.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    tslice : slice
        Time slice to select (default=slice(150, 500))
    kpc : bool
        If True, convert z coordinates to kpc (default=True)
    savefig : bool
        If True, save figure (default=True)
    """
    sims = simgroup[gr]
    models = list(sims.keys())
    nmodels = len(models)

    fig, axes = plt.subplots(
        4, 2, figsize=(8, 10), sharex=True, sharey="row", constrained_layout=True
    )
    for i, m in enumerate(models):
        s = sims[m]
        color = model_color[m]
        dset = s.zp_ph.sel(time=tslice).sum(dim="vz_dir")
        dset = update_stress(s, dset)
        # dset["Etot"] *= (s.u.energy_density/ac.k_B).cgs.value
        for axs, ph in zip(axes.T, ["wc", "hot"]):
            for ax, field in zip(axs, ["rho", "Pok_kin", "Pok_th", "Pok_B"]):
                plt.sca(ax)
                plot_zprof_frac(
                    dset, field, ph, kpc=kpc, color=color, label=model_name[m]
                )
                plot_zprof_field(dset, field, ph, kpc=kpc, color=color, lw=1)
                plt.yscale("log")
            # plt.ylim(1.0e-5, 10)
            # plt.title(f"ph={ph}")
    plt.sca(axes[0, 0])
    plt.ylabel(r"$\langle n_H \rangle\,[{\rm cm^{-3}}]$")
    plt.xlim(-4, 4)
    plt.ylim(bottom=1.0e-5)
    plt.sca(axes[1, 0])
    plt.ylabel(r"$\langle \mathcal{E}_{\rm tot} \rangle\,[{\rm erg\,cm^{-3}}]$")
    # plt.ylim(bottom=1.e-2)
    plt.setp(axes[1, :], "xlabel", r"$z\,[{\rm kpc}]$")
    if savefig:
        plt.savefig(osp.join(fig_outdir, f"{gr}_rho_z.pdf"))
    return fig


def plot_rhov_z(simgroup, gr, tslice=slice(150, 500), kpc=True, savefig=True):
    """Plot density and velocity profiles for warm cloud phase.

    Creates a 4-panel plot showing density, vertical velocity, Alfvén
    velocity, and CR drift velocity profiles for warm cloud phase.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    tslice : slice
        Time slice to select (default=slice(150, 500))
    kpc : bool
        If True, convert z coordinates to kpc (default=True)
    savefig : bool
        If True, save figure (default=True)
    """
    sims = simgroup[gr]
    models = list(sims.keys())
    nmodels = len(models)

    fig, axes = plt.subplots(4, 1, figsize=(5, 8), sharex=True, constrained_layout=True)
    for i, m in enumerate(models):
        s = sims[m]
        c = model_color[m]
        dset = s.zp_ph.sel(time=tslice).sum(dim="vz_dir")

        plt.sca(axes[0])
        ph = "wc"
        plot_zprof_field(
            dset, "rho", ["wc", "hot"], kpc=kpc, color=c, label=model_name[m]
        )
        plt.yscale("log")
        plt.ylim(1.0e-3, 10)
        plt.ylabel(r"$\langle \rho \rangle$")

        plt.legend()

        plt.sca(axes[1])
        plot_zprof_field(dset, "vel3", ph, kpc=kpc, color=c)
        plt.ylabel(r"$\langle v_z \rangle_{\rm wc}$")

        plt.sca(axes[2])
        if s.options["cosmic_ray"]:
            plot_zprof_field(dset, "0-Vs3", ph, kpc=kpc, color=c)
        plt.ylabel(r"$\langle v_{A,i} \rangle_{\rm wc}$")

        plt.sca(axes[3])
        if s.options["cosmic_ray"]:
            plot_zprof_field(dset, "0-Vd3", ph, kpc=kpc, color=c)
        plt.xlabel("z")
        plt.ylabel(r"$\langle v_d \rangle_{\rm wc}$")

    plt.setp(axes[1:], "ylim", (-50, 50))
    if savefig:
        plt.savefig(osp.join(fig_outdir, f"{gr}_rho_vz.pdf"))
    return fig


def plot_cr_velocity_sigma_z(
    simgroup, gr, tslice=slice(150, 500), kpc=True, savefig=True
):
    """Plot CR transport properties and energy densities for a simulation group.

    Creates a 5x2 grid showing effective sound speed, diffusion coefficient,
    diffusivity, heating rate, and CR work for both warm cloud and hot phases.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    tslice : slice
        Time slice to select (default=slice(150, 500))
    kpc : bool
        If True, convert z coordinates to kpc (default=True)
    savefig : bool
        If True, save figure (default=True)
    """
    sims = simgroup[gr]
    fig, axes = plt.subplots(
        5, 2, figsize=(6, 10), sharex=True, constrained_layout=True
    )
    print(len(axes.T))
    for m, s in sims.items():
        color = model_color[m]
        if s.options["cosmic_ray"]:
            dset = s.zp_ph.sel(time=tslice).sum(dim="vz_dir")
            dset["cs"] = np.sqrt(dset["press"] / dset["dens"])
            if "0-rhoCcr2" in dset:
                dset["Cc"] = np.sqrt(dset["0-rhoCcr2"] / dset["dens"])
                dset["Ceff"] = np.sqrt(
                    (dset["0-rhoCcr2"] + dset["press"]) / dset["dens"]
                )
            dset["vz"] = np.sqrt(dset["Pturbz"] / dset["dens"])
            vmax_kms = s.par["cr"]["vmax"] / 1.0e5
            dset["sigma"] = dset["0-Sigma_diff1"] / vmax_kms / s.u.cm**2 * s.u.s
            if "0-kappac" in dset:
                dset["kappa"] = dset["0-kappac"] * (s.u.cm**2 / s.u.s)
            if "0-heating_cr" in dset:
                dset["cr_heating"] = (
                    -dset["0-heating_cr"] * (s.u.energy_density / s.u.time).cgs.value
                )
            if "0-work_cr" in dset:
                dset["cr_work"] = (
                    dset["0-work_cr"] * (s.u.energy_density / s.u.time).cgs.value
                )
            ls_sigma = "-"
            for axs, ph in zip(axes.T, ["wc", "hot"]):
                len(axs)
                plt.sca(axs[0])
                for pfield, ls in zip(["Ceff", "vz"], ["-", ":"]):
                    plot_zprof(
                        dset, pfield, ph, kpc=kpc, color=color, label=pfield, ls=ls
                    )
                plt.ylim(bottom=0)
                plt.title(f"ph={ph}")
                plt.sca(axs[1])
                plot_zprof_field(
                    dset, "sigma", ph, kpc=kpc, color=color, label=model_name[m]
                )
                plt.sca(axs[2])
                plot_zprof_field(
                    dset, "kappa", ph, kpc=kpc, color=color, label=model_name[m]
                )
                plt.sca(axs[3])
                plot_zprof_field(
                    dset, "cr_heating", ph, kpc=kpc, color=color, label=ph, ls=ls_sigma
                )
                plt.sca(axs[4])
                if "cr_work" in dset:
                    plot_zprof_field(
                        dset, "cr_work", ph, kpc=kpc, color=color, label=ph
                    )
    plt.sca(axes[0, 0])
    lines, labels = axes[0, 0].get_legend_handles_labels()
    custom_lines = [lines[0], lines[1]]
    plt.legend(
        custom_lines,
        [r"$\overline{C}_{\rm eff}$", r"$\overline{v}_z$"],
        fontsize="x-small",
    )
    plt.ylabel(r"velocity $[{\rm km/s}]$")
    plt.ylim(0, 100)
    plt.sca(axes[0, 1])
    plt.ylabel(r"velocity $[{\rm km/s}]$")
    plt.ylim(0, 350)

    plt.sca(axes[1, 0])
    # plt.legend(fontsize="x-small")
    plt.ylabel(r"$\tilde{\sigma}_{\parallel}[{\rm cm^{-2}\,s}]$")
    plt.ylim(0, 5.0e-28)
    plt.sca(axes[1, 1])
    plt.ylabel(r"$\tilde{\sigma}_{\parallel}[{\rm cm^{-2}\,s}]$")
    plt.ylim(0, 5.0e-28)

    plt.sca(axes[2, 0])
    plt.ylabel(r"$\tilde{\kappa}_{\parallel}[{\rm cm^{2}\,s^{-1}}]$")
    plt.yscale("log")
    plt.sca(axes[2, 1])
    plt.ylabel(r"$\tilde{\kappa}_{\parallel}[{\rm cm^{2}\,s^{-1}}]$")
    plt.yscale("log")
    plt.ylim(1.0e27, 1.0e29)

    plt.sca(axes[3, 0])
    plt.ylabel(r"$\tilde{\Gamma}_{\rm cr}[{\rm erg\,s^{-1}\,cm^{-3}}]$")
    plt.xlabel(r"$z\,[{\rm kpc}]$")
    # plt.ylim(0, 7.0e-28)
    plt.xlim(-4, 4)
    plt.sca(axes[3, 1])
    plt.ylabel(r"$\tilde{\Gamma}_{\rm cr}[{\rm erg\,s^{-1}\,cm^{-3}}]$")
    plt.xlabel(r"$z\,[{\rm kpc}]$")
    # plt.ylim(0, 7.0e-28)
    plt.xlim(-4, 4)
    plt.sca(axes[4, 0])
    plt.ylabel(r"$\tilde{PdV}_{\rm cr}[{\rm erg\,s^{-1}\,cm^{-3}}]$")
    plt.xlabel(r"$z\,[{\rm kpc}]$")
    # plt.ylim(0, 7.0e-28)
    plt.xlim(-4, 4)
    plt.sca(axes[4, 1])
    plt.ylabel(r"$\tilde{PdV}_{\rm cr}[{\rm erg\,s^{-1}\,cm^{-3}}]$")
    plt.xlabel(r"$z\,[{\rm kpc}]$")
    # plt.ylim(0, 7.0e-28)
    plt.xlim(-4, 4)

    if savefig:
        plt.savefig(osp.join(fig_outdir, f"{gr}_cr_velocity_sigma.pdf"))
    return fig


# def plot_cr_field(s,field="sigma",ph="wc"):
#     area = np.prod(s.domain["Lx"][:-1])
#     if s.options["cosmic_ray"]:
#         dset = s.zp_ph.sel(time=slice(150,500)).sum(dim="vz_dir")
#         vmax_kms = s.par["cr"]["vmax"]/1.e5
#         dset["sigma"] = dset["0-Sigma_diff1"]/vmax_kms/s.u.cm**2*s.u.s
#         dset["cr_heating"] = -dset["0-heating_cr"]*(s.u.energy_density/s.u.time).cgs.value
#         # dset["kappa"] = 1/dset["0-kappac"]/(s.u.cm**2*s.u.s)
#         for j,ph in enumerate(["wc","hot"]):
#             plot_zprof_field(dset,field,ph,color=f"C{j}",label=ph)
#         plt.legend(fontsize="x-small")


def plot_flux_z(
    simgroup, gr, both=True, vz_dir=None, tslice=slice(150, 500), kpc=True, savefig=True
):
    """Plot mass, pressure, and energy fluxes for a simulation group.

    Creates a 3x2 grid showing mass flux, MHD pressure flux, and energy flux
    for warm cloud and hot phases. Optionally separates outflow and inflow.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    both : bool
        If True, fold both hemispheres (default=True)
    vz_dir : int or None
        Filter by vertical velocity direction (default=None for both)
    tslice : slice
        Time slice to select (default=slice(150, 500))
    kpc : bool
        If True, convert z coordinates to kpc (default=True)
    savefig : bool
        If True, save figure (default=True)
    """
    sims = simgroup[gr]
    fig, axes = plt.subplots(
        3, 2, figsize=(8, 7), sharey="row", sharex="col", constrained_layout=True
    )
    for m, s in sims.items():
        color = model_color[m]
        dset_ = s.zp_ph.sel(time=tslice)
        dset = update_flux(s, dset_, vz_dir=vz_dir, both=both)
        if vz_dir is not None:
            dset_outin = [dset]
            dset_outin.append(update_flux(s, dset_, vz_dir=-vz_dir, both=both))

        for axs, ph in zip(axes.T, ["wc", "hot"]):
            plt.sca(axs[0])
            plt.title(f"ph={ph}")
            if vz_dir == 1:
                dset = dset_outin[0]
            elif vz_dir == -1:
                dset = dset_outin[1]
            plot_zprof(
                dset, "mflux", ph, kpc=kpc, line="mean", color=color, label="out"
            )
            z = dset.z / 1.0e3 if kpc else dset.z
            if vz_dir == 1:
                mf_in = dset_outin[1]["mflux"].sel(phase=ph).mean(dim="time")
                plt.plot(z, mf_in, color=color, ls=":", label="in")
            elif vz_dir == -1:
                mf_in = dset_outin[0]["mflux"].sel(phase=ph).mean(dim="time")
                plt.plot(z, mf_in, color=color, ls=":", label="in")
            plt.ylim(1.0e-4, 1e-1)
            plt.yscale("log")

            plt.sca(axs[1])
            plot_zprof(
                dset,
                "pflux_MHD",
                ph,
                kpc=kpc,
                line="mean",
                color=color,
                label=model_name[m],
            )
            if s.options["cosmic_ray"]:
                plot_zprof(
                    dset,
                    "pflux_CR",
                    ph,
                    kpc=kpc,
                    line="mean",
                    color=color,
                    lw=1,
                    ls="--",
                )
            plt.yscale("log")
            plt.ylim(1.0e-2, 10)

            plt.sca(axs[2])
            plot_zprof(
                dset, "eflux_MHD", ph, kpc=kpc, line="mean", color=color, label="MHD"
            )
            if s.options["cosmic_ray"]:
                plot_zprof(
                    dset,
                    "eflux_CR",
                    ph,
                    kpc=kpc,
                    line="mean",
                    color=color,
                    label="CR",
                    lw=1,
                    ls="--",
                )
            plt.yscale("log")
            plt.ylim(1.0e43, 1.0e47)
            plt.xlim(0, 4)
    axs = axes[:, 0]
    plt.sca(axs[0])
    if vz_dir is not None:
        lines, labels = axs[0].get_legend_handles_labels()
        custom_lines = [lines[0], lines[1]]
        plt.legend(custom_lines, ["outflow", "inflow"], fontsize="x-small")
    # plt.ylabel(r"$\langle n_H\rangle\,[{\rm cm^{-3}}]$")
    # plt.sca(axs[1])
    vout_label = "out" if vz_dir == 1 else "net"
    plt.ylabel(
        f"$\\mathcal{{F}}_M^{{\\rm {vout_label}}}$"
        r"$\,[M_\odot{\rm \,kpc^{-2}\,yr^{-1}}]$"
    )
    plt.sca(axs[1])
    plt.legend(fontsize="x-small")
    plt.ylabel(
        f"$\\mathcal{{F}}_p^{{\\rm {vout_label}}}$"
        r"$\,[M_\odot{\rm \,(km/s)\,kpc^{-2}\,yr^{-1}}]$"
    )
    plt.sca(axs[2])
    plt.ylabel(
        f"$\\mathcal{{F}}_E^{{\\rm {vout_label}}}$"
        r"$\,[{\rm erg\,kpc^{-2}\,yr^{-1}}]$"
    )
    lines, labels = axs[2].get_legend_handles_labels()
    if len(lines) > 2:
        custom_lines = [lines[1], lines[2]]
        plt.legend(custom_lines, ["MHD flux", "CR flux"], fontsize="x-small")
    zunit_label = r"$\,[{\rm kpc}]$" if kpc else r"$\,[{\rm pc}]$"
    if both:
        plt.setp(axes[-1, :], "xlabel", r"$|z|$" + zunit_label)
    else:
        plt.setp(axes[-1, :], "xlabel", r"$z$" + zunit_label)

    if savefig:
        plt.savefig(osp.join(fig_outdir, f"{gr}_flux_{vout_label}_z.pdf"))
    return fig


def plot_loading_z(
    simgroup, gr, vz_dir=None, both=True, tslice=slice(150, 500), kpc=True, savefig=True
):
    """Plot flux loading (normalized by reference star formation) for a simulation group.

    Creates a 3x2 grid showing normalized mass, pressure, and energy fluxes
    relative to star formation-driven input rates for warm cloud and hot phases.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    vz_dir : int or None
        Filter by vertical velocity direction (default=None for both)
    both : bool
        If True, fold both hemispheres (default=True)
    tslice : slice
        Time slice to select (default=slice(150, 500))
    kpc : bool
        If True, convert z coordinates to kpc (default=True)
    savefig : bool
        If True, save figure (default=True)
    """
    sims = simgroup[gr]
    fig, axes = plt.subplots(
        3, 2, figsize=(8, 7), sharey="row", sharex="col", constrained_layout=True
    )
    for m, s in sims.items():
        Zsn = s.par["feedback"]["Z_SN"]
        Mej = s.par["feedback"]["M_ej"]
        dt = 0.1
        mstar = 1 / np.sum(s.pop_synth["snrate"] * dt)
        field = "sfr40"
        h = s.read_hst()
        sfr_avg = h[field].loc[tslice].mean()
        sfr_std = h[field].loc[tslice].std()
        ref_flux = dict(
            mflux=sfr_avg / mstar * mstar,
            pflux_MHD=sfr_avg / mstar * 1.25e5,
            pflux_CR=sfr_avg / mstar * 1.25e5,
            eflux_MHD=sfr_avg / mstar * 1.0e51,
            eflux_CR=sfr_avg / mstar * 1.0e51,
            mZflux=sfr_avg / mstar * Mej * Zsn,
        )

        color = model_color[m]
        dset_ = s.zp_ph.sel(time=tslice)
        dset = update_flux(s, dset_, vz_dir=vz_dir, both=both)
        if vz_dir is not None:
            dset_outin = [dset]
            dset_outin.append(update_flux(s, dset_, vz_dir=-vz_dir, both=both))

        for axs, ph in zip(axes.T, ["wc", "hot"]):
            # ph = ["wc", "hot"]
            plt.sca(axs[0])
            plt.title(f"ph={ph}")
            if vz_dir == 1:
                dset = dset_outin[0]
            elif vz_dir == -1:
                dset = dset_outin[1]
            plot_zprof(
                dset,
                "mflux",
                ph,
                kpc=kpc,
                norm=ref_flux["mflux"],
                line="mean",
                color=color,
                label=model_name[m],
            )
            plt.ylim(1.0e-2, 10)
            plt.yscale("log")

            plt.sca(axs[1])
            plot_zprof(
                dset,
                "pflux_MHD",
                ph,
                kpc=kpc,
                norm=ref_flux["pflux_MHD"],
                line="mean",
                color=color,
                label=model_name[m],
            )
            if s.options["cosmic_ray"]:
                plot_zprof(
                    dset,
                    "pflux_CR",
                    ph,
                    kpc=kpc,
                    norm=ref_flux["pflux_CR"],
                    line="mean",
                    color=color,
                    label="CR",
                    lw=1,
                    ls="--",
                )
            plt.yscale("log")
            plt.ylim(1.0e-2, 1)

            plt.sca(axs[2])
            plot_zprof(
                dset,
                "eflux_MHD",
                ph,
                kpc=kpc,
                norm=ref_flux["eflux_MHD"],
                line="mean",
                color=color,
                label="MHD",
            )
            if s.options["cosmic_ray"]:
                plot_zprof(
                    dset,
                    "eflux_CR",
                    ph,
                    kpc=kpc,
                    norm=ref_flux["eflux_CR"],
                    line="mean",
                    color=color,
                    label="CR",
                    lw=1,
                    ls="--",
                )
            plt.yscale("log")
            plt.ylim(1.0e-3, 5.0)
            plt.xlim(0, 4)
    axs = axes[:, 0]
    plt.sca(axs[0])
    plt.legend(fontsize="x-small")

    # lines, labels = axs[0].get_legend_handles_labels()
    # custom_lines = [lines[0], lines[1]]
    # plt.legend(custom_lines, ["outflow", "inflow"], fontsize="x-small")
    # plt.ylabel(r"$\langle n_H\rangle\,[{\rm cm^{-3}}]$")
    # plt.sca(axs[1])
    vout_label = "out" if vz_dir == 1 else "net"
    plt.ylabel(
        f"$\\eta_M^{{\\rm {vout_label}}}$"
        # r"$\,[M_\odot{\rm \,kpc^{-2}\,yr^{-1}}]$"
    )
    plt.sca(axs[1])
    plt.ylabel(
        f"$\\eta_p^{{\\rm {vout_label}}}$"
        # r"$\,[M_\odot{\rm \,km/s\,kpc^{-2}\,yr^{-1}}]$"
    )
    plt.sca(axs[2])
    plt.ylabel(
        f"$\\eta_E^{{\\rm {vout_label}}}$"
        # r"$\,[{\rm erg\,kpc^{-2}\,yr^{-1}}]$"
    )
    lines, labels = axs[2].get_legend_handles_labels()
    custom_lines = [lines[1], lines[2]]
    plt.legend(custom_lines, ["MHD flux", "CR flux"], fontsize="x-small")
    # plt.setp(axs[-1], "xlabel", r"$z\,[{\rm kpc}]$")
    zunit_label = r"$\,[{\rm kpc}]$" if kpc else r"$\,[{\rm pc}]$"
    if both:
        plt.setp(axes[-1, :], "xlabel", r"$|z|$" + zunit_label)
    else:
        plt.setp(axes[-1, :], "xlabel", r"$z$" + zunit_label)

    if savefig:
        plt.savefig(osp.join(fig_outdir, f"{gr}_loading_{vout_label}_z.pdf"))
    return fig


def plot_area_mass_fraction_z(
    simgroup, gr, tslice=slice(150, 500), kpc=True, savefig=True
):
    """Plot area and density fractions by phase with outflow contributions.

    Creates a 2xN grid (where N is number of simulations) showing area fraction
    (top row) and density (bottom row) by phase, including outflow-only fractions.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    tslice : slice
        Time slice to select (default=slice(150, 500))
    kpc : bool
        If True, convert z coordinates to kpc (default=True)
    savefig : bool
        If True, save figure (default=True)
    """
    sims = simgroup[gr]
    models = list(sims.keys())
    nmodels = len(models)
    fig, axes = plt.subplots(
        2,
        nmodels,
        figsize=(4 * nmodels, 5),
        sharey="row",
        sharex=True,
        constrained_layout=True,
    )
    for i, m in enumerate(models):
        s = sims[m]

        dset = s.zprof.sum(dim="vz_dir").sel(time=slice(150, 500))
        dset_upper = (
            s.zprof.sel(vz_dir=1)
            .sel(time=slice(150, 500))
            .sel(z=slice(0, s.domain["re"][2]))
        )
        dset_lower = (
            s.zprof.sel(vz_dir=-1)
            .sel(time=slice(150, 500))
            .sel(z=slice(s.domain["le"][2], 0))
        )
        dset_out = xr.concat([dset_lower, dset_upper], dim="z")

        for axs, field in zip(axes, ["area", "rho"]):
            plt.sca(axs[i])
            for ph, color, label in zip(
                [["CNM", "UNM"], "WNM", ["WHIM", "HIM"]],
                ["C0", "C2", "C1"],
                ["CNM+UNM", "WNM", "WHIM+HIM"],
            ):
                plot_zprof_frac(
                    dset, field, ph, kpc=kpc, line="mean", color=color, label=label
                )
                if field == "area":
                    if "whole" in dset_out["area"].phase:
                        totarea = dset_out["area"].sel(phase="whole")
                    else:
                        totarea = dset_out["area"].sum(dim="phase")
                    fA_out = (
                        dset_out[field].sel(phase=np.atleast_1d(ph)).sum(dim="phase")
                        / totarea
                    ).mean(dim="time")
                    z = fA_out.z / 1.0e3 if kpc else fA_out.z
                    plt.plot(z, fA_out, color=color, ls="--", label="outflow")

        plt.sca(axes[0, i])
        # plt.annotate(model_name[m],(0.05,0.95),xycoords="axes fraction",ha="left",va="top",color=model_color[m])
        plt.title(model_name[m], color=model_color[m])

    plt.sca(axes[0, 0])
    plt.ylim(0, 1)
    plt.xlim(-4, 4)
    plt.ylabel(r"$f_A, f_A^{\rm out}$")
    lines, labels = axes[0, 0].get_legend_handles_labels()
    custom_lines = [lines[4], lines[5]]
    plt.legend(
        custom_lines,
        [r"total", r"outflow"],
        fontsize="x-small",
    )
    plt.sca(axes[1, 0])
    plt.legend(fontsize="x-small")
    plt.yscale("log")
    plt.ylabel(r"$\langle{n}_{\rm H}\rangle\,[{\rm cm^{-3}}]$")
    plt.ylim(1.0e-5, 1)

    zunit_label = r"$\,[{\rm kpc}]$" if kpc else r"$\,[{\rm pc}]$"
    plt.setp(axes[-1, :], "xlabel", r"$z$" + zunit_label)
    if savefig:
        plt.savefig(osp.join(fig_outdir, f"{gr}_area_nH_profile_frac_z.pdf"))
    return fig


def plot_vertical_proflies_separate(simgroup, gr):
    """Plot separate panels for vertical profiles (area, density) and flow directions.

    Creates multiple plots showing area and density fraction profiles,
    separated by overall vs. outflow-only contributions.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    """
    field = "area"
    fig = plot_profile_frac_z(simgroup, gr, field=field, line="mean", savefig=False)
    plt.ylabel(r"$f_A$")
    plt.legend().remove()
    plt.yscale("linear")
    plt.ylim(0, 1)
    plt.savefig(osp.join(fig_outdir, f"{gr}_{field}_profile_frac_z.pdf"))

    field = "area"
    fig = plot_profile_frac_z(
        simgroup, gr, field=field, vz_dir=1, line="mean", savefig=False
    )
    plt.ylabel(r"$f_A^{\rm out}$")
    plt.legend().remove()
    plt.yscale("linear")
    plt.ylim(0, 1)
    plt.savefig(osp.join(fig_outdir, f"{gr}_{field}_profile_frac_out_z.pdf"))

    field = "rho"
    fig = plot_profile_frac_z(simgroup, gr, field=field, line="mean", savefig=False)
    plt.ylabel(r"$\langle{n}_{\rm H}\rangle\,[{\rm cm^{-3}}]$")
    plt.legend(fontsize="xx-small")
    plt.ylim(1.0e-5, 5)
    plt.savefig(osp.join(fig_outdir, f"{gr}_{field}_profile_frac_z.pdf"))

    field = "rho"
    fig = plot_profile_frac_z(
        simgroup, gr, field=field, vz_dir=1, line="mean", savefig=False
    )
    plt.ylabel(r"$\langle{n}_{\rm H}\rangle^{\rm out}\,[{\rm cm^{-3}}]$")
    plt.legend(fontsize="xx-small")
    plt.ylim(1.0e-5, 5)
    plt.savefig(osp.join(fig_outdir, f"{gr}_{field}_profile_frac_out_z.pdf"))

    field = "rho"
    fig = plot_profile_z(simgroup, gr, field=field, savefig=False)
    plt.ylabel(r"$\overline{n}_{\rm H}\,[{\rm cm^{-3}}]$")
    plt.legend().remove()
    plt.ylim(1.0e-4, 10)
    plt.savefig(osp.join(fig_outdir, f"{gr}_{field}_profile_z.pdf"))


def plot_velocity_z(
    simgroup, gr, ph="wc", tslice=slice(150, 500), kpc=True, savefig=True
):
    """Plot vertical velocity profiles for a simulation group.

    Creates a 4-panel plot showing bulk vertical velocity, CR Alfvén velocity,
    CR drift velocity, and effective CR velocity for warm cloud phase,
    separated into outflow and net contributions.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    ph : str
        Phase selection (default='wc' for warm cloud)
    savefig : bool
        If True, save figure (default=True)
    """
    sims = simgroup[gr]
    models = list(sims.keys())

    fig, axs = plt.subplots(4, 1, figsize=(3, 7), sharex=True, constrained_layout=True)

    for i, m in enumerate(models):
        s = sims[m]
        c = model_color[m]
        dnet = s.zp_ph.sel(time=tslice, z=slice(0, s.zp_ph.z.max())).sum(dim="vz_dir")
        dout = s.zp_ph.sel(time=tslice, z=slice(0, s.zp_ph.z.max()), vz_dir=1)

        if s.options["cosmic_ray"] and ("0-Veff3" not in dnet):
            crzp_net = s.zp_pp_ph.sel(time=tslice, z=slice(0, s.zp_ph.z.max())).sum(
                dim="vz_dir"
            )
            crzp_out = s.zp_pp_ph.sel(
                time=tslice, z=slice(0, s.zp_ph.z.max()), vz_dir=1
            )
        plt.sca(axs[0])
        plot_zprof_field(
            dout, "vel3", ph, kpc=kpc, color=c, line="median", label=model_name[m]
        )
        plot_zprof_field(
            dnet,
            "vel3",
            ph,
            kpc=kpc,
            color=c,
            line="median",
            quantile=False,
            lw=1,
            ls=":",
        )
        # plt.ylim(-25, 75)
        plt.title(f"ph={ph}")

        plt.sca(axs[1])
        if s.options["cosmic_ray"]:
            plot_zprof_field(
                dout, "0-Vs3", ph, kpc=kpc, color=c, line="median", label="out"
            )
            plot_zprof_field(
                dnet,
                "0-Vs3",
                ph,
                kpc=kpc,
                color=c,
                line="median",
                quantile=False,
                lw=1,
                ls=":",
                label="net",
            )
        # plt.ylim(0, 50)

        plt.sca(axs[2])
        if s.options["cosmic_ray"]:
            plot_zprof_field(dout, "0-Vd3", ph, kpc=kpc, color=c, line="median")
            plot_zprof_field(
                dnet,
                "0-Vd3",
                ph,
                kpc=kpc,
                color=c,
                line="median",
                quantile=False,
                lw=1,
                ls=":",
            )
        plt.ylim(-50, 50)

        plt.sca(axs[3])
        if s.options["cosmic_ray"]:
            if "0-Veff3" in dout:
                plot_zprof_field(
                    dout, "0-Veff3", ph, kpc=kpc, color=c, line="median", label="out"
                )
            else:
                plot_zprof_field(
                    crzp_out,
                    "0-Veff3",
                    ph,
                    kpc=kpc,
                    color=c,
                    line="median",
                    label="out",
                )
            if "0-Veff3" in dnet:
                plot_zprof_field(
                    dnet,
                    "0-Veff3",
                    ph,
                    kpc=kpc,
                    color=c,
                    line="median",
                    quantile=False,
                    lw=1,
                    ls=":",
                    label="net",
                )
            else:
                plot_zprof_field(
                    crzp_net,
                    "0-Veff3",
                    ph,
                    kpc=kpc,
                    color=c,
                    line="median",
                    quantile=False,
                    lw=1,
                    ls=":",
                    label="net",
                )

    plt.sca(axs[0])
    plt.ylabel(r"$\overline{v}_z\,[{\rm km/s}]$")
    plt.legend(fontsize="x-small")
    plt.sca(axs[1])
    plt.ylabel(r"$\overline{v}_{s,z}\,[{\rm km/s}]$")
    # adding custom legend for two line styles
    lines, labels = axs[1].get_legend_handles_labels()
    if len(lines) > 1:
        custom_lines = [lines[0], lines[1]]
        plt.legend(custom_lines, ["out", "net"], fontsize="x-small")

    plt.sca(axs[2])
    plt.ylabel(r"$\overline{v}_{d,z}\,[{\rm km/s}]$")

    plt.sca(axs[3])
    plt.ylabel(r"$\overline{v}_{{\rm eff},z}\,[{\rm km/s}]$")
    zunit_label = r"$\,[{\rm kpc}]$" if kpc else r"$\,[{\rm pc}]$"
    plt.xlabel(r"$z$" + zunit_label)
    plt.xlim(0, 4)
    if savefig:
        plt.savefig(osp.join(fig_outdir, f"{gr}_velocity_z_{ph}.pdf"))
    return fig


def plot_kappa_z(
    simgroup, gr, phases=["wc", "hot"], tslice=slice(150, 500), kpc=True, savefig=True
):
    """Plot CR diffusion coefficient and inverse (scattering rate) profiles.

    Creates a 2xN grid (where N is number of phases) showing effective
    diffusivity (kappa) and effective scattering rate (1/kappa) as a
    function of height.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    phases : list
        Phase names to plot (default=['wc', 'hot'])
    tslice : slice
        Time slice to select (default=slice(150, 500))
    kpc : bool
        If True, convert z coordinates to kpc (default=True)
    savefig : bool
        If True, save figure (default=True)
    """
    nph = len(phases)
    sims = simgroup[gr]
    fig, axes = plt.subplots(
        2,
        nph,
        figsize=(2.5 * nph + 1, 4),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )
    for m, s in sims.items():
        color = model_color[m]
        if s.options["cosmic_ray"]:
            if "0-Fd_B" not in s.zprof:
                # use post-processing zprof
                if phases[0] == "wc" and phases[1] == "hot":
                    dset_pp = s.zp_pp_ph.sel(time=tslice).sum(dim="vz_dir")
                else:
                    dset_pp = s.zp_pp.sel(time=tslice).sum(dim="vz_dir")
                dset_pp["kappa_eff"] = (
                    dset_pp["Fcr_diff_parallel"]
                    / dset_pp["GradPcr_parallel"]
                    * dset_pp["area"]
                ) * (s.u.cm**2 / s.u.s)
                dset_pp["sigma_eff"] = (
                    dset_pp["GradPcr_parallel"]
                    / dset_pp["Fcr_diff_parallel"]
                    * dset_pp["area"]
                ) / (s.u.cm**2 / s.u.s)
            else:
                if phases[0] == "wc" and phases[1] == "hot":
                    dset_pp = s.zp_ph.sel(time=tslice).sum(dim="vz_dir")
                else:
                    dset_pp = s.zprof.sel(time=tslice).sum(dim="vz_dir")
                dset_pp["kappa_eff"] = (
                    dset_pp["0-Fd_B"] / dset_pp["0-GradPc_B"] * dset_pp["area"]
                ) * (s.u.cm**2 / s.u.s)
                dset_pp["sigma_eff"] = (
                    dset_pp["0-GradPc_B"] / dset_pp["0-Fd_B"] * dset_pp["area"]
                ) / (s.u.cm**2 / s.u.s)
            for axs, ph in zip(axes.T, phases):
                plt.sca(axs[0])
                if isinstance(ph, list):
                    plt.title(f"ph={'+'.join(ph)}")
                else:
                    plt.title(f"ph={ph}")

                plot_zprof_field(
                    dset_pp, "kappa_eff", ph, color=color, label=r"$\kappa_{\rm eff}$"
                )
                plt.sca(axs[1])
                plot_zprof_field(
                    dset_pp,
                    "sigma_eff",
                    ph,
                    kpc=kpc,
                    color=color,
                    label=r"$\kappa_{\rm eff}^{-1}$",
                )

    plt.sca(axes[0, 0])
    plt.ylabel(r"${\kappa}_{\parallel, {\rm avg}}[{\rm cm^{2}\,s^{-1}}]$")
    plt.yscale("log")
    plt.ylim(1.0e27, 1.0e30)

    plt.sca(axes[1, 0])
    plt.ylabel(r"${\sigma}_{\parallel, {\rm avg}}[{\rm cm^{-2}\,s^{1}}]$")
    plt.yscale("log")
    plt.ylim(1.0e-30, 1.0e-27)

    zunit_label = r"$\,[{\rm kpc}]$" if kpc else r"$\,[{\rm pc}]$"
    plt.setp(axes[1, :], xlabel=r"$z$" + zunit_label, xlim=(-4, 4))

    if savefig:
        plt.savefig(osp.join(fig_outdir, f"{gr}_kappa_z_{nph}ph.pdf"))

    return fig


def plot_gainloss_z_each(
    s, m, phases=["wc", "hot"], tslice=slice(150, 500), kpc=True, savefig=True
):
    """Plot thermal energy loss/gain and CR energy loss/gain rates.

    Creates a 2xN grid showing radiation cooling/heating and CR work/losses
    (top row) vs. CR losses/injection and CR work (bottom row) as functions
    of height for specified phases.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    phases : list
        Phase names to plot (default=['wc', 'hot'])
    tslice : slice
        Time slice to select (default=slice(150, 500))
    kpc : bool
        If True, convert z coordinates to kpc (default=True)
    savefig : bool
        If True, save figure (default=True)
    """
    nph = len(phases)
    fig, axes = plt.subplots(
        2,
        nph,
        figsize=(2.5 * nph + 1, 4),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )

    color = model_color[m]
    name = model_name[m]
    is_zp_pp = hasattr(s, "zp_pp")

    if phases[0] == "wc" and phases[1] == "hot":
        if is_zp_pp:
            dset_pp = s.zp_pp_ph.sel(time=tslice).sum(dim="vz_dir")
        dset = s.zp_ph.sel(time=tslice).sum(dim="vz_dir")
    else:
        if is_zp_pp:
            dset_pp = s.zp_pp.sel(time=tslice).sum(dim="vz_dir")
        dset = s.zprof.sel(time=tslice).sum(dim="vz_dir")
    if s.options["cosmic_ray"]:
        if is_zp_pp:
            dset_pp["cr_heating"] = (
                -dset_pp["Gamma_cr_stream"] * (s.u.energy_density / s.u.time).cgs.value
            )
            dset_pp["cr_work"] = (
                dset_pp["CRwork_total"] * (s.u.energy_density / s.u.time).cgs.value
            )
            dset_pp["cr_loss"] = (
                -dset_pp["CRLosses"] * (s.u.energy_density / s.u.time).cgs.value
            )
        if "0-cooling_cr" in dset:
            dset["cr_loss"] = (
                -dset["0-cooling_cr"] * (s.u.energy_density / s.u.time).cgs.value
            )
        if "0-heating_cr" in dset:
            dset["cr_heating"] = (
                -dset["0-heating_cr"] * (s.u.energy_density / s.u.time).cgs.value
            )
        if "0-work_cr" in dset:
            dset["cr_work"] = (
                dset["0-work_cr"] * (s.u.energy_density / s.u.time).cgs.value
            )
        tdec_scr = s.par["feedback"]["tdec_scr"] * s.u.Myr
        dset["CRinj_rate"] = (dset["sCR"] / tdec_scr) * (
            s.u.energy_density / s.u.time
        ).cgs.value
    if not is_zp_pp:
        dset_pp = dset
    for axs, ph in zip(axes.T, phases):
        plt.sca(axs[0])
        if isinstance(ph, list):
            plt.title(f"ph={'+'.join(ph)}")
        else:
            plt.title(f"ph={ph}")
        plot_zprof_frac(
            dset_pp,
            "cool_rate",
            ph,
            kpc=kpc,
            color="C0",
            label=r"$n_H^2 \Lambda$",
        )
        if s.options["cosmic_ray"]:
            plot_zprof_frac(
                dset_pp,
                "cr_heating",
                ph,
                kpc=kpc,
                color="C1",
                label=r"$\mathcal{G}_{\rm cr, st}$",
            )
        plot_zprof_frac(
            dset_pp, "heat_rate", ph, kpc=kpc, color="C2", label=r"$n_H \Gamma$"
        )
        if s.options["cosmic_ray"]:
            plt.sca(axs[1])
            plot_zprof_frac(
                dset_pp,
                "cr_loss",
                ph,
                kpc=kpc,
                color="C0",
                label=r"$\mathcal{L}_{\rm cr}$",
            )
            plot_zprof_frac(
                dset,
                "CRinj_rate",
                ph,
                kpc=kpc,
                color="C1",
                label=r"$e_{\rm cr}/t_{\rm dec}$",
            )
            plot_zprof_frac(
                dset_pp,
                "cr_work",
                ph,
                kpc=kpc,
                color="C2",
                label=r"$W_{\rm gas-cr}$",
            )

    plt.sca(axes[0, 0])
    plt.ylabel("Energy Loss/Gain\n" + r"$[{\rm erg\,s^{-1}\,cm^{-3}}]$")
    plt.yscale("log")
    plt.ylim(1.0e-30, 1.0e-25)
    plt.legend(fontsize="x-small")

    plt.sca(axes[1, 0])
    plt.ylabel("Energy Loss/Gain\n" + r"$[{\rm erg\,s^{-1}\,cm^{-3}}]$")
    plt.yscale("log")
    plt.ylim(1.0e-30, 1.0e-25)
    plt.legend(fontsize="x-small")
    plt.sca(axes[1, 1])

    zunit_label = r"$\,[{\rm kpc}]$" if kpc else r"$\,[{\rm pc}]$"
    plt.setp(axes[1, :], xlabel=r"$z$" + zunit_label, xlim=(-4, 4))

    if savefig:
        plt.savefig(osp.join(fig_outdir, f"{name}_heatcool_z_{nph}ph.pdf"))
    return fig


def plot_gainloss_z(
    simgroup,
    gr,
    phases=["wc", "hot"],
    grav_work=False,
    tslice=slice(150, 500),
    kpc=True,
    savefig=True,
):
    """Plot thermal energy loss/gain and CR energy loss/gain rates.

    Creates a 2xN grid showing radiation cooling/heating and CR work/losses
    (top row) vs. CR losses/injection and CR work (bottom row) as functions
    of height for specified phases.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    phases : list
        Phase names to plot (default=['wc', 'hot'])
    tslice : slice
        Time slice to select (default=slice(150, 500))
    kpc : bool
        If True, convert z coordinates to kpc (default=True)
    savefig : bool
        If True, save figure (default=True)
    """
    sims = simgroup[gr]
    nph = len(phases)
    fig, axes = plt.subplots(
        2,
        nph,
        figsize=(2.5 * nph + 1, 4),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )
    labels = dict(
        cool_rate=r"$\mathcal{L}$",
        heat_rate=r"$\mathcal{G}$",
        cr_heating=r"$\mathcal{G}_{\rm st}$",
        cr_loss=r"$\mathcal{L}_{\rm cr}$",
        CRinj_rate=r"$\dot{e}_{\rm cr,SN}$",
        cr_work=r"$W_{\rm gas-cr}$",
        grav_work=r"$W_{\rm grav}$",
    )
    for m, s in sims.items():
        color = model_color[m]
        name = model_name[m]
        is_zp_pp = hasattr(s, "zp_pp")
        if phases[0] == "wc" and phases[1] == "hot":
            if is_zp_pp:
                dset_pp = s.zp_pp_ph.sel(time=tslice).sum(dim="vz_dir")
            dset = s.zp_ph.sel(time=tslice).sum(dim="vz_dir")
        else:
            if is_zp_pp:
                dset_pp = s.zp_pp.sel(time=tslice).sum(dim="vz_dir")
            dset = s.zprof.sel(time=tslice).sum(dim="vz_dir")
        if not is_zp_pp:
            dset_pp = dset
        if s.options["cosmic_ray"]:
            if is_zp_pp:
                dset_pp["cr_heating"] = (
                    -dset_pp["Gamma_cr_stream"]
                    * (s.u.energy_density / s.u.time).cgs.value
                )
                dset_pp["cr_work"] = (
                    dset_pp["CRwork_total"] * (s.u.energy_density / s.u.time).cgs.value
                )
                dset_pp["cr_loss"] = (
                    -dset_pp["CRLosses"] * (s.u.energy_density / s.u.time).cgs.value
                )
            if "0-heating_cr" in dset:
                dset["cr_heating"] = (
                    -dset["0-heating_cr"] * (s.u.energy_density / s.u.time).cgs.value
                )
            if "0-work_cr" in dset:
                dset["cr_work"] = (
                    dset["0-work_cr"] * (s.u.energy_density / s.u.time).cgs.value
                )
            if "0-cooling_cr" in dset:
                dset["cr_loss"] = (
                    dset["0-cooling_cr"] * (s.u.energy_density / s.u.time).cgs.value
                )
            tdec_scr = s.par["feedback"]["tdec_scr"] * s.u.Myr
            dset["CRinj_rate"] = (dset["sCR"] / tdec_scr) * (
                s.u.energy_density / s.u.time
            ).cgs.value
        dset["grav_work"] = (
            -(dset["Egflux1"] + dset["Egflux2"] + dset["Egflux3"])
            * (s.u.energy_density / s.u.time).cgs.value
        )
        for axs, ph in zip(axes.T, phases):
            plt.sca(axs[0])
            if isinstance(ph, list):
                plt.title(f"ph={'+'.join(ph)}")
            else:
                plt.title(f"ph={ph}")
            for f, c in zip(
                ["cool_rate", "heat_rate", "cr_heating", "cr_work", "grav_work"],
                ["C0", "C1", "C2", "C3", "C4"],
            ):
                kwargs = dict(ls=":" if name == "mhd" else "-", label=labels[f])
                if f == "grav_work":
                    if grav_work:
                        plot_zprof_frac(dset, f, ph, kpc=kpc, color=c, **kwargs)
                else:
                    plot_zprof_frac(dset_pp, f, ph, kpc=kpc, color=c, **kwargs)

            plt.sca(axs[1])
            for f, c in zip(
                ["cr_loss", "CRinj_rate", "cr_heating", "cr_work"],
                ["C0", "C1", "C2", "C3"],
            ):
                kwargs = dict(ls=":" if name == "mhd" else "-", label=labels[f])
                if f == "CRinj_rate":
                    plot_zprof_frac(dset, f, ph, kpc=kpc, color=c, **kwargs)
                else:
                    plot_zprof_frac(dset_pp, f, ph, kpc=kpc, color=c, **kwargs)

    nmhd = 3 if grav_work else 2  # number of mhd lines plotted
    plt.sca(axes[0, 0])
    plt.ylabel("Loss/Gain for Gas\n" + r"$[{\rm erg\,s^{-1}\,cm^{-3}}]$")
    plt.yscale("log")
    plt.ylim(1.0e-30, 1.0e-25)
    lines, labels = axes[0, 0].get_legend_handles_labels()
    custom_lines = [lines[0], lines[nmhd]]
    custom_labels = ["mhd", "crmhd"]
    plt.legend(
        custom_lines,
        custom_labels,
        fontsize="x-small",
        title="model",
        title_fontsize="x-small",
        frameon=False,
        loc=1,
    )

    plt.sca(axes[0, 1])
    lines, labels = axes[0, 1].get_legend_handles_labels()
    custom_lines = [lines[nmhd], lines[nmhd + 3]]
    custom_labels = [labels[nmhd], labels[nmhd + 3]]
    if grav_work:
        custom_lines.append(lines[nmhd + 4])
        custom_labels.append(labels[nmhd + 4])
    plt.legend(
        custom_lines,
        custom_labels,
        fontsize="x-small",
        title="gas loss",
        title_fontsize="x-small",
        frameon=False,
        loc=1,
    )

    plt.sca(axes[0, 2])
    lines, labels = axes[0, 2].get_legend_handles_labels()
    custom_lines = [lines[nmhd + 1], lines[nmhd + 2]]
    custom_labels = [labels[nmhd + 1], labels[nmhd + 2]]
    plt.legend(
        custom_lines,
        custom_labels,
        fontsize="x-small",
        title="gas gain",
        title_fontsize="x-small",
        frameon=False,
        loc=1,
    )
    # plt.legend(fontsize="x-small")

    plt.sca(axes[1, 0])
    plt.ylabel("Loss/Gain for CRs\n" + r"$[{\rm erg\,s^{-1}\,cm^{-3}}]$")
    plt.yscale("log")
    plt.ylim(1.0e-30, 1.0e-25)
    plt.sca(axes[1, 1])
    lines, labels = axes[1, 1].get_legend_handles_labels()
    custom_lines = [lines[0], lines[2]]
    custom_labels = [labels[0], labels[2]]
    plt.legend(
        custom_lines,
        custom_labels,
        fontsize="x-small",
        title="CR loss",
        title_fontsize="x-small",
        loc=1,
        frameon=False,
    )

    plt.sca(axes[1, 2])
    lines, labels = axes[1, 2].get_legend_handles_labels()
    custom_lines = [lines[1], lines[3]]
    custom_labels = [labels[1], labels[3]]
    plt.legend(
        custom_lines,
        custom_labels,
        fontsize="x-small",
        title="CR gain",
        title_fontsize="x-small",
        frameon=False,
        loc=1,
    )

    zunit_label = r"$\,[{\rm kpc}]$" if kpc else r"$\,[{\rm pc}]$"
    plt.setp(axes[1, :], xlabel=r"$z$" + zunit_label, xlim=(-4, 4))

    if savefig:
        plt.savefig(osp.join(fig_outdir, f"{gr}_heatcool_z_{nph}ph.pdf"))
    return fig


def plot_momentum_transfer_z(
    simgroup,
    gr,
    hot="full",
    showall=False,
    zmin=1000,
    tslice=slice(150, 500),
    kpc=True,
    savefig=True,
):
    """Plot vertical momentum transfer balance and pressure contributions.

    Creates a 2x2 grid showing momentum balance calculations in lower and upper
    hemispheres, including MHD pressure gradients, turbulent pressure, weight,
    and CR pressure contributions.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    hot : str
        How to show hot phase contribution: 'full' for total or 'trb' for turbulent only
    showall : bool
        If True, show all contributions (default=False)
    zmin : float
        Minimum z height to consider (default=1000)
    tslice : slice
        Time slice to select (default=slice(150, 500))
    kpc : bool
        If True, convert z coordinates to kpc (default=True)
    savefig : bool
        If True, save figure (default=True)
    """
    fig, axes = plt.subplots(
        2, 2, figsize=(8, 6), sharey=True, sharex="col", constrained_layout=True
    )
    sims = simgroup[gr]
    for i, (m, s) in enumerate(sims.items()):
        color = model_color[m]

        # read zprof/merge velocity
        dset = s.zp_ph.sum(dim="vz_dir").sel(time=tslice)
        dset = update_stress(s, dset)

        # setup gzext
        gzext = np.interp(dset.z, s.extgrav["z"], s.extgrav["gz"])
        dz = s.domain["dx"][2]

        # total area
        area = np.prod(s.domain["Lx"][:2])

        # total pressure
        PMHD = (
            (
                dset["Pturbz"]
                + dset["press"]
                + dset["Pmag1"]
                + dset["Pmag2"]
                - dset["Pmag3"]
            )
            * s.u.pok
            / area
        )
        # turbulent pressure
        Ptrb = dset["Pturbz"] * s.u.pok / area

        # weight
        dW = (dset["rhogz"] + dset["rho"] * gzext) * s.u.pok / area

        # CR pressure
        if s.options["cosmic_ray"]:
            Pcr = dset["0-Ec"] / 3.0 * s.u.pok / area

        # calculate pressure differences w.r.t. z=1kpc
        zmax = 5000
        Pu = PMHD.sel(z=zmin, method="nearest").sel(phase="wc").mean(dim="time")
        Pl = PMHD.sel(z=-zmin, method="nearest").sel(phase="wc").mean(dim="time")

        # upper half
        dFMHD_upper = (
            PMHD.sel(z=slice(zmin, zmax)) - PMHD.sel(z=zmin, method="nearest")
        ) / Pu
        dFtrb_upper = (
            Ptrb.sel(z=slice(zmin, zmax)) - Ptrb.sel(z=zmin, method="nearest")
        ) / Pu
        W_upper = dW.sel(z=slice(zmin, zmax)).cumsum(dim="z") * dz / Pu
        upper_fields = [dFMHD_upper, dFtrb_upper, W_upper]
        # lower half
        dFMHD_lower = (
            PMHD.sel(z=slice(-zmax, -zmin)) - PMHD.sel(z=-zmin, method="nearest")
        ) / Pl
        dFtrb_lower = (
            Ptrb.sel(z=slice(-zmax, -zmin)) - Ptrb.sel(z=-zmin, method="nearest")
        ) / Pl
        W_lower = (
            -dW.sel(z=slice(-zmax, -zmin)).isel(z=slice(None, None, -1)).cumsum(dim="z")
            * dz
            / Pl
        )
        W_lower = W_lower.isel(z=slice(None, None, -1))
        lower_fields = [dFMHD_lower, dFtrb_lower, W_lower]
        if s.options["cosmic_ray"]:
            dPcr_upper = (
                Pcr.sel(z=slice(zmin, zmax)) - Pcr.sel(z=zmin, method="nearest")
            ) / Pu
            dPcr_lower = (
                Pcr.sel(z=slice(-zmax, -zmin)) - Pcr.sel(z=-zmin, method="nearest")
            ) / Pl
            upper_fields.append(dPcr_upper)
            lower_fields.append(dPcr_lower)
        # annotate model
        # i for crmhd and mhd
        plt.sca(axes[i, 0])
        plt.annotate(
            model_name[m],
            (0.05, 0.95),
            xycoords="axes fraction",
            ha="left",
            va="top",
            color=model_color[m],
        )
        for fields, ax in zip([lower_fields, upper_fields], axes[i, :]):
            if s.options["cosmic_ray"]:
                dFMHD, dFtrb, W, dPcr = fields
            else:
                dFMHD, dFtrb, W = fields
            plt.sca(ax)

            # calculating RHS from hot
            RHS = -(dFMHD - W).sel(phase="hot")
            RHS_hot = -(dFMHD - W).sel(phase="hot")
            RHS_hot_label = r"$-(\Delta_z \mathcal{F}_{\rm MHD}^{\rm hot}-\Delta_z \mathcal{W}^{\rm hot})$"
            if s.options["cosmic_ray"]:
                RHS_hot += (-dPcr).sel(phase="hot")
                RHS += (-dPcr).sum(dim="phase")
                RHS_hot_label += r"$-\Delta_z P_{\rm cr}^{\rm hot}$"

            if showall:
                plot_zprof_mean_quantile(
                    RHS, kpc=kpc, color=color, ls=":", lw=3, quantile=False, label="RHS"
                )
            else:
                # dflux - weight - P_CR from hot phase only
                plot_zprof_mean_quantile(
                    RHS_hot,
                    kpc=kpc,
                    label=RHS_hot_label,
                    color=color,
                    ls="-.",
                    quantile=False,
                )
            # taking into account weight
            plot_zprof_mean_quantile(
                (dFMHD - W).sel(phase="wc"),
                kpc=kpc,
                label=r"$\Delta_z \mathcal{F}_{\rm MHD}^{\rm wc}-\Delta_z \mathcal{W}^{\rm wc}$",
                color=color,
                lw=3,
            )
            # Flux difference alone
            plot_zprof_mean_quantile(
                (dFMHD).sel(phase="wc"),
                kpc=kpc,
                label=r"$\Delta_z \mathcal{F}_{\rm MHD}^{\rm wc}$",
                color=color,
                lw=1,
                quantile=False,
            )
            if s.options["cosmic_ray"]:
                plot_zprof_mean_quantile(
                    (-dPcr).sel(phase="wc"),
                    kpc=kpc,
                    label=r"$-\Delta_z P_{\rm cr}^{\rm wc}$",
                    color=color,
                    ls="--",
                    quantile=False,
                )
                # separtely show CR pressure in hot
                if showall:
                    plot_zprof_mean_quantile(
                        (-dPcr).sel(phase="hot"),
                        kpc=kpc,
                        label=r"$-\Delta_z P_{\rm cr}^{\rm hot}$",
                        color=color,
                        ls="-.",
                        quantile=False,
                    )
            if showall:
                if hot == "full":
                    # hot contribution
                    plot_zprof_mean_quantile(
                        -(dFMHD - W).sel(phase="hot"),
                        kpc=kpc,
                        label=r"$-(\Delta_z \mathcal{F}_{\rm MHD}^{\rm hot}-\Delta_z \mathcal{W}^{\rm hot})$",
                        color=color,
                        ls=":",
                        quantile=False,
                    )
                elif hot == "trb":
                    plot_zprof_mean_quantile(
                        -dFtrb.sel(phase="hot"),
                        kpc=kpc,
                        label=r"$-\Delta_z \mathcal{F}_{\rm trb}^{\rm hot}$",
                        color=color,
                        ls=":",
                        quantile=False,
                    )

    plt.sca(axes[0, 0])
    plt.title("lower")
    plt.xlim(-4, -1)
    plt.ylim(-1, 2.5)
    plt.sca(axes[0, 1])
    plt.title("upper")
    plt.legend(frameon=False, loc=2)
    plt.xlim(1, 4)
    plt.sca(axes[1, 1])
    plt.legend(frameon=False, loc=2)
    plt.setp(
        axes[:, 0],
        ylabel=r"$\langle\Delta_z P\rangle/\langle\mathcal{F}_{\rm MHD}^{\rm wc}(z=1 {\rm kpc})\rangle$",
    )
    zunit_label = r"$\,[{\rm kpc}]$" if kpc else r"$\,[{\rm pc}]$"
    plt.setp(axes[1, :], xlabel=r"$z$" + zunit_label)
    if savefig:
        if showall:
            plt.savefig(osp.join(fig_outdir, f"{gr}_dflux_{hot}.pdf"))
        else:
            plt.savefig(osp.join(fig_outdir, f"{gr}_dflux.pdf"))
    return fig


# ----------------------------------------
# plotting functions time evolution
# ----------------------------------------
def plot_mass_fraction_t(simgroup, gr, zslice=slice(-50, 50), savefig=True):
    """Plot mass fractions in different phases as a function of time.

    Shows the evolution of cold, warm, and hot phase mass fractions
    over the simulation time for all simulations in a group.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    zslice : slice
        Height range for data selection (default=slice(-50, 50))
    savefig : bool
        If True, save figure (default=True)
    """
    sims = simgroup[gr]
    models = list(sims.keys())
    nmodels = len(models)
    fig, axes = plt.subplots(
        1,
        nmodels,
        figsize=(3 * nmodels, 4),
        sharey=True,
        sharex=True,
        constrained_layout=True,
    )
    for i, m in enumerate(models):
        s = sims[m]

        dset = s.zprof.sum(dim="vz_dir").sel(z=zslice)
        plt.sca(axes[i])
        total_mass = dset["rho"].sum(dim="phase").sum(dim="z")
        for ph, color, label in zip(
            [["CNM", "UNM"], "WNM", "WHIM", "HIM"],
            ["tab:blue", "tab:olive", "tab:red"],
            ["Cold", "Warm", "Hot"],
        ):
            frac = dset["rho"].sel(phase=ph).sum(dim="phase").sum(dim="z") / total_mass
            plt.plot(dset.time * s.u.Myr, frac, color=color, label=label)
        plt.title(model_name[m])
    plt.sca(axes[0])
    plt.legend()
    plt.ylim(0, 1)
    plt.xlabel(r"$t\,[{\rm Myr}]$")
    plt.ylabel(r"Mass Fraction")
    if savefig:
        plt.savefig(osp.join(fig_outdir, f"{gr}_mass_fraction_t.pdf"))
    return fig


def plot_history(simgroup, gr, tslice=slice(150, 500), savefig=True):
    """Plot simulation history (star formation rate and other quantities).

    Creates a 2-panel plot showing star formation rate and vertical momentum
    transfer over time, with time-averaged values indicated.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    tslice : slice
        Time slice to select (default=slice(150, 500))
    savefig : bool
        If True, save figure (default=True)
    """
    sims = simgroup[gr]
    fig, axes = plt.subplots(
        1, 2, figsize=(8, 2.5), sharex=True, constrained_layout=True
    )

    plt.sca(axes[0])
    field = "sfr40"
    for m, s in sims.items():
        color = model_color[m]
        name = model_name[m]
        h = s.hst
        plt.plot(h["time"], h[field], label=name, color=color)

        avg = h[field].loc[tslice].mean()
        std = h[field].loc[tslice].std()
        plt.axhline(avg, color=color, lw=1, ls="--")
        plt.axhspan(avg - std, avg + std, color=color, alpha=0.1, lw=0)
        print(m, avg, std)

    plt.sca(axes[1])
    for field, label, ls in zip(
        ["Sigma_gas", "Sigma_out"],
        [r"$\Sigma_{\rm gas}$", r"$\Sigma_{\rm out}$"],
        ["-", "--"],
    ):
        i = 0
        for m, s in sims.items():
            color = model_color[m]
            name = model_name[m]
            h = s.hst
            plt.plot(
                h["time"], h[field], label=label if i == 0 else None, ls=ls, color=color
            )
            i += 1
    plt.sca(axes[0])
    plt.ylabel(r"$\Sigma_{\rm SFR}\,[M_\odot\,{\rm kpc^{-2}\,yr^{-1}}]$")
    plt.ylim(bottom=0)
    # plt.ylim(1.0e-3, 1.0e-2)
    # plt.yscale("log")
    plt.legend(fontsize="x-small")
    plt.xlabel(r"$t\,[{\rm Myr}]$")
    plt.annotate("(a)", xy=(0.05, 0.95), xycoords="axes fraction", ha="left", va="top")
    plt.sca(axes[1])
    plt.ylabel(r"$\Sigma\,[M_\odot\,{\rm pc^{-2}}]$")
    plt.ylim(0, 15)
    plt.legend(fontsize="x-small")
    plt.xlabel(r"$t\,[{\rm Myr}]$")
    plt.xlim(0, 500)
    plt.annotate("(b)", xy=(0.05, 0.95), xycoords="axes fraction", ha="left", va="top")
    if savefig:
        plt.savefig(osp.join(fig_outdir, f"{gr}_history.pdf"))

    return fig


def plot_pressure_t(simgroup, gr, ph="wc", zslice=slice(-50, 50), savefig=True):
    """Plot pressure components vs. time in the disk midplane.

    Creates a 4-panel plot showing CR, thermal, kinetic, and magnetic pressure
    evolution at z=0 over simulation time for all simulations.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    ph : str
        Phase selection (default='wc' for warm cloud)
    zslice : slice
        Height range for data selection (default=slice(-50, 50))
    savefig : bool
        If True, save figure (default=True)
    """
    sims = simgroup[gr]
    fig, axes = plt.subplots(
        1, 4, figsize=(8, 2), sharey=True, sharex=True, constrained_layout=True
    )
    for m, s in sims.items():
        color = model_color[m]
        name = model_name[m]
        dset = s.zp_ph.sel(z=zslice).sum(dim="vz_dir")
        dset = update_stress(s, dset)
        for ax, pfield in zip(axes, ["Pok_cr", "Pok_th", "Pok_kin", "Pi_B"]):
            plt.sca(ax)
            lab = pfield.split("_")[-1]
            if pfield not in dset:
                continue
            # if pfield == "Pok_cr":
            #     pok = (dset[pfield].sum(dim="phase")/dset["area"].sum(dim="phase")).mean(dim="z")
            #     plt.plot(pok.time * s.u.Myr, pok, color=color, lw=1)

            pok = (dset[pfield] / dset["area"]).sel(phase=ph).mean(dim="z")
            plt.plot(pok.time * s.u.Myr, pok, label=name, color=color)

            if pfield.startswith("Pok_"):
                label = f"$P_{{\\rm {lab}}}$"
            else:
                label = f"$\\Pi_{{\\rm {lab}}}$"
            # plt.annotate(label,(0.05,0.95),
            #              xycoords="axes fraction",ha="left",va="top")
            plt.title(label)
            plt.yscale("log")
            plt.axvline(150, color="k", ls=":")
            plt.ylim(5.0e2, 1.0e5)
            plt.xlabel(r"$t\,[{\rm Myr}]$")
            plt.xlim(0, 500)

    plt.sca(axes[1])
    plt.legend(fontsize="small")
    plt.sca(axes[0])
    plt.ylabel(r"$\overline{P}^{\rm \;wc}(z=0)/k_B\,[{\rm cm^{-3}\,K}]$")
    if savefig:
        plt.savefig(osp.join(fig_outdir, f"{gr}_pressure_t.pdf"))
    return fig


def plot_vertical_equilibrium_t(
    simgroup, gr, ph="wc", exclude=[], zmax=1000, savefig=True
):
    """Plot vertical pressure equilibrium evolution over time.

    Creates plots showing pressure balance between midplane and 1 kpc altitude
    for specified phase, organized by simulation. Computes pressure differences
    to assess vertical equilibrium.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    ph : str
        Phase selection (default='wc' for warm cloud)
    exclude : list
        List of simulation names to exclude (default=[])
    zmax : float
        Maximum z height to consider (default=1000)
    savefig : bool
        If True, save figure (default=True)
    """
    sims = simgroup[gr]
    for m in exclude:
        sims = dict(sims)
        sims.pop(m, None)
    nmodels = len(sims)
    fig, axes = plt.subplots(
        1,
        nmodels,
        figsize=(4 * nmodels, 2.5),
        sharey=True,
        sharex=True,
        constrained_layout=True,
    )
    for (m, s), ax in zip(sims.items(), axes):
        plt.sca(ax)
        color = model_color[m]
        name = model_name[m]
        # plt.annotate(name, xy=(0.05, 0.95), xycoords="axes fraction",
        #              ha="left", va="top", color=color)
        plt.title(name, color=color)
        dset = s.zp_ph.sum(dim="vz_dir").sel(phase=ph)
        # dset = s.zp_ph.sum(dim=["phase","vz_dir"])
        dset = update_stress(s, dset)

        # calculate delta between z=0 and z=1kpc
        z0 = zmax
        dz = s.domain["dx"][2]

        # total pressure
        Ptot = dset["Pok_tot"] / dset["area"]
        Ptot_mid = Ptot.sel(z=slice(-dz, dz)).mean(dim="z")
        Ptot_1kpc = 0.5 * (
            Ptot.sel(z=slice(z0 - dz, z0 + dz)).mean(dim="z")
            + Ptot.sel(z=slice(-z0 - dz, -z0 + dz)).mean(dim="z")
        )
        delta = Ptot_mid - Ptot_1kpc
        plt.plot(dset.time * s.u.Myr, delta * 1.0e-4, label=r"$P_{\rm MHD}$")

        # total weight
        area_tot = s.domain["Lx"][0] * s.domain["Lx"][1]
        Wtot = dset["Wtot"] / area_tot
        Wtot_mid = Wtot.sel(z=slice(-dz, dz)).mean(dim="z")
        Wtot_1kpc = 0.5 * (
            Wtot.sel(z=slice(z0 - dz, z0 + dz)).mean(dim="z")
            + Wtot.sel(z=slice(-z0 - dz, -z0 + dz)).mean(dim="z")
        )
        delta = Wtot_mid - Wtot_1kpc
        plt.plot(dset.time * s.u.Myr, delta * 1.0e-4, label=r"$\mathcal{W}$")

        # CR
        if s.options["cosmic_ray"]:
            Pcr = dset["Pok_cr"] / dset["area"]
            Pcr_mid = Pcr.sel(z=slice(-dz, dz)).mean(dim="z")
            Pcr_1kpc = 0.5 * (
                Pcr.sel(z=slice(z0 - dz, z0 + dz)).mean(dim="z")
                + Pcr.sel(z=slice(-z0 - dz, -z0 + dz)).mean(dim="z")
            )
            delta = Pcr_mid - Pcr_1kpc
            plt.plot(dset.time * s.u.Myr, delta * 1.0e-4, label=r"$P_{\rm cr}$")
        # plt.yscale("log")
        # plt.ylim(5.0e2, 5.0e4)
        plt.xlabel(r"$t\,[{\rm Myr}]$")
    plt.xlim(0, 500)
    plt.sca(axes[0])
    plt.ylabel(  # r"$\langle P_{\rm tot} \rangle^{\rm wc}/f_A^{\rm wc}$"
        # r"$\,\langle \mathcal{W}_{\rm tot}\rangle^{\rm wc}$"
        r"$\Delta_{1{\rm kpc}} P\,[10^4 k_B{\rm cm^{-3}\,K}]$"
    )
    plt.sca(axes[1])
    plt.legend(fontsize="small", loc=1)
    if savefig:
        plt.savefig(osp.join(fig_outdir, f"{gr}_vertical_equilibrium_t.pdf"))
    return fig


# ----------------------------------------
# plotting functions outflow velocity PDFs
# ----------------------------------------
def plot_jointpdf(simgroup, gr, kpc=True, savefig=True):
    """Plot 2D joint PDF of outflow velocity and sound speed.

    Creates a 2D histogram showing the probability distribution of outflow
    velocity vs. sound speed at multiple altitudes for each simulation.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    kpc : bool
        If True, convert z coordinates to kpc (default=True)
    savefig : bool
        If True, save figure (default=True)
    """
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(10, 4.8),
        sharey="row",
        sharex="col",
        constrained_layout=True,
        gridspec_kw={"hspace": 0.01, "wspace": 0.01},
    )
    for (m, s), axs in zip(simgroup[gr].items(), axes):
        outflux = s.outflux
        dbin = outflux.logvz[1] - outflux.logvz[0]
        dbinsq = dbin**2
        name = model_name[m]
        for z0, ax in zip(outflux.z, axs):
            z = z0 / 1.0e3 if kpc else z0
            zunit_label = r"$\,{\rm kpc}$" if kpc else r"$\,{\rm pc}$"
            plt.sca(ax)
            plt.pcolormesh(
                outflux.logvz,
                outflux.logcs,
                outflux.sel(z=z0, flux="mflux") / dbinsq,
                norm=LogNorm(1.0e-5, 1.0),
                cmap=cmr.fall_r,
            )
            if name == "mhd":
                plt.title(f"$|z|={z:3.1f}$" + zunit_label)
            if z0 == 500:
                plt.annotate(
                    name,
                    xy=(0.05, 0.95),
                    xycoords="axes fraction",
                    ha="left",
                    va="top",
                    color=model_color[m],
                )
            plt.xlim(0, 3.5)
            plt.ylim(0, 3.3)
            ax.set_aspect("equal")
    plt.setp(axes[:, 0], "ylabel", r"$\log c_s\,[{\rm km/s}]$")
    plt.setp(axes[1, :], "xlabel", r"$\log v_{\rm out}\,[{\rm km/s}]$")
    plt.colorbar(
        mappable=axes[0, 0].collections[0],
        ax=axes[:, -1],
        label=r"$d^2\mathcal{F}_M/d\log v_{\rm out}d\log c_s\,[M_\odot\,{\rm kpc^{-2}\,yr^{-1}\,dex^{-2}}]$",
    )
    if savefig:
        plt.savefig(osp.join(fig_outdir, f"{gr}_jointpdfs.png"))
    return fig


def plot_voutpdf(simgroup, gr, kpc=True, savefig=True):
    """Plot 1D PDFs of outflow velocity separated by phase.

    Creates a 2x2 grid showing 1D probability distributions of outflow velocity
    and energy flux for warm cloud and hot phases across different altitudes.

    Parameters
    ----------
    simgroup : dict
        Nested dictionary of grouped simulations
    gr : str
        Group name to plot
    kpc : bool
        If True, convert z coordinates to kpc (default=True)
    savefig : bool
        If True, save figure (default=True)
    """
    fig, axes = plt.subplots(
        2, 2, figsize=(8, 6), sharex="col", sharey="row", constrained_layout=True
    )
    for (m, s), axs in zip(simgroup[gr].items(), axes.T):
        outflux = s.outflux
        dbin = outflux.logvz[1] - outflux.logvz[0]
        name = model_name[m]
        color = model_color[m]
        axs[0].set_title(name, color=color)
        for z0 in outflux.z:
            z = z0 / 1.0e3 if kpc else z0
            lw = z0 / 1.0e3
            zunit_label = r"$\,{\rm kpc}$" if kpc else r"$\,{\rm pc}$"
            plt.sca(axs[0])
            plt.plot(
                outflux.logvz,
                outflux.sel(logcs=slice(0, 1.2), z=z0, flux="mflux").sum(dim="logcs")
                / dbin,
                color="C0",
                lw=lw,
                label=f"${z:3.1f}$" + zunit_label,
            )
            plt.plot(
                outflux.logvz,
                outflux.sel(logcs=slice(1.2, 4), z=z0, flux="mflux").sum(dim="logcs")
                / dbin,
                color="C1",
                lw=lw,
            )
            plt.yscale("log")
            plt.ylim(1.0e-5, 1.0e-1)
            plt.sca(axs[1])
            plt.plot(
                outflux.logvz,
                outflux.sel(logcs=slice(0, 1.2), z=z0, flux="eflux").sum(dim="logcs")
                / dbin,
                color="C0",
                lw=lw,
                label="wc" if z0 == 3000 else None,
            )
            plt.plot(
                outflux.logvz,
                outflux.sel(logcs=slice(1.2, 4), z=z0, flux="eflux").sum(dim="logcs")
                / dbin,
                color="C1",
                lw=lw,
                label="hot" if z0 == 3000 else None,
            )
            plt.yscale("log")
            plt.ylim(1.0e42, 5.0e46)
    plt.sca(axes[0, 0])
    plt.ylabel(
        r"$d\mathcal{F}_M/d\log v_{\rm out}$"
        + "\n"
        + r"$[M_\odot\,{\rm kpc^{-2}\,yr^{-1}\,dex^{-1}}]$"
    )
    plt.legend(fontsize="x-small")
    plt.xlim(0, 3.5)
    plt.sca(axes[0, 1])
    plt.xlim(0, 3.5)
    plt.sca(axes[1, 0])
    plt.legend(fontsize="x-small")
    plt.xlabel(r"$\log v_{\rm out}\,[{\rm km/s}]$")
    plt.ylabel(
        r"$d\mathcal{F}_{E,{\rm MHD}}/d\log v_{\rm out}$"
        + "\n"
        + r"$[{\rm erg}\,{\rm kpc^{-2}\,yr^{-1}\,dex^{-1}}]$"
    )
    plt.sca(axes[1, 1])
    plt.xlabel(r"$\log v_{\rm out}\,[{\rm km/s}]$")
    if savefig:
        plt.savefig(osp.join(fig_outdir, f"{gr}_voutpdf.pdf"))

    return fig
