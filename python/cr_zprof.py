import os
import os.path as osp
import glob

import xarray as xr
import numpy as np

from scipy.optimize import curve_fit

model_name = {
    "crmhd-16pc-b1-diode-lngrad_out": "crmhd_v1",
    "crmhd_v2-16pc-b1-diode-lngrad_out": "crmhd_v2",
    "crmhd_v2-8pc-b1-diode-lngrad_out": "crmhd",
    "mhd-16pc-b1-diode": "mhd_v1",
    "mhd-16pc-b1-lngrad_out": "mhd_lngrad",
    "mhd_v2-16pc-b1-diode": "mhd_v2",
    "mhd_v2-8pc-b1-diode": "mhd",
    "crmhd-16pc-b0.1-diode-lngrad_out": "b0.1",
    "crmhd-16pc-b1-diode-lngrad_out-Vmax10": "Vmax10",
    "crmhd-16pc-b1-lngrad_out-lngrad_out-sigma27": "σ27",
    "crmhd-16pc-b1-lngrad_out-lngrad_out-sigma29": "σ29",
    "crmhd-16pc-b1-diode-lngrad_out-sigma28": "σ28",
    "crmhd-16pc-b1-diode-lngrad_out-sigma27_va0": "σ27_vA",
    "crmhd-16pc-b1-diode-lngrad_out-sigma28_va0": "σ28_vA",
    "crmhd-16pc-b1-diode-lngrad_out-sigma29_va0": "σ29_vA",
    "crmhd-16pc-b1-diode-lngrad_out-sigma27_va1": "σ27_vAi",
    "crmhd-16pc-b1-diode-lngrad_out-sigma28_va1": "σ28_vAi",
    "crmhd-16pc-b1-diode-lngrad_out-sigma29_va1": "σ29_vAi",
    "crmhd-16pc-b10-diode-lngrad_out": "b10",
    "crmhd-16pc-tallbox-b1-diode-lngrad_out": "tall",
    "crmhd-16pc-tallbox-b1-diode-diode": "tall-diode",
    "crmhd-16pc-fullgrav-b1-diode-lngrad_out": "fullgrav",
}

model_color = {
    "crmhd-16pc-b1-diode-lngrad_out": "#0504aa",
    "crmhd_v2-16pc-b1-diode-lngrad_out": "cornflowerblue",
    "crmhd_v2-8pc-b1-diode-lngrad_out": "#E77500",
    "crmhd-16pc-b1-diode-lngrad_out-sigma28": "gold",
    "mhd-16pc-b1-diode": "crimson",
    "mhd-16pc-b1-lngrad_out": "salmon",
    "mhd_v2-16pc-b1-diode": "salmon",
    "mhd_v2-8pc-b1-diode": "#000000",
    "crmhd-16pc-b0.1-diode-lngrad_out": "turquoise",
    "crmhd-16pc-b1-diode-lngrad_out-Vmax10": "orchid",
    "crmhd-16pc-b1-lngrad_out-lngrad_out-sigma27": "sienna",
    "crmhd-16pc-b1-lngrad_out-lngrad_out-sigma29": "darkorange",
    "crmhd-16pc-b1-diode-lngrad_out-sigma27_va0": "sienna",
    "crmhd-16pc-b1-diode-lngrad_out-sigma28_va0": "gold",
    "crmhd-16pc-b1-diode-lngrad_out-sigma29_va0": "darkorange",
    "crmhd-16pc-b1-diode-lngrad_out-sigma27_va1": "sienna",
    "crmhd-16pc-b1-diode-lngrad_out-sigma28_va1": "gold",
    "crmhd-16pc-b1-diode-lngrad_out-sigma29_va1": "darkorange",
    "crmhd-16pc-b10-diode-lngrad_out": "teal",
    "crmhd-16pc-tallbox-b1-diode-lngrad_out": "indigo",
    "crmhd-16pc-tallbox-b1-diode-diode": "violet",
    "crmhd-16pc-fullgrav-b1-diode-lngrad_out": "orange",
}
model_edge_color = {
    "mhd_v2-8pc-b1-diode": "#E77500",
    "crmhd_v2-8pc-b1-diode-lngrad_out": "#000000",
}

model_default = [
    "crmhd_v2-8pc-b1-diode-lngrad_out",
    "mhd_v2-8pc-b1-diode",
]

model_crmhd_compare = [
    "crmhd-16pc-b1-diode-lngrad_out",
    "crmhd_v2-16pc-b1-diode-lngrad_out",
    "crmhd_v2-8pc-b1-diode-lngrad_out",
]

model_mhd_compare = [
    "mhd-16pc-b1-diode",
    "mhd_v2-16pc-b1-diode",
    "mhd_v2-8pc-b1-diode",
]

model_beta = [
    "crmhd-16pc-b1-diode-lngrad_out",
    "crmhd-16pc-b0.1-diode-lngrad_out",
    "crmhd-16pc-b10-diode-lngrad_out",
    "crmhd-16pc-b1-diode-lngrad_out-Vmax10",
]
model_sigma = [
    "crmhd-16pc-b1-diode-lngrad_out",
    "crmhd-16pc-b1-diode-lngrad_out-sigma27_va1",
    "crmhd-16pc-b1-diode-lngrad_out-sigma28_va1",
    "crmhd-16pc-b1-diode-lngrad_out-sigma29_va1",
    "crmhd-16pc-b1-diode-lngrad_out-sigma27_va0",
    "crmhd-16pc-b1-diode-lngrad_out-sigma28_va0",
    "crmhd-16pc-b1-diode-lngrad_out-sigma29_va0",
    # "crmhd-16pc-b1-lngrad_out-lngrad_out-sigma27",
    # "crmhd-16pc-b1-lngrad_out-lngrad_out-sigma29",
    # "crmhd-16pc-b1-diode-lngrad_out-sigma28",
]
model_tall = [
    "crmhd-16pc-b1-diode-lngrad_out",
    "crmhd-16pc-tallbox-b1-diode-lngrad_out",
    "crmhd-16pc-tallbox-b1-diode-diode",
    "crmhd-16pc-fullgrav-b1-diode-lngrad_out",
]


# Add this function to fit exponential profiles
def fit_exponential_profile(z, P, return_all=False, zmin=0.1, zmax=1):
    """Fit P(z) with exponential profile: P(z) = P0 * exp(-|z|/H)

    Parameters
    ----------
    z : array-like
        Height array (can be negative)
    P : array-like
        Pressure values corresponding to z
    return_all : bool
        If True, return dict with P0, H, and covariance

    Returns
    -------
    popt : tuple (P0, H)
        Fitted parameters
    """

    def exp_profile(z, P0, H):
        return P0 * np.exp(-np.abs(z) / H)

    # Remove NaN/inf values
    mask = np.isfinite(P) & np.isfinite(z) & (np.abs(z) > zmin) & (np.abs(z) < zmax)
    z_clean = z[mask]
    P_clean = P[mask]

    # Initial guess: P0 from midplane, H from scale height
    P0_guess = (
        P_clean[np.argmin(np.abs(z_clean))].values
        if hasattr(P_clean, "values")
        else P_clean[np.argmin(np.abs(z_clean))]
    )
    H_guess = 1.0  # kpc

    try:
        popt, pcov = curve_fit(
            exp_profile, z_clean, P_clean, p0=[P0_guess, H_guess], maxfev=1000
        )
        if return_all:
            return {"P0": popt[0], "H": popt[1], "covariance": pcov}
        return popt
    except RuntimeError:
        print("Fit failed")
        return None


def get_model_table_line(s):
    """Generate a formatted table line describing simulation parameters.

    Extracts and formats key simulation parameters (physics, resolution,
    boundary conditions, etc.) into a LaTeX-formatted table row. Also
    classifies the simulation into a group (default, mhd, crmhd, sigma, etc.).

    Parameters
    ----------
    s : LoadSim
        Simulation object with parameters

    Returns
    -------
    name : str
        Simulation name
    line : str
        Formatted LaTeX table row
    group : str
        Classification group for organizing simulations
    """
    par = s.par
    prob = s.par["problem"]
    mesh = s.par["mesh"]

    # varied parameters
    beta = prob["beta0"]
    Lz = mesh["x3max"] - mesh["x3min"]
    dx = int(Lz) / int(mesh["nx3"])
    mhdbc = mesh["mhd_outflow_bc"]
    grav = "skipped" if par["gravity"]["solve_grav_hyperbolic_dt"] == "true" else "full"

    physics = (
        "crmhd" if par["configure"]["Cosmic_Ray_Transport"] == "Multigroups" else "mhd"
    )

    if physics == "crmhd":
        cr = s.par["cr"]
        crbc = mesh["cr_outflow_bc"]
        vmax = f"{cr['vmax'] / 1.0e9:<5.0f}"
        sigma = f"{cr['sigma']}" if cr["self_consistent_flag"] == 0 else "full"
    else:
        crbc = "\\nodata"
        vmax = "\\nodata"
        sigma = "\\nodata"
    name = s.basename.replace("mhdbc_", "").replace("crbc_", "").replace("-icpx", "")
    try:
        name = model_name[name]
    except KeyError:
        pass
    line = (
        f"{name:<20s} & {physics:<10s} & {beta:<5.1f} & {Lz:<10.0f} & {dx:<5.0f} & "
        f"{mhdbc:<10s} & {crbc:<10s} & {vmax:<10s} & {sigma:<10s} & {grav:<10s} \\\\"
    )

    if beta == 0.1:
        group = "b0.1"
    elif beta == 10:
        group = "b10"
    else:
        if physics == "mhd":
            group = "mhd"
        elif sigma != "full":
            group = "sigma"
        elif cr["vmax"] / 1.0e9 == 10:
            group = "vmax"
        elif Lz != 8192:
            group = "tall"
        # elif mhdbc == "diode":
        # group = "diode"
        # elif mhdbc == "lngrad_out":
        # group = "lngrad"
        else:
            group = "bcs"
    return name, line, group


def cr_data_load(
    basedir="/scratch/gpfs/EOST/changgoo/tigress_classic/", pattern="*mhd*"
):
    """Load simulation folders and initialize color/name mappings.

    Scans a base directory for simulation folders matching a pattern,
    creates a dictionary mapping normalized names to folder paths, and
    assigns colors/names to new simulations not in predefined mappings.

    Parameters
    ----------
    basedir : str
        Base directory containing simulation folders
    pattern : str
        Glob pattern for selecting simulation folders

    Returns
    -------
    model_dict : dict
        Mapping from simulation name to folder path
    """
    folders = sorted(glob.glob(osp.join(basedir, pattern)))
    icolor = 0
    model_dict = dict()
    print(f"Available models in {basedir}:")
    for folder in folders:
        name = os.path.basename(folder)
        name = name.replace("mhdbc_", "").replace("crbc_", "").replace("-icpx", "")
        model_dict[name] = folder
        if name not in model_color:
            print(f"  - {name}")
            model_color[name] = f"C{icolor}"
            model_name[name] = name
            icolor += 1

    return model_dict


def cr_data_group(sims):
    """Group simulations by classification based on parameters.

    Classifies each simulation into groups (default, mhd, crmhd, sigma, etc.)
    using get_model_table_line(). Creates nested dictionary structure.

    Parameters
    ----------
    sims : LoadSimAll
        Container with list of simulation models

    Returns
    -------
    sim_group : dict
        Nested dictionary: sim_group[group_name][model_name] = LoadSim object
    """
    sim_group = dict()
    sim_group["default"] = dict()
    for m in sims.models:
        s = sims.set_model(m)
        name, line, group = get_model_table_line(s)
        if group not in sim_group:
            sim_group[group] = dict()
        sim_group[group][name] = s
        if name in ["crmhd-b1-diode-lngrad_out"]:
            sim_group["default"][name] = s

    return sim_group


def load_group(sim_group, group="default"):
    """Load and process z-profile data for a group of simulations.

    Loads zprof and hst (history) data for all simulations in a group,
    and creates phase-aggregated z-profile data (warm cloud, hot phases).

    Parameters
    ----------
    sim_group : dict
        Nested dictionary of simulations by group
    group : str
        Name of the group to load (default="default")
    """
    sg = sim_group[group]

    for name, s in sg.items():
        print(f"loading {name}...")
        if not hasattr(s, "zprof"):
            s.zprof = s.load_zprof()
        s.zp_ph = xr.concat(
            [
                s.zprof.sel(phase=["CNM", "UNM", "WNM"])
                .sum(dim="phase")
                .assign_coords(phase="wc"),
                s.zprof.sel(phase=["WHIM", "HIM"])
                .sum(dim="phase")
                .assign_coords(phase="hot"),
            ],
            dim="phase",
        )
        if not hasattr(s, "hst"):
            s.hst = s.read_hst()


def load_windpdf(s, tslice=slice(150, 500), both=True):
    """Load and process wind probability distribution functions.

    Loads outflow and inflow PDFs, normalizes by volume and area,
    computes reference fluxes from star formation, and extracts
    flux values at specific altitudes (500-3000 pc).

    Parameters
    ----------
    s : LoadSim
        Simulation object; populates s.outflux and s.influx attributes
    """
    pdf_outdir = os.path.join(s.savdir, "windpdf")
    oufpdf_fname = os.path.join(pdf_outdir, "outpdf.nc")
    inpdf_fname = os.path.join(pdf_outdir, "inpdf.nc")

    if not os.path.isfile(oufpdf_fname) or not os.path.isfile(inpdf_fname):
        s.create_windpdf(pdf_outdir=pdf_outdir)

    with xr.open_dataarray(oufpdf_fname) as da:
        s.outpdf = (
            da.sel(flux=["mflux", "eflux", "mflux_Z"]).sel(time=tslice).mean(dim="time")
        )
    with xr.open_dataarray(inpdf_fname) as da:
        s.inpdf = (
            da.sel(flux=["mflux", "eflux", "mflux_Z"]).sel(time=tslice).mean(dim="time")
        )
    zfc = np.linspace(s.domain["le"][2], s.domain["re"][2], s.domain["Nx"][2] + 1)
    zcc = 0.5 * (zfc[1:] + zfc[:-1])
    dnz = len(zcc[(zcc > 950) & (zcc < 1050)])
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
        eflux=sfr_avg / mstar * 1.0e51,
        mZflux=sfr_avg / mstar * Mej * Zsn,
    )
    ref_flux = xr.Dataset(ref_flux).to_array("flux")

    outflux = s.outpdf / np.prod(s.domain["Nx"][:-1]) / dnz
    influx = s.inpdf / np.prod(s.domain["Nx"][:-1]) / dnz

    # mean both sides
    if both:
        zabs = [500, 1000, 2000, 3000]
        outflux_zabs = []
        influx_zabs = []
        for z_ in zabs:
            outflux_zabs.append(
                outflux.sel(z=[-z_, z_]).mean(dim="z").assign_coords(z=z_)
            )
            influx_zabs.append(
                influx.sel(z=[-z_, z_]).mean(dim="z").assign_coords(z=z_)
            )
        s.outflux = xr.concat(outflux_zabs, dim="z")
        s.influx = xr.concat(influx_zabs, dim="z")
    else:
        s.outflux = outflux
        s.influx = influx


def print_sim_table(sims):
    """Print a formatted table of simulation parameters to stdout.

    Generates and prints a LaTeX-formatted table with parameters
    for all simulations, including group classification.

    Parameters
    ----------
    sims : LoadSimAll
        Container with list of simulation models
    """
    line = (
        f"{'name':<40s} & {'physics':<10s} & {'beta':<5.0s} & {'Lz':<10.0s} & {'dx':<5.0s} & "
        f"{'mhdbc':<10s} & {'crbc':<10s} & {'vmax':<10s} & {'sigma':<10s} & {'grav':<10s} \\\\"
    )
    print(line)
    for m in sims.models:
        s = sims.set_model(m)
        name, line, group = get_model_table_line(s)
        print(line + f" -- {group}")


def update_stress(s, dset):
    """Calculate and add stress/pressure components to dataset.

    Computes derived pressure quantities from raw fields:
    - Thermal and kinetic pressure from temperature and turbulence
    - Magnetic pressure tensor components
    - Total pressure

    Parameters
    ----------
    s : LoadSim
        Simulation object with unit information
    dset : xr.Dataset
        Z-profile dataset to update in-place

    Returns
    -------
    xr.Dataset
        Updated dataset with new stress fields
    """
    dset["Pok_th"] = dset["press"] * s.u.pok
    dset["Pok_kin"] = dset["Pturbz"] * s.u.pok
    dset["Pok_tot"] = dset["Pok_th"] + dset["Pok_kin"]
    if s.options["mhd"]:
        dset["Pi_B"] = (dset["Pmag1"] + dset["Pmag2"] - dset["Pmag3"]) * s.u.pok
        dset["Pok_B"] = (dset["Pmag1"] + dset["Pmag2"] + dset["Pmag3"]) * s.u.pok
        dset["Pok_tot"] += dset["Pi_B"]
    if s.options["cosmic_ray"]:
        dset["Pok_cr"] = dset["0-Ec"] / 3.0 * s.u.pok

    # weights
    gzext = np.interp(dset.z, s.extgrav["z"], s.extgrav["gz"])
    dz = s.domain["dx"][2]
    Wsg_from_bot = (dset["rhogz"] * dz).cumsum(dim="z") * s.u.pok
    Wsg_from_top = (-dset["rhogz"] * dz).sel(z=slice(None, None, -1)).cumsum(
        dim="z"
    ).sel(z=slice(None, None, -1)) * s.u.pok
    # Wsg_mean = 0.5 * (Wsg_from_bot + Wsg_from_top)
    dset["Wsg"] = xr.concat(
        (
            Wsg_from_bot.sel(z=slice(s.domain["le"][2], 0)),
            Wsg_from_top.sel(z=slice(0, s.domain["re"][2])),
        ),
        dim="z",
    )
    Wext_from_bot = (dset["rho"] * gzext * dz).cumsum(dim="z") * s.u.pok
    Wext_from_top = (-dset["rho"] * gzext * dz).sel(z=slice(None, None, -1)).cumsum(
        dim="z"
    ).sel(z=slice(None, None, -1)) * s.u.pok
    # Wext_mean = 0.5 * (Wext_from_bot + Wext_from_top)
    dset["Wext"] = xr.concat(
        (
            Wext_from_bot.sel(z=slice(s.domain["le"][2], 0)),
            Wext_from_top.sel(z=slice(0, s.domain["re"][2])),
        ),
        dim="z",
    )
    dset["Wtot"] = dset["Wsg"] + dset["Wext"]

    return dset


def update_flux(s, dset_, vz_dir=None, both=False):
    """Calculate and add flux components to dataset.

    Computes mass, pressure, and energy fluxes in vertical direction.
    Handles optional filtering by vertical velocity direction (inflow/outflow)
    and folding of lower/upper hemisphere data.

    Parameters
    ----------
    s : LoadSim
        Simulation object with domain and unit information
    dset_ : xr.Dataset
        Z-profile dataset with flux input fields
    vz_dir : int or None
        Vertical velocity direction filter (1 for outflow, -1 for inflow, None for both)
    both : bool
        If True, fold and sum both hemispheres (default=False)

    Returns
    -------
    xr.Dataset
        Dataset with computed flux fields
    """
    # fluxes
    # total area
    area = np.prod(s.domain["Lx"][:-1])
    mflux_units = (s.u.density * s.u.velocity).to("Msun/(kpc**2*yr)").value
    pflux_units = (s.u.pressure).to("(Msun*km)/(kpc**2*yr*s)").value
    eflux_units = (s.u.energy_density * s.u.velocity).to("(erg)/(kpc**2*yr)").value
    dset_upper = dset_.sel(z=slice(0, s.domain["re"][2]))
    dset_lower = dset_.sel(z=slice(s.domain["le"][2], 0))
    for zsgn, dset in zip([1, -1], [dset_upper, dset_lower]):
        dset["mflux"] = dset["mom3"] / area * mflux_units * zsgn
        dset["pflux_MHD"] = (dset["Pturbz"] + dset["press"]) / area * pflux_units
        dset["eflux_MHD"] = (
            (
                dset["Ekin_flux1"]
                + dset["Ekin_flux2"]
                + dset["Ekin_flux3"]
                + dset["Eth_flux"]
            )
            / area
            * eflux_units
            * zsgn
        )
        if s.options["mhd"]:
            dset["pflux_MHD"] += (
                (dset["Pmag1"] + dset["Pmag2"] - dset["Pmag3"]) / area * pflux_units
            )
            dset["eflux_MHD"] += (
                (dset["Sz_Bpress"] + dset["Sz_Btens"]) / area * eflux_units * zsgn
            )
        if s.options["cosmic_ray"]:
            vmax_kms = s.par["cr"]["vmax"] / 1.0e5
            dset["pflux_CR"] = dset["0-Ec"] / 3.0 / area * pflux_units
            dset["eflux_CR"] = dset["0-Fc3"] * vmax_kms / area * eflux_units * zsgn
    if vz_dir is not None:
        dset_upper = dset_upper.sel(vz_dir=vz_dir) * vz_dir
        dset_lower = dset_lower.sel(vz_dir=-vz_dir) * vz_dir
    if both:
        # fold lower half
        dset_lower = dset_lower.isel(z=slice(None, None, -1))
        # reassign z coord
        dset_lower = dset_lower.assign_coords(z=dset_lower.z * (-1))

        dset = dset_lower + dset_upper
    else:
        dset = xr.concat([dset_lower, dset_upper], dim="z")

    if vz_dir is None:
        dset = dset.sum(dim="vz_dir")

    return dset
