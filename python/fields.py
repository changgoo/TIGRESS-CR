"""
pyathena.fields.fields

Registry of derived fields and their plotting metadata.

Each `set_derived_fields_*` function builds and returns six dictionaries:
  - func[name]      : callable(d, u) -> array-like
  - field_dep[name] : set/list of primitive fields required for func[name]
  - label[name]     : label string (typically LaTeX) for colorbars/axes
  - cmap[name]      : colormap (string or matplotlib/cmasher colormap)
  - vminmax[name]   : (vmin, vmax) defaults for visualization
  - take_log[name]  : bool; if True use LogNorm, else Normalize
"""

import matplotlib as mpl
import numpy as np
import xarray as xr
import astropy.constants as ac
import astropy.units as au
import cmasher as cmr

from matplotlib.colors import Normalize, LogNorm


def set_derived_fields_user(par):
    func = dict()
    field_dep = dict()
    label = dict()
    cmap = dict()
    vminmax = dict()
    take_log = dict()

    gamma_cr = 4.0 / 3.0
    vmax = par["cr"]["vmax"]  # in cm/s

    # cr advection + streaming
    f = "Vtotz"
    field_dep[f] = set(["velocity", "0-Vs3"])

    def _Vtotz(d, u):
        return (d["velocity3"] + d["0-Vs3"]) * u.kms

    func[f] = _Vtotz
    label[f] = r"$v_{\rm tot,z}\;[{\rm km\,s^{-1}}]$"
    cmap[f] = "bwr"
    vminmax[f] = (-100.0, 100.0)
    take_log[f] = False

    # veff
    f = "0-Veff3"
    field_dep[f] = set(['0-Ec','0-Fc3'])

    def _Veffz(d, u):
        vmax_ = vmax/(u.cm/u.s)
        return d["0-Fc3"]*vmax_/(gamma_cr*d['0-Ec'])*u.kms

    func[f] = _Veffz
    label[f] = r"$v_{\rm eff,z}\;[{\rm km\,s^{-1}}]$"
    cmap[f] = "bwr"
    vminmax[f] = (-100.0, 100.0)
    take_log[f] = False

    # kappa_para
    f = "kappa_para"
    field_dep[f] = set(['0-Sigma_diff1'])

    def _kappa_para(d, u):
        vmax_ = vmax/(u.cm/u.s)
        kappa = vmax_/d["0-Sigma_diff1"]
        return kappa * (u.cm**2 / u.s)

    func[f] = _kappa_para
    label[f] = r"$\kappa_{\parallel}\;[{\rm cm^{2}\,s^{-1}}]$"
    cmap[f] = "viridis"
    vminmax[f] = [1.e27,1.e30]
    take_log[f] = True

    return func, field_dep, label, cmap, vminmax, take_log


def add_fields(self, dfi):
    par = self.par
    func, field_dep, label, cmap, vminmax, take_log = set_derived_fields_user(par)

    # Set colormap normalization and scale
    norm = dict()
    scale = dict()
    imshow_args = dict()
    for f in func:
        if take_log[f]:
            norm[f] = LogNorm(*vminmax[f])
            scale[f] = "log"
        else:
            norm[f] = Normalize(*vminmax[f])
            scale[f] = "linear"
        imshow_args[f] = dict(
            norm=norm[f], cmap=cmap[f], cbar_kwargs=dict(label=label[f])
        )

    for f in func:
        dfi.dfi[f] = dict(
            field_dep=field_dep[f],
            func=func[f],
            label=label[f],
            norm=norm[f],
            vminmax=vminmax[f],
            cmap=cmap[f],
            scale=scale[f],
            take_log=take_log[f],
            imshow_args=imshow_args[f],
        )
