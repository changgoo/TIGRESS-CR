# slc_prj.py

import os
import os.path as osp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr
from matplotlib import cm

# import astropy.units as au
# import astropy.constants as ac
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
import xarray as xr

from pyathena.load_sim import LoadSim


cpp_to_cc = {
    "rho": "density",
    "press": "pressure",
    "vel1": "velocity1",
    "vel2": "velocity2",
    "vel3": "velocity3",
    "Bcc1": "cell_centered_B1",
    "Bcc2": "cell_centered_B2",
    "Bcc3": "cell_centered_B3",
    "rHI": "xHI",
    "rH2": "xH2",
    "rEL": "xe",
}


class PDF:
    def get_pdf(
        self,
        dchunk,
        xf,
        yf,
        wf,
        xlim,
        ylim,
        Nx=128,
        Ny=128,
        logx=False,
        logy=False,
        phase=None,
    ):
        try:
            xdata = dchunk[xf]
        except KeyError:
            xdata = self.dfi(dchunk, self.u)
        try:
            ydata = dchunk[yf]
        except KeyError:
            ydata = self.dfi(dchunk, self.u)
        if wf is not None:
            try:
                wdata = dchunk[wf]
            except KeyError:
                wdata = self.dfi(dchunk, self.u)
            if phase is not None:
                wdata = wdata * phase.data.flatten()
            name = f"{wf}-pdf"
        else:
            name = "vol-pdf"

        if logx:
            xdata = np.log10(np.abs(xdata))
            xf = f"log_{xf}"
        if logy:
            ydata = np.log10(np.abs(ydata))
            yf = f"log_{yf}"

        b1 = np.linspace(xlim[0], xlim[1], Nx)
        b2 = np.linspace(ylim[0], ylim[1], Ny)
        h, b1, b2 = np.histogram2d(
            xdata.data.flatten(),
            ydata.data.flatten(),
            weights=wdata.data.flatten() if wf is not None else None,
            bins=[b1, b2],
        )
        dx = b1[1] - b1[0]
        dy = b2[1] - b2[0]
        pdf = h.T / dx / dy
        da = xr.DataArray(
            pdf,
            coords=[0.5 * (b2[1:] + b2[:-1]), 0.5 * (b1[1:] + b1[:-1])],
            dims=[yf, xf],
            name=name,
        )
        return da

    @LoadSim.Decorators.check_netcdf
    def get_jointpdf(
        self,
        num,
        prefix,
        savdir=None,
        force_override=False,
        filebase=None,
        xf="nH",
        yf="pok",
        xlim=(-10, 10),
        ylim=(-10, 10),
        wlist=[None, "nH"],
        Nx=256,
        Ny=256,
        logx=True,
        logy=True,
        zslice=None,
        outid=None,
        dryrun=False,
    ):
        """
        a warpper function to make data reading easier
        """
        ds = self.load_hdf5(num=num, outid=outid, file_only=True)

        if dryrun:
            return osp.getmtime(self.fhdf5)
            # return max(osp.getmtime(self.fhdf5),osp.getmtime(__file__))

        data = self.get_data(num, outid=outid, load_derived=True)

        dset = xr.Dataset()

        data = data[[xf, yf] + wlist[1:]].load()
        if zslice is not None:
            data = data.sel(z=zslice)
        for wf in wlist:
            if wf is None:
                total = np.prod(self.domain["Nx"])
                coord_dict = {"vol": total}
            else:
                total = data[wf].sum().data
                coord_dict = {wf: total}
            da = self.get_pdf(
                data, xf, yf, wf, xlim, ylim, Nx=Nx, Ny=Ny, logx=logx, logy=logy
            )
            dset[da.name] = da
            dset = dset.assign_coords(coord_dict)
        return dset.assign_coords(time=data.attrs["Time"])

    def get_nPpdf(self, num):
        dset = self.get_jointpdf(num, "pdf", filebase="nP")
        return dset

    def get_nTpdf(self, num):
        dset = self.get_jointpdf(
            num, "pdf", filebase="nT", yf="T", ylim=(0, 10), Ny=128
        )
        return dset

    def get_crpdf(self, num, force_override=False):
        xf = "T"
        for yf in ["velocity3", "0-Vs3", "0-Vd3", "0-Veff3", "Vtotz"]:
            pdf = self.get_jointpdf(
                num,
                "pdf",
                filebase="-".join([xf, yf]),
                xf=xf,
                yf=yf,
                wlist=[None, "nH", "pok_cr"],
                xlim=(1, 9),
                ylim=(-1, 4),
                force_override=force_override,
            )
        for yf in ["kappa_para"]:
            pdf = self.get_jointpdf(
                num,
                "pdf",
                filebase="-".join([xf, yf]),
                xf=xf,
                yf=yf,
                wlist=[None, "nH", "pok_cr"],
                xlim=(1, 9),
                ylim=(25, 35),
                force_override=force_override,
            )
        for yf in ["sigma_para"]:
            pdf = self.get_jointpdf(
                num,
                "pdf",
                filebase="-".join([xf, yf]),
                xf=xf,
                yf=yf,
                wlist=[None, "nH", "pok_cr"],
                xlim=(1, 9),
                ylim=(-35, -25),
                force_override=force_override,
            )
        xf = "nH"
        for yf in ["kappa_para"]:
            pdf = self.get_jointpdf(
                num,
                "pdf",
                filebase="-".join([xf, yf]),
                xf=xf,
                yf=yf,
                wlist=[None, "nH", "pok_cr"],
                xlim=(-6, 4),
                ylim=(25, 35),
                force_override=force_override,
            )
        for yf in ["sigma_para"]:
            pdf = self.get_jointpdf(
                num,
                "pdf",
                filebase="-".join([xf, yf]),
                xf=xf,
                yf=yf,
                wlist=[None, "nH", "pok_cr"],
                xlim=(-6, 4),
                ylim=(-35, -25),
                force_override=force_override,
            )

    @LoadSim.Decorators.check_netcdf
    def get_windpdf(
        self,
        num,
        prefix,
        savdir=None,
        force_override=False,
        filebase=None,
        dryrun=False,
        zlist=[-3000, -2000, -1000, -500, 0, 500, 1000, 2000, 3000],
        dz=50,
    ):
        ds = self.load_hdf5(num=num, file_only=True)

        if dryrun:
            return osp.getmtime(self.fhdf5)
            # return max(osp.getmtime(self.fhdf5),osp.getmtime(__file__))

        ds = self.get_data(num, load_derived=False)

        pdf = {"out": xr.Dataset(), "in": xr.Dataset()}

        bin = np.logspace(0, 4, 201)
        dbin = np.log10(bin[1] / bin[0])
        bcc = np.log10(bin)[:-1] + dbin
        mfunits = (self.u.density * self.u.velocity).to("Msun/(kpc**2*yr)").value
        pfunits = (self.u.energy_density).to("(Msun*km)/(kpc**2*yr*s)").value
        efunits = (self.u.energy_density * self.u.velocity).to("erg/(kpc**2*yr)").value
        for z0 in zlist:
            ds_sel = ds.sel(z=slice(z0 - dz, z0 + dz)).stack(xyz=["x", "y", "z"])
            cs = np.sqrt(ds_sel["press"] / ds_sel["rho"])
            zsgn = ds_sel.z / np.abs(ds_sel.z)
            vout = ds_sel["vel3"] * zsgn
            vol = ds_sel["rho"] / ds_sel["rho"]
            rho = ds_sel["rho"]
            press = ds_sel["press"]

            # mass flux
            mflux_out = rho * vout
            vsq = ds_sel["vel1"] ** 2 + ds_sel["vel2"] ** 2 + ds_sel["vel3"] ** 2
            csq = self.par["hydro"]["gamma"] / (self.par["hydro"]["gamma"] - 1) * cs**2
            vBsq = vsq + csq
            # momentum flux
            pflux_kin = mflux_out * vout
            pflux_th = press
            pflux = pflux_kin + pflux_th
            if self.options["mhd"]:
                pflux_mag = 0.5 * (
                    ds_sel["Bcc1"] ** 2 + ds_sel["Bcc2"] ** 2 - ds_sel["Bcc3"] ** 2
                )
                pflux += pflux_mag

            # energy flux
            eflux_kin = 0.5 * mflux_out * vsq
            eflux_th = mflux_out * csq
            eflux = eflux_kin + eflux_th
            if self.options["mhd"]:
                Bout = ds_sel["Bcc3"] * zsgn
                vAsq = (
                    ds_sel["Bcc1"] ** 2 + ds_sel["Bcc2"] ** 2 + ds_sel["Bcc3"] ** 2
                ) / rho
                eflux_mag = mflux_out * vAsq
                eflux_magt = (
                    ds_sel["Bcc1"] * ds_sel["vel1"]
                    + ds_sel["Bcc2"] * ds_sel["vel2"]
                    + ds_sel["Bcc3"] * ds_sel["vel3"]
                ) * Bout
                eflux += eflux_mag - eflux_magt

            # metal flux
            mZflux_out = mflux_out * ds_sel["rmetal"]
            fluxlist = {
                "vol": vol,
                "mass": rho,
                "mflux": mflux_out * mfunits,
                "mflux_Z": mZflux_out * mfunits,
                "pflux": pflux * pfunits,
                "eflux": eflux * efunits,
                "pflux_kin": pflux_kin * pfunits,
                "pflux_th": pflux_th * pfunits,
                "eflux_kin": eflux_kin * efunits,
                "eflux_th": eflux_th * efunits,
            }
            if self.options["mhd"]:
                fluxlist["eflux_mag"] = (eflux_mag - eflux_magt) * efunits
                fluxlist["eflux_magp"] = eflux_mag * efunits
                fluxlist["eflux_magt"] = eflux_magt * efunits
                fluxlist["pflux_mag"] = pflux_mag * pfunits
            outpdflist = []
            inpdflist = []
            for f in fluxlist:
                outpdf, _, _ = np.histogram2d(
                    vout, cs, bins=[bin, bin], weights=fluxlist[f]
                )
                vsgn = -1 if ("mflux" in f) or ("eflux" in f) else 1
                inpdf, _, _ = np.histogram2d(
                    -vout, cs, bins=[bin, bin], weights=vsgn * fluxlist[f]
                )

                outpdf = xr.DataArray(
                    outpdf.T, dims=["logcs", "logvz"], coords=[bcc, bcc]
                )
                inpdf = xr.DataArray(
                    inpdf.T, dims=["logcs", "logvz"], coords=[bcc, bcc]
                )
                outpdflist.append(outpdf.assign_coords(flux=f))
                inpdflist.append(inpdf.assign_coords(flux=f))
            pdf["out"][z0] = xr.concat(outpdflist, dim="flux")
            pdf["in"][z0] = xr.concat(inpdflist, dim="flux")

        pdf["out"] = pdf["out"].to_array("z")
        pdf["in"] = pdf["in"].to_array("z")

        return xr.Dataset(pdf).assign_coords(time=ds.attrs["Time"])

    def create_windpdf(self, pdf_outdir=None):
        self.logger.info("Merged wind pdfs are not found. Creating new wind pdfs...")

        outpdf = []
        inpdf = []
        for num in self.nums:
            pdf = self.get_windpdf(num, "windpdf")
            outpdf.append(pdf["out"])
            inpdf.append(pdf["in"])
        if pdf_outdir is None:
            pdf_outdir = os.path.join(self.savdir, "windpdf")
        xr.concat(outpdf, dim="time").to_netcdf(os.path.join(pdf_outdir, "outpdf.nc"))
        xr.concat(inpdf, dim="time").to_netcdf(os.path.join(pdf_outdir, "inpdf.nc"))
