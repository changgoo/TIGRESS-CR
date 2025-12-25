import struct
import numpy as np
import re
from pyathena.io.read_athinput import read_athinput_from_lines


class AthenaRestartReader:
    def __init__(self, filename, verbose=False):
        self.filename = filename
        self.header_text = ""
        self.verbose = verbose

        self.blocks = []
        self.mesh_size = {}

        # Binary types
        self.real_type = np.float64
        self.int_type = np.int32
        self.real_size = 8
        self.int_size = 4

        # Results storage
        self.user_mesh_data_int = []
        self.user_mesh_data_real = []

        self._read_global_header()

    def _read_global_header(self):
        """Step 1: Read up to ncycle and record the position of User Mesh Data."""
        with open(self.filename, "rb") as f:
            header_accum = b""
            while True:
                chunk = f.read(4096)
                if not chunk:
                    break
                header_accum += chunk
                loc = header_accum.find(b"<par_end>")
                if loc != -1:
                    binary_start = loc + 10
                    break

            self.header_text = header_accum[:loc].decode("utf-8", errors="ignore")
            self.par = read_athinput_from_lines(self.header_text.splitlines())

            f.seek(binary_start)
            self.nbtotal = struct.unpack("<i", f.read(4))[0]
            self.root_level = struct.unpack("<i", f.read(4))[0]

            # RegionSize (108 bytes + 4 bytes padding)
            rs = struct.unpack("<12d3i", f.read(108))
            self.mesh_size = {
                "x1min": rs[0],
                "x2min": rs[1],
                "x3min": rs[2],
                "x1max": rs[3],
                "x2max": rs[4],
                "x3max": rs[5],
                "x1len": rs[6],
                "x2len": rs[7],
                "x3len": rs[8],
                "x1rat": rs[9],
                "x2rat": rs[10],
                "x3rat": rs[11],
                "nx1": rs[12],
                "nx2": rs[13],
                "nx3": rs[14],
            }
            f.read(4)

            self.time = struct.unpack("<d", f.read(8))[0]
            self.dt = struct.unpack("<d", f.read(8))[0]
            self.ncycle = struct.unpack("<i", f.read(4))[0]

            if self.verbose:
                print(f"Time: {self.time:.4f}, Blocks: {self.nbtotal}")
                print(
                    f"Root Grid: {self.mesh_size['nx1']}x{self.mesh_size['nx2']}x{self.mesh_size['nx3']}"
                )
                print(
                    f"Bounds: X1[{self.mesh_size['x1min']:.2f}, {self.mesh_size['x1max']:.2f}]"
                )
            # Store pointer for Step 2
            self._user_data_start_pos = f.tell()
            self._setup_params()

    def _setup_params(self):
        par = self.par
        mesh = par["mesh"]
        meshblock = par["meshblock"]
        isothermal = par["configure"]["Equation_of_state"] == "isothermal"
        self.NHYDRO = 4 if isothermal else 5
        mhd = par["configure"]["Magnetic_fields"] == "ON"
        self.NFIELD = 3 if mhd else 0
        self.NGHOST = par["configure"]["Number_of_ghost_cells"]
        self.PARTICLE = False
        for b in self.par:
            if b.startswith("particle"):
                self.PARTICLE |= self.par[b]["type"] != "none"
        self.NSCALARS = par["configure"]["Number_of_scalars"]
        cosmic_ray = par["configure"]["Cosmic_Ray_Transport"] == "Multigroups"
        self.NCR = 4 if cosmic_ray else 0
        self.NCRG = par["configure"]["Cosmic_Ray_energy_groups"]

        # meshblock data
        nx3, nx2, nx1 = [meshblock["nx3"], meshblock["nx2"], meshblock["nx1"]]
        ncells3, ncells2, ncells1 = (
            nx3 + 2 * self.NGHOST,
            nx2 + 2 * self.NGHOST,
            nx1 + 2 * self.NGHOST,
        )
        mb_size = dict()
        mb_size["cons"] = (self.NHYDRO, ncells3, ncells2, ncells1)
        mb_size["b1f"] = (ncells3, ncells2, ncells1 + 1)
        mb_size["b2f"] = (ncells3, ncells2 + 1, ncells1)
        mb_size["b3f"] = (ncells3 + 1, ncells2, ncells1)
        mb_size["cr"] = (self.NCRG, self.NCR, ncells3, ncells2, ncells1)
        mb_size["scalar"] = (self.NSCALARS, ncells3, ncells2, ncells1)
        self.mb_data_size = mb_size

        # mesh data
        Nx3, Nx2, Nx1 = [mesh["nx3"], mesh["nx2"], mesh["nx1"]]
        Ncells3, Ncells2, Ncells1 = (
            Nx3 + 2 * self.NGHOST,
            Nx2 + 2 * self.NGHOST,
            Nx1 + 2 * self.NGHOST,
        )
        data_size = dict()
        data_size["cons"] = (self.NHYDRO, Ncells3, Ncells2, Ncells1)
        data_size["b1f"] = (Ncells3, Ncells2, Ncells1 + 1)
        data_size["b2f"] = (Ncells3, Ncells2 + 1, Ncells1)
        data_size["b3f"] = (Ncells3 + 1, Ncells2, Ncells1)
        self.data_size = data_size

    def read_user_mesh_data(self, n_int_elements=[], n_real_elements=[]):
        """
        Step 2: Read variable global user data arrays.
        n_int_elements: list of ints, e.g., [100, 50] for two arrays of those lengths.
        n_real_elements: list of ints, e.g., [1000] for one real array.
        """
        with open(self.filename, "rb") as f:
            f.seek(self._user_data_start_pos)

            # 1. Read Integer User Mesh Data
            for n_elements in n_int_elements:
                bytes_to_read = n_elements * self.int_size
                raw = f.read(bytes_to_read)
                self.user_mesh_data_int.append(np.frombuffer(raw, dtype=np.int32))

            # 2. Read Real User Mesh Data
            for n_elements in n_real_elements:
                bytes_to_read = n_elements * self.real_size
                raw = f.read(bytes_to_read)
                self.user_mesh_data_real.append(
                    np.frombuffer(raw, dtype=self.real_type)
                )

            # After User Mesh Data, we reach the Block ID list
            self._id_list_start_pos = f.tell()
            self._parse_block_ids()

    def _parse_block_ids(self):
        """
        Step 2b: Parse the block metadata list.
        LogicalLocation: 3 x int64 (24 bytes) + 1 x int32 (4 bytes) + 4 bytes padding = 32 bytes
        Cost: 1 x double (8 bytes)
        Offset/Size: 1 x uint64 (8 bytes)
        Total entry size = 48 bytes
        """
        self.blocks = []
        # listsize = sizeof(LogicalLocation) + sizeof(double) + sizeof(IOWrapperSizeT)
        # Based on your definition: 32 + 8 + 8 = 48 bytes per block
        entry_size = 48

        with open(self.filename, "rb") as f:
            f.seek(self._id_list_start_pos)
            for i in range(self.nbtotal):
                raw = f.read(entry_size)
                if len(raw) < entry_size:
                    break

                # Unpack LogicalLocation (lx1, lx2, lx3 are q=int64, level is i=int32)
                # We add '4x' to skip the 4 bytes of padding after 'level'
                # to align the next double to 8 bytes.
                lx1, lx2, lx3, level = struct.unpack("<qqqi", raw[:28])

                # Cost starts at offset 32
                cost = struct.unpack("<d", raw[32:40])[0]

                # Block size (IOWrapperSizeT) starts at offset 40
                blk_offset = struct.unpack("<Q", raw[40:48])[0]

                self.blocks.append(
                    {
                        "lx1": lx1,
                        "lx2": lx2,
                        "lx3": lx3,
                        "level": level,
                        "offset": blk_offset,
                        "cost": cost,
                    }
                )

            # The binary data for the first block starts exactly where the ID list ends
            data_pos = f.tell()
            for blk in self.blocks:
                blk["data_offset"] = data_pos
                # We increment by the size reported in the file to stay perfectly aligned
                data_pos += blk["offset"]

    def read_cons_data(self, block_idx, offset=None):
        """Step 3: Read Hydro conserved variables for a specific block index."""
        blk = self.blocks[block_idx]
        # Conserved variables are (Density, M1, M2, M3, Energy) = 5
        # Use header nx1, nx2, nx3 if consistent across blocks
        read_size = np.prod(self.mb_data_size["cons"]) * self.real_size
        if offset is None:
            offset = blk["data_offset"]

        with open(self.filename, "rb") as f:
            f.seek(offset)
            data = np.frombuffer(f.read(read_size), dtype=self.real_type)
            # Athena layout: (variable, k, j, i)
            blk["cons"] = data.reshape(self.mb_data_size["cons"])
            return f.tell()

    def read_field_data(self, block_idx, offset=None):
        """Step 3: Read Hydro conserved variables for a specific block index."""
        blk = self.blocks[block_idx]
        if offset is None:
            offset = (
                blk["data_offset"] + np.prod(self.mb_data_size["cons"]) * self.real_size
            )

        with open(self.filename, "rb") as f:
            bfield = dict()
            for b_ in ["b1f", "b2f", "b3f"]:
                read_size = np.prod(self.mb_data_size[b_]) * self.real_size
                f.seek(offset)
                data = np.frombuffer(f.read(read_size), dtype=self.real_type)
                bfield[b_] = data.reshape(self.mb_data_size[b_])
                offset += read_size
            blk["field"] = bfield
            return f.tell()

    def read_particle_data(self, block_idx, nint=2, nreal=15, offset=None):
        blk = self.blocks[block_idx]
        if offset is None:
            offset = blk["data_offset"]
            offset += np.prod(self.mb_data_size["cons"]) * self.real_size
            if self.NFIELD != 0:
                for b_ in ["b1f", "b2f", "b3f"]:
                    offset += np.prod(self.mb_data_size[b_]) * self.real_size

        particle = dict()
        with open(self.filename, "rb") as f:
            f.seek(offset)
            npar_ = struct.unpack("<i", f.read(4))[0]
            idmax_ = struct.unpack("<i", f.read(4))[0]

            particle["npar"] = npar_
            particle["idmax"] = idmax_

            read_size = npar_ * nint * self.int_size
            intprop = np.frombuffer(f.read(read_size), dtype=self.int_type)

            read_size = npar_ * nreal * self.real_size
            realprop = np.frombuffer(f.read(read_size), dtype=self.real_type)

            if npar_ > 0:
                intprop = intprop.reshape(nint, npar_)
                realprop = realprop.reshape(nreal, npar_)
            particle["intprop"] = intprop
            particle["realprop"] = realprop

            blk["particle"] = particle

            return f.tell()

    def read_cr_data(self, block_idx, offset=None):
        """Step 3: Read Hydro conserved variables for a specific block index."""
        blk = self.blocks[block_idx]
        read_size = np.prod(self.mb_data_size["cr"]) * self.real_size
        if offset is None:
            offset = blk["data_offset"]
            offset += np.prod(self.mb_data_size["cons"]) * self.real_size
            if self.NFIELD != 0:
                for b_ in ["b1f", "b2f", "b3f"]:
                    offset += np.prod(self.mb_data_size[b_]) * self.real_size
            if self.PARTICLE:
                offset = self.read_particle_data(block_idx, offset=offset)

        with open(self.filename, "rb") as f:
            f.seek(offset)
            data = np.frombuffer(f.read(read_size), dtype=self.real_type)
            # Athena layout: (variable, k, j, i)
            blk["cr"] = data.reshape(self.mb_data_size["cr"])
            return f.tell()

    def read_scalar_data(self, block_idx, offset=None):
        """Step 3: Read Hydro conserved variables for a specific block index."""
        blk = self.blocks[block_idx]
        read_size = np.prod(self.mb_data_size["scalar"]) * self.real_size
        if offset is None:
            offset = blk["data_offset"]
            offset += np.prod(self.mb_data_size["cons"]) * self.real_size
            if self.NFIELD != 0:
                for b_ in ["b1f", "b2f", "b3f"]:
                    offset += np.prod(self.mb_data_size[b_]) * self.real_size
            if self.PARTICLE:
                offset = self.read_particle_data(block_idx, offset=offset)
            if self.NCR > 0:
                offset += np.prod(self.mb_data_size["cr"]) * self.real_size

        with open(self.filename, "rb") as f:
            f.seek(offset)
            data = np.frombuffer(f.read(read_size), dtype=self.real_type)
            # Athena layout: (variable, k, j, i)
            blk["scalar"] = data.reshape(self.mb_data_size["scalar"])
            return f.tell()

    def read_block_data(self, block_idx):
        """Step 3: Read Hydro conserved variables for a specific block index."""
        offset = self.read_cons_data(block_idx)
        if self.NFIELD > 0:
            offset = self.read_field_data(block_idx, offset=offset)
        if self.PARTICLE:
            offset = self.read_particle_data(block_idx, offset=offset)
        if self.NCR > 0:
            offset = self.read_cr_data(block_idx, offset=offset)
        if self.NSCALARS > 0:
            offset = self.read_scalar_data(block_idx, offset=offset)
        # init user meshblock data (1 int [3], 4 real [5,11,2+2*NHYDRO+2(MHD)+4(SHEAR),10+4(CR)])
        # random number


# Example usage:
# reader = AthenaRestartReader("my_file.rst")
# # Suppose you defined:
# # 1 int array of size 10 and 1 real array of size 50
# reader.read_user_mesh_data(n_int_elements=[10], n_real_elements=[50])
#
# # Access the data
# print(reader.user_mesh_data_real[0])
# u = reader.read_hydro_data(0)
