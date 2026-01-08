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
        if self.NFIELD > 0:
            mb_size["b1f"] = (ncells3, ncells2, ncells1 + 1)
            mb_size["b2f"] = (ncells3, ncells2 + 1, ncells1)
            mb_size["b3f"] = (ncells3 + 1, ncells2, ncells1)
        if self.NCR > 0:
            mb_size["cr"] = (self.NCRG, self.NCR, ncells3, ncells2, ncells1)
        if self.NSCALARS > 0:
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
        if self.NFIELD > 0:
            data_size["b1f"] = (Ncells3, Ncells2, Ncells1 + 1)
            data_size["b2f"] = (Ncells3, Ncells2 + 1, Ncells1)
            data_size["b3f"] = (Ncells3 + 1, Ncells2, Ncells1)
        if self.NCR > 0:
            data_size["cr"] = (self.NCRG, self.NCR, Ncells3, Ncells2, Ncells1)
        if self.NSCALARS > 0:
            data_size["scalar"] = (self.NSCALARS, Ncells3, Ncells2, Ncells1)
        self.data_size = data_size
        self.mesh = mesh
        self.meshblock = meshblock

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
            int_raw = f.read(npar_ * nint * self.int_size)
            intprop = np.frombuffer(int_raw, dtype=self.int_type)

            real_raw = f.read(npar_ * nreal * self.real_size)
            realprop = np.frombuffer(real_raw, dtype=self.real_type)

            if npar_ > 0:
                intprop = intprop.reshape(nint, npar_)
                realprop = realprop.reshape(nreal, npar_)
            else:
                # Ensure consistent 2D shape when there are zero particles
                intprop = np.empty((nint, 0), dtype=self.int_type)
                realprop = np.empty((nreal, 0), dtype=self.real_type)
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

    def set_global_array(self):
        self.data = dict()
        self.data["cons"] = np.zeros(self.data_size["cons"], dtype=self.real_type)
        if self.NFIELD > 0:
            self.data["b1f"] = np.zeros(self.data_size["b1f"], dtype=self.real_type)
            self.data["b2f"] = np.zeros(self.data_size["b2f"], dtype=self.real_type)
            self.data["b3f"] = np.zeros(self.data_size["b3f"], dtype=self.real_type)
        if self.NCR > 0:
            self.data["cr"] = np.zeros(self.data_size["cr"], dtype=self.real_type)
        if self.NSCALARS > 0:
            self.data["scalar"] = np.zeros(
                self.data_size["scalar"], dtype=self.real_type
            )

    def fill_global_array(self, block_idx):
        blk = self.blocks[block_idx]
        mb = self.meshblock
        if not hasattr(self, "data"):
            self.set_global_array()

        nmb1 = int(self.mesh["nx1"] / self.meshblock["nx1"])
        nmb2 = int(self.mesh["nx2"] / self.meshblock["nx2"])
        nmb3 = int(self.mesh["nx3"] / self.meshblock["nx3"])

        left = blk["lx1"] == 0
        right = blk["lx1"] == (nmb1 - 1)
        bottom = blk["lx2"] == 0
        top = blk["lx2"] == (nmb2 - 1)
        front = blk["lx3"] == 0
        back = blk["lx3"] == (nmb3 - 1)

        # local indices of active cells
        lis = 0 if left else self.NGHOST
        lie = mb["nx1"] + 2 * self.NGHOST
        lie -= 0 if right else self.NGHOST
        ljs = 0 if bottom else self.NGHOST
        lje = mb["nx2"] + 2 * self.NGHOST
        lje -= 0 if top else self.NGHOST
        lks = 0 if front else self.NGHOST
        lke = mb["nx3"] + 2 * self.NGHOST
        lke -= 0 if back else self.NGHOST

        # corresponding global indices of active cells
        gis = blk["lx1"] * mb["nx1"] + lis
        gie = blk["lx1"] * mb["nx1"] + lie
        gjs = blk["lx2"] * mb["nx2"] + ljs
        gje = blk["lx2"] * mb["nx2"] + lje
        gks = blk["lx3"] * mb["nx3"] + lks
        gke = blk["lx3"] * mb["nx3"] + lke

        self.data["cons"][:, gks:gke, gjs:gje, gis:gie] = blk["cons"][
            :, lks:lke, ljs:lje, lis:lie
        ]
        if self.NFIELD > 0:
            self.data["b1f"][gks:gke, gjs:gje, gis : gie + 1] = blk["field"]["b1f"][
                lks:lke, ljs:lje, lis : lie + 1
            ]
            self.data["b2f"][gks:gke, gjs : gje + 1, gis:gie] = blk["field"]["b2f"][
                lks:lke, ljs : lje + 1, lis:lie
            ]
            self.data["b3f"][gks : gke + 1, gjs:gje, gis:gie] = blk["field"]["b3f"][
                lks : lke + 1, ljs:lje, lis:lie
            ]
        if self.NCR > 0:
            self.data["cr"][:, :, gks:gke, gjs:gje, gis:gie] = blk["cr"][
                :, :, lks:lke, ljs:lje, lis:lie
            ]
        if self.NSCALARS > 0:
            self.data["scalar"][:, gks:gke, gjs:gje, gis:gie] = blk["scalar"][
                :, lks:lke, ljs:lje, lis:lie
            ]

    def merge_particle(self):
        """Step 4b: Merge particle data from all blocks."""
        all_particles_int = []
        all_particles_real = []
        for blk in self.blocks:
            if "particle" in blk:
                all_particles_int.append(blk["particle"]["intprop"])
                all_particles_real.append(blk["particle"]["realprop"])
        if all_particles_int:
            self.particle_int = np.hstack(all_particles_int)
        else:
            self.particle_int = np.array([], dtype=self.int_type).reshape(0, 0)
        if all_particles_real:
            self.particle_real = np.hstack(all_particles_real)
        else:
            self.particle_real = np.array([], dtype=self.real_type).reshape(0, 0)

    def read_data(self):
        """Step 4: Read all block data and fill global arrays."""
        for i in range(self.nbtotal):
            self.read_block_data(i)
            self.fill_global_array(i)
        self.merge_particle()


# Example usage:
# reader = AthenaRestartReader("my_file.rst")
# # Suppose you defined:
# # 1 int array of size 10 and 1 real array of size 50
# reader.read_user_mesh_data(n_int_elements=[10], n_real_elements=[50])
#
# # Access the data
# print(reader.user_mesh_data_real[0])
# u = reader.read_hydro_data(0)
