import os
import pandas as pd
import numpy as np
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import Element
from tqdm import tqdm
from itertools import combinations
from mendeleev import element as elm
from bond_a import BondAnalyzer2  # Your updated bond analyzer
import math

class Feature2:
    def __init__(self, api_key, input_csv, output_csv, cif_folder, flush_interval=1):
        """
        Initialize Feature2 extraction with resume capability.
        flush_interval: number of rows after which CSV is saved (default: 1 → saves every row)
        """
        self.api_key = api_key
        self.input_csv = input_csv
        self.output_csv = os.path.abspath(output_csv)
        self.cif_folder = cif_folder
        self.flush_interval = flush_interval
        self.mpr = MPRester(api_key)
        self.cnn = CrystalNN()
        
        # Load input CSV
        print(f"🔧 Loading input CSV: {self.input_csv}")
        self.df = pd.read_csv(input_csv)
        print(f"✅ Loaded {len(self.df)} rows from {self.input_csv}\n")

        # Load existing output (resume support)
        if os.path.exists(self.output_csv):
            self.existing_df = pd.read_csv(self.output_csv)
            self.completed_ids = set(self.existing_df["identifier"].astype(str))
            print(f"♻️ Resuming: {len(self.completed_ids)} rows already processed.")
            self.features_list = self.existing_df.to_dict("records")
        else:
            self.completed_ids = set()
            self.features_list = []

        print("🚀 Starting feature extraction...\n")
        self.extract_features()

    def get_atomic_properties(self, symbol):
        """Return basic atomic properties."""
        el = Element(symbol)
        return {
            "electronegativity": getattr(el, "X", None),
            "atomic_radius": getattr(el, "atomic_radius", None),
            "ionization_energy": getattr(el, "ionization_energy", None),
            "electron_affinity": getattr(el, "electron_affinity", None),
            "oxidation_states": getattr(el, "common_oxidation_states", None)
        }
    
    def compute_bond_angles(self, structure, site_index):
        """Compute bond angles for a given site."""
        try:
            neighbors = self.cnn.get_nn_info(structure, site_index)
            coords = [structure[i].coords for i in [site_index] + [n['site_index'] for n in neighbors]]
            angles = []
            origin = coords[0]
            for p1, p2 in combinations(coords[1:], 2):
                v1 = np.array(p1) - np.array(origin)
                v2 = np.array(p2) - np.array(origin)
                cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angles.append(np.arccos(cos_theta) * 180 / np.pi)
            return np.mean(angles) if angles else None, np.std(angles) if angles else None
        except Exception:
            return None, None
    
    def extract_features(self):
        """Extract features with resume and real-time saving."""
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Extracting features"):
            identifier = str(row["identifier"])

            # ✅ Skip if already processed
            if identifier in self.completed_ids:
                continue

            try:
                formula = row["formula"]
                species = row["Species"]
                structure = None
                spg_num, volume_per_atom = None, None

                # 1️⃣ Try Materials Project API
                try:
                    if identifier.startswith("mp-"):
                        docs = self.mpr.materials.summary.search(
                            material_ids=[identifier],
                            fields=["structure", "symmetry", "volume", "nsites"]
                        )
                        if docs:
                            doc = docs[0]
                            structure = doc.structure
                            volume_per_atom = doc.volume / doc.nsites
                except Exception as e:
                    print(f"⚠️ MP lookup failed for {identifier}: {e}")

                # 2️⃣ Fallback to CIF
                if structure is None:
                    cif_path = os.path.join(self.cif_folder, f"{identifier}.cif")
                    if not os.path.exists(cif_path):
                        cif_path = os.path.join(self.cif_folder, f"{formula}.cif")
                    if os.path.exists(cif_path):
                        try:
                            structure = Structure.from_file(cif_path)
                            volume_per_atom = structure.volume / structure.num_sites
                        except Exception as e:
                            print(f"❌ Failed to read CIF for {identifier} ({formula}): {e}")
                            continue
                    else:
                        print(f"⚠️ No structure found for {identifier} ({formula}), skipping...")
                        continue

                # 3️⃣ Symmetry info
                try:
                    sga = SpacegroupAnalyzer(structure, symprec=0.01)
                    spg_symbol = sga.get_space_group_symbol()
                    spg_num = sga.get_space_group_number()
                    bravais = sga.get_lattice_type()
                    lattice = structure.lattice
                    a, b, c = lattice.a, lattice.b, lattice.c
                    alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
                except Exception as e:
                    print(f"⚠️ Symmetry analysis failed for {identifier}: {e}")
                    spg_symbol = spg_num = bravais = None
                    a = b = c = alpha = beta = gamma = None

                # 4️⃣ Average electronegativity
                comp = Composition(formula)
                total_atoms = comp.num_atoms
                weighted_sum = sum(Element(el).X * amt for el, amt in comp.get_el_amt_dict().items() if Element(el).X)
                avg_en = weighted_sum / total_atoms

                # 5️⃣ Atomic properties
                atomic_props = self.get_atomic_properties(species)

                # 6️⃣ NN analysis
                try:
                    analyzer = BondAnalyzer2(cif_file=None, structure=structure, tolerance=0.4, n_shortest=4)
                    final_dfs, avg_nn_bl_dict, neighbor_summary_all, neighbor_counts_all = analyzer.analyze([species])
                    avg_l_for_species = avg_nn_bl_dict.get(species, None)
                except Exception as e:
                    print(f"⚠️ NN analysis failed for {species}: {e}")
                    avg_l_for_species = None
                    neighbor_counts_all = {}

                # 7️⃣ Sum of neighbor atomic numbers
                try:
                    s_stm_v = 0
                    if neighbor_counts_all and species in neighbor_counts_all:
                        atm_no_nn = {}
                        for atom, counts in neighbor_counts_all[species].items():
                            tot_atm_no = sum(elm(e).atomic_number * count for e, count in counts.items())
                            atm_no_nn[atom] = tot_atm_no
                        s_stm_v = sum(atm_no_nn.values())
                except Exception as e:
                    print(f"⚠️ Neighbor atomic number calc failed: {e}")
                    s_stm_v = None

                # 8️⃣ Covalent radii
                c_radii = {el.symbol: elm(el.symbol).covalent_radius for el in comp.elements}
                c_r_s = c_radii.get(species, None)

                # 9️⃣ Save features
                feature_dict = {
                    "identifier": identifier,
                    "formula": formula,
                    "Species": species,
                    "spacegroup_number": spg_num,
                    "spacegroup_symbol": spg_symbol,
                    "volume_per_atom": volume_per_atom,
                    "bravais_lattice": bravais,
                    "lattice_a": a,
                    "lattice_b": b,
                    "lattice_c": c,
                    "lattice_alpha": alpha,
                    "lattice_beta": beta,
                    "lattice_gamma": gamma,
                    "avg_nn_dist": avg_l_for_species,
                    "sum_atn_nn": s_stm_v,
                    "avg_en_formula": avg_en,
                    "electronegativity": atomic_props["electronegativity"],
                    "ionization_energy": atomic_props["ionization_energy"],
                    "electron_affinity": atomic_props["electron_affinity"],
                    "oxidation_states": str(atomic_props["oxidation_states"]),
                }

                self.features_list.append(feature_dict)
                self.completed_ids.add(identifier)

                # 🔄 Incremental save
                if len(self.features_list) % self.flush_interval == 0:
                    pd.DataFrame(self.features_list).to_csv(self.output_csv, index=False)
                    print(f"💾 Progress saved → {self.output_csv} (rows: {len(self.features_list)})")

            except Exception as e:
                print(f"⚠️ Error processing {identifier}: {e}")

        # Final save
        pd.DataFrame(self.features_list).to_csv(self.output_csv, index=False)
        print(f"\n✅ Feature extraction completed. Total rows: {len(self.features_list)}")
        print(f"🔸 CSV saved at: {self.output_csv}")

