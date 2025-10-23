from pymatgen.core import Structure
from mendeleev import element
import itertools
import pandas as pd
from collections import Counter
import re

class BondAnalyzer2:
    def __init__(self, cif_file=None, structure=None, tolerance=0.4, n_shortest=4):
        """
        Initialize the bond analyzer.
        Parameters:
            cif_file (str): Path to the CIF file.
            structure (Structure): Optional pymatgen Structure object.
            tolerance (float): Bond length tolerance.
            n_shortest (int): Number of shortest bonds to consider.
        """
        if cif_file is not None:
            try:
                self.cif_file = cif_file
                self.struct = Structure.from_file(cif_file)
            except Exception as e:
                print(f"⚠️ Reading CIF file failed: {e}")
        elif structure is not None:
            self.struct = structure
        else:
            raise ValueError("You must provide either a CIF file or a pymatgen Structure object.")

        self.tolerance = tolerance
        self.n_shortest = n_shortest

        # Metals list
        self.metals = [
            "Li", "Be", "Na", "Mg", "K", "Ca", "Al", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Ag", "Au", "Ga", "In", "Sn", "Tl", "Pb", "Bi",
            "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
            "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"
        ]

        # Assign atom labels and generate bonds once
        self.atom_labels = self._assign_atom_labels()
        self.bonds_list = self._generate_bonds()

    def _assign_atom_labels(self):
        """Assign numbered labels like Ag1, Ag2, O1, etc."""
        element_counts = {}
        labels = []
        for site in self.struct:
            symbol = site.specie.symbol
            element_counts[symbol] = element_counts.get(symbol, 0) + 1
            labels.append(f"{symbol}{element_counts[symbol]}")
        return labels

    def _generate_bonds(self):
        """Generate all possible bonds using fast pairwise iteration."""
        bonds_list = []
        seen = set()

        # Use pymatgen's get_sites_in_sphere style by computing all distances once
        coords = [site.coords for site in self.struct]

        for i, j in itertools.combinations(range(len(self.struct)), 2):
            label_i, label_j = self.atom_labels[i], self.atom_labels[j]
            d = self.struct[i].distance(self.struct[j])
            d_round = round(d, 3)

            el_i = element(self.struct[i].specie.symbol)
            el_j = element(self.struct[j].specie.symbol)

            r_i = el_i.metallic_radius if el_i.symbol in self.metals else el_i.covalent_radius
            r_j = el_j.metallic_radius if el_j.symbol in self.metals else el_j.covalent_radius
            if r_i is None or r_j is None:
                continue

            if d <= (r_i + r_j + self.tolerance):
                # Determine bond type
                if el_i.symbol in self.metals and el_j.symbol in self.metals:
                    bond_type = "metallic"
                else:
                    en_i = el_i.electronegativity("pauling")
                    en_j = el_j.electronegativity("pauling")
                    if en_i is None or en_j is None:
                        bond_type = "unknown"
                    else:
                        delta_en = abs(en_i - en_j)
                        if delta_en < 0.4:
                            bond_type = "covalent"
                        elif delta_en < 1.7:
                            bond_type = "polar covalent"
                        else:
                            bond_type = "ionic"

                key = tuple(sorted([label_i, label_j]))
                if key in seen:
                    continue
                seen.add(key)

                bonds_list.append({
                    "Atom 1": label_i,
                    "Atom 2": label_j,
                    "Distance (Å)": d_round,
                    "Bond type": bond_type
                })
        return bonds_list

    def analyze(self, target_species):
        """
        Analyze shortest bonds for one or more target species.
        - Exact atom (Li1) → only that atom
        - Element-only (Li) → all atoms of that element
        """
        df = pd.DataFrame(self.bonds_list)
        if df.empty:
            print("⚠️ No bonds found. Check tolerance or CIF structure.")
            return {}, {}, {}, {}

        final_dfs = {}
        avg_nn_bl_dict = {}
        neighbor_summary_all = {}
        neighbor_counts_all = {}

        for species in target_species:
            results = []
            neighbor_summary = {}
            has_digit = any(char.isdigit() for char in species)

            for atom in df["Atom 1"].unique():
                if (has_digit and atom == species) or (not has_digit and re.match(rf"^{species}\d+$", atom)):
                    atom_bonds = df[(df["Atom 1"] == atom) | (df["Atom 2"] == atom)]
                    shortest = atom_bonds.nsmallest(self.n_shortest, "Distance (Å)")
                    results.append(shortest)

                    neighbors = []
                    for _, row in shortest.iterrows():
                        other_atom = row["Atom 2"] if row["Atom 1"] == atom else row["Atom 1"]
                        neighbors.append(other_atom)
                    neighbor_summary[atom] = neighbors

            if not results:
                print(f"⚠️ No bonds found for species {species}.")
                continue

            final_df = pd.concat(results).drop_duplicates().reset_index(drop=True)
            avg_nn_bl = final_df["Distance (Å)"].mean()
            avg_nn_bl_dict[species] = avg_nn_bl

            neighbor_counts = {}
            for atom, neigh_list in neighbor_summary.items():
                symbols = [''.join(filter(str.isalpha, x)) for x in neigh_list]
                neighbor_counts[atom] = dict(Counter(symbols))

            output_file = f"shortest_bonds_{species}.csv"
            final_df.to_csv(output_file, index=False)

            final_dfs[species] = final_df
            neighbor_summary_all[species] = neighbor_summary
            neighbor_counts_all[species] = neighbor_counts

        # Print grouped summary
        print("\n🔹 Average nearest-neighbor bond lengths:")
        grouped_summary = {}
        for sp in avg_nn_bl_dict:
            base_el = ''.join(filter(str.isalpha, sp))
            grouped_summary.setdefault(base_el, []).append((sp, avg_nn_bl_dict[sp]))

        for el, values in grouped_summary.items():
            print(f"\n  🧩 Element: {el}")
            for sp, avg_val in values:
                print(f"     {sp:<8} → {avg_val:.3f} Å")

        return final_dfs, avg_nn_bl_dict, neighbor_summary_all, neighbor_counts_all
