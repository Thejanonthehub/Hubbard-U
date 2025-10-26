# ==============================
# Usage example
# ==============================
from feature import Feature2

def main():
    
    api = "7yVAfZcHPJ1CYixiI4v4XgXcEUg31DPm"
    input_csv = "../hubbard_u_j_values.csv"      # your input CSV
    output_csv = "x3_corr4.csv"               # ✅ valid filename
    cif_folder = "./hubbard_structures_cifs" # your CIF folder

    print("🔍 Starting Feature2 execution...\n")
    f = Feature2(
        api_key=api,
        input_csv=input_csv,
        output_csv=output_csv,
        cif_folder=cif_folder,
        flush_interval=1  # save after every row
    )

if __name__ == "__main__":
    main()