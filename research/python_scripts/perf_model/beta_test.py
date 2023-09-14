import sys
from pet_perf_model import BetaModel

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python " + sys.argv[0] + " db_path")
        quit()
    db_path = sys.argv[1]

    beta_model = BetaModel(db_path)
    print(beta_model.query(0, 1, 4))
