import sys
from pet_perf_model import AlphaModel

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python " + sys.argv[0] + " db_path")
        quit()
    db_path = sys.argv[1]

    alpha_model = AlphaModel(db_path)
    print(alpha_model.query(3, 1))
