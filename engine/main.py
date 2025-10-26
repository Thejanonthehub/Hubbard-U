# main.py
from dataset import HubbardDataset
from models import MLP, RandomForestModel
from training import Trainer
from evaluation import Evaluator

def main():
    # === User input ===
    train_pct = float(input("Enter training size as decimal (e.g., 0.9 for 90%): ").strip())
    u_j_file = "../hubbard_u_j_values.csv"
    features_file = "../features_list.csv"
    alpha = 0.7

    # === Dataset ===
    dataset = HubbardDataset(u_j_file, features_file, train_size=train_pct)

    # === Models ===
    input_size = dataset.X_train.shape[1]
    mlp = MLP(input_size=input_size, hidden_size=128)
    rf = RandomForestModel()

    # === Train MLP ===
    trainer = Trainer(mlp, dataset.train_loader)
    trainer.train(num_epochs=500, patience=50)

    # === Train Random Forest ===
    rf.fit(dataset.X_train, dataset.y_train)
    print(f"RF OOB Score: {rf.model.oob_score_:.3f}")

    # === Evaluate ===
    evaluator = Evaluator(mlp, rf.model, dataset.poly, dataset.scaler, alpha)
    evaluator.evaluate(dataset.X_test, dataset.y_test)

if __name__ == "__main__":
    main()
