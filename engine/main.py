from data_loader import DataModule
from mlp_model import MLP
from rf_model import RFModel
from trainer import Trainer
from predictor import Predictor
from evaluator import Evaluator
from saver import Saver

# Input files
u_j_file = '../hubbard_u_j_values.csv'
features_file = '../features_list.csv'

# Initialize data module (training size will ask user as in original)
data_module = DataModule(u_j_file, features_file)

# Initialize models
mlp_model = MLP(input_size=data_module.X_train.shape[1])
rf_model = RFModel()

# Training
trainer = Trainer(mlp_model, rf_model, data_module)
trainer.train()

# Prediction
predictor = Predictor(mlp_model, rf_model, data_module)

# Evaluation
evaluator = Evaluator(predictor, data_module)
results = evaluator.evaluate()
print(results)

# Save models
saver = Saver(mlp_model, rf_model, data_module)
saver.save()
