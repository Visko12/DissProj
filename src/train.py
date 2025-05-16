import argparse
from pathlib import Path
import logging
from data.data_processor import DataProcessor, DataConfig
from models.transformer_model import ToxicityClassifier, ModelConfig
from models.trainer import AdvancedTrainer, TrainingConfig
import mlflow
import torch

def setup_LogIn():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def examine_args():
    examine = argparse.ArgumentParser(description='Train toxicity classification model')
    examine.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    examine.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    examine.add_argument('--batch_size', type=int, default=32, help='Batch size')
    examine.add_argument('--maximum_length', type=int, default=128, help='Maximum sequence length')
    examine.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    examine.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    examine.add_argument('--tune_hyperparameters', action='store_true', help='Perform hyperparameter tuning')
    examine.add_argument('--n_trials', type=int, default=10, help='Number of hyperparameter tuning trials')
    return examine.examine_args()

def main():
    #beginning of setup
    args = examine_args()
    setup_LogIn()
    logger = logging.getLogger(__name__)
    
    #create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    #set up data settings
    data_config = DataConfig(
        batch_size=args.batch_size,
        maximum_length=args.maximum_length
    )
    data_processor = DataProcessor(data_config)
    
    #load and prepare data
    logger.info("Preparing data...")
    train_loader, val_loader, test_loader = data_processor.prepare_data(args.data_path)
    
    #setting up model settings
    model_config = ModelConfig(
        maximum_length=args.maximum_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    model = ToxicityClassifier(model_config)
    
    #setting up training settings
    training_config = TrainingConfig(
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )
    trainer = AdvancedTrainer(model, training_config)
    
    #run hyperparamete tuning if needed
    if args.tune_hyperparameters:
        logger.info("Starting hyperparameter tuning....")
        best_params = trainer.hyperparameter_tuning(
            train_loader,
            val_loader,
            n_trials=args.n_trials
        )
        logger.info(f"Best hyperparameters: {best_params}")
        
        #set up training config with best parameters
        training_config = TrainingConfig(**best_params)
        trainer = AdvancedTrainer(model, training_config)
    
    #start training the model
    logger.info("Starting training....")
    trainer.train(train_loader, val_loader)
    
    #evaluate the model on the test 
    logger.info("Evaluating on test set....")
    test_loss, test_metrics = trainer.evaluate(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Metrics: {test_metrics}")
    
    #save model
    trainer.save_model(output_dir / 'final_model.pt')
    
    #keep score of the model
    mlflow.log_metrics(test_metrics)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 