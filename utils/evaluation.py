import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class ModelEvaluator:
    @staticmethod
    def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, epochs=10, batch_size=32):
        """Evaluate a Keras model"""
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        # Make predictions
        y_pred_test = model.predict(X_test)
        y_pred_val = model.predict(X_val)
        
        # Calculate metrics
        r2_test = r2_score(y_test, y_pred_test)
        r2_val = r2_score(y_val, y_pred_val)
        mse_test = mean_squared_error(y_test, y_pred_test)
        mse_val = mean_squared_error(y_val, y_pred_val)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        mae_val = mean_absolute_error(y_val, y_pred_val)
        
        # Get loss values
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        val_loss = model.evaluate(X_val, y_val, verbose=0)
        
        return {
            'history': history,
            'r2_test': r2_test,
            'r2_val': r2_val,
            'mse_test': mse_test,
            'mse_val': mse_val,
            'mae_test': mae_test,
            'mae_val': mae_val,
            'test_loss': test_loss,
            'val_loss': val_loss,
            'predictions': {
                'y_pred_test': y_pred_test,
                'y_pred_val': y_pred_val
            }
        }
    
    @staticmethod
    def plot_training_history(history, title="Model Training History"):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{title} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def print_evaluation_results(results, model_name):
        """Print evaluation results in a formatted way"""
        print(f"\n{'='*50}")
        print(f"{model_name} EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"R² Score (Test):      {results['r2_test']:.4f}")
        print(f"R² Score (Val):       {results['r2_val']:.4f}")
        print(f"MSE (Test):           {results['mse_test']:.6f}")
        print(f"MSE (Val):            {results['mse_val']:.6f}")
        print(f"MAE (Test):           {results['mae_test']:.6f}")
        print(f"MAE (Val):            {results['mae_val']:.6f}")
        print(f"Test Loss:            {results['test_loss']:.6f}")
        print(f"Val Loss:             {results['val_loss']:.6f}")
        print(f"{'='*50}")
