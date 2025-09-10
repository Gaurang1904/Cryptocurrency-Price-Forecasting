import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Install required packages (uncomment if not installed)
# !pip install neuralforecast
# !pip install datasetsforecast

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_directional_accuracy(y_true, y_pred):
    """Calculate directional accuracy"""
    # Calculate actual and predicted direction changes
    actual_direction = np.diff(y_true) > 0
    predicted_direction = np.diff(y_pred) > 0
    
    # Calculate accuracy
    correct_predictions = np.sum(actual_direction == predicted_direction)
    total_predictions = len(actual_direction)
    
    return (correct_predictions / total_predictions) * 100

def load_and_prepare_data(file_path):
    """Load and prepare the Ethereum dataset"""
    print("Loading dataset...")
    df = pd.read_csv('ETH_Final_ready.csv')
    
    # Convert date column to datetime
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Sort by date
    df = df.sort_values('ds').reset_index(drop=True)
    
    # Prepare data for NeuralForecast (requires specific format)
    # NeuralForecast expects columns: unique_id, ds, y
    forecast_df = df[['unique_id', 'ds', 'y']].copy()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    print(f"Price range: ${df['y'].min():.2f} to ${df['y'].max():.2f}")
    
    return df, forecast_df

def split_data(df, train_ratio=0.8):
    """Split data into train and test sets"""
    n_train = int(len(df) * train_ratio)
    train_df = df[:n_train].copy()
    test_df = df[n_train:].copy()
    
    print(f"Training set: {len(train_df)} samples ({df['ds'].iloc[0]} to {df['ds'].iloc[n_train-1]})")
    print(f"Test set: {len(test_df)} samples ({df['ds'].iloc[n_train]} to {df['ds'].iloc[-1]})")
    
    return train_df, test_df

def train_nhits_model(train_df, horizon=1):
    """Train N-HiTS model"""
    print("Training N-HiTS model...")
    
    # Initialize N-HiTS model
    models = [
        NHITS(
            h=horizon,                    # Forecast horizon
            input_size=49,               # Input sequence length (lookback window)
            max_steps=1000,               # Training epochs
            val_check_steps=10,          # Validation frequency
            early_stop_patience_steps=5, # Early stopping patience
            learning_rate=1e-3,          # Learning rate
            n_blocks=[1, 1, 1],         # Number of blocks per stack
            mlp_units=[[512, 512], [512, 512], [512, 512]], # MLP units per stack
            n_pool_kernel_size=[2, 2, 1], # Pooling kernel sizes
            n_freq_downsample=[4, 2, 1],  # Frequency downsampling
            interpolation_mode='linear',   # Interpolation mode
            dropout_prob_theta=0.1,       # Dropout probability
            batch_size=32,               # Batch size
            random_seed=42               # Random seed for reproducibility
        )
    ]
    
    # Initialize NeuralForecast
    nf = NeuralForecast(models=models, freq='D')  # Daily frequency
    
    # Fit the model with validation size (10% of training data for validation)
    val_size = int(len(train_df) * 0.1)
    nf.fit(train_df, val_size=val_size)
    
    return nf

def evaluate_model(nf, test_df, train_df, horizon=7):
    """Evaluate the trained model"""
    print("Evaluating model...")
    
    # For N-HiTS, we need to predict step by step for the test period
    # We'll use the last part of training data as context and predict the test period
    
    predictions_list = []
    
    # Get the last input_size points from training data as initial context
    input_size = 49
    context_df = train_df.tail(input_size).copy()
    
    # Predict each point in the test set iteratively
    for i in range(len(test_df)):
        # Make prediction
        pred_df = nf.predict(df=context_df)
        
        # Get the prediction for the next step
        next_pred = pred_df['NHITS'].iloc[-1]
        predictions_list.append(next_pred)
        
        # Update context: remove oldest, add actual next value for next prediction
        if i < len(test_df) - 1:  # Don't do this for the last prediction
            new_row = test_df.iloc[i:i+1].copy()
            context_df = pd.concat([context_df.iloc[1:], new_row], ignore_index=True)
    
    # Create results dataframe
    results = test_df.copy()
    results['NHITS'] = predictions_list
    
    # Extract actual and predicted values
    y_true = results['y'].values
    y_pred = results['NHITS'].values
    
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    directional_accuracy = calculate_directional_accuracy(y_true, y_pred)
    
    # Print results
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"Mean Squared Error (MSE): ${mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${np.sqrt(mse):.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Directional Accuracy: {directional_accuracy:.2f}%")
    print("="*50)
    
    return results, {
        'r2': r2,
        'mae': mae,
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mape': mape,
        'directional_accuracy': directional_accuracy
    }

def print_prediction_summary(results):
    """Print a summary of all predictions"""
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    
    actual_prices = results['y'].values
    predicted_prices = results['NHITS'].values
    
    print(f"Total Predictions: {len(results)}")
    print(f"Date Range: {results['ds'].min().strftime('%Y-%m-%d')} to {results['ds'].max().strftime('%Y-%m-%d')}")
    print(f"\nActual Price Range: ${actual_prices.min():.2f} - ${actual_prices.max():.2f}")
    print(f"Predicted Price Range: ${predicted_prices.min():.2f} - ${predicted_prices.max():.2f}")
    print(f"\nMean Actual Price: ${actual_prices.mean():.2f}")
    print(f"Mean Predicted Price: ${predicted_prices.mean():.2f}")
    print(f"\nLargest Prediction Error: ${abs(actual_prices - predicted_prices).max():.2f}")
    print(f"Smallest Prediction Error: ${abs(actual_prices - predicted_prices).min():.2f}")
    
    # Count correct directional predictions
    actual_direction = np.diff(actual_prices) > 0
    predicted_direction = np.diff(predicted_prices) > 0
    correct_directions = np.sum(actual_direction == predicted_direction)
    
    print(f"\nCorrect Direction Predictions: {correct_directions}/{len(actual_direction)}")
    print("="*60)

def print_predictions(results, num_predictions=20):
    """Print predictions in a formatted table"""
    print("\n" + "="*80)
    print("DETAILED PREDICTIONS")
    print("="*80)
    print(f"{'Date':<12} {'Actual Price':<15} {'Predicted Price':<17} {'Error':<12} {'Error %':<10}")
    print("-" * 80)
    
    for i in range(min(num_predictions, len(results))):
        date = results['ds'].iloc[i].strftime('%Y-%m-%d')
        actual = results['y'].iloc[i]
        predicted = results['NHITS'].iloc[i]
        error = actual - predicted
        error_pct = (error / actual) * 100
        
        print(f"{date:<12} ${actual:<14.2f} ${predicted:<16.2f} ${error:<11.2f} {error_pct:<9.2f}%")
    
    if len(results) > num_predictions:
        print(f"... and {len(results) - num_predictions} more predictions")
    
    print("="*80)

def print_recent_predictions(results, days=10):
    """Print the most recent predictions"""
    print(f"\n" + "="*70)
    print(f"LAST {days} PREDICTIONS")
    print("="*70)
    
    recent_results = results.tail(days)
    
    print(f"{'Date':<12} {'Actual':<12} {'Predicted':<12} {'Difference':<12} {'Direction':<10}")
    print("-" * 70)
    
    for i in range(len(recent_results)):
        row = recent_results.iloc[i]
        date = row['ds'].strftime('%Y-%m-%d')
        actual = row['y']
        predicted = row['NHITS']
        diff = actual - predicted
        
        # Determine if prediction direction was correct
        if i > 0:
            prev_actual = recent_results.iloc[i-1]['y']
            prev_predicted = recent_results.iloc[i-1]['NHITS']
            actual_up = actual > prev_actual
            pred_up = predicted > prev_predicted
            direction = "✓" if actual_up == pred_up else "✗"
        else:
            direction = "-"
        
        print(f"{date:<12} ${actual:<11.2f} ${predicted:<11.2f} ${diff:<11.2f} {direction:<10}")
    
    print("="*70)

def export_predictions_to_csv(results, filename="eth_predictions.csv"):
    """Export predictions to CSV file"""
    # Create a clean dataframe for export
    export_df = results[['ds', 'y', 'NHITS']].copy()
    export_df.columns = ['Date', 'Actual_Price', 'Predicted_Price']
    export_df['Error'] = export_df['Actual_Price'] - export_df['Predicted_Price']
    export_df['Error_Percentage'] = (export_df['Error'] / export_df['Actual_Price']) * 100
    export_df['Absolute_Error'] = abs(export_df['Error'])
    
    # Save to CSV
    export_df.to_csv(filename, index=False)
    print(f"\nPredictions exported to {filename}")
    return export_df

def print_future_predictions(model, train_df, days=7):
    """Generate and print future predictions"""
    print(f"\n" + "="*50)
    print(f"FUTURE PREDICTIONS (Next {days} days)")
    print("="*50)
    
    try:
        # Get the last part of training data for context
        context_df = train_df.tail(24).copy()
        
        # Generate future predictions
        future_pred = model.predict(df=context_df, h=days)
        
        # Get the last date from training data
        last_date = train_df['ds'].max()
        
        print(f"{'Date':<12} {'Predicted Price':<15} {'Change from Previous':<20}")
        print("-" * 50)
        
        prev_price = train_df['y'].iloc[-1]  # Last actual price
        
        for i in range(days):
            pred_date = last_date + pd.Timedelta(days=i+1)
            pred_price = future_pred['NHITS'].iloc[i] if i < len(future_pred) else future_pred['NHITS'].iloc[-1]
            change = pred_price - prev_price
            change_pct = (change / prev_price) * 100
            
            print(f"{pred_date.strftime('%Y-%m-%d'):<12} ${pred_price:<14.2f} ${change:+.2f} ({change_pct:+.1f}%)")
            prev_price = pred_price
        
        print("="*50)
        return future_pred
        
    except Exception as e:
        print(f"Error generating future predictions: {e}")
        print("Skipping future predictions...")
        return None

def plot_ethereum_price(df):
    """Plot Ethereum price over time"""
    plt.figure(figsize=(15, 8))
    plt.plot(df['ds'], df['y'], linewidth=1.5, color='#1f77b4', alpha=0.8)
    plt.title('Ethereum Price Over Time', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Add some styling
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Format y-axis to show currency
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.show()

def plot_predictions(results):
    """Plot actual vs predicted values"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Time series plot
    ax1.plot(results['ds'], results['y'], label='Actual', linewidth=2, color='#1f77b4', alpha=0.8)
    ax1.plot(results['ds'], results['NHITS'], label='Predicted', linewidth=2, color='#ff7f0e', alpha=0.8)
    ax1.set_title('Actual vs Predicted Ethereum Prices', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Remove top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Format y-axis
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Scatter plot
    ax2.scatter(results['y'], results['NHITS'], alpha=0.6, color='#2ca02c', s=50)
    
    # Add perfect prediction line
    min_val = min(results['y'].min(), results['NHITS'].min())
    max_val = max(results['y'].max(), results['NHITS'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, label='Perfect Prediction')
    
    ax2.set_xlabel('Actual Price (USD)', fontsize=12)
    ax2.set_ylabel('Predicted Price (USD)', fontsize=12)
    ax2.set_title('Actual vs Predicted Price Scatter Plot', fontsize=16, fontweight='bold', pad=20)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Remove top and right spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Format axes
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function"""
    # Load and prepare data
    df, forecast_df = load_and_prepare_data('ETH_Final_ready.csv')
    
    # Split data (80% train, 20% test)
    train_df, test_df = split_data(forecast_df, train_ratio=0.8)
    
    # Plot Ethereum price over time
    print("Plotting Ethereum price history...")
    plot_ethereum_price(df)
    
    # Train N-HiTS model
    nf = train_nhits_model(train_df)
    
    # Evaluate model
    results, metrics = evaluate_model(nf, test_df, train_df)
    
    # Print detailed predictions
    print_prediction_summary(results)
    print_predictions(results, num_predictions=20)  # Print first 20 predictions
    print_recent_predictions(results, days=10)       # Print last 10 predictions
    
    # Export predictions to CSV
    export_df = export_predictions_to_csv(results, "ethereum_predictions.csv")
    
    # Print future predictions
    print_future_predictions(nf, train_df, days=7)
    
    # Plot predictions
    print("Plotting prediction results...")
    plot_predictions(results)
    
    return nf, results, metrics

# Run the complete pipeline
if __name__ == "__main__":
    # Execute the main pipeline
    trained_model, prediction_results, evaluation_metrics = main()
    
    print("\nModel training and evaluation completed successfully!")
    print("The trained N-HiTS model is ready for making future predictions.")
    
    # Print final summary of first 10 predictions
    print("\nSample Predictions (First 10):")
    print("-" * 50)
    for i in range(min(10, len(prediction_results))):
        date = prediction_results['ds'].iloc[i].strftime('%Y-%m-%d')
        actual = prediction_results['y'].iloc[i]
        predicted = prediction_results['NHITS'].iloc[i]
        error = abs(actual - predicted)
        print(f"{date}: Actual ${actual:8.2f} | Predicted ${predicted:8.2f} | Error ${error:6.2f}")
    
    print(f"\nTotal predictions made: {len(prediction_results)}")
    print(f"R² Score: {evaluation_metrics['r2']:.4f}")
    print(f"MAPE: {evaluation_metrics['mape']:.2f}%")
    print(f"Directional Accuracy: {evaluation_metrics['directional_accuracy']:.2f}%")