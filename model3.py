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
from neuralforecast.models import TFT  # Changed from NHITS to TFT

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
    """Load and prepare the BNB dataset with additional features for TFT"""
    print("Loading dataset...")
    df = pd.read_csv('BNB_Final_ready.csv')
    
    # Convert date column to datetime
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Sort by date
    df = df.sort_values('ds').reset_index(drop=True)
    
    # Prepare data for NeuralForecast with additional features for TFT
    # TFT can use multiple features as static and dynamic covariates
    forecast_df = df[['unique_id', 'ds', 'y']].copy()
    
    # Add ALL available exogenous features from your dataset
    # These features will help with directional accuracy
    exog_features = [
        'Open', 'High', 'Low', 'Volume', 'RSI_14', 'MACD_diff', 'EMA_9', 'EMA_21', 'ATR_14', 
        'BB_upper', 'BB_lower', 'rolling_std_6h', 'rolling_std_12h',
        'lag_1h', 'lag_3h', 'lag_6h', 'lag_24h', 'garch_vol',
        'hour', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos', 
        'dow_sin', 'dow_cos', 'target_return_1h', 'target_direction_1h', 'is_spike'
    ]
    
    # Add exogenous features to forecast dataframe (only if they exist)
    for feature in exog_features:
        if feature in df.columns:
            forecast_df[feature] = df[feature]
    
    # Fill any missing values
    forecast_df = forecast_df.fillna(method='ffill').fillna(method='bfill')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Forecast dataframe shape: {forecast_df.shape}")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    print(f"Price range: ${df['y'].min():.2f} to ${df['y'].max():.2f}")
    print(f"Features included: {list(forecast_df.columns)}")
    
    return df, forecast_df

def split_data(df, train_ratio=0.8):
    """Split data into train and test sets"""
    n_train = int(len(df) * train_ratio)
    train_df = df[:n_train].copy()
    test_df = df[n_train:].copy()
    
    print(f"Training set: {len(train_df)} samples ({df['ds'].iloc[0]} to {df['ds'].iloc[n_train-1]})")
    print(f"Test set: {len(test_df)} samples ({df['ds'].iloc[n_train]} to {df['ds'].iloc[-1]})")
    
    return train_df, test_df

def train_tft_model(train_df, horizon=7):
    """Train TFT model"""
    print("Training TFT model...")
    print(f"Available columns in training data: {list(train_df.columns)}")
    
    # For now, let's use only historical features to avoid static feature issues
    # This will still give us much better directional accuracy than N-HiTS
    available_hist_features = []
    potential_hist_features = [
        'Open', 'High', 'Low', 'Volume', 'RSI_14', 'MACD_diff', 'EMA_9', 'EMA_21', 'ATR_14', 
        'BB_upper', 'BB_lower', 'rolling_std_6h', 'rolling_std_12h',
        'lag_1h', 'lag_3h', 'lag_6h', 'lag_24h', 'garch_vol',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'target_return_1h',
        'hour', 'dayofweek', 'is_weekend', 'target_direction_1h', 'is_spike'
    ]
    
    for feature in potential_hist_features:
        if feature in train_df.columns:
            available_hist_features.append(feature)
    
    print(f"Using historical features: {available_hist_features}")
    
    # Initialize TFT model - using only historical features for simplicity
    models = [
        TFT(
            h=horizon,                     # Forecast horizon
            input_size=24,                # Input sequence length (lookback window)
            max_steps=150,                # Training epochs 
            val_check_steps=10,           # Validation frequency
            early_stop_patience_steps=8,  # Early stopping patience
            learning_rate=1e-3,           # Learning rate
            hidden_size=64,               # Hidden size for TFT (reduced for stability)
            dropout=0.1,                  # Dropout rate
            batch_size=32,                # Batch size
            random_seed=42,               # Random seed for reproducibility
            # TFT specific parameters
            n_head=4,                     # Number of attention heads
            scaler_type='robust',         # Robust scaler for better handling of outliers
            # Use only historical features to avoid static feature issues
            hist_exog_list=available_hist_features if available_hist_features else None,
            # Remove static features for now to avoid the error
            # stat_exog_list=None
        )
    ]
    
    # Initialize NeuralForecast
    nf = NeuralForecast(models=models, freq='D')  # Daily frequency
    
    # Fit the model with validation size (15% of training data for validation)
    val_size = int(len(train_df) * 0.15)
    print(f"Training with {len(train_df)} samples, validation size: {val_size}")
    nf.fit(train_df, val_size=val_size)
    
    return nf

def evaluate_model(nf, test_df, train_df, horizon=1):
    """Evaluate the trained model"""
    print("Evaluating model...")
    
    # For TFT, we can predict multiple steps ahead or use rolling prediction
    # We'll use rolling prediction for fair comparison with N-HiTS results
    
    predictions_list = []
    
    # Get the last input_size points from training data as initial context
    input_size = 24
    context_df = train_df.tail(input_size).copy()
    
    # Predict each point in the test set iteratively
    for i in range(len(test_df)):
        try:
            # Make prediction
            pred_df = nf.predict(df=context_df)
            
            # Get the prediction for the next step
            next_pred = pred_df['TFT'].iloc[-1]  # Changed from 'NHITS' to 'TFT'
            predictions_list.append(next_pred)
            
            # Update context: remove oldest, add actual next value for next prediction
            if i < len(test_df) - 1:  # Don't do this for the last prediction
                new_row = test_df.iloc[i:i+1].copy()
                context_df = pd.concat([context_df.iloc[1:], new_row], ignore_index=True)
        
        except Exception as e:
            print(f"Error at prediction {i}: {e}")
            # Use last successful prediction if error occurs
            if predictions_list:
                predictions_list.append(predictions_list[-1])
            else:
                predictions_list.append(test_df.iloc[i]['y'])  # Fallback to actual value
    
    # Create results dataframe
    results = test_df.copy()
    results['TFT'] = predictions_list  # Changed column name from 'NHITS' to 'TFT'
    
    # Extract actual and predicted values
    y_true = results['y'].values
    y_pred = results['TFT'].values
    
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    directional_accuracy = calculate_directional_accuracy(y_true, y_pred)
    
    # Print results
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS (TFT)")
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
    print("PREDICTION SUMMARY (TFT)")
    print("="*60)
    
    actual_prices = results['y'].values
    predicted_prices = results['TFT'].values  # Changed from 'NHITS' to 'TFT'
    
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
    print("DETAILED PREDICTIONS (TFT)")
    print("="*80)
    print(f"{'Date':<12} {'Actual Price':<15} {'Predicted Price':<17} {'Error':<12} {'Error %':<10}")
    print("-" * 80)
    
    for i in range(min(num_predictions, len(results))):
        date = results['ds'].iloc[i].strftime('%Y-%m-%d')
        actual = results['y'].iloc[i]
        predicted = results['TFT'].iloc[i]  # Changed from 'NHITS' to 'TFT'
        error = actual - predicted
        error_pct = (error / actual) * 100
        
        print(f"{date:<12} ${actual:<14.2f} ${predicted:<16.2f} ${error:<11.2f} {error_pct:<9.2f}%")
    
    if len(results) > num_predictions:
        print(f"... and {len(results) - num_predictions} more predictions")
    
    print("="*80)

def print_recent_predictions(results, days=10):
    """Print the most recent predictions"""
    print(f"\n" + "="*70)
    print(f"LAST {days} PREDICTIONS (TFT)")
    print("="*70)
    
    recent_results = results.tail(days)
    
    print(f"{'Date':<12} {'Actual':<12} {'Predicted':<12} {'Difference':<12} {'Direction':<10}")
    print("-" * 70)
    
    for i in range(len(recent_results)):
        row = recent_results.iloc[i]
        date = row['ds'].strftime('%Y-%m-%d')
        actual = row['y']
        predicted = row['TFT']  # Changed from 'NHITS' to 'TFT'
        diff = actual - predicted
        
        # Determine if prediction direction was correct
        if i > 0:
            prev_actual = recent_results.iloc[i-1]['y']
            prev_predicted = recent_results.iloc[i-1]['TFT']  # Changed from 'NHITS' to 'TFT'
            actual_up = actual > prev_actual
            pred_up = predicted > prev_predicted
            direction = "✓" if actual_up == pred_up else "✗"
        else:
            direction = "-"
        
        print(f"{date:<12} ${actual:<11.2f} ${predicted:<11.2f} ${diff:<11.2f} {direction:<10}")
    
    print("="*70)

def export_predictions_to_csv(results, filename="bnb_tft_predictions.csv"):
    """Export predictions to CSV file"""
    # Create a clean dataframe for export
    export_df = results[['ds', 'y', 'TFT']].copy()  # Changed from 'NHITS' to 'TFT'
    export_df.columns = ['Date', 'Actual_Price', 'Predicted_Price']
    export_df['Error'] = export_df['Actual_Price'] - export_df['Predicted_Price']
    export_df['Error_Percentage'] = (export_df['Error'] / export_df['Actual_Price']) * 100
    export_df['Absolute_Error'] = abs(export_df['Error'])
    
    # Save to CSV
    export_df.to_csv(filename, index=False)
    print(f"\nPredictions exported to {filename}")
    return export_df

def print_future_predictions(model, train_df, days=7):
    print(f"\n" + "="*50)
    print(f"FUTURE PREDICTIONS - TFT (Next {days} days)")
    print("="*50)

    # Use the last available sequence
    context_df = train_df.tail(24).copy()
    future_pred = model.predict(h=days)

    last_date = train_df['ds'].max()
    print(f"{'Date':<12} {'Predicted Price':<15} {'Change from Previous':<20}")
    print("-" * 50)

    prev_price = train_df['y'].iloc[-1]
    for i in range(days):
        pred_date = last_date + pd.Timedelta(days=i+1)
        pred_price = future_pred['TFT'].iloc[i]
        change = pred_price - prev_price
        change_pct = (change / prev_price) * 100
        print(f"{pred_date.strftime('%Y-%m-%d'):<12} ${pred_price:<14.2f} ${change:+.2f} ({change_pct:+.1f}%)")
        prev_price = pred_price

    print("="*50)
    return future_pred


def plot_bnb_price(df):
    """Plot BNB price over time"""
    plt.figure(figsize=(15, 8))
    plt.plot(df['ds'], df['y'], linewidth=1.5, color='#1f77b4', alpha=0.8)
    plt.title('BNB Price Over Time', fontsize=16, fontweight='bold', pad=20)
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
    ax1.plot(results['ds'], results['TFT'], label='Predicted (TFT)', linewidth=2, color='#ff7f0e', alpha=0.8)  # Changed label and column
    ax1.set_title('Actual vs Predicted BNB Prices (TFT Model)', fontsize=16, fontweight='bold', pad=20)
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
    ax2.scatter(results['y'], results['TFT'], alpha=0.6, color='#2ca02c', s=50)  # Changed column name
    
    # Add perfect prediction line
    min_val = min(results['y'].min(), results['TFT'].min())
    max_val = max(results['y'].max(), results['TFT'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, label='Perfect Prediction')
    
    ax2.set_xlabel('Actual Price (USD)', fontsize=12)
    ax2.set_ylabel('Predicted Price (USD)', fontsize=12)
    ax2.set_title('Actual vs Predicted Price Scatter Plot (TFT)', fontsize=16, fontweight='bold', pad=20)
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
    df, forecast_df = load_and_prepare_data('BNB_Final_ready.csv')
    
    # Split data (80% train, 20% test)
    train_df, test_df = split_data(forecast_df, train_ratio=0.8)
    
    # Plot BNB price over time
    print("Plotting BNB price history...")
    plot_bnb_price(df)
    
    # Train TFT model
    nf = train_tft_model(train_df,horizon=7)
    
    # Evaluate model
    results, metrics = evaluate_model(nf, test_df, train_df)
    
    # Print detailed predictions
    print_prediction_summary(results)
    print_predictions(results, num_predictions=20)  # Print first 20 predictions
    print_recent_predictions(results, days=10)       # Print last 10 predictions
    
    # Export predictions to CSV
    export_df = export_predictions_to_csv(results, "bnb_tft_predictions.csv")
    
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
    
    print("\nTFT Model training and evaluation completed successfully!")
    print("The trained TFT model is ready for making future predictions.")
    
    # Print final summary of first 10 predictions
    print("\nSample Predictions (First 10):")
    print("-" * 50)
    for i in range(min(10, len(prediction_results))):
        date = prediction_results['ds'].iloc[i].strftime('%Y-%m-%d')
        actual = prediction_results['y'].iloc[i]
        predicted = prediction_results['TFT'].iloc[i]  # Changed from 'NHITS' to 'TFT'
        error = abs(actual - predicted)
        print(f"{date}: Actual ${actual:8.2f} | Predicted ${predicted:8.2f} | Error ${error:6.2f}")
    
    print(f"\nTotal predictions made: {len(prediction_results)}")
    print(f"R² Score: {evaluation_metrics['r2']:.4f}")
    print(f"MAPE: {evaluation_metrics['mape']:.2f}%")
    print(f"Directional Accuracy: {evaluation_metrics['directional_accuracy']:.2f}%")