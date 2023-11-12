"""Drive application."""

import datetime, numpy as np, os, pandas as pd, torch, torch.nn as nn, traceback
from matplotlib             import pyplot as plt
from sklearn.preprocessing  import MinMaxScaler
from termcolor              import colored
from tqdm                   import tqdm

from data.stock             import Stock
from models.trader_lstm     import TraderLSTM
from utils                  import ARGS, LOGGER

if __name__ == '__main__':
    try:
        LOGGER.info(f"\n{open('utils/banner.txt', 'r').read()}")


        # INITIALIZATION ==========================================================================

        # Initialize output directory
        output_dir = f"{ARGS.output_path}/{ARGS.ticker}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)

        # Initialize dataset
        data = Stock(ARGS.ticker, ARGS.start_date, ARGS.end_date)

        # Initialize model
        model = TraderLSTM(1, hidden_size=ARGS.hidden_size, num_layers=ARGS.layers)

        LOGGER.debug(f"Model:\n{model}")

        # Place model on GPU if available
        if torch.cuda.is_available(): model = model.cuda()
        LOGGER.info(f"Using device: {torch.cuda.get_device_name()}")

        # Define loss function
        loss_func = nn.MSELoss(reduction="mean")

        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=ARGS.learning_rate)

        # Fetch data loaders
        train, test = data.get_loaders(ARGS.batch_size)

        LOGGER.debug(f"Train loader:\n{vars(train)}")
        LOGGER.debug(f"Test loader:\n{vars(test)}")

        # TRAINING ================================================================================

        # Initialize history
        train_history, val_history = [], []

        for epoch in range(ARGS.epochs):

            LOGGER.info(f"BEGIN EPOCH {epoch + 1:>3} ================================================")

            # Initialize running loss
            running_loss_train, running_loss_val = 0.0, 0.0

            # Put model in training mode
            model.train()

            with tqdm(total=len(train) + len(test), desc=f"Epoch {epoch + 1:>3}/{ARGS.epochs}", leave=True, colour="magenta") as pbar:

                pbar.set_postfix(status=colored("Training", "cyan"))

                for batch_X, batch_y in train:

                    # Move data set to GPU if available
                    if torch.cuda.is_available(): batch_X, batch_y = batch_X.cuda(), batch_y.cuda()

                    # Feed input to model and make predictions
                    predictions = model(batch_X)

                    # Calculate loss
                    loss = loss_func(predictions, batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss_train += loss.item()

                    # Update progress bar
                    pbar.update(1)

                avg_loss = running_loss_train / len(train)
                train_history.append(avg_loss)

        # VALIDATION ==============================================================================

                # Put model in evaluation mode
                pbar.set_postfix(status=colored("Validating", "yellow"))
                model.eval()

                with torch.no_grad():

                    for batch_X_val, batch_y_val in test:

                        # Move data set to GPU if available
                        if torch.cuda.is_available(): batch_X_val, batch_y_val = batch_X_val.cuda(), batch_y_val.cuda()

                        # Make predictions
                        predictions_val = model(batch_X_val)

                        # Calculate loss
                        val_loss = loss_func(predictions_val, batch_y_val)
                        running_loss_val += val_loss.item()

                        # Update progress bar
                        pbar.update(1)

                    avg_val_loss = running_loss_val / len(test)
                    val_history.append(avg_val_loss)

                pbar.set_postfix(status=colored(f"Loss: {avg_loss:.4f}", "green"))

            LOGGER.info(f"END EPOCH {epoch + 1} | Train loss: {avg_loss:.4f} | Validation loss: {avg_val_loss:.4f}")

        # Plot loss
        x = np.linspace(1, ARGS.epochs, ARGS.epochs)
        plt.plot(x, train_history, scalex=True, label="Training Loss")
        plt.plot(x, val_history,   label="Validation Loss")
        plt.title(f"${ARGS.ticker} ({ARGS.start_date} - {ARGS.end_date})")
        plt.legend()
        # plt.show()
        plt.savefig(f"{output_dir}/loss.jpg")
        LOGGER.info(f"Loss plot saved to {output_dir}/loss.jpg")

        # PREDICTION ==============================================================================
        LOGGER.info(f"Making forecast predictions for {ARGS.look_ahead} days")

        # Convert to np-array
        seq_to_plot = data.get_x_test().squeeze().cpu().numpy()

        # Grab last month of data
        historical_data = seq_to_plot[-1]

        # Initialize forecast
        forecast = []

        with tqdm(total=ARGS.look_ahead * 2, desc=f"Forecasting", leave=True, colour="magenta") as pbar:
                
            pbar.set_postfix(status=colored("Forecasting", "green"))

            # Use model to forecast
            with torch.no_grad():

                for i in range(ARGS.look_ahead * 2):

                    # Prepare historical data
                    history_tensor = torch.as_tensor(historical_data).view(1, -1, 1).float().cuda()

                    # Make forecasting prediction
                    prediction = model(history_tensor).cpu().numpy()[0, 0]
                    forecast.append(prediction[0])
                    pbar.set_postfix(status=colored(prediction[0], "green"))

                    # Roll sequence
                    historical_data = np.roll(historical_data, shift=-1)
                    historical_data[-1] = prediction

                    # Update progress bar
                    pbar.update(1)

        # Generate & append future dates
        last_date = data.get_last_date()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(1), periods=ARGS.look_ahead)
        combined_index = data.raw_test.index.append(future_dates)

        # Plot forecast
        plt.clf()
        plt.rcParams['figure.figsize'] = [14, 4]
        plt.plot(data.raw_test.index[-100:-30], data.raw_test.Open[-100:-30], label="Test data", color='b')

        original_cases = data.scaler.inverse_transform(np.expand_dims(seq_to_plot[-1], axis=0)).flatten()
        plt.plot(data.raw_test.index[-30:], original_cases, label="Actual values", color='green')

        forecasted_cases = data.scaler.inverse_transform(np.expand_dims(forecast, axis=0)).flatten()
        plt.plot(combined_index[-60:], forecasted_cases, label="Forecast", color='red')

        plt.xlabel("Time Step")
        plt.ylabel("Value ($)")
        plt.legend()
        plt.grid(True)
        plt.title(f"${ARGS.ticker} ({data.first_date} - {data.last_date})")
        plt.savefig(f"{output_dir}/forecast.jpg")
        LOGGER.info(f"Forecast plot saved to {output_dir}/forecast.jpg")

    except KeyboardInterrupt:
        LOGGER.critical("Keyboard interrupt detected...")

    except Exception as e:
        LOGGER.error(e)
        traceback.print_exc()

    finally:
        LOGGER.info("Exiting...")