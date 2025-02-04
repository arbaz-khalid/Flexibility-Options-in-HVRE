# solar_analysis_script.py
# solar_analysis_script.py

import pandas as pd
import numpy as np
import logging
from plotly.subplots import make_subplots
import plotly.graph_objs as go

initial_soc = 50
battery_capacity_per_unit = 57  # Capacity of each battery in kWh

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pv_data(pv_file_path: str,
                time_start_col: str = 'time',
                time_end_col: str = 'local_time',
                electricity_col: str = 'electricity',
                sep: str = ',') -> pd.DataFrame:
    """
    Load and preprocess PV production data from a CSV file.
    Converts PV production from MWh to kWh.
    
    Parameters:
        pv_file_path (str): Path to the PV production CSV file.
        time_start_col (str): Column name for the start time.
        time_end_col (str): Column name for the end time.
        electricity_col (str): Column name for electricity production.
        sep (str): Separator used in the CSV file.
    
    Returns:
        pd.DataFrame: Preprocessed PV production DataFrame with datetime index in kWh.
    """
    logger.info("Loading PV production data from file: %s", pv_file_path)

    try:
        # Read the CSV file with specified delimiter
        pv_data = pd.read_csv(pv_file_path, sep=sep)
    except FileNotFoundError:
        logger.error("PV production file '%s' not found.", pv_file_path)
        raise
    except Exception as e:
        logger.error("Error reading PV production file '%s': %s", pv_file_path, e)
        raise

    # Check if expected columns exist
    expected_columns = [time_start_col, time_end_col, electricity_col]
    missing_columns = [col for col in expected_columns if col not in pv_data.columns]
    if missing_columns:
        logger.error("Missing columns in PV production data: %s", missing_columns)
        raise KeyError(f"Missing columns in PV production data: {missing_columns}")

    # Rename columns for consistency
    pv_data.rename(columns={
        time_start_col: 'Time_Start',
        time_end_col: 'Time_End',
        electricity_col: 'PV_Production_MWh'  # Original unit: MWh
    }, inplace=True)

    # Convert 'Time_Start' to datetime with correct format
    pv_data['Time_Start'] = pd.to_datetime(pv_data['Time_Start'], format='%Y-%m-%d %H:%M', errors='coerce')

    # Drop rows with invalid dates
    initial_row_count = pv_data.shape[0]
    pv_data.dropna(subset=['Time_Start'], inplace=True)
    dropped_rows = initial_row_count - pv_data.shape[0]
    if dropped_rows > 0:
        logger.warning("Dropped %d rows due to invalid 'Time_Start' format in PV data.", dropped_rows)

    # Set 'Time_Start' as the index
    pv_data.set_index('Time_Start', inplace=True)

    # Drop the 'Time_End' column as it's not needed
    if 'Time_End' in pv_data.columns:
        pv_data.drop(columns=['Time_End'], inplace=True)
        logger.info("'Time_End' column dropped from PV production data.")

    # Convert PV production from MWh to kWh
    pv_data['PV_Production_kWh'] = pv_data['PV_Production_MWh']  # 1 MWh = 1000 kWh
    pv_data.drop(columns=['PV_Production_MWh'], inplace=True)

    logger.info("PV production data successfully loaded, converted to kWh, and processed.")
    return pv_data

def extend_pv_data(pv_data: pd.DataFrame, consumption_start_date: str, consumption_end_date: str) -> pd.DataFrame:
    """
    Extend PV production data to cover the consumption period by repeating the data.

    Parameters:
        pv_data (pd.DataFrame): Original PV production DataFrame.
        consumption_start_date (str): Start date for the consumption data (inclusive).
        consumption_end_date (str): End date for the consumption data (exclusive).

    Returns:
        pd.DataFrame: Extended PV production DataFrame covering the consumption period.
    """
    logger.info("Extending PV production data to cover %s to %s.", consumption_start_date, consumption_end_date)

    # Convert consumption start and end dates to datetime
    consumption_start_dt = pd.to_datetime(consumption_start_date)
    consumption_end_dt = pd.to_datetime(consumption_end_date)

    # Calculate the duration of the original PV data
    pv_duration = pv_data.index.max() - pv_data.index.min()
    pv_hours = pv_data.shape[0]
    logger.info("Original PV data duration: %s (%d hours)", pv_duration, pv_hours)

    # Calculate total duration to cover
    total_duration = consumption_end_dt - consumption_start_dt
    total_hours = int(total_duration.total_seconds() / 3600)
    logger.info("Total duration to cover: %s (%d hours)", total_duration, total_hours)

    # Calculate the number of repeats needed
    num_repeats = int(np.ceil(total_hours / pv_hours))
    logger.info("Repeating PV data %d times to cover the desired period.", num_repeats)

    # Repeat the PV data
    pv_data_repeated = pd.concat([pv_data] * num_repeats)

    # Reset the index and set a new date range starting from consumption_start_dt
    pv_data_repeated = pv_data_repeated.reset_index(drop=True)
    pv_data_repeated.index = pd.date_range(start=consumption_start_dt, periods=len(pv_data_repeated), freq='H')

    # Truncate the PV data to match the end_date
    pv_data_truncated = pv_data_repeated.loc[consumption_start_dt:consumption_end_dt - pd.Timedelta(hours=1)]

    # Log the new date range and number of records
    logger.info("PV production data extended from %s to %s.", pv_data_truncated.index.min(), pv_data_truncated.index.max())
    logger.info("Total PV production records after extension: %d", len(pv_data_truncated))

    return pv_data_truncated

def load_data(consumption_file_path: str, pv_file_path: str, consumption_start_date: str, consumption_end_date: str,
              consumption_sep: str = ';',
              consumption_time_col: str = 'Time',
              consumption_value_col: str = 'Consumption (kWh)',
              pv_sep: str = ',') -> pd.DataFrame:
    """
    Load and preprocess energy consumption and PV production data from CSV files.

    Parameters:
        consumption_file_path (str): Path to the consumption CSV file.
        pv_file_path (str): Path to the PV production CSV file.
        consumption_start_date (str): Start date for filtering consumption data (inclusive).
        consumption_end_date (str): End date for filtering consumption data (exclusive).
        consumption_sep (str): Separator used in the consumption CSV file.
        consumption_time_col (str): Column name for time in consumption data.
        consumption_value_col (str): Column name for consumption values.
        pv_sep (str): Separator used in the PV production CSV file.

    Returns:
        pd.DataFrame: Cleaned and merged DataFrame containing both consumption and PV production data.
    """
    logger.info("Loading energy consumption data from file: %s", consumption_file_path)

    try:
        # Read the consumption CSV file with specified delimiter
        data = pd.read_csv(consumption_file_path, sep=consumption_sep)
    except FileNotFoundError:
        logger.error("Consumption file '%s' not found.", consumption_file_path)
        raise
    except Exception as e:
        logger.error("Error reading consumption file '%s': %s", consumption_file_path, e)
        raise

    # Check if expected columns exist
    expected_columns = [consumption_time_col, consumption_value_col]

    # Since the CSV has a single column 'Time;Consumption (kWh)', split it
    if len(expected_columns) == 2 and len(data.columns) == 1:
        logger.info("Splitting single column into two based on delimiter.")
        split_data = data[data.columns[0]].str.split(';', expand=True)
        if split_data.shape[1] >= 2:
            data['Time'] = split_data[0].str.strip()
            data['Consumption (kWh)'] = split_data[1].str.strip()
        else:
            logger.error("Unable to split the consumption data into two columns.")
            raise ValueError("Consumption CSV does not contain two columns separated by ';'.")
    else:
        missing_columns = [col for col in expected_columns if col not in data.columns]
        if missing_columns:
            logger.error("Missing columns in consumption data: %s", missing_columns)
            raise KeyError(f"Missing columns in consumption data: {missing_columns}")

    # Parse the specified time column to datetime with correct format
    data['Time'] = pd.to_datetime(data['Time'], format='%Y/%m/%d %H:%M', errors='coerce')

    # Drop rows with invalid dates
    initial_row_count = data.shape[0]
    data.dropna(subset=['Time'], inplace=True)
    dropped_rows = initial_row_count - data.shape[0]
    if dropped_rows > 0:
        logger.warning("Dropped %d rows due to invalid 'Time' format in consumption data.", dropped_rows)

    # Remove non-numeric characters from the consumption column and convert to numeric
    try:
        data['Consumption_kWh'] = pd.to_numeric(
            data['Consumption (kWh)'].str.replace(' kWh', '', regex=False).str.replace(',', '', regex=False),
            errors='coerce'
        )
    except Exception as ve:
        logger.error("Error converting consumption values to float: %s", ve)
        raise

    # Drop rows with NaN values in 'Consumption_kWh'
    data.dropna(subset=['Consumption_kWh'], inplace=True)

    # Filter data for the specified date range
    data = data[(data['Time'] >= consumption_start_date) & (data['Time'] < consumption_end_date)]

    # Ensure filtered data is not empty
    if data.empty:
        logger.error("No consumption data available for the specified date range: %s to %s.", consumption_start_date, consumption_end_date)
        raise ValueError("No consumption data available for the specified date range.")

    # Set 'Time' as the index
    data.set_index('Time', inplace=True)

    logger.info("Energy consumption data successfully loaded and filtered.")

    # Load PV production data
    pv_data = load_pv_data(pv_file_path, sep=pv_sep)

    # Extend PV data to match the consumption data date range
    pv_data_extended = extend_pv_data(pv_data, consumption_start_date, consumption_end_date)

    # Merge consumption and PV data on the timestamp index
    merged_data = data.join(pv_data_extended, how='left')

    # Handle missing PV production data by filling with zeros
    merged_data['PV_Production_kWh'].fillna(0, inplace=True)

    # Drop the redundant 'Consumption (kWh)' column
    if 'Consumption (kWh)' in merged_data.columns:
        merged_data.drop(columns=['Consumption (kWh)'], inplace=True)
        logger.info("'Consumption (kWh)' column dropped from merged data.")

    logger.info("Consumption and PV production data successfully merged and cleaned.")
    return merged_data


def calculate_annual_metrics(merged_data: pd.DataFrame, num_panels: int, panel_capacity: float) -> dict:
    """
    Calculate annual energy metrics using actual PV production data.

    Parameters:
        merged_data (pd.DataFrame): Merged DataFrame containing consumption and PV production data.
        num_panels (int): Number of solar panels.
        panel_capacity (float): Capacity of each panel in kilowatts (kW).

    Returns:
        dict: Dictionary containing various annual energy metrics.
    """
    logger.info("Calculating annual metrics using actual PV production data...")

    # Calculate total PV production (kWh)
    annual_pv_produced_kwh = merged_data['PV_Production_kWh'].sum() * num_panels  # kWh per panel * number of panels

    # Calculate total energy consumed (kWh)
    total_energy_consumed = merged_data['Consumption_kWh'].sum()

    # Calculate energy delivered to load directly from PV (kWh)
    energy_delivered_directly = merged_data.apply(
        lambda row: min(row['Consumption_kWh'], row['PV_Production_kWh'] * num_panels), axis=1
    ).sum()

    # Calculate energy imported from the grid (kWh)
    energy_imported = merged_data.apply(
        lambda row: max(row['Consumption_kWh'] - (row['PV_Production_kWh'] * num_panels), 0), axis=1
    ).sum()

    # Calculate energy exported to the grid (kWh)
    energy_exported = merged_data.apply(
        lambda row: max((row['PV_Production_kWh'] * num_panels) - row['Consumption_kWh'], 0), axis=1
    ).sum()

    # Calculate performance indicator (%)
    performance_indicator = (annual_pv_produced_kwh / total_energy_consumed) * 100 if total_energy_consumed > 0 else 0

    logger.info("Annual metrics calculation completed.")

    return {
        "annual_pv_produced_kwh": round(annual_pv_produced_kwh, 1),
        "annual_imported_kwh": round(energy_imported, 1),
        "annual_exported_kwh": round(energy_exported, 1),
        "total_energy_consumed": round(total_energy_consumed, 1),
        "energy_delivered_to_load": round(energy_delivered_directly, 1),
        "performance_indicator": round(performance_indicator, 1)
    }

def calculate_annual_metrics_with_battery(merged_data: pd.DataFrame, num_panels: int, panel_capacity: float, 
                                         num_batteries: int, initial_soc: float, battery_capacity_per_unit: float, 
                                         converter_efficiency: float, temperature_coefficient: float, NOCT: float) -> dict:
    """
    Calculate annual energy metrics with battery storage using actual PV production data.

    Parameters:
        merged_data (pd.DataFrame): Merged DataFrame containing consumption and PV production data.
        num_panels (int): Number of solar panels.
        panel_capacity (float): Capacity of each panel in kilowatts (kW).
        num_batteries (int): Number of battery units.
        initial_soc (float): Initial state of charge (percentage).
        battery_capacity_per_unit (float): Capacity of each battery unit in kilowatt-hours (kWh).
        converter_efficiency (float): Efficiency of the converter.
        temperature_coefficient (float): Temperature coefficient for PV panels.
        NOCT (float): Nominal Operating Cell Temperature.

    Returns:
        dict: Dictionary containing various annual energy metrics with battery storage.
    """
    logger.info("Calculating annual metrics with battery storage using actual PV production data...")

    battery_capacity = battery_capacity_per_unit * num_batteries  # Total battery capacity in kWh
    soc = initial_soc  # State of Charge

    annual_pv_produced_kwh = 0
    annual_storage_supplied_kwh = 0
    annual_storage_consumed_kwh = 0
    annual_imported_kwh = 0
    annual_exported_kwh = 0
    total_energy_consumed = 0

    for timestamp, row in merged_data.iterrows():
        consumption = row['Consumption_kWh']
        pv_production = row['PV_Production_kWh'] * num_panels  # Total PV production in kWh

        # Accumulate total PV production and energy consumed
        annual_pv_produced_kwh += pv_production
        total_energy_consumed += consumption

        if consumption > pv_production:
            remaining_load = consumption - pv_production
            # Discharge battery if SOC > 20%
            if soc > 20:
                discharge_energy = min((soc - 20) / 100 * battery_capacity, remaining_load)
                annual_storage_supplied_kwh += discharge_energy
                soc -= (discharge_energy / battery_capacity) * 100
                remaining_load -= discharge_energy
            # Import remaining load from the grid
            if remaining_load > 0:
                annual_imported_kwh += remaining_load
        else:
            excess_energy = pv_production - consumption
            # Charge battery with excess energy if SOC < 100%
            if soc < 100:
                charge_energy = min(excess_energy * converter_efficiency, (100 - soc) / 100 * battery_capacity)
                annual_storage_consumed_kwh += charge_energy
                soc += (charge_energy / battery_capacity) * 100
            # Export to grid if battery is full
            if excess_energy > 0:
                annual_exported_kwh += excess_energy - charge_energy / converter_efficiency

        # Ensure SOC remains within 0-100%
        soc = max(0, min(100, soc))

    # Calculate performance indicator (%)
    performance_indicator = (annual_pv_produced_kwh / total_energy_consumed) * 100 if total_energy_consumed > 0 else 0

    logger.info("Annual metrics with battery storage calculation completed.")

    return {
        "annual_pv_produced_kwh": round(annual_pv_produced_kwh, 1),
        "annual_storage_supplied_kwh": round(annual_storage_supplied_kwh, 1),
        "annual_storage_consumed_kwh": round(annual_storage_consumed_kwh, 1),
        "annual_imported_kwh": round(annual_imported_kwh, 1),
        "annual_exported_kwh": round(annual_exported_kwh, 1),
        "total_energy_consumed": round(total_energy_consumed, 1),
        "energy_delivered_to_load": round((annual_pv_produced_kwh - annual_exported_kwh + annual_imported_kwh + annual_storage_supplied_kwh), 1),
        "performance_indicator": round(performance_indicator, 1)
    }

def plot_daily_distribution(merged_data: pd.DataFrame, specific_date: str, num_panels: int):
    """
    Plot the hourly PV production and consumption for a specific day.

    Parameters:
        merged_data (pd.DataFrame): Merged DataFrame containing consumption and PV production data.
        specific_date (str): Date to plot in 'YYYY-MM-DD' format.
        num_panels (int): Number of solar panels.

    Returns:
        None
    """
    # Convert specific_date to datetime
    try:
        specific_datetime = pd.to_datetime(specific_date)
    except Exception as e:
        logger.error("Invalid date format for specific_date: %s", e)
        print(f"Invalid date format for specific_date: {specific_date}")
        return

    # Filter data for the specific date using .loc
    try:
        daily_data = merged_data.loc[specific_datetime.strftime('%Y-%m-%d')]
    except KeyError:
        logger.warning("No data available for the specified date: %s", specific_date)
        print(f"No data available for the specified date: {specific_date}")
        return

    if daily_data.empty:
        logger.warning("No data available for the specified date: %s", specific_date)
        print(f"No data available for the specified date: {specific_date}")
        return

    # Create a plotly figure
    fig = go.Figure()

    # Add Consumption trace
    fig.add_trace(
        go.Bar(
            x=daily_data.index.hour,
            y=daily_data['Consumption_kWh'],
            name='Consumption (kWh)',
            marker_color='indianred'
        )
    )

    # Add PV Production trace
    fig.add_trace(
        go.Bar(
            x=daily_data.index.hour,
            y=daily_data['PV_Production_kWh'],
            name='PV Production (kWh)',
            marker_color='lightsalmon'
        )
    )

    # Update layout
    fig.update_layout(
        title_text=f'Hourly Consumption vs. PV Production on {specific_date}',
        xaxis_title='Hour of Day',
        yaxis_title='Energy (kWh)',
        barmode='group',
        template='plotly_dark'
    )

    fig.show()

def plot_multiple_days_distribution(merged_data: pd.DataFrame, dates: list, num_panels: int):
    """
    Plot the hourly PV production and consumption for multiple specified days.

    Parameters:
        merged_data (pd.DataFrame): Merged DataFrame containing consumption and PV production data.
        dates (list): List of dates to plot in 'YYYY-MM-DD' format.
        num_panels (int): Number of solar panels.

    Returns:
        None
    """
    num_days = len(dates)
    fig = make_subplots(rows=num_days, cols=1, shared_xaxes=True, 
                        subplot_titles=dates, vertical_spacing=0.05)

    for i, date in enumerate(dates, 1):
        try:
            specific_datetime = pd.to_datetime(date)
        except Exception as e:
            logger.error("Invalid date format for date '%s': %s", date, e)
            continue

        # Filter data for the specific date
        daily_data = merged_data[specific_datetime.strftime('%Y-%m-%d')]

        if daily_data.empty:
            logger.warning("No data available for the specified date: %s", date)
            continue

        # Add Consumption trace
        fig.add_trace(
            go.Bar(
                x=daily_data.index.hour,
                y=daily_data['Consumption_kWh'],
                name='Consumption (kWh)',
                marker_color='indianred'
            ),
            row=i, col=1
        )

        # Add PV Production trace
        fig.add_trace(
            go.Bar(
                x=daily_data.index.hour,
                y=daily_data['PV_Production_kWh'],
                name='PV Production (kWh)',
                marker_color='lightsalmon'
            ),
            row=i, col=1
        )

    fig.update_layout(
        height=300 * num_days,  # Adjust height based on number of days
        title_text='Hourly Consumption vs. PV Production for Selected Days',
        xaxis_title='Hour of Day',
        yaxis_title='Energy (kWh)',
        barmode='group',
        template='plotly_dark'
    )

    fig.show()

def calculate_monthly_pv_production(month: int, num_panels: int, panel_capacity: float, merged_data: pd.DataFrame) -> pd.Series:
    """
    Calculate average PV production per hour for the specified month, number of panels, and panel capacity.

    Parameters:
        month (int): Month number (1-12).
        num_panels (int): Number of solar panels.
        panel_capacity (float): Capacity of each panel in kilowatts (kW).
        merged_data (pd.DataFrame): DataFrame containing 'PV_Production_kWh' with DatetimeIndex.

    Returns:
        pd.Series: PV production per hour indexed by hour (0-23).
    """
    logger.info(f"Calculating monthly PV production for month: {month}, with {num_panels} panels and {panel_capacity} kW capacity each.")
    
    # Validate inputs
    if month < 1 or month > 12:
        logger.error("Invalid month value. Must be between 1 and 12.")
        raise ValueError("Month must be an integer between 1 and 12.")
    
    if num_panels <= 0:
        logger.error("Number of panels must be a positive integer.")
        raise ValueError("Number of panels must be a positive integer.")
    
    if panel_capacity <= 0:
        logger.error("Panel capacity must be a positive float.")
        raise ValueError("Panel capacity must be a positive float.")
    
    # Filter data for the specified month
    pv_data_month = merged_data[merged_data.index.month == month]['PV_Production_kWh']
    
    if pv_data_month.empty:
        logger.warning(f"No PV production data available for month: {month}. Returning zeros.")
        return pd.Series([0]*24, index=range(24))
    
    # Group by hour and calculate average PV production per hour
    pv_avg_per_hour = pv_data_month.groupby(pv_data_month.index.hour).mean()
    
    # Ensure all 24 hours are represented
    pv_avg_per_hour = pv_avg_per_hour.reindex(range(24), fill_value=0)
    
    # Scale PV production based on number of panels and their capacity
    # Assuming 'PV_Production_kWh' is already for the original number of panels
    # If 'PV_Production_kWh' is per panel, uncomment the following line:
    # pv_avg_per_hour = pv_avg_per_hour * num_panels * panel_capacity
    
    # If 'PV_Production_kWh' is for a different number of panels, adjust accordingly
    # For example, if the original data is for 100 panels:
    # original_num_panels = 100
    # scaling_factor = (num_panels / original_num_panels)
    # pv_avg_per_hour = pv_avg_per_hour * scaling_factor
    
    # For this example, let's assume 'PV_Production_kWh' is for 1 panel
    # Therefore, multiply by number of panels and panel capacity
    pv_avg_per_hour = pv_avg_per_hour * num_panels * panel_capacity
    
    logger.info(f"Monthly PV production calculation completed for month: {month}.")
    
    return pv_avg_per_hour