# solar_analysis_script.py

import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(level=logging.ERROR)  # Use DEBUG for detailed logs
logger = logging.getLogger(__name__)

# ----------------------------
# System Parameters
# ----------------------------
INITIAL_NUM_PANELS = 1000      # Default number of PV panels
INITIAL_NUM_TURBINES = 1       # Default number of wind turbines
BATTERY_CAPACITY_PER_UNIT = 57 # kWh per battery
CONVERTER_EFFICIENCY = 0.9     # 90% efficiency
INITIAL_SOC = 0                # Initial SOC (as %)

# ----------------------------
# BatterySOCTracker Class
# ----------------------------
class BatterySOCTracker:
    """
    Tracks and computes the battery State of Charge (SOC) across multiple dates.
    """
    
    def __init__(self, initial_soc=50, start_date='2019-01-01'):
        self.battery_soc_tracking = {}
        self.initialize_battery_soc(initial_soc, start_date)

    def initialize_battery_soc(self, initial_soc=50, start_date='2019-01-01'):
        self.battery_soc_tracking = {}
        self.battery_soc_tracking[pd.Timestamp(start_date)] = initial_soc
        logger.info("Initialized Battery SOC at %s%% on %s.", initial_soc, start_date)

    def get_battery_soc(self, date: pd.Timestamp, initial_soc: float = 50) -> float:
        return self.battery_soc_tracking.get(date, initial_soc)

    def update_battery_soc(self, date: pd.Timestamp, soc: float):
        self.battery_soc_tracking[date] = soc
        logger.info("Updated Battery SOC on %s to %s%%.", date.strftime('%Y-%m-%d'), soc)

    def reset_soc_from_date(self, start_date: pd.Timestamp):
        keys_to_remove = [d for d in self.battery_soc_tracking if d >= start_date]
        for d in keys_to_remove:
            del self.battery_soc_tracking[d]
        logger.info("Reset SOC tracking from %s onward.", start_date.strftime('%Y-%m-%d'))

    def compute_soc_up_to_date(
        self, 
        merged_data: pd.DataFrame, 
        selected_date: pd.Timestamp, 
        num_panels: int, 
        num_turbines: int, 
        num_batteries: int, 
        initial_soc: float, 
        battery_capacity_per_unit: float, 
        converter_efficiency: float, 
        start_date: pd.Timestamp = None
    ):
        """
        Compute SOC values up to the selected_date.
        (Note: This function is not called directly by update_graph below.)
        """
        battery_capacity = battery_capacity_per_unit * num_batteries
        sorted_dates = merged_data.index.normalize().unique().sort_values()

        for current_date in sorted_dates:
            if current_date > selected_date:
                break

            already_computed = (current_date in self.battery_soc_tracking)
            within_reset_range = (start_date and current_date >= start_date)
            if already_computed and not within_reset_range:
                continue

            previous_day = current_date - pd.Timedelta(days=1)
            soc = self.get_battery_soc(previous_day, initial_soc=initial_soc)

            try:
                daily_data = merged_data.loc[current_date]
            except KeyError:
                logger.warning("No data for %s. Setting consumption/production to zero.", current_date)
                daily_data = pd.DataFrame(
                    {
                        'Consumption_kWh':[0]*96,
                        'PV_Production_kWh':[0]*96,
                        'Wind_Production_kWh':[0]*96
                    },
                    index=pd.date_range(current_date, periods=96, freq='15T')
                )

            if isinstance(daily_data, pd.Series):
                daily_data = daily_data.to_frame().T

            if daily_data.shape[0] != 96:
                logger.warning("Expected 96 intervals (15-min) but found %d for %s.", daily_data.shape[0], current_date)
                daily_data = daily_data.reindex(
                    pd.date_range(start=current_date, periods=96, freq='15T'), 
                    fill_value=0
                )

            for t in daily_data.index:
                consumption = daily_data.loc[t, 'Consumption_kWh']
                pv_value = daily_data.loc[t, 'PV_Production_kWh']
                wind_value = daily_data.loc[t, 'Wind_Production_kWh']
                total_production = pv_value + wind_value

                if total_production >= consumption:
                    excess = total_production - consumption
                    if (num_batteries > 0) and (soc < 100):
                        charge = min(excess * converter_efficiency, (100 - soc) / 100 * battery_capacity)
                        soc += (charge / battery_capacity) * 100
                else:
                    deficit = consumption - total_production
                    if (num_batteries > 0) and (soc > 0):
                        discharge = min(deficit, (soc / 100) * battery_capacity)
                        soc -= (discharge / battery_capacity) * 100

                soc = max(0, min(100, soc))

            self.update_battery_soc(current_date, soc)

# Instantiate a single global tracker to be reused
soc_tracker = BatterySOCTracker(initial_soc=INITIAL_SOC, start_date='2019-01-01')

# ----------------------------
# Data Loading Functions
# ----------------------------
def load_city_consumption(
    consumption_file_path: str,
    consumption_sep: str = ';',
    consumption_date_col: str = 'Date',
    consumption_time_col: str = 'Heures',
    consumption_value_col: str = 'Consommation(MWh)'
) -> pd.Series:
    """
    Load city consumption from a CSV, parse the datetime, and convert from MWh to kWh.
    (No resampling is performed so that each 15-minute reading is preserved.)
    """
    logger.info("Loading consumption data from %s...", consumption_file_path)
    try:
        data = pd.read_csv(consumption_file_path, sep=consumption_sep)
        logger.debug("Consumption data loaded successfully.")
    except FileNotFoundError as err:
        logger.error("File not found: %s", consumption_file_path)
        raise err
    except Exception as err:
        logger.error("Error reading consumption file %s: %s", consumption_file_path, err)
        raise err

    data['Datetime'] = pd.to_datetime(
        data[consumption_date_col] + ' ' + data[consumption_time_col],
        format='%Y/%m/%d %H:%M',
        errors='coerce'
    )
    data.dropna(subset=['Datetime'], inplace=True)
    data.set_index('Datetime', inplace=True)

    data[consumption_value_col] = pd.to_numeric(
        data[consumption_value_col].replace('ND', np.nan),
        errors='coerce'
    )
    data[consumption_value_col].fillna(0, inplace=True)
    # Convert from MWh to kWh
    data[consumption_value_col] = data[consumption_value_col] * 1000

    data.rename(columns={consumption_value_col: 'Consumption_kWh'}, inplace=True)
    logger.info("Consumption data loaded and converted from MWh to kWh without resampling.")
    return data['Consumption_kWh']


def load_city_pv_production(
    pv_file_path: str,
    consumption_start_date: str,
    consumption_end_date: str,
    sep: str = ',',
    time_start_col: str = 'time',
    time_end_col: str = 'local_time',
    electricity_col: str = 'electricity'
) -> pd.Series:
    """
    Load and filter PV production data to match the consumption date range.
    """
    logger.info("Loading PV production data from %s...", pv_file_path)
    try:
        pv_data = pd.read_csv(pv_file_path, sep=sep)
        logger.debug("PV production data loaded successfully.")
    except FileNotFoundError as err:
        logger.error("File not found: %s", pv_file_path)
        raise err
    except Exception as err:
        logger.error("Error reading PV production file %s: %s", pv_file_path, err)
        raise err

    pv_data.rename(columns={
        time_start_col: 'Time_Start',
        time_end_col: 'Time_End',
        electricity_col: 'PV_Production_kWh'
    }, inplace=True)

    if 'PV_Production_kWh' not in pv_data.columns:
        logger.error("'PV_Production_kWh' column not found after renaming.")
        raise KeyError("Column 'PV_Production_kWh' not found in PV production data.")

    pv_data['Time_Start'] = pd.to_datetime(
        pv_data['Time_Start'], 
        format='%Y-%m-%d %H:%M', 
        errors='coerce'
    )
    pv_data.dropna(subset=['Time_Start'], inplace=True)
    pv_data.set_index('Time_Start', inplace=True)

    if 'Time_End' in pv_data.columns:
        pv_data.drop(columns=['Time_End'], inplace=True)

    pv_data['PV_Production_kWh'].fillna(0, inplace=True)
    pv_production = pv_data['PV_Production_kWh'][
        (pv_data.index >= consumption_start_date) & (pv_data.index <= consumption_end_date)
    ]

    logger.info("PV production data loaded and filtered to the date range.")
    logger.debug(f"PV production data sample:\n{pv_production.head()}")
    return pv_production

def load_city_wind_production(
    wind_file_path: str,
    consumption_start_date: str,
    consumption_end_date: str,
    sep: str = ',',
    time_start_col: str = 'time',
    time_end_col: str = 'local_time',
    electricity_col: str = 'wind_electricity'
) -> pd.Series:
    """
    Load and filter Wind production data to match the consumption date range.
    """
    logger.info("Loading wind production data from %s...", wind_file_path)
    try:
        wind_data = pd.read_csv(wind_file_path, sep=sep)
        logger.debug("Wind production data loaded successfully.")
    except FileNotFoundError as err:
        logger.error("File not found: %s", wind_file_path)
        raise err
    except Exception as err:
        logger.error("Error reading wind production file %s: %s", wind_file_path, err)
        raise err

    wind_data.rename(columns={
        time_start_col: 'Time_Start',
        time_end_col: 'Time_End',
        electricity_col: 'Wind_Production_kWh'
    }, inplace=True)

    if 'Wind_Production_kWh' not in wind_data.columns:
        logger.error("'Wind_Production_kWh' column not found after renaming.")
        raise KeyError("Column 'Wind_Production_kWh' not found in wind production data.")

    wind_data['Time_Start'] = pd.to_datetime(
        wind_data['Time_Start'], 
        format='%m/%d/%Y %H:%M', 
        errors='coerce'
    )
    wind_data.dropna(subset=['Time_Start'], inplace=True)
    wind_data.set_index('Time_Start', inplace=True)

    if 'Time_End' in wind_data.columns:
        wind_data.drop(columns=['Time_End'], inplace=True)

    wind_data['Wind_Production_kWh'].fillna(0, inplace=True)
    wind_production = wind_data['Wind_Production_kWh'][
        (wind_data.index >= consumption_start_date) & (wind_data.index <= consumption_end_date)
    ]

    logger.info("Wind production data loaded and filtered to the date range.")
    logger.debug(f"Wind production data sample:\n{wind_production.head()}")
    return wind_production

def merge_city_data(
    consumption: pd.Series,
    pv_production: pd.Series,
    wind_production: pd.Series,
    num_panels: int,
    num_turbines: int
) -> pd.DataFrame:
    """
    Merge consumption and scaled PV + Wind production into a single DataFrame.
    """
    logger.info("Merging consumption, PV production, and wind production data.")
    pv_total   = pv_production * num_panels
    wind_total = wind_production * num_turbines

    logger.debug(f"Scaling PV Production by {num_panels} panels.")
    logger.debug(f"Scaling Wind Production by {num_turbines} turbines.")

    logger.debug(f"Sample PV Production after scaling:\n{pv_total.head()}")
    logger.debug(f"Sample Wind Production after scaling:\n{wind_total.head()}")

    merged_df = pd.DataFrame({
        'Consumption_kWh': consumption,
        'PV_Production_kWh': pv_total,
        'Wind_Production_kWh': wind_total
    })
    merged_df.fillna(0, inplace=True)

    logger.info("Merging complete (PV + Wind included).")
    logger.debug(f"Merged data sample:\n{merged_df.head()}")
    return merged_df

# ----------------------------
# Update Graph Function
# ----------------------------
def update_graph(
    num_panels: int,
    num_turbines: int,
    num_batteries: int,
    selected_date: str,
    city_merged_data: pd.DataFrame,
    soc_tracker: BatterySOCTracker,
    initial_num_panels: int,
    initial_num_turbines: int,
    initial_soc: float,
    battery_capacity_per_unit: float,
    converter_efficiency: float
):
    """
    Generate a Plotly figure and a SOC message for the selected date.
    Only data points at full-hour intervals (e.g. 00:00, 01:00, etc.) are used for plotting.
    """
    selected_date_ts = pd.Timestamp(selected_date)

    # Attempt to retrieve all data for the selected day
    try:
        daily_data = city_merged_data.loc[selected_date_ts.strftime('%Y-%m-%d')]
        logger.debug(f"Retrieved data for {selected_date}.")
    except KeyError:
        msg = f"[Error] No data available for the selected date: {selected_date}"
        logger.error(msg)
        return None, msg

    if daily_data.empty:
        msg = f"[Warning] No data available for the selected date: {selected_date}"
        logger.warning(msg)
        return None, msg

    # Ensure daily_data is a DataFrame
    if isinstance(daily_data, pd.Series):
        daily_data = daily_data.to_frame().T

    # Filter to only include rows at full-hour intervals (minute == 0)
    daily_data = daily_data[daily_data.index.minute == 0]

    # Use the filtered time index for plotting
    time_points = daily_data.index
    time_labels = [tp.strftime('%H:%M') for tp in time_points]

    # Retrieve previous period SOC from the tracker
    previous_period = selected_date_ts - pd.Timedelta(days=1)
    battery_soc = soc_tracker.get_battery_soc(
        previous_period,
        initial_soc=(initial_soc if selected_date_ts > pd.Timestamp('2019-01-01') else 0)
    )
    battery_capacity = battery_capacity_per_unit * num_batteries

    energy_charged = []
    energy_discharged = []
    unmet_demand = []
    excess_energy = []
    soc_history = []

    # Calculate the time step in hours (should be 1 hour after filtering)
    if len(time_points) > 1:
        dt = (time_points[1] - time_points[0]).total_seconds() / 3600.0
    else:
        dt = 1

    for t in time_points:
        consumption = daily_data.loc[t, 'Consumption_kWh']
        # Scale production based on the number of panels and turbines
        pv_value = daily_data.loc[t, 'PV_Production_kWh'] * (num_panels / initial_num_panels)
        wind_value = daily_data.loc[t, 'Wind_Production_kWh'] * (num_turbines / initial_num_turbines)
        total_generation = pv_value + wind_value

        if total_generation >= consumption:
            excess = total_generation - consumption
            if num_batteries > 0 and battery_soc < 100:
                charge = min(excess * converter_efficiency, (100 - battery_soc) / 100 * battery_capacity)
                energy_charged.append(charge)
                battery_soc += (charge / battery_capacity) * 100
                remainder = excess - (charge / converter_efficiency) if converter_efficiency > 0 else excess
                excess_energy.append(remainder)
            else:
                energy_charged.append(0)
                excess_energy.append(excess)
            energy_discharged.append(0)
            unmet_demand.append(0)
        else:
            deficit = consumption - total_generation
            if num_batteries > 0 and battery_soc > 0:
                discharge = min(deficit, (battery_soc / 100) * battery_capacity)
                energy_discharged.append(discharge)
                battery_soc -= (discharge / battery_capacity) * 100
                unmet_demand.append(deficit - discharge)
            else:
                energy_discharged.append(0)
                unmet_demand.append(deficit)
            energy_charged.append(0)
            excess_energy.append(0)

        battery_soc = max(0, min(100, battery_soc))
        soc_history.append(battery_soc)

    # Update the SOC tracker with the SOC at the end of the day
    soc_tracker.update_battery_soc(selected_date_ts, battery_soc)

    # Build the Plotly figure using the hourly data points
    fig = go.Figure()

    pv_scaled = daily_data['PV_Production_kWh'] * (num_panels / initial_num_panels)
    fig.add_trace(
        go.Bar(
            x=time_labels,
            y=pv_scaled,
            name='Solar PV (kWh)',
            marker_color='orange'
        )
    )

    wind_scaled = daily_data['Wind_Production_kWh'] * (num_turbines / initial_num_turbines)
    fig.add_trace(
        go.Bar(
            x=time_labels,
            y=wind_scaled,
            name='Wind (kWh)',
            marker_color='lightblue'
        )
    )

    fig.add_trace(
        go.Bar(
            x=time_labels,
            y=unmet_demand,
            name='Unmet Demand (kWh)',
            marker_color='red'
        )
    )

    if num_batteries > 0:
        fig.add_trace(
            go.Bar(
                x=time_labels,
                y=[-val for val in energy_charged],
                name='Energy Charged to Battery (kWh)',
                marker_color='green'
            )
        )
        fig.add_trace(
            go.Bar(
                x=time_labels,
                y=energy_discharged,
                name='Energy Discharged (kWh)',
                marker_color='purple'
            )
        )

    fig.add_trace(
        go.Bar(
            x=time_labels,
            y=[-val for val in excess_energy],
            name='Excess Energy (kWh)',
            marker_color='cyan'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=time_labels,
            y=daily_data['Consumption_kWh'].values,
            mode='lines+markers',
            name='Electricity Demand (kWh)',
            line=dict(color='blue', dash='dash')
        )
    )

    fig.update_layout(
        title=f'Solar + Wind Production, Battery Usage, and Demand on {selected_date_ts.strftime("%Y-%m-%d")}',
        xaxis_title='Time of Day',
        yaxis_title='Power (kWh)',
        template='plotly_white',
        barmode='relative',
        height=400
    )

    soc_message = (
        f"State of Charge at the end of {selected_date_ts.strftime('%Y-%m-%d')}: "
        f"{battery_soc:.2f}%"
    )

    logger.info(f"Generated plot for {selected_date} with SOC: {battery_soc:.2f}%.")
    
    return fig, soc_message
