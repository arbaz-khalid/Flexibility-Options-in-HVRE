# solar_analysis_script.py

import pandas as pd
import numpy as np
import logging
from plotly.subplots import make_subplots
import plotly.graph_objs as go

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# ----------------------------
# System Parameters
# ----------------------------
INITIAL_NUM_PANELS = 1000      # Default number of PV panels
INITIAL_NUM_TURBINES = 1       # Default number of wind turbines
BATTERY_CAPACITY_PER_UNIT = 57 # kWh per battery
CONVERTER_EFFICIENCY = 0.9     # 90% efficiency
INITIAL_SOC = 0              # Initial State of Charge (SOC)

panel_capacity = 610
global_irradiance = 4560
NOCT = 43
temperature_coefficient = 0.003
monthly_solar_energy = {1: 2.35, 2: 3.53, 3: 5.02, 4: 5.48, 5: 5.61, 6: 6.16,
                        7: 6.55, 8: 6.14, 9: 5.4, 10: 3.91, 11: 2.5, 12: 2.02}
monthly_temperatures = {1: 5.2, 2: 6.3, 3: 10.2, 4: 13.5, 5: 17.2, 6: 21.2,
                        7: 22.6, 8: 22.4, 9: 18, 10: 15.3, 11: 9.8, 12: 5.9}
initial_soc = 50  # For example, starting at 50%
battery_capacity_per_unit = 57  # kWh per battery

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
                        'Consumption_kWh':[0]*24,
                        'PV_Production_kWh':[0]*24,
                        'Wind_Production_kWh':[0]*24
                    },
                    index=pd.date_range(current_date, periods=24, freq='H')
                )

            if isinstance(daily_data, pd.Series):
                daily_data = daily_data.to_frame().T

            if daily_data.shape[0] != 24:
                logger.warning("Expected 24 hours but found %d for %s.", daily_data.shape[0], current_date)
                daily_data = daily_data.reindex(
                    pd.date_range(start=current_date, periods=24, freq='H'), 
                    fill_value=0
                )

            for hour in range(24):
                consumption = daily_data.iloc[hour]['Consumption_kWh']
                pv_hour   = daily_data.iloc[hour]['PV_Production_kWh']
                wind_hour = daily_data.iloc[hour]['Wind_Production_kWh']
                total_generation = pv_hour + wind_hour

                if total_generation >= consumption:
                    excess = total_generation - consumption
                    if (num_batteries > 0) and (soc < 100):
                        charge = min(excess * converter_efficiency, (100 - soc) / 100 * battery_capacity)
                        soc += (charge / battery_capacity) * 100
                else:
                    deficit = consumption - total_generation
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
    Load city consumption data from a CSV file.
    Note: The data in the file is in MWh. This function converts the values to kWh.
    Instead of aggregating (resampling) to hourly sums, we filter to keep only the rows
    at full-hour timestamps.
    """
    logger.info("Loading consumption data from %s...", consumption_file_path)
    try:
        data = pd.read_csv(consumption_file_path, sep=consumption_sep)
    except Exception as err:
        logger.error("Error reading consumption file %s: %s", consumption_file_path, err)
        raise err

    # Combine date and time columns into a single Datetime column
    data['Datetime'] = pd.to_datetime(
        data[consumption_date_col] + ' ' + data[consumption_time_col],
        format='%Y/%m/%d %H:%M',
        errors='coerce'
    )
    data.dropna(subset=['Datetime'], inplace=True)
    data.set_index('Datetime', inplace=True)

    # Convert the consumption values (originally in MWh) to numeric and then to kWh
    data[consumption_value_col] = pd.to_numeric(
        data[consumption_value_col].replace('ND', np.nan),
        errors='coerce'
    )
    data[consumption_value_col].fillna(0, inplace=True)
    # Conversion: 1 MWh = 1000 kWh
    data[consumption_value_col] = data[consumption_value_col] * 1000

    # Instead of resampling (which sums the 15-min intervals), keep only the rows at full-hour times.
    data_hourly = data[data.index.minute == 0].copy()
    data_hourly.rename(columns={consumption_value_col: 'Consumption_kWh'}, inplace=True)

    logger.info("Consumption data loaded, converted from MWh to kWh, and filtered to full-hour timestamps.")
    return data_hourly['Consumption_kWh']

def load_city_pv_production(
    pv_file_path: str,
    consumption_start_date: str,
    consumption_end_date: str,
    sep: str = ',',
    time_start_col: str = 'time',
    time_end_col: str = 'local_time',
    electricity_col: str = 'electricity'
) -> pd.Series:
    logger.info("Loading PV production data from %s...", pv_file_path)
    try:
        pv_data = pd.read_csv(pv_file_path, sep=sep)
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
    logger.info("Loading wind production data from %s...", wind_file_path)
    try:
        wind_data = pd.read_csv(wind_file_path, sep=sep)
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
    return wind_production

def merge_city_data(
    consumption: pd.Series,
    pv_production: pd.Series,
    wind_production: pd.Series,
    num_panels: int,
    num_turbines: int
) -> pd.DataFrame:
    logger.info("Merging consumption, PV production, and wind production data.")
    pv_total = pv_production * num_panels
    wind_total = wind_production * num_turbines
    merged_df = pd.DataFrame({
        'Consumption_kWh': consumption,
        'PV_Production_kWh': pv_total,
        'Wind_Production_kWh': wind_total
    })
    merged_df.fillna(0, inplace=True)
    logger.info("Merging complete (PV + Wind included).")
    return merged_df

def update_weekly_graph(
    city_consumption: pd.Series,
    city_pv_production: pd.Series,
    city_wind_production: pd.Series,
    soc_tracker,
    num_panels: int,
    num_turbines: int,
    num_batteries: int,
    selected_week_start: str,
    battery_capacity_per_unit: float = BATTERY_CAPACITY_PER_UNIT,
    converter_efficiency: float = CONVERTER_EFFICIENCY,
    initial_soc: float = INITIAL_SOC
) -> dict:
    """
    Updates the weekly graph based on the selected parameters.
    Returns a dictionary with a Plotly figure and summary metrics.
    
    Note: Although the internal calculations are in kWh, the graph and summary 
    display energy values in MWh (i.e. by dividing by 1000).
    """
    # Convert selected_week_start to Timestamp
    selected_week_start_ts = pd.Timestamp(selected_week_start)
    selected_week_end_ts = selected_week_start_ts + pd.Timedelta(days=7)

    # Merge the data using the module-level merge_city_data function
    try:
        current_merged_data = merge_city_data(
            consumption=city_consumption,
            pv_production=city_pv_production,
            wind_production=city_wind_production,
            num_panels=num_panels,
            num_turbines=num_turbines
        )
    except Exception as err:
        logger.error("Failed to merge data: %s", err)
        raise

    try:
        weekly_data = current_merged_data.loc[selected_week_start_ts:selected_week_end_ts - pd.Timedelta(hours=1)]
    except KeyError:
        logger.error("No data available for the selected week starting on: %s", selected_week_start)
        raise ValueError(f"No data available for the selected week starting on: {selected_week_start}")

    if weekly_data.empty:
        logger.warning("No data available for the selected week starting on: %s", selected_week_start)
        raise ValueError(f"No data available for the selected week starting on: {selected_week_start}")

    battery_capacity = battery_capacity_per_unit * num_batteries
    soc = initial_soc
    soc_tracker.reset_soc_from_date(selected_week_start_ts)
    soc_tracker.initialize_battery_soc(initial_soc, selected_week_start_ts)

    # Lists to track hourly metrics (in kWh)
    energy_charged    = []
    energy_discharged = []
    unmet_demand      = []
    excess_energy     = []
    daily_consumption = []
    daily_pv          = []
    daily_wind        = []
    time_stamps       = []  # Store the actual timestamps

    for day in range(7):
        current_day = selected_week_start_ts + pd.Timedelta(days=day)
        for hour in range(24):
            timestamp = current_day + pd.Timedelta(hours=hour)
            time_stamps.append(timestamp)
            if timestamp not in weekly_data.index:
                consumption = 0
                pv_production = 0
                wind_production = 0
            else:
                consumption     = weekly_data.loc[timestamp]['Consumption_kWh']
                pv_production   = weekly_data.loc[timestamp]['PV_Production_kWh']
                wind_production = weekly_data.loc[timestamp]['Wind_Production_kWh']

            total_generation = pv_production + wind_production

            if total_generation >= consumption:
                excess = total_generation - consumption
                if num_batteries > 0 and soc < 100:
                    charge = min(excess * converter_efficiency, (100 - soc) / 100 * battery_capacity)
                    energy_charged.append(charge)
                    soc += (charge / battery_capacity) * 100
                    excess_after_charge = excess - (charge / converter_efficiency)
                    excess_energy.append(excess_after_charge)
                    energy_discharged.append(0)
                    unmet_demand.append(0)
                else:
                    energy_charged.append(0)
                    excess_energy.append(excess)
                    energy_discharged.append(0)
                    unmet_demand.append(0)
            else:
                deficit = consumption - total_generation
                if num_batteries > 0 and soc > 0:
                    discharge = min(deficit, (soc / 100) * battery_capacity)
                    energy_discharged.append(discharge)
                    soc -= (discharge / battery_capacity) * 100
                    unmet_demand.append(deficit - discharge)
                    energy_charged.append(0)
                    excess_energy.append(0)
                else:
                    energy_discharged.append(0)
                    unmet_demand.append(deficit)
                    energy_charged.append(0)
                    excess_energy.append(0)

            daily_consumption.append(consumption)
            daily_pv.append(pv_production)
            daily_wind.append(wind_production)
            soc = max(0, min(100, soc))

    total_pv           = sum(daily_pv)
    total_wind         = sum(daily_wind)
    total_consumption  = sum(daily_consumption)
    total_charged      = sum(energy_charged)
    total_discharged   = sum(energy_discharged)
    total_unmet        = sum(unmet_demand)
    total_excess       = sum(excess_energy)
    total_energy_supplied = total_pv + total_wind + total_discharged
    ssr = ((total_pv + total_wind + total_discharged) / total_consumption) * 100 if total_consumption > 0 else 0

    # Conversion factor: 1000 kWh = 1 MWh
    cf = 1000

    # Build a list of labels using the actual hourly timestamps (formatted with full time)
    hour_labels = [ts.strftime('%H:%M') for ts in time_stamps]

    # For the x-axis, we want to show a date label only at the start of each day.
    # We'll build tick labels so that if the timestamp is the first hour (i.e. 00:00), we show the date.
    tickvals = list(range(len(time_stamps)))
    ticktext = []
    for ts in time_stamps:
        if ts.hour == 0:
            ticktext.append(ts.strftime('%a %d-%b'))
        else:
            ticktext.append('')

    # Build the Plotly figure using MWh values
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=list(range(len(time_stamps))),
            y=[val/cf for val in daily_pv],
            name='Solar PV (MWh)',
            marker_color='orange'
        )
    )

    fig.add_trace(
        go.Bar(
            x=list(range(len(time_stamps))),
            y=[val/cf for val in daily_wind],
            name='Wind (MWh)',
            marker_color='lightblue'
        )
    )

    fig.add_trace(
        go.Bar(
            x=list(range(len(time_stamps))),
            y=[val/cf for val in unmet_demand],
            name='Unmet Demand (MWh)',
            marker_color='red'
        )
    )

    if num_batteries > 0:
        fig.add_trace(
            go.Bar(
                x=list(range(len(time_stamps))),
                y=[-val/cf for val in energy_charged],
                name='Energy Charged to Battery (MWh)',
                marker_color='green'
            )
        )
        fig.add_trace(
            go.Bar(
                x=list(range(len(time_stamps))),
                y=[val/cf for val in energy_discharged],
                name='Energy Discharged (MWh)',
                marker_color='purple'
            )
        )

    fig.add_trace(
        go.Bar(
            x=list(range(len(time_stamps))),
            y=[-val/cf for val in excess_energy],
            name='Excess Energy (MWh)',
            marker_color='cyan'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(len(time_stamps))),
            y=[val/cf for val in daily_consumption],
            mode='lines+markers',
            name='Electricity Demand (MWh)',
            line=dict(color='blue', dash='dash')
        )
    )

    fig.update_layout(
        title=f'Solar + Wind Production, Battery Usage, and Demand from {selected_week_start_ts.strftime("%Y-%m-%d")} to {(selected_week_start_ts + pd.Timedelta(days=6)).strftime("%Y-%m-%d")}',
        xaxis_title='Time',
        yaxis_title='Power (MWh)',
        xaxis=dict(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=-45
        ),
        template='plotly_white',
        barmode='relative',
        height=900,
        margin=dict(b=200)
    )

    # Build summary metrics in MWh
    summary_metrics = {
        "Total Solar PV Produced (MWh)": round(total_pv/cf, 1),
        "Total Wind Produced (MWh)": round(total_wind/cf, 1),
        "Total Energy Charged to Battery (MWh)": round(total_charged/cf, 1),
        "Total Energy Discharged from Battery (MWh)": round(total_discharged/cf, 1),
        "Total Unmet Demand (MWh)": round(total_unmet/cf, 1),
        "Total Excess Energy (MWh)": round(total_excess/cf, 1),
        "Total Energy Supplied to Demand (MWh)": round(total_energy_supplied/cf, 1),
        "Total Energy Demand (MWh)": round(total_consumption/cf, 1),
        "Self-Sufficiency Rate (SSR)": round(ssr, 1)
    }

    return {
        "figure": fig,
        "metrics": summary_metrics
    }
