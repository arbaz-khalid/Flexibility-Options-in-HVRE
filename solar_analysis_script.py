# Solar Analysis Script

import pandas as pd
import numpy as np

# Constants and Monthly Solar Data
panel_capacity = 610
converter_efficiency = 0.90
global_irradiance = 4560
NOCT = 43
temperature_coefficient = 0.003
monthly_solar_energy = {1: 2.35, 2: 3.53, 3: 5.02, 4: 5.48, 5: 5.61, 6: 6.16, 7: 6.55, 8: 6.14, 9: 5.4, 10: 3.91, 11: 2.5, 12: 2.02}
monthly_temperatures = {1: 5.2, 2: 6.3, 3: 10.2, 4: 13.5, 5: 17.2, 6: 21.2, 7: 22.6, 8: 22.4, 9: 18, 10: 15.3, 11: 9.8, 12: 5.9}
initial_soc = 50
battery_capacity_per_unit = 57  # Capacity of each battery in kWh

# Load Data
def load_data(file_path, start_date, end_date):
    """
    Loads and preprocesses the CSV data, filters it for the given date range.
    """
    print("Loading data from file: ", file_path)
    # Read the CSV file with semicolon as the delimiter and specify column names
    data = pd.read_csv(file_path, sep=';', header=0)
    
    # Rename columns to remove unwanted characters and standardize
    data.columns = ['Time', 'Consumption']
    
    # Parse dates in the 'Time' column
    data['Time'] = pd.to_datetime(data['Time'], format='%Y/%m/%d %H:%M', errors='coerce')
    
    # Remove rows where 'Time' could not be parsed
    data.dropna(subset=['Time'], inplace=True)
    
    # Filter data for the specified date range
    filtered_data = data[(data['Time'] >= start_date) & (data['Time'] < end_date)]
    
    # Remove non-numeric characters from the 'Consumption' column and convert to numeric
    filtered_data['Consumption'] = (
        filtered_data['Consumption']
        .astype(str)                           # Ensure all values are treated as strings
        .str.replace(' kWh', '', regex=False)  # Remove " kWh"
        .str.replace(',', '', regex=False)     # Remove commas
        .astype(float)                         # Convert to float
    )
    
    # Drop rows with NaN values in 'Consumption'
    filtered_data.dropna(subset=['Consumption'], inplace=True)
    
    # Ensure filtered data is not empty
    if filtered_data.empty:
        raise ValueError("No data available for the specified date range.")
    
    # Set the date as the index
    filtered_data.set_index('Time', inplace=True)
    
    print("Data successfully loaded and filtered.")
    return filtered_data

# PV Production Calculation 
def calculate_monthly_pv_production(month, panels, panel_capacity):
    """
    Calculate PV production per hour for a given month, number of panels, and panel capacity.
    Uses a normal distribution profile to simulate daily production from 6 AM to 6 PM.
    """
    print(f"Calculating monthly PV production for month: {month} with {panels} panels and panel capacity {panel_capacity}W")
    
    T_ambient = monthly_temperatures[month]
    actual_pv_temp = T_ambient + (NOCT - 20) * (global_irradiance / 800)
    actual_energy_per_panel = (monthly_solar_energy[month] / 1000) * panel_capacity * ((actual_pv_temp - 25) * temperature_coefficient + 1)
    production_per_hour = np.zeros(24)
    hours = np.arange(6, 19)
    mu, sigma = 12, 3
    prodPV_normal = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((hours - mu) / sigma) ** 2)
    prodPV_watts = panels * prodPV_normal * converter_efficiency * actual_energy_per_panel
    production_per_hour[hours] = prodPV_watts
    return pd.Series(production_per_hour, index=np.arange(24))

# Annual Metrics Calculation without Battery
def calculate_annual_metrics(filtered_data, num_panels, panel_capacity):
    """
    Calculate annual energy metrics without battery storage.
    """
    print("Calculating annual metrics without battery storage...")
    annual_pv_produced_kwh = 0
    annual_imported_kwh = 0
    annual_exported_kwh = 0
    total_energy_consumed = 0

    # Loop through each row in the filtered_data (which is hourly)
    for timestamp, row in filtered_data.iterrows():
        consumption = row['Consumption']
        month = timestamp.month

        # Calculate PV production for the corresponding month (based on the time)
        pv_production = calculate_monthly_pv_production(month, num_panels, panel_capacity)[timestamp.hour]

        # Calculate energy balance
        if consumption > pv_production:
            remaining_load = consumption - pv_production
            annual_imported_kwh += remaining_load
        else:
            excess_energy = pv_production - consumption
            annual_exported_kwh += excess_energy

        # Total energy consumed (sum of all consumption values)
        total_energy_consumed += consumption

        # Accumulate PV production
        annual_pv_produced_kwh += pv_production

    return {
        "annual_pv_produced_kwh": round(annual_pv_produced_kwh / 1000, 1),
        "annual_imported_kwh": round(annual_imported_kwh / 1000, 1),
        "annual_exported_kwh": round(annual_exported_kwh / 1000, 1),
        "total_energy_consumed": round(total_energy_consumed / 1000, 1),
        "energy_delivered_to_load": round((annual_pv_produced_kwh - annual_exported_kwh + annual_imported_kwh) / 1000, 1),
        "performance_indicator": round((annual_pv_produced_kwh / total_energy_consumed) * 100, 1)
    }

# Annual Metrics Calculation with Battery
def calculate_annual_metrics_with_battery(filtered_data, num_panels, panel_capacity, num_batteries, initial_soc):
    """
    Calculate annual energy metrics with battery storage.
    """
    print("Calculating annual metrics with battery storage...")
    battery_capacity = battery_capacity_per_unit * num_batteries  # Total battery capacity in kWh
    annual_pv_produced_kwh = 0
    annual_storage_supplied_kwh = 0
    annual_storage_consumed_kwh = 0
    annual_imported_kwh = 0
    annual_exported_kwh = 0
    total_energy_consumed = 0
    soc = initial_soc

    # Iterate directly over each hourly timestamp in filtered_data
    for timestamp, row in filtered_data.iterrows():
        current_load = row['Consumption']
        month = timestamp.month

        # Calculate PV production for the specific hour
        hourly_pv_production = calculate_monthly_pv_production(month, num_panels, panel_capacity)[timestamp.hour]
        annual_pv_produced_kwh += hourly_pv_production

        if current_load > hourly_pv_production:
            # Load exceeds PV production
            remaining_load = current_load - hourly_pv_production
            if soc > 20:
                # Discharge battery
                discharge_energy = min((soc / 100) * battery_capacity, remaining_load)
                annual_storage_supplied_kwh += discharge_energy
                soc -= (discharge_energy / battery_capacity) * 100
                remaining_load -= discharge_energy
            # Import remaining load from the grid if necessary
            if remaining_load > 0:
                annual_imported_kwh += remaining_load
            total_energy_consumed += current_load
        else:
            # PV production exceeds load
            excess_energy = hourly_pv_production - current_load
            if soc < 100:
                # Charge battery with excess energy
                charge_energy = min(excess_energy * converter_efficiency, (100 - soc) / 100 * battery_capacity)
                annual_storage_consumed_kwh += charge_energy
                soc += (charge_energy / battery_capacity) * 100
            else:
                # Export to grid if battery is full
                annual_exported_kwh += excess_energy
            total_energy_consumed += current_load

        # Ensure SOC remains within 0-100%
        soc = max(0, min(100, soc))

    return {
        "annual_pv_produced_kwh": round(annual_pv_produced_kwh / 1000, 1),
        "annual_storage_supplied_kwh": round(annual_storage_supplied_kwh / 1000, 1),
        "annual_storage_consumed_kwh": round(annual_storage_consumed_kwh / 1000, 1),
        "annual_imported_kwh": round(annual_imported_kwh / 1000, 1),
        "annual_exported_kwh": round(annual_exported_kwh / 1000, 1),
        "total_energy_consumed": round(total_energy_consumed / 1000, 1),
        "energy_delivered_to_load": round((annual_pv_produced_kwh - annual_exported_kwh + annual_imported_kwh + annual_storage_supplied_kwh) / 1000, 1),
        "performance_indicator": round((annual_pv_produced_kwh / total_energy_consumed) * 100, 1)
    }

# Monthly Import/Export Calculation
def monthly_import_export(filtered_data, num_panels, panel_capacity):
    """
    Calculate monthly energy import and export based on average daily PV production.
    """
    import_kwh = []
    export_kwh = []
    for month in range(1, 13):
        monthly_avg_load = filtered_data[filtered_data.index.month == month].resample('H').mean().groupby(lambda x: x.hour).mean()['Consumption']
        monthly_pv_production = calculate_monthly_pv_production(month, num_panels, panel_capacity)
        total_import, total_export = 0, 0

        for hour in range(24):
            pv_power = monthly_pv_production[hour]
            consumption = monthly_avg_load[hour]

            if pv_power > consumption:
                total_export += pv_power - consumption
            else:
                total_import += consumption - pv_power

        import_kwh.append(total_import)
        export_kwh.append(total_export)

    return import_kwh, export_kwh


