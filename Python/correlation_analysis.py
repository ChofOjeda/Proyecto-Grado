import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

def load_comprehensive_data():
    """Load and prepare all data for comprehensive correlation analysis"""
    print("="*80)
    print("COMPREHENSIVE CORRELATION ANALYSIS")
    print("="*80)
    print("Loading data from 'Datos Unificados1.ods'...")
    
    # Read all sheets
    sheets = pd.read_excel('Datos Unificados1.ods', sheet_name=None, engine='odf')
    
    print(f"Available sheets: {list(sheets.keys())}")
    
    # Load averages data
    averages_data = sheets['Promedios y desviacion'].copy()
    
    # Load raw interior data for more detailed analysis
    ied_interior = sheets['IED Interior'].copy() if 'IED Interior' in sheets else pd.DataFrame()
    sta_interior = sheets['Santa Juanita Interior'].copy() if 'Santa Juanita Interior' in sheets else pd.DataFrame()
    ied_exterior = sheets['IED Exterior'].copy() if 'IED Exterior' in sheets else pd.DataFrame()
    sta_exterior = sheets['Santa Juanita Exterior'].copy() if 'Santa Juanita Exterior' in sheets else pd.DataFrame()
    
    # Add school identifiers to raw data
    if not ied_interior.empty:
        ied_interior['School'] = 'IED'
        ied_interior['Environment'] = 'Indoor'
    if not sta_interior.empty:
        sta_interior['School'] = 'STA_Juanita'
        sta_interior['Environment'] = 'Indoor'
    if not ied_exterior.empty:
        ied_exterior['School'] = 'IED'
        ied_exterior['Environment'] = 'Outdoor'
    if not sta_exterior.empty:
        sta_exterior['School'] = 'STA_Juanita'
        sta_exterior['Environment'] = 'Outdoor'
    
    # Combine raw data
    raw_data_list = [df for df in [ied_interior, sta_interior, ied_exterior, sta_exterior] if not df.empty]
    raw_data = pd.concat(raw_data_list, ignore_index=True) if raw_data_list else pd.DataFrame()
    
    print(f"\nData loaded successfully:")
    print(f"  Averages data shape: {averages_data.shape}")
    print(f"  Raw data shape: {raw_data.shape}")
    
    return averages_data, raw_data, sheets

def standardize_column_names(df):
    """Standardize column names for easier analysis"""
    # Create a mapping for common variations
    column_mapping = {
        'PM2.5(ug/m3)': 'PM25',
        'PM2.5(ug/m3) MEAN': 'PM25',
        'PM10(ug/m3)': 'PM10', 
        'PM10(ug/m3) MEAN': 'PM10',
        'TEMP': 'Temperature',
        'TEMP MEAN': 'Temperature',
        'HUMI(%RH)': 'Humidity',
        'HUMI(%RH) MEAN': 'Humidity',
        'HCHO(mg/m3)': 'HCHO',
        'HCHO(mg/m3) MEAN': 'HCHO',
        'TVOC(mg/m3)': 'TVOC',
        'TVOC(mg/m3) MEAN': 'TVOC',
        'CO2(ppm)': 'CO2',
        'CO(ppm)': 'CO',
        'NO2(ppb)': 'NO2',
        'PARTICLES(per/L)': 'Particles',
        'AQI': 'AQI'
    }
    
    # Apply mapping
    df_copy = df.copy()
    df_copy.columns = [column_mapping.get(col, col) for col in df_copy.columns]
    
    return df_copy

def calculate_correlation_with_significance(x, y, method='pearson'):
    """Calculate correlation with significance test"""
    # Convert to pandas Series if not already
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    
    # Convert to numeric, coercing errors to NaN
    x = pd.to_numeric(x, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    
    # Remove NaN values
    mask = ~(x.isna() | y.isna())
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 3:
        return np.nan, np.nan, 0
    
    try:
        if method == 'pearson':
            corr, p_value = pearsonr(x_clean, y_clean)
        elif method == 'spearman':
            corr, p_value = spearmanr(x_clean, y_clean)
        else:
            raise ValueError("Method must be 'pearson' or 'spearman'")
        
        return corr, p_value, len(x_clean)
    except Exception as e:
        print(f"  ⚠️ Error calculating correlation: {e}")
        return np.nan, np.nan, len(x_clean)

def physical_correlations(averages_data, raw_data):
    """Analyze physical correlations: PM2.5 vs Temperature, PM2.5 vs Humidity, PM10 vs PM2.5"""
    print("\n" + "="*60)
    print("PHYSICAL CORRELATIONS")
    print("="*60)
    
    # Standardize column names
    avg_data = standardize_column_names(averages_data)
    raw_data_std = standardize_column_names(raw_data) if not raw_data.empty else pd.DataFrame()
    
    correlations_results = {}
    
    # 1. PM2.5 vs Temperature
    print("\n1. PM2.5 vs TEMPERATURE")
    print("-" * 30)
    
    if 'PM25' in avg_data.columns and 'Temperature' in avg_data.columns:
        corr, p_val, n = calculate_correlation_with_significance(
            avg_data['PM25'].values, avg_data['Temperature'].values
        )
        
        print(f"Using averages data (n={n}):")
        print(f"  Pearson correlation: r = {corr:.4f}, p = {p_val:.6f}")
        
        if p_val < 0.05:
            print("  ✓ SIGNIFICANT correlation")
        else:
            print("  ✗ Not statistically significant")
        
        correlations_results['PM25_vs_Temperature'] = {
            'correlation': corr, 'p_value': p_val, 'n': n, 'data_source': 'averages'
        }
    
    # Check raw data if available
    if not raw_data_std.empty and 'PM25' in raw_data_std.columns and 'Temperature' in raw_data_std.columns:
        corr_raw, p_val_raw, n_raw = calculate_correlation_with_significance(
            raw_data_std['PM25'].values, raw_data_std['Temperature'].values
        )
        
        print(f"\nUsing raw data (n={n_raw}):")
        print(f"  Pearson correlation: r = {corr_raw:.4f}, p = {p_val_raw:.6f}")
        
        if p_val_raw < 0.05:
            print("  ✓ SIGNIFICANT correlation")
        else:
            print("  ✗ Not statistically significant")
    
    # 2. PM2.5 vs Humidity
    print("\n2. PM2.5 vs HUMIDITY")
    print("-" * 30)
    
    if 'PM25' in avg_data.columns and 'Humidity' in avg_data.columns:
        corr, p_val, n = calculate_correlation_with_significance(
            avg_data['PM25'].values, avg_data['Humidity'].values
        )
        
        print(f"Using averages data (n={n}):")
        print(f"  Pearson correlation: r = {corr:.4f}, p = {p_val:.6f}")
        
        if p_val < 0.05:
            print("  ✓ SIGNIFICANT correlation")
        else:
            print("  ✗ Not statistically significant")
        
        correlations_results['PM25_vs_Humidity'] = {
            'correlation': corr, 'p_value': p_val, 'n': n, 'data_source': 'averages'
        }
    
    # Check raw data if available
    if not raw_data_std.empty and 'PM25' in raw_data_std.columns and 'Humidity' in raw_data_std.columns:
        corr_raw, p_val_raw, n_raw = calculate_correlation_with_significance(
            raw_data_std['PM25'].values, raw_data_std['Humidity'].values
        )
        
        print(f"\nUsing raw data (n={n_raw}):")
        print(f"  Pearson correlation: r = {corr_raw:.4f}, p = {p_val_raw:.6f}")
        
        if p_val_raw < 0.05:
            print("  ✓ SIGNIFICANT correlation")
        else:
            print("  ✗ Not statistically significant")
    
    # 3. PM10 vs PM2.5
    print("\n3. PM10 vs PM2.5")
    print("-" * 30)
    
    if 'PM25' in avg_data.columns and 'PM10' in avg_data.columns:
        corr, p_val, n = calculate_correlation_with_significance(
            avg_data['PM10'].values, avg_data['PM25'].values
        )
        
        print(f"Using averages data (n={n}):")
        print(f"  Pearson correlation: r = {corr:.4f}, p = {p_val:.6f}")
        
        if p_val < 0.05:
            print("  ✓ SIGNIFICANT correlation")
        else:
            print("  ✗ Not statistically significant")
        
        correlations_results['PM10_vs_PM25'] = {
            'correlation': corr, 'p_value': p_val, 'n': n, 'data_source': 'averages'
        }
    
    # Check raw data if available
    if not raw_data_std.empty and 'PM25' in raw_data_std.columns and 'PM10' in raw_data_std.columns:
        corr_raw, p_val_raw, n_raw = calculate_correlation_with_significance(
            raw_data_std['PM10'].values, raw_data_std['PM25'].values
        )
        
        print(f"\nUsing raw data (n={n_raw}):")
        print(f"  Pearson correlation: r = {corr_raw:.4f}, p = {p_val_raw:.6f}")
        
        if p_val_raw < 0.05:
            print("  ✓ SIGNIFICANT correlation")
        else:
            print("  ✗ Not statistically significant")
    
    return correlations_results

def environment_correlations(averages_data, raw_data):
    """Analyze environment-based correlations: PM2.5 Indoor vs Outdoor, CO2 Indoor vs Outdoor"""
    print("\n" + "="*60)
    print("ENVIRONMENT-BASED CORRELATIONS")
    print("="*60)
    
    # Standardize column names
    avg_data = standardize_column_names(averages_data)
    
    correlations_results = {}
    
    # Prepare data by environment
    if 'Espacio' in averages_data.columns:
        indoor_data = avg_data[averages_data['Espacio'] == 'Interior'].copy()
        outdoor_data = avg_data[averages_data['Espacio'] == 'Exterior'].copy()
    else:
        print("⚠️ Environment information not available in averages data")
        return correlations_results
    
    print(f"Indoor locations: {len(indoor_data)}")
    print(f"Outdoor locations: {len(outdoor_data)}")
    
    # 1. PM2.5 Indoor vs PM2.5 Outdoor
    print("\n1. PM2.5 INDOOR vs OUTDOOR COMPARISON")
    print("-" * 40)
    
    if 'PM25' in indoor_data.columns and 'PM25' in outdoor_data.columns:
        indoor_pm25 = indoor_data['PM25'].dropna()
        outdoor_pm25 = outdoor_data['PM25'].dropna()
        
        print(f"Indoor PM2.5 statistics:")
        print(f"  Mean: {indoor_pm25.mean():.3f} μg/m³")
        print(f"  Std: {indoor_pm25.std():.3f} μg/m³")
        print(f"  Range: {indoor_pm25.min():.3f} - {indoor_pm25.max():.3f} μg/m³")
        
        print(f"\nOutdoor PM2.5 statistics:")
        print(f"  Mean: {outdoor_pm25.mean():.3f} μg/m³")
        print(f"  Std: {outdoor_pm25.std():.3f} μg/m³")
        print(f"  Range: {outdoor_pm25.min():.3f} - {outdoor_pm25.max():.3f} μg/m³")
        
        # Paired comparison if possible (same number of measurements)
        if len(indoor_pm25) == len(outdoor_pm25) and len(indoor_pm25) > 0:
            corr, p_val, n = calculate_correlation_with_significance(
                indoor_pm25.values, outdoor_pm25.values
            )
            
            print(f"\nCorrelation between Indoor and Outdoor PM2.5:")
            print(f"  Pearson correlation: r = {corr:.4f}, p = {p_val:.6f}")
            
            if p_val < 0.05:
                print("  ✓ SIGNIFICANT correlation")
            else:
                print("  ✗ Not statistically significant")
            
            correlations_results['PM25_Indoor_vs_Outdoor'] = {
                'correlation': corr, 'p_value': p_val, 'n': n
            }
        
        # Statistical test for difference
        if len(indoor_pm25) > 0 and len(outdoor_pm25) > 0:
            t_stat, t_p = stats.ttest_ind(indoor_pm25, outdoor_pm25)
            print(f"\nT-test for difference between Indoor and Outdoor PM2.5:")
            print(f"  t-statistic: {t_stat:.4f}, p = {t_p:.6f}")
            
            if t_p < 0.05:
                print("  ✓ SIGNIFICANT difference between environments")
            else:
                print("  ✗ No significant difference between environments")
    
    # 2. CO2 Indoor vs Outdoor (if available)
    print("\n2. CO2 INDOOR vs OUTDOOR COMPARISON")
    print("-" * 40)
    
    # Check if CO2 data is available
    co2_cols = [col for col in avg_data.columns if 'CO2' in col.upper()]
    
    if co2_cols:
        co2_col = co2_cols[0]
        indoor_co2 = indoor_data[co2_col].dropna()
        outdoor_co2 = outdoor_data[co2_col].dropna()
        
        if len(indoor_co2) > 0 and len(outdoor_co2) > 0:
            print(f"Indoor CO2 statistics:")
            print(f"  Mean: {indoor_co2.mean():.3f} ppm")
            print(f"  Std: {indoor_co2.std():.3f} ppm")
            print(f"  Range: {indoor_co2.min():.3f} - {indoor_co2.max():.3f} ppm")
            
            print(f"\nOutdoor CO2 statistics:")
            print(f"  Mean: {outdoor_co2.mean():.3f} ppm")
            print(f"  Std: {outdoor_co2.std():.3f} ppm")
            print(f"  Range: {outdoor_co2.min():.3f} - {outdoor_co2.max():.3f} ppm")
            
            # Correlation if possible
            if len(indoor_co2) == len(outdoor_co2):
                corr, p_val, n = calculate_correlation_with_significance(
                    indoor_co2.values, outdoor_co2.values
                )
                
                print(f"\nCorrelation between Indoor and Outdoor CO2:")
                print(f"  Pearson correlation: r = {corr:.4f}, p = {p_val:.6f}")
                
                correlations_results['CO2_Indoor_vs_Outdoor'] = {
                    'correlation': corr, 'p_value': p_val, 'n': n
                }
        else:
            print("CO2 column found but insufficient data for comparison")
    else:
        print("⚠️ CO2 data not available in the dataset")
    
    return correlations_results

def pollutant_correlations(averages_data, raw_data):
    """Analyze pollutant correlations: CO vs PM2.5, NO2 vs PM2.5, HCHO vs TVOC"""
    print("\n" + "="*60)
    print("POLLUTANT CORRELATIONS")
    print("="*60)
    
    # Standardize column names
    avg_data = standardize_column_names(averages_data)
    raw_data_std = standardize_column_names(raw_data) if not raw_data.empty else pd.DataFrame()
    
    correlations_results = {}
    
    # 1. CO vs PM2.5
    print("\n1. CO vs PM2.5")
    print("-" * 20)
    
    # Check in averages data
    co_cols = [col for col in avg_data.columns if 'CO' in col.upper() and 'CO2' not in col.upper()]
    
    if co_cols and 'PM25' in avg_data.columns:
        co_col = co_cols[0]
        print(f"Using CO column: {co_col}")
        
        # Ensure both columns are numeric
        co_values = pd.to_numeric(avg_data[co_col], errors='coerce')
        pm25_values = pd.to_numeric(avg_data['PM25'], errors='coerce')
        
        corr, p_val, n = calculate_correlation_with_significance(co_values, pm25_values)
        
        if not np.isnan(corr):
            print(f"Using averages data (n={n}):")
            print(f"  Pearson correlation: r = {corr:.4f}, p = {p_val:.6f}")
            
            if p_val < 0.05:
                print("  ✓ SIGNIFICANT correlation")
            else:
                print("  ✗ Not statistically significant")
            
            correlations_results['CO_vs_PM25'] = {
                'correlation': corr, 'p_value': p_val, 'n': n, 'data_source': 'averages'
            }
        else:
            print(f"  ⚠️ Could not calculate correlation (insufficient data)")
    else:
        print("⚠️ CO data not available")
    
    # Check raw data
    if not raw_data_std.empty:
        co_cols_raw = [col for col in raw_data_std.columns if 'CO' in col.upper() and 'CO2' not in col.upper()]
        if co_cols_raw and 'PM25' in raw_data_std.columns:
            co_col_raw = co_cols_raw[0]
            print(f"\nUsing raw data CO column: {co_col_raw}")
            
            co_values_raw = pd.to_numeric(raw_data_std[co_col_raw], errors='coerce')
            pm25_values_raw = pd.to_numeric(raw_data_std['PM25'], errors='coerce')
            
            corr_raw, p_val_raw, n_raw = calculate_correlation_with_significance(co_values_raw, pm25_values_raw)
            
            if not np.isnan(corr_raw):
                print(f"Using raw data (n={n_raw}):")
                print(f"  Pearson correlation: r = {corr_raw:.4f}, p = {p_val_raw:.6f}")
                
                if p_val_raw < 0.05:
                    print("  ✓ SIGNIFICANT correlation")
                else:
                    print("  ✗ Not statistically significant")
    
    # 2. NO2 vs PM2.5
    print("\n2. NO2 vs PM2.5")
    print("-" * 20)
    
    # Check in averages data
    no2_cols = [col for col in avg_data.columns if 'NO2' in col.upper()]
    
    if no2_cols and 'PM25' in avg_data.columns:
        no2_col = no2_cols[0]
        print(f"Using NO2 column: {no2_col}")
        
        no2_values = pd.to_numeric(avg_data[no2_col], errors='coerce')
        pm25_values = pd.to_numeric(avg_data['PM25'], errors='coerce')
        
        corr, p_val, n = calculate_correlation_with_significance(no2_values, pm25_values)
        
        if not np.isnan(corr):
            print(f"Using averages data (n={n}):")
            print(f"  Pearson correlation: r = {corr:.4f}, p = {p_val:.6f}")
            
            if p_val < 0.05:
                print("  ✓ SIGNIFICANT correlation")
            else:
                print("  ✗ Not statistically significant")
            
            correlations_results['NO2_vs_PM25'] = {
                'correlation': corr, 'p_value': p_val, 'n': n, 'data_source': 'averages'
            }
        else:
            print(f"  ⚠️ Could not calculate correlation (insufficient data)")
    else:
        print("⚠️ NO2 data not available")
    
    # Check raw data
    if not raw_data_std.empty:
        no2_cols_raw = [col for col in raw_data_std.columns if 'NO2' in col.upper()]
        if no2_cols_raw and 'PM25' in raw_data_std.columns:
            no2_col_raw = no2_cols_raw[0]
            print(f"\nUsing raw data NO2 column: {no2_col_raw}")
            
            no2_values_raw = pd.to_numeric(raw_data_std[no2_col_raw], errors='coerce')
            pm25_values_raw = pd.to_numeric(raw_data_std['PM25'], errors='coerce')
            
            corr_raw, p_val_raw, n_raw = calculate_correlation_with_significance(no2_values_raw, pm25_values_raw)
            
            if not np.isnan(corr_raw):
                print(f"Using raw data (n={n_raw}):")
                print(f"  Pearson correlation: r = {corr_raw:.4f}, p = {p_val_raw:.6f}")
                
                if p_val_raw < 0.05:
                    print("  ✓ SIGNIFICANT correlation")
                else:
                    print("  ✗ Not statistically significant")
    
    # 3. HCHO vs TVOC
    print("\n3. HCHO vs TVOC")
    print("-" * 20)
    
    if 'HCHO' in avg_data.columns and 'TVOC' in avg_data.columns:
        hcho_values = pd.to_numeric(avg_data['HCHO'], errors='coerce')
        tvoc_values = pd.to_numeric(avg_data['TVOC'], errors='coerce')
        
        corr, p_val, n = calculate_correlation_with_significance(hcho_values, tvoc_values)
        
        if not np.isnan(corr):
            print(f"Using averages data (n={n}):")
            print(f"  Pearson correlation: r = {corr:.4f}, p = {p_val:.6f}")
            
            if p_val < 0.05:
                print("  ✓ SIGNIFICANT correlation")
            else:
                print("  ✗ Not statistically significant")
            
            correlations_results['HCHO_vs_TVOC'] = {
                'correlation': corr, 'p_value': p_val, 'n': n, 'data_source': 'averages'
            }
        else:
            print(f"  ⚠️ Could not calculate correlation (insufficient data)")
    else:
        print("⚠️ HCHO and/or TVOC data not available")
    
    # Check raw data
    if not raw_data_std.empty and 'HCHO' in raw_data_std.columns and 'TVOC' in raw_data_std.columns:
        hcho_values_raw = pd.to_numeric(raw_data_std['HCHO'], errors='coerce')
        tvoc_values_raw = pd.to_numeric(raw_data_std['TVOC'], errors='coerce')
        
        corr_raw, p_val_raw, n_raw = calculate_correlation_with_significance(hcho_values_raw, tvoc_values_raw)
        
        if not np.isnan(corr_raw):
            print(f"\nUsing raw data (n={n_raw}):")
            print(f"  Pearson correlation: r = {corr_raw:.4f}, p = {p_val_raw:.6f}")
            
            if p_val_raw < 0.05:
                print("  ✓ SIGNIFICANT correlation")
            else:
                print("  ✗ Not statistically significant")
    
    return correlations_results

def school_correlations(averages_data, raw_data):
    """Analyze correlations by school and compare correlation matrices"""
    print("\n" + "="*60)
    print("SCHOOL-SPECIFIC CORRELATIONS")
    print("="*60)
    
    # Standardize column names
    avg_data = standardize_column_names(averages_data)
    
    # Identify schools
    if 'Colegio' in averages_data.columns:
        schools = averages_data['Colegio'].unique()
        print(f"Schools found: {schools}")
    else:
        print("⚠️ School information not available")
        return {}
    
    school_matrices = {}
    
    # Define key variables for correlation analysis
    key_variables = ['PM25', 'PM10', 'Temperature', 'Humidity', 'HCHO', 'TVOC']
    available_vars = [var for var in key_variables if var in avg_data.columns]
    
    print(f"Variables available for correlation: {available_vars}")
    
    if len(available_vars) < 2:
        print("⚠️ Insufficient variables for correlation analysis")
        return {}
    
    # Analyze each school separately
    for school in schools:
        print(f"\n{'='*40}")
        print(f"SCHOOL: {school}")
        print('='*40)
        
        school_data = avg_data[averages_data['Colegio'] == school].copy()
        print(f"Sample size: {len(school_data)} observations")
        
        if len(school_data) < 3:
            print(f"⚠️ Insufficient data for {school} (n={len(school_data)})")
            continue
        
        # Calculate correlation matrix for this school
        school_corr_data = school_data[available_vars].select_dtypes(include=[np.number])
        
        if school_corr_data.empty:
            print(f"⚠️ No numeric data available for {school}")
            continue
        
        # Remove columns with all NaN
        school_corr_data = school_corr_data.dropna(axis=1, how='all')
        available_vars_school = school_corr_data.columns.tolist()
        
        if len(available_vars_school) < 2:
            print(f"⚠️ Insufficient variables for {school}")
            continue
        
        print(f"Variables for {school}: {available_vars_school}")
        
        # Calculate correlation matrix
        corr_matrix = school_corr_data.corr()
        school_matrices[school] = corr_matrix
        
        print(f"\nCorrelation Matrix for {school}:")
        print(corr_matrix.round(3))
        
        # Highlight strong correlations (|r| > 0.7)
        print(f"\nStrong correlations (|r| > 0.7) in {school}:")
        strong_corrs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                
                if abs(corr_val) > 0.7 and not np.isnan(corr_val):
                    # Calculate significance
                    x = school_corr_data[var1].dropna()
                    y = school_corr_data[var2].dropna()
                    
                    # Find common indices
                    common_idx = x.index.intersection(y.index)
                    if len(common_idx) >= 3:
                        _, p_val = pearsonr(x.loc[common_idx], y.loc[common_idx])
                        sig_mark = "✓" if p_val < 0.05 else "✗"
                        strong_corrs.append(f"  {var1} ↔ {var2}: r = {corr_val:.3f} ({sig_mark})")
        
        if strong_corrs:
            for corr_str in strong_corrs:
                print(corr_str)
        else:
            print("  No strong correlations found")
        
        # Moderate correlations (0.4 < |r| < 0.7)
        print(f"\nModerate correlations (0.4 < |r| < 0.7) in {school}:")
        moderate_corrs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                
                if 0.4 < abs(corr_val) <= 0.7 and not np.isnan(corr_val):
                    # Calculate significance
                    x = school_corr_data[var1].dropna()
                    y = school_corr_data[var2].dropna()
                    
                    # Find common indices
                    common_idx = x.index.intersection(y.index)
                    if len(common_idx) >= 3:
                        _, p_val = pearsonr(x.loc[common_idx], y.loc[common_idx])
                        sig_mark = "✓" if p_val < 0.05 else "✗"
                        moderate_corrs.append(f"  {var1} ↔ {var2}: r = {corr_val:.3f} ({sig_mark})")
        
        if moderate_corrs:
            for corr_str in moderate_corrs:
                print(corr_str)
        else:
            print("  No moderate correlations found")
    
    # Compare correlation matrices between schools
    if len(school_matrices) >= 2:
        print(f"\n{'='*60}")
        print("COMPARISON BETWEEN SCHOOLS")
        print('='*60)
        
        school_names = list(school_matrices.keys())
        
        # Find common variables across schools
        common_vars = set(school_matrices[school_names[0]].columns)
        for school in school_names[1:]:
            common_vars = common_vars.intersection(set(school_matrices[school].columns))
        
        common_vars = list(common_vars)
        print(f"Common variables across schools: {common_vars}")
        
        if len(common_vars) >= 2:
            # Compare specific correlations
            print(f"\nCorrelation comparison between schools:")
            print("-" * 50)
            
            for i in range(len(common_vars)):
                for j in range(i+1, len(common_vars)):
                    var1, var2 = common_vars[i], common_vars[j]
                    
                    print(f"\n{var1} ↔ {var2}:")
                    
                    correlations = {}
                    for school in school_names:
                        if var1 in school_matrices[school].columns and var2 in school_matrices[school].columns:
                            corr_val = school_matrices[school].loc[var1, var2]
                            correlations[school] = corr_val
                            print(f"  {school}: r = {corr_val:.3f}")
                    
                    # Calculate difference between schools
                    if len(correlations) == 2:
                        school_list = list(correlations.keys())
                        diff = abs(correlations[school_list[0]] - correlations[school_list[1]])
                        print(f"  Difference: |Δr| = {diff:.3f}")
                        
                        if diff > 0.3:
                            print("  → LARGE difference between schools")
                        elif diff > 0.1:
                            print("  → Moderate difference between schools")
                        else:
                            print("  → Similar across schools")
    
    return school_matrices

def create_correlation_visualizations(averages_data, all_correlations):
    """Create comprehensive correlation visualizations"""
    print("\n" + "="*60)
    print("CREATING CORRELATION VISUALIZATIONS")
    print("="*60)
    
    # Standardize column names
    avg_data = standardize_column_names(averages_data)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("coolwarm")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Overall correlation heatmap
    plt.subplot(2, 3, 1)
    
    # Select numeric columns for correlation
    numeric_cols = avg_data.select_dtypes(include=[np.number]).columns
    corr_vars = [col for col in ['PM25', 'PM10', 'Temperature', 'Humidity', 'HCHO', 'TVOC'] 
                 if col in numeric_cols]
    
    if len(corr_vars) >= 2:
        corr_matrix = avg_data[corr_vars].corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.3f')
        plt.title('Overall Correlation Matrix\n(All Variables)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    # 2. PM2.5 vs Temperature scatter plot
    plt.subplot(2, 3, 2)
    if 'PM25' in avg_data.columns and 'Temperature' in avg_data.columns:
        # Color by environment if available
        if 'Espacio' in averages_data.columns:
            environments = averages_data['Espacio'].map({'Interior': 'Indoor', 'Exterior': 'Outdoor'})
            scatter = plt.scatter(avg_data['Temperature'], avg_data['PM25'], 
                                c=['red' if env == 'Indoor' else 'blue' for env in environments],
                                alpha=0.7, s=60)
            
            # Add legend
            indoor_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                    markersize=8, label='Indoor')
            outdoor_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                                     markersize=8, label='Outdoor')
            plt.legend(handles=[indoor_patch, outdoor_patch])
        else:
            plt.scatter(avg_data['Temperature'], avg_data['PM25'], alpha=0.7, s=60)
        
        plt.xlabel('Temperature')
        plt.ylabel('PM2.5 (μg/m³)')
        plt.title('PM2.5 vs Temperature')
        
        # Add correlation coefficient
        corr_val = avg_data['PM25'].corr(avg_data['Temperature'])
        plt.text(0.05, 0.95, f'r = {corr_val:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 3. PM2.5 vs Humidity scatter plot
    plt.subplot(2, 3, 3)
    if 'PM25' in avg_data.columns and 'Humidity' in avg_data.columns:
        if 'Espacio' in averages_data.columns:
            environments = averages_data['Espacio'].map({'Interior': 'Indoor', 'Exterior': 'Outdoor'})
            plt.scatter(avg_data['Humidity'], avg_data['PM25'], 
                       c=['red' if env == 'Indoor' else 'blue' for env in environments],
                       alpha=0.7, s=60)
        else:
            plt.scatter(avg_data['Humidity'], avg_data['PM25'], alpha=0.7, s=60)
        
        plt.xlabel('Humidity (%RH)')
        plt.ylabel('PM2.5 (μg/m³)')
        plt.title('PM2.5 vs Humidity')
        
        # Add correlation coefficient
        corr_val = avg_data['PM25'].corr(avg_data['Humidity'])
        plt.text(0.05, 0.95, f'r = {corr_val:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 4. PM10 vs PM2.5 scatter plot
    plt.subplot(2, 3, 4)
    if 'PM25' in avg_data.columns and 'PM10' in avg_data.columns:
        if 'Espacio' in averages_data.columns:
            environments = averages_data['Espacio'].map({'Interior': 'Indoor', 'Exterior': 'Outdoor'})
            plt.scatter(avg_data['PM25'], avg_data['PM10'], 
                       c=['red' if env == 'Indoor' else 'blue' for env in environments],
                       alpha=0.7, s=60)
        else:
            plt.scatter(avg_data['PM25'], avg_data['PM10'], alpha=0.7, s=60)
        
        plt.xlabel('PM2.5 (μg/m³)')
        plt.ylabel('PM10 (μg/m³)')
        plt.title('PM10 vs PM2.5')
        
        # Add correlation coefficient
        corr_val = avg_data['PM25'].corr(avg_data['PM10'])
        plt.text(0.05, 0.95, f'r = {corr_val:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 5. HCHO vs TVOC scatter plot
    plt.subplot(2, 3, 5)
    if 'HCHO' in avg_data.columns and 'TVOC' in avg_data.columns:
        if 'Espacio' in averages_data.columns:
            environments = averages_data['Espacio'].map({'Interior': 'Indoor', 'Exterior': 'Outdoor'})
            plt.scatter(avg_data['HCHO'], avg_data['TVOC'], 
                       c=['red' if env == 'Indoor' else 'blue' for env in environments],
                       alpha=0.7, s=60)
        else:
            plt.scatter(avg_data['HCHO'], avg_data['TVOC'], alpha=0.7, s=60)
        
        plt.xlabel('HCHO (mg/m³)')
        plt.ylabel('TVOC (mg/m³)')
        plt.title('HCHO vs TVOC')
        
        # Add correlation coefficient
        corr_val = avg_data['HCHO'].corr(avg_data['TVOC'])
        plt.text(0.05, 0.95, f'r = {corr_val:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 6. Indoor vs Outdoor PM2.5 comparison
    plt.subplot(2, 3, 6)
    if 'Espacio' in averages_data.columns and 'PM25' in avg_data.columns:
        indoor_pm25 = avg_data[averages_data['Espacio'] == 'Interior']['PM25']
        outdoor_pm25 = avg_data[averages_data['Espacio'] == 'Exterior']['PM25']
        
        # Box plot comparison
        data_for_box = []
        labels_for_box = []
        
        if len(indoor_pm25) > 0:
            data_for_box.extend(indoor_pm25.values)
            labels_for_box.extend(['Indoor'] * len(indoor_pm25))
        
        if len(outdoor_pm25) > 0:
            data_for_box.extend(outdoor_pm25.values)
            labels_for_box.extend(['Outdoor'] * len(outdoor_pm25))
        
        if data_for_box:
            df_box = pd.DataFrame({'PM25': data_for_box, 'Environment': labels_for_box})
            sns.boxplot(data=df_box, x='Environment', y='PM25')
            plt.title('PM2.5: Indoor vs Outdoor')
            plt.ylabel('PM2.5 (μg/m³)')
    
    plt.tight_layout()
    plt.savefig('correlation_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    print("Comprehensive correlation visualizations saved as 'correlation_analysis_comprehensive.png'")

def generate_correlation_summary(all_correlations):
    """Generate comprehensive correlation summary"""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS SUMMARY")
    print("="*60)
    
    print("\n📊 CORRELATION STRENGTH INTERPRETATION:")
    print("  |r| < 0.3: Weak correlation")
    print("  0.3 ≤ |r| < 0.7: Moderate correlation") 
    print("  |r| ≥ 0.7: Strong correlation")
    print("  p < 0.05: Statistically significant")
    
    print("\n📈 KEY FINDINGS:")
    
    # Summarize significant correlations
    significant_correlations = []
    strong_correlations = []
    
    for category, correlations in all_correlations.items():
        print(f"\n{category.upper()}:")
        
        if isinstance(correlations, dict) and correlations:
            for corr_name, corr_data in correlations.items():
                if isinstance(corr_data, dict) and 'correlation' in corr_data:
                    corr_val = corr_data['correlation']
                    p_val = corr_data['p_value']
                    n = corr_data['n']
                    
                    if not np.isnan(corr_val):
                        strength = "Strong" if abs(corr_val) >= 0.7 else "Moderate" if abs(corr_val) >= 0.3 else "Weak"
                        significance = "✓" if p_val < 0.05 else "✗"
                        
                        print(f"  {corr_name}: r = {corr_val:.3f} ({strength}, {significance}, n={n})")
                        
                        if p_val < 0.05:
                            significant_correlations.append((corr_name, corr_val, p_val))
                        
                        if abs(corr_val) >= 0.7:
                            strong_correlations.append((corr_name, corr_val, p_val))
        else:
            print("  No correlations calculated")
    
    # Highlight most important findings
    print(f"\n🎯 MOST IMPORTANT FINDINGS:")
    
    if strong_correlations:
        print(f"\nStrong correlations found:")
        for name, corr, p_val in strong_correlations:
            sig_mark = "✓" if p_val < 0.05 else "✗"
            print(f"  • {name}: r = {corr:.3f} ({sig_mark})")
    else:
        print("\n  • No strong correlations (|r| ≥ 0.7) found")
    
    if significant_correlations:
        print(f"\nStatistically significant correlations:")
        for name, corr, p_val in significant_correlations:
            strength = "Strong" if abs(corr) >= 0.7 else "Moderate" if abs(corr) >= 0.3 else "Weak"
            print(f"  • {name}: r = {corr:.3f} ({strength})")
    else:
        print("\n  • No statistically significant correlations found")
    
    print(f"\n💡 PRACTICAL IMPLICATIONS:")
    print("  • Strong positive correlations suggest variables increase together")
    print("  • Strong negative correlations suggest one variable increases as the other decreases")
    print("  • Significant correlations help identify important relationships for air quality management")
    print("  • School-specific differences suggest location-dependent factors")
    
    print(f"\n📁 OUTPUTS GENERATED:")
    print(f"  • Comprehensive visualizations: correlation_analysis_comprehensive.png")
    print(f"  • Detailed correlation results: Console output above")

def main():
    """Main correlation analysis function"""
    try:
        # Load data
        averages_data, raw_data, sheets = load_comprehensive_data()
        
        # Store all correlation results
        all_correlations = {}
        
        # 1. Physical correlations
        phys_corr = physical_correlations(averages_data, raw_data)
        all_correlations['Physical'] = phys_corr
        
        # 2. Environment correlations
        env_corr = environment_correlations(averages_data, raw_data)
        all_correlations['Environment'] = env_corr
        
        # 3. Pollutant correlations
        poll_corr = pollutant_correlations(averages_data, raw_data)
        all_correlations['Pollutant'] = poll_corr
        
        # 4. School-specific correlations
        school_corr = school_correlations(averages_data, raw_data)
        all_correlations['School_Matrices'] = school_corr
        
        # 5. Create visualizations
        create_correlation_visualizations(averages_data, all_correlations)
        
        # 6. Generate summary
        generate_correlation_summary(all_correlations)
        
        print(f"\n✅ Comprehensive correlation analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during correlation analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()