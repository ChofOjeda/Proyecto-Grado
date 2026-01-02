import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare data for ANOVA analysis"""
    print("Loading data from 'Datos Unificados1.ods'...")
    
    # Read the averages sheet
    df = pd.read_excel('Datos Unificados1.ods', sheet_name='Promedios y desviacion', engine='odf')
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Clean and prepare data
    df_clean = df.copy()
    
    # Standardize school names
    df_clean['School'] = df_clean['Colegio'].map({
        'Santa Juanita Campestre': 'STA_Juanita',
        'IED San Isidro': 'IED'
    })
    
    # Standardize space types
    df_clean['Space_Type'] = df_clean['Espacio'].map({
        'Interior': 'Indoor',
        'Exterior': 'Outdoor'
    })
    
    # Filter only PM2.5 data (present in both indoor and outdoor)
    pm25_data = df_clean[['School', 'Space_Type', 'Punto de Monitoreo', 
                          'PM2.5(ug/m3) MEAN', 'PM2.5(ug/m3) DESVEST',
                          'TEMP MEAN', 'TEMP DESVEST',
                          'HUMI(%RH) MEAN', 'HUMI(%RH) DESVEST']].copy()
    
    # Remove rows with missing values
    pm25_data = pm25_data.dropna()
    
    print(f"\nCleaned data shape: {pm25_data.shape}")
    print(f"Schools: {pm25_data['School'].unique()}")
    print(f"Space types: {pm25_data['Space_Type'].unique()}")
    
    return pm25_data

def descriptive_statistics(data):
    """Generate descriptive statistics"""
    print("\n" + "="*60)
    print("DESCRIPTIVE STATISTICS")
    print("="*60)
    
    # Overall summary
    print("\nOverall PM2.5 Summary:")
    print(data['PM2.5(ug/m3) MEAN'].describe())
    
    # By School and Space Type
    print("\nPM2.5 by School and Space Type:")
    summary = data.groupby(['School', 'Space_Type'])['PM2.5(ug/m3) MEAN'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(3)
    print(summary)
    
    # Temperature and Humidity summary
    print("\nTemperature Summary by School and Space Type:")
    temp_summary = data.groupby(['School', 'Space_Type'])['TEMP MEAN'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(3)
    print(temp_summary)
    
    print("\nHumidity Summary by School and Space Type:")
    humi_summary = data.groupby(['School', 'Space_Type'])['HUMI(%RH) MEAN'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(3)
    print(humi_summary)
    
    return summary

def two_way_anova(data):
    """Perform two-way ANOVA for PM2.5 or appropriate alternative based on available data"""
    print("\n" + "="*60)
    print("A. INDOOR VS OUTDOOR ANALYSIS")
    print("="*60)
    
    print("\nFactors:")
    print("- Space_Type: Indoor vs Outdoor")
    print("- School: School comparison (if multiple schools available)")
    print("- Dependent Variable: PM2.5(ug/m3) MEAN")
    
    # Prepare data for ANOVA
    anova_data = data[['School', 'Space_Type', 'PM2.5(ug/m3) MEAN']].copy()
    anova_data.columns = ['School', 'Space_Type', 'PM25']
    
    print(f"\nSample sizes:")
    print(anova_data.groupby(['School', 'Space_Type']).size())
    
    n_schools = anova_data['School'].nunique()
    n_space_types = anova_data['Space_Type'].nunique()
    
    print(f"\nNumber of schools: {n_schools}")
    print(f"Number of space types: {n_space_types}")
    
    # Check assumptions
    print("\n" + "-"*40)
    print("ASSUMPTION CHECKS")
    print("-"*40)
    
    # 1. Normality test
    print("\n1. Normality Test (Shapiro-Wilk):")
    stat, p = stats.shapiro(anova_data['PM25'])
    print(f"   Statistic: {stat:.4f}, p-value: {p:.4f}")
    if p > 0.05:
        print("   ✓ Data appears to be normally distributed (p > 0.05)")
    else:
        print("   ⚠ Data may not be normally distributed (p ≤ 0.05)")
    
    # 2. Homogeneity of variance (Levene's test)
    print("\n2. Homogeneity of Variance (Levene's test):")
    groups = [group['PM25'].values for name, group in anova_data.groupby(['School', 'Space_Type'])]
    stat, p = stats.levene(*groups)
    print(f"   Statistic: {stat:.4f}, p-value: {p:.4f}")
    if p > 0.05:
        print("   ✓ Variances appear to be homogeneous (p > 0.05)")
    else:
        print("   ⚠ Variances may not be homogeneous (p ≤ 0.05)")
    
    # Determine appropriate analysis based on available data
    print("\n" + "-"*40)
    print("STATISTICAL ANALYSIS")
    print("-"*40)
    
    if n_schools >= 2 and n_space_types >= 2:
        # Two-way ANOVA
        print("\nPerforming Two-way ANOVA (School × Space Type)")
        
        model = ols('PM25 ~ C(School) + C(Space_Type) + C(School):C(Space_Type)', 
                    data=anova_data).fit()
        anova_table = anova_lm(model, typ=2)
        
        print("\nANOVA Table:")
        print(anova_table.round(6))
        
        # Interpretation
        alpha = 0.05
        print("\n" + "-"*40)
        print("INTERPRETATION")
        print("-"*40)
        
        # Main effect: School
        p_school = anova_table.loc['C(School)', 'PR(>F)']
        print(f"\n1. Main Effect - School:")
        print(f"   F = {anova_table.loc['C(School)', 'F']:.4f}, p = {p_school:.6f}")
        if p_school < alpha:
            print(f"   ✓ SIGNIFICANT difference between schools (p < {alpha})")
        else:
            print(f"   ✗ No significant difference between schools (p ≥ {alpha})")
        
        # Main effect: Space Type
        p_space = anova_table.loc['C(Space_Type)', 'PR(>F)']
        print(f"\n2. Main Effect - Space Type (Indoor vs Outdoor):")
        print(f"   F = {anova_table.loc['C(Space_Type)', 'F']:.4f}, p = {p_space:.6f}")
        if p_space < alpha:
            print(f"   ✓ SIGNIFICANT difference between Indoor and Outdoor (p < {alpha})")
        else:
            print(f"   ✗ No significant difference between Indoor and Outdoor (p ≥ {alpha})")
        
        # Interaction effect
        p_interaction = anova_table.loc['C(School):C(Space_Type)', 'PR(>F)']
        print(f"\n3. Interaction Effect - School × Space Type:")
        print(f"   F = {anova_table.loc['C(School):C(Space_Type)', 'F']:.4f}, p = {p_interaction:.6f}")
        if p_interaction < alpha:
            print(f"   ✓ SIGNIFICANT interaction effect (p < {alpha})")
            print("   → The effect of Indoor/Outdoor varies by school")
        else:
            print(f"   ✗ No significant interaction effect (p ≥ {alpha})")
            print("   → The effect of Indoor/Outdoor is consistent across schools")
        
    elif n_schools == 1 and n_space_types >= 2:
        # One-way ANOVA: Space Type within single school
        print(f"\nPerforming One-way ANOVA: Space Type within {anova_data['School'].iloc[0]} school")
        print("(Only one school available)")
        
        space_groups = [group['PM25'].values for name, group in anova_data.groupby('Space_Type')]
        f_stat, p_value = stats.f_oneway(*space_groups)
        
        print(f"\nOne-way ANOVA Results:")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"p-value: {p_value:.6f}")
        
        alpha = 0.05
        if p_value < alpha:
            print(f"✓ SIGNIFICANT difference between Indoor and Outdoor (p < {alpha})")
        else:
            print(f"✗ No significant difference between Indoor and Outdoor (p ≥ {alpha})")
        
        # Create a simplified anova table for consistency
        anova_table = pd.DataFrame({
            'sum_sq': [f_stat],  # Simplified
            'df': [1],
            'F': [f_stat],
            'PR(>F)': [p_value]
        }, index=['C(Space_Type)'])
        
        model = None  # No OLS model for one-way ANOVA
        
    elif n_schools >= 2 and n_space_types == 1:
        # One-way ANOVA: School comparison within single space type
        space_type = anova_data['Space_Type'].iloc[0]
        print(f"\nPerforming One-way ANOVA: School comparison within {space_type} environments")
        print("(Only one space type available)")
        
        school_groups = [group['PM25'].values for name, group in anova_data.groupby('School')]
        f_stat, p_value = stats.f_oneway(*school_groups)
        
        print(f"\nOne-way ANOVA Results:")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"p-value: {p_value:.6f}")
        
        alpha = 0.05
        if p_value < alpha:
            print(f"✓ SIGNIFICANT difference between schools (p < {alpha})")
        else:
            print(f"✗ No significant difference between schools (p ≥ {alpha})")
        
        # Create a simplified anova table for consistency
        anova_table = pd.DataFrame({
            'sum_sq': [f_stat],  # Simplified
            'df': [1],
            'F': [f_stat],
            'PR(>F)': [p_value]
        }, index=['C(School)'])
        
        model = None
        
    else:
        # Not enough groups for ANOVA
        print("\n⚠️ Insufficient groups for ANOVA analysis")
        print("Need at least 2 groups to compare")
        
        # Perform descriptive comparison
        print("\nDescriptive Comparison:")
        group_stats = anova_data.groupby(['School', 'Space_Type'])['PM25'].agg(['count', 'mean', 'std']).round(4)
        print(group_stats)
        
        anova_table = pd.DataFrame()  # Empty table
        model = None
    
    # Effect sizes if applicable
    if not anova_table.empty and len(anova_table) > 0:
        print("\n" + "-"*40)
        print("EFFECT SIZES (η²)")
        print("-"*40)
        
        if 'sum_sq' in anova_table.columns:
            ss_total = anova_table['sum_sq'].sum() if anova_table['sum_sq'].sum() > 0 else 1
            
            for effect in anova_table.index:
                eta2 = anova_table.loc[effect, 'sum_sq'] / ss_total if ss_total > 0 else 0
                print(f"{effect} effect (η²): {eta2:.4f}")
                
                # Interpretation of effect sizes
                def interpret_eta2(eta2):
                    if eta2 < 0.01:
                        return "negligible"
                    elif eta2 < 0.06:
                        return "small"
                    elif eta2 < 0.14:
                        return "medium"
                    else:
                        return "large"
                
                print(f"  → {interpret_eta2(eta2)} effect")
    
    return model, anova_table, anova_data

def post_hoc_analysis(data):
    """Perform post-hoc analysis"""
    print("\n" + "="*60)
    print("POST-HOC ANALYSIS")
    print("="*60)
    
    # Mean differences
    print("\nMean PM2.5 concentrations by group:")
    means = data.groupby(['School', 'Space_Type'])['PM2.5(ug/m3) MEAN'].mean()
    print(means.round(3))
    
    # Pairwise t-tests
    print("\nPairwise comparisons (t-tests):")
    
    # Create groups for comparison
    groups = {}
    for (school, space), group_data in data.groupby(['School', 'Space_Type']):
        groups[f"{school}_{space}"] = group_data['PM2.5(ug/m3) MEAN'].values
    
    group_names = list(groups.keys())
    
    print(f"\nNumber of observations per group:")
    for name, values in groups.items():
        print(f"  {name}: n = {len(values)}")
    
    # Perform pairwise t-tests
    print(f"\nPairwise t-test results:")
    for i in range(len(group_names)):
        for j in range(i+1, len(group_names)):
            name1, name2 = group_names[i], group_names[j]
            stat, p = stats.ttest_ind(groups[name1], groups[name2])
            
            # Calculate Cohen's d
            pooled_std = np.sqrt(((len(groups[name1])-1)*np.var(groups[name1], ddof=1) + 
                                  (len(groups[name2])-1)*np.var(groups[name2], ddof=1)) / 
                                  (len(groups[name1]) + len(groups[name2]) - 2))
            cohens_d = (np.mean(groups[name1]) - np.mean(groups[name2])) / pooled_std
            
            print(f"  {name1} vs {name2}:")
            print(f"    t = {stat:.4f}, p = {p:.6f}, Cohen's d = {cohens_d:.4f}")
            
            if p < 0.05:
                print(f"    ✓ SIGNIFICANT difference (p < 0.05)")
            else:
                print(f"    ✗ No significant difference (p ≥ 0.05)")

def load_indoor_data():
    """Load and prepare indoor-only data for comprehensive ANOVA analysis"""
    print("\n" + "="*60)
    print("LOADING INDOOR-ONLY DATA")
    print("="*60)
    
    # Read the interior data from both schools
    sheets = pd.read_excel('Datos Unificados1.ods', sheet_name=None, engine='odf')
    
    # Combine IED and Santa Juanita interior data
    ied_interior = sheets['IED Interior'].copy()
    sta_interior = sheets['Santa Juanita Interior'].copy()
    
    # Add school identifier
    ied_interior['School'] = 'IED'
    sta_interior['School'] = 'STA_Juanita'
    
    # Combine data
    indoor_data = pd.concat([ied_interior, sta_interior], ignore_index=True)
    
    # Clean column names
    indoor_data.columns = [col.strip() for col in indoor_data.columns]
    
    # Rename location column for consistency
    if 'Lugar' in indoor_data.columns:
        indoor_data['Location'] = indoor_data['Lugar']
    elif 'Location' not in indoor_data.columns:
        indoor_data['Location'] = indoor_data['Punto de Monitoreo'] if 'Punto de Monitoreo' in indoor_data.columns else 'Unknown'
    
    # Remove rows with missing values in key variables
    key_vars = ['PM2.5(ug/m3)', 'PM10(ug/m3)', 'School', 'Location']
    available_vars = [var for var in key_vars if var in indoor_data.columns]
    indoor_data = indoor_data.dropna(subset=available_vars)
    
    print(f"Indoor data loaded successfully:")
    print(f"  Total observations: {len(indoor_data)}")
    print(f"  Schools: {indoor_data['School'].unique()}")
    print(f"  Locations: {indoor_data['Location'].unique()}")
    print(f"  Available pollutants: {[col for col in indoor_data.columns if any(pollutant in col for pollutant in ['PM2.5', 'PM10', 'HCHO', 'TVOC', 'CO2'])]}")
    
    return indoor_data

def indoor_only_anova(indoor_data):
    """Perform comprehensive ANOVA analysis for indoor environments only"""
    print("\n" + "="*60)
    print("B. ANOVA INDOORS ONLY")
    print("="*60)
    
    print("Objective: Compare pollution levels between indoor locations and schools")
    print("Pollutants analyzed: PM₂.₅, PM₁₀, HCHO, TVOC, and other available variables")
    
    # Define pollutants to analyze
    pollutant_columns = {
        'PM2.5(ug/m3)': 'PM₂.₅ (μg/m³)',
        'PM10(ug/m3)': 'PM₁₀ (μg/m³)', 
        'HCHO(mg/m3)': 'HCHO (mg/m³)',
        'TVOC(mg/m3)': 'TVOC (mg/m³)',
        'PARTICLES(per/L)': 'Particles (per/L)',
        'AQI': 'Air Quality Index'
    }
    
    # Check which pollutants are available
    available_pollutants = {}
    for col, label in pollutant_columns.items():
        if col in indoor_data.columns:
            available_pollutants[col] = label
    
    print(f"\nAvailable pollutants for analysis: {list(available_pollutants.values())}")
    
    if not available_pollutants:
        print("❌ No pollutant data available for indoor analysis")
        return None
    
    results = {}
    
    for pollutant_col, pollutant_label in available_pollutants.items():
        print(f"\n" + "-"*50)
        print(f"ANALYZING: {pollutant_label}")
        print("-"*50)
        
        # Prepare data for this pollutant
        pollutant_data = indoor_data[['School', 'Location', pollutant_col]].copy()
        pollutant_data = pollutant_data.dropna()
        
        if len(pollutant_data) == 0:
            print(f"No data available for {pollutant_label}")
            continue
        
        print(f"Sample size: {len(pollutant_data)} observations")
        print(f"Schools: {pollutant_data['School'].nunique()}")
        print(f"Locations: {pollutant_data['Location'].nunique()}")
        
        # Descriptive statistics
        print(f"\nDescriptive Statistics for {pollutant_label}:")
        desc_stats = pollutant_data.groupby(['School', 'Location'])[pollutant_col].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        print(desc_stats)
        
        # Check if we have enough data for ANOVA
        location_counts = pollutant_data['Location'].value_counts()
        school_counts = pollutant_data['School'].value_counts()
        
        if len(location_counts) < 2 and len(school_counts) < 2:
            print(f"⚠️ Insufficient groups for ANOVA analysis of {pollutant_label}")
            continue
        
        # 1. ONE-WAY ANOVA: Locations within schools
        print(f"\n1. ONE-WAY ANOVA: Locations (within schools)")
        
        # Separate analysis for each school if we have multiple schools
        for school in pollutant_data['School'].unique():
            school_data = pollutant_data[pollutant_data['School'] == school]
            unique_locations = school_data['Location'].nunique()
            
            if unique_locations < 2:
                print(f"   {school}: Only {unique_locations} location(s) - skipping ANOVA")
                continue
                
            print(f"\n   {school} School - Locations Comparison:")
            location_groups = [group[pollutant_col].values for name, group in school_data.groupby('Location')]
            
            try:
                f_stat, p_value = stats.f_oneway(*location_groups)
                print(f"   F-statistic: {f_stat:.4f}")
                print(f"   p-value: {p_value:.6f}")
                
                if p_value < 0.05:
                    print(f"   ✓ SIGNIFICANT differences between locations (p < 0.05)")
                else:
                    print(f"   ✗ No significant differences between locations (p ≥ 0.05)")
                
                # Location means
                location_means = school_data.groupby('Location')[pollutant_col].mean().sort_values(ascending=False)
                print(f"   Location means (highest to lowest):")
                for loc, mean_val in location_means.items():
                    print(f"     {loc}: {mean_val:.3f}")
                    
            except Exception as e:
                print(f"   Error in ANOVA: {e}")
        
        # 2. TWO-WAY ANOVA: Locations × Schools
        print(f"\n2. TWO-WAY ANOVA: Locations × Schools")
        
        # Check if we have balanced design or sufficient data
        cross_tab = pd.crosstab(pollutant_data['School'], pollutant_data['Location'])
        print(f"   Data distribution:")
        print(cross_tab)
        
        # Only proceed if we have multiple schools and locations
        if pollutant_data['School'].nunique() >= 2 and pollutant_data['Location'].nunique() >= 2:
            try:
                # Prepare data for two-way ANOVA
                model_formula = f"`{pollutant_col}` ~ C(School) + C(Location) + C(School):C(Location)"
                model = ols(model_formula, data=pollutant_data).fit()
                anova_table = anova_lm(model, typ=2)
                
                print(f"\n   Two-way ANOVA Results:")
                print(anova_table.round(6))
                
                # Interpretation
                alpha = 0.05
                
                if 'C(School)' in anova_table.index:
                    p_school = anova_table.loc['C(School)', 'PR(>F)']
                    print(f"\n   School Effect: F = {anova_table.loc['C(School)', 'F']:.4f}, p = {p_school:.6f}")
                    if p_school < alpha:
                        print(f"   ✓ SIGNIFICANT school differences")
                    else:
                        print(f"   ✗ No significant school differences")
                
                if 'C(Location)' in anova_table.index:
                    p_location = anova_table.loc['C(Location)', 'PR(>F)']
                    print(f"\n   Location Effect: F = {anova_table.loc['C(Location)', 'F']:.4f}, p = {p_location:.6f}")
                    if p_location < alpha:
                        print(f"   ✓ SIGNIFICANT location differences")
                    else:
                        print(f"   ✗ No significant location differences")
                
                if 'C(School):C(Location)' in anova_table.index:
                    p_interaction = anova_table.loc['C(School):C(Location)', 'PR(>F)']
                    print(f"\n   Interaction Effect: F = {anova_table.loc['C(School):C(Location)', 'F']:.4f}, p = {p_interaction:.6f}")
                    if p_interaction < alpha:
                        print(f"   ✓ SIGNIFICANT interaction effect")
                        print(f"   → Location effects vary by school")
                    else:
                        print(f"   ✗ No significant interaction effect")
                        print(f"   → Location effects are consistent across schools")
                
                results[pollutant_col] = {
                    'anova_table': anova_table,
                    'model': model,
                    'data': pollutant_data
                }
                
            except Exception as e:
                print(f"   Error in two-way ANOVA: {e}")
                print("   This may be due to insufficient data or unbalanced design")
        else:
            print(f"   Insufficient schools ({pollutant_data['School'].nunique()}) or locations ({pollutant_data['Location'].nunique()}) for two-way ANOVA")
        
        # 3. Post-hoc analysis if significant differences found
        if pollutant_data['Location'].nunique() >= 3:  # Need at least 3 groups for meaningful post-hoc
            print(f"\n3. POST-HOC ANALYSIS: Pairwise Comparisons")
            
            locations = pollutant_data['Location'].unique()
            if len(locations) >= 2:
                print(f"   Pairwise t-tests between locations:")
                
                for i in range(len(locations)):
                    for j in range(i+1, len(locations)):
                        loc1, loc2 = locations[i], locations[j]
                        
                        group1 = pollutant_data[pollutant_data['Location'] == loc1][pollutant_col]
                        group2 = pollutant_data[pollutant_data['Location'] == loc2][pollutant_col]
                        
                        if len(group1) > 0 and len(group2) > 0:
                            t_stat, p_val = stats.ttest_ind(group1, group2)
                            mean1, mean2 = group1.mean(), group2.mean()
                            
                            print(f"     {loc1} vs {loc2}:")
                            print(f"       Means: {mean1:.3f} vs {mean2:.3f}")
                            print(f"       t = {t_stat:.4f}, p = {p_val:.6f}")
                            
                            if p_val < 0.05:
                                print(f"       ✓ SIGNIFICANT difference")
                            else:
                                print(f"       ✗ No significant difference")
    
    return results

def environmental_variables_analysis(data):
    """Analyze environmental variables as context"""
    print("\n" + "="*60)
    print("ENVIRONMENTAL VARIABLES ANALYSIS")
    print("="*60)
    
    print("Analyzing Temperature and Humidity as context variables")
    print("(These are not included in the ANOVA but analyzed through correlation)")
    
    # Correlation analysis
    print("\n" + "-"*40)
    print("CORRELATION ANALYSIS")
    print("-"*40)
    
    # Calculate correlations
    corr_vars = ['PM2.5(ug/m3) MEAN', 'TEMP MEAN', 'HUMI(%RH) MEAN']
    correlation_matrix = data[corr_vars].corr()
    
    print("\nCorrelation Matrix:")
    print(correlation_matrix.round(4))
    
    # Specific correlations with PM2.5
    pm25_temp_corr = data['PM2.5(ug/m3) MEAN'].corr(data['TEMP MEAN'])
    pm25_humi_corr = data['PM2.5(ug/m3) MEAN'].corr(data['HUMI(%RH) MEAN'])
    
    print(f"\nCorrelations with PM2.5:")
    print(f"  PM2.5 ↔ Temperature: r = {pm25_temp_corr:.4f}")
    print(f"  PM2.5 ↔ Humidity: r = {pm25_humi_corr:.4f}")
    
    # Significance tests
    from scipy.stats import pearsonr
    
    temp_stat, temp_p = pearsonr(data['PM2.5(ug/m3) MEAN'], data['TEMP MEAN'])
    humi_stat, humi_p = pearsonr(data['PM2.5(ug/m3) MEAN'], data['HUMI(%RH) MEAN'])
    
    print(f"\nSignificance tests:")
    print(f"  PM2.5 ↔ Temperature: p = {temp_p:.6f}")
    if temp_p < 0.05:
        print("    ✓ SIGNIFICANT correlation")
    else:
        print("    ✗ Not significant")
    
    print(f"  PM2.5 ↔ Humidity: p = {humi_p:.6f}")
    if humi_p < 0.05:
        print("    ✓ SIGNIFICANT correlation")
    else:
        print("    ✗ Not significant")
    
    # Environmental conditions by group
    print("\n" + "-"*40)
    print("ENVIRONMENTAL CONDITIONS BY GROUP")
    print("-"*40)
    
    print("\nTemperature by School and Space Type:")
    temp_by_group = data.groupby(['School', 'Space_Type'])['TEMP MEAN'].agg(['mean', 'std']).round(3)
    print(temp_by_group)
    
    print("\nHumidity by School and Space Type:")
    humi_by_group = data.groupby(['School', 'Space_Type'])['HUMI(%RH) MEAN'].agg(['mean', 'std']).round(3)
    print(humi_by_group)

def create_visualizations(data, anova_data, indoor_results=None):
    """Create visualizations for both analyses"""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # PART A: Indoor vs Outdoor Analysis
    print("Creating plots for Indoor vs Outdoor analysis...")
    
    # 1. Box plot of PM2.5 by School and Space Type
    plt.subplot(3, 4, 1)
    sns.boxplot(data=anova_data, x='School', y='PM25', hue='Space_Type')
    plt.title('A. PM2.5: Indoor vs Outdoor')
    plt.ylabel('PM2.5 (μg/m³)')
    plt.xticks(rotation=45)
    
    # 2. Bar plot of means
    plt.subplot(3, 4, 2)
    means = anova_data.groupby(['School', 'Space_Type'])['PM25'].mean().reset_index()
    sns.barplot(data=means, x='School', y='PM25', hue='Space_Type')
    plt.title('A. Mean PM2.5 Concentrations')
    plt.ylabel('Mean PM2.5 (μg/m³)')
    plt.xticks(rotation=45)
    
    # 3. Interaction plot
    plt.subplot(3, 4, 3)
    school_means = anova_data.groupby(['School', 'Space_Type'])['PM25'].mean().unstack()
    if not school_means.empty:
        school_means.plot(kind='line', marker='o', ax=plt.gca())
        plt.title('A. Interaction: School × Space Type')
        plt.ylabel('Mean PM2.5 (μg/m³)')
        plt.xlabel('School')
        plt.legend(title='Space Type')
    
    # 4. Correlation plot: PM2.5 vs Temperature
    plt.subplot(3, 4, 4)
    if 'TEMP MEAN' in data.columns:
        sns.scatterplot(data=data, x='TEMP MEAN', y='PM2.5(ug/m3) MEAN', 
                       hue='School', style='Space_Type')
        plt.title('A. PM2.5 vs Temperature')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('PM2.5 (μg/m³)')
    
    # PART B: Indoor Only Analysis
    if indoor_results:
        print("Creating plots for Indoor-only analysis...")
        
        # Get the first available pollutant data for visualization
        first_pollutant = list(indoor_results.keys())[0] if indoor_results else None
        
        if first_pollutant and 'data' in indoor_results[first_pollutant]:
            indoor_data = indoor_results[first_pollutant]['data']
            
            # 5. Box plot by School and Location (Indoor only)
            plt.subplot(3, 4, 5)
            if len(indoor_data['Location'].unique()) <= 10:  # Only if not too many locations
                sns.boxplot(data=indoor_data, x='School', y=first_pollutant, hue='Location')
                plt.title(f'B. {first_pollutant} by Location (Indoor)')
                plt.ylabel(f'{first_pollutant}')
                plt.xticks(rotation=45)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                sns.boxplot(data=indoor_data, x='School', y=first_pollutant)
                plt.title(f'B. {first_pollutant} by School (Indoor)')
                plt.ylabel(f'{first_pollutant}')
            
            # 6. Bar plot of means by location
            plt.subplot(3, 4, 6)
            location_means = indoor_data.groupby(['School', 'Location'])[first_pollutant].mean().reset_index()
            if len(location_means) <= 20:  # Only if manageable number of bars
                sns.barplot(data=location_means, x='Location', y=first_pollutant, hue='School')
                plt.title(f'B. Mean {first_pollutant} by Location')
                plt.ylabel(f'Mean {first_pollutant}')
                plt.xticks(rotation=45)
            else:
                sns.barplot(data=location_means, x='School', y=first_pollutant)
                plt.title(f'B. Mean {first_pollutant} by School')
                plt.ylabel(f'Mean {first_pollutant}')
            
            # 7. Violin plot for distribution comparison
            plt.subplot(3, 4, 7)
            if len(indoor_data['Location'].unique()) <= 8:
                sns.violinplot(data=indoor_data, x='Location', y=first_pollutant)
                plt.title(f'B. Distribution of {first_pollutant}')
                plt.xticks(rotation=45)
            else:
                sns.violinplot(data=indoor_data, x='School', y=first_pollutant)
                plt.title(f'B. Distribution by School')
    
    # Additional plots for multiple pollutants if available
    if indoor_results and len(indoor_results) > 1:
        pollutant_list = list(indoor_results.keys())[:4]  # Max 4 pollutants
        
        for i, pollutant in enumerate(pollutant_list):
            if 'data' in indoor_results[pollutant]:
                subplot_idx = 8 + i
                if subplot_idx <= 12:
                    plt.subplot(3, 4, subplot_idx)
                    poll_data = indoor_results[pollutant]['data']
                    
                    # Simple box plot by school
                    sns.boxplot(data=poll_data, x='School', y=pollutant)
                    plt.title(f'B. {pollutant} by School')
                    plt.ylabel(pollutant)
                    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('anova_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("Comprehensive visualizations saved as 'anova_comprehensive_analysis.png'")

def generate_summary_report(data, anova_table, indoor_results=None):
    """Generate a comprehensive summary report for both analyses"""
    print("\n" + "="*60)
    print("COMPREHENSIVE SUMMARY REPORT")
    print("="*60)
    
    print("\n📊 RESEARCH QUESTIONS:")
    print("A. Is there a significant difference in PM2.5 concentrations between indoor and outdoor environments across schools?")
    print("B. Are there significant differences in pollutant concentrations between indoor locations within and across schools?")
    
    print("\n📋 METHODOLOGY:")
    print("A. ANOVA for Indoor vs Outdoor PM2.5 comparison (method depends on available data)")
    print("B. One-way and Two-way ANOVA for Indoor-only analysis of multiple pollutants")
    print("   - One-way: Locations within each school")
    print("   - Two-way: Locations × Schools (if applicable)")
    print("- Environmental variables analyzed through correlation")
    
    print("\n" + "="*50)
    print("A. INDOOR VS OUTDOOR ANALYSIS")
    print("="*50)
    
    if not anova_table.empty:
        # Extract key results from Part A
        alpha = 0.05
        
        print(f"\n📈 KEY FINDINGS:")
        
        # Check which effects are available in the ANOVA table
        if 'C(School)' in anova_table.index:
            p_school = anova_table.loc['C(School)', 'PR(>F)']
            print(f"\n1. MAIN EFFECT - SCHOOL:")
            if p_school < alpha:
                print(f"   ✓ SIGNIFICANT difference between schools (p = {p_school:.6f})")
            else:
                print(f"   ✗ No significant difference between schools (p = {p_school:.6f})")
        
        if 'C(Space_Type)' in anova_table.index:
            p_space = anova_table.loc['C(Space_Type)', 'PR(>F)']
            print(f"\n2. MAIN EFFECT - SPACE TYPE (Indoor vs Outdoor):")
            if p_space < alpha:
                print(f"   ✓ SIGNIFICANT difference between indoor and outdoor (p = {p_space:.6f})")
            else:
                print(f"   ✗ No significant difference between indoor and outdoor (p = {p_space:.6f})")
        
        if 'C(School):C(Space_Type)' in anova_table.index:
            p_interaction = anova_table.loc['C(School):C(Space_Type)', 'PR(>F)']
            print(f"\n3. INTERACTION EFFECT:")
            if p_interaction < alpha:
                print(f"   ✓ SIGNIFICANT interaction (p = {p_interaction:.6f})")
                print("   → The indoor/outdoor effect varies by school")
            else:
                print(f"   ✗ No significant interaction (p = {p_interaction:.6f})")
                print("   → The indoor/outdoor effect is consistent across schools")
    else:
        print(f"\n⚠️ ANOVA could not be performed due to insufficient data")
    
    # Calculate group means for Part A
    means = data.groupby(['School', 'Space_Type'])['PM2.5(ug/m3) MEAN'].mean()
    
    print(f"\n📊 GROUP MEANS (PM2.5):")
    for (school, space), mean_val in means.items():
        print(f"   {school} - {space}: {mean_val:.2f} μg/m³")
    
    # Part B Results
    if indoor_results:
        print("\n" + "="*50)
        print("B. INDOOR-ONLY ANALYSIS")
        print("="*50)
        
        print(f"\n📈 POLLUTANTS ANALYZED: {len(indoor_results)}")
        
        for pollutant, results in indoor_results.items():
            print(f"\n--- {pollutant} ---")
            
            if 'data' in results and len(results['data']) > 0:
                poll_data = results['data']
                
                # Show descriptive statistics
                overall_mean = poll_data[pollutant].mean()
                overall_std = poll_data[pollutant].std()
                print(f"Overall: {overall_mean:.3f} ± {overall_std:.3f}")
                
                # School comparison
                school_means = poll_data.groupby('School')[pollutant].mean()
                print(f"By School:")
                for school, mean_val in school_means.items():
                    print(f"  {school}: {mean_val:.3f}")
                
                # Location comparison (top 3 highest)
                location_means = poll_data.groupby('Location')[pollutant].mean().sort_values(ascending=False)
                print(f"Top 3 Locations (highest concentrations):")
                for i, (location, mean_val) in enumerate(location_means.head(3).items()):
                    print(f"  {i+1}. {location}: {mean_val:.3f}")
                
                # ANOVA results summary if available
                if 'anova_table' in results:
                    anova_res = results['anova_table']
                    
                    if 'C(School)' in anova_res.index:
                        p_school_indoor = anova_res.loc['C(School)', 'PR(>F)']
                        if p_school_indoor < 0.05:
                            print(f"  ✓ Significant school differences (p = {p_school_indoor:.4f})")
                        else:
                            print(f"  ✗ No significant school differences (p = {p_school_indoor:.4f})")
                    
                    if 'C(Location)' in anova_res.index:
                        p_location = anova_res.loc['C(Location)', 'PR(>F)']
                        if p_location < 0.05:
                            print(f"  ✓ Significant location differences (p = {p_location:.4f})")
                        else:
                            print(f"  ✗ No significant location differences (p = {p_location:.4f})")
    
    print(f"\n🌡️ ENVIRONMENTAL CONTEXT:")
    if 'TEMP MEAN' in data.columns and 'HUMI(%RH) MEAN' in data.columns:
        pm25_temp_corr = data['PM2.5(ug/m3) MEAN'].corr(data['TEMP MEAN'])
        pm25_humi_corr = data['PM2.5(ug/m3) MEAN'].corr(data['HUMI(%RH) MEAN'])
        
        print(f"   Temperature correlation with PM2.5: r = {pm25_temp_corr:.3f}")
        print(f"   Humidity correlation with PM2.5: r = {pm25_humi_corr:.3f}")
    
    print(f"\n💡 PRACTICAL IMPLICATIONS:")
    
    # Indoor vs Outdoor implications
    if not anova_table.empty and 'C(Space_Type)' in anova_table.index:
        p_space = anova_table.loc['C(Space_Type)', 'PR(>F)']
        if p_space < 0.05:
            indoor_mean = means[means.index.get_level_values('Space_Type') == 'Indoor'].mean()
            outdoor_mean = means[means.index.get_level_values('Space_Type') == 'Outdoor'].mean()
            if indoor_mean > outdoor_mean:
                print("   - Indoor environments show higher PM2.5 concentrations")
                print("   - Consider improving indoor air quality measures")
            else:
                print("   - Outdoor environments show higher PM2.5 concentrations")
                print("   - External pollution sources may be affecting air quality")
    
    if not anova_table.empty and 'C(School)' in anova_table.index:
        p_school = anova_table.loc['C(School)', 'PR(>F)']
        if p_school < 0.05:
            print("   - Significant differences between schools suggest location-specific factors")
            print("   - School-specific interventions may be needed")
    
    # Indoor-only implications
    if indoor_results:
        print("\n   📍 INDOOR LOCATION INSIGHTS:")
        
        # Find consistently problematic locations
        problematic_locations = set()
        for pollutant, results in indoor_results.items():
            if 'data' in results:
                poll_data = results['data']
                # Get top 2 locations for this pollutant
                top_locations = poll_data.groupby('Location')[pollutant].mean().nlargest(2).index
                problematic_locations.update(top_locations)
        
        if problematic_locations:
            print(f"   - Locations with consistently higher pollution: {', '.join(problematic_locations)}")
            print("   - These areas may require targeted interventions")
    
    print(f"\n📁 OUTPUTS GENERATED:")
    print(f"   - Comprehensive visualizations: anova_comprehensive_analysis.png")
    print(f"   - Detailed analysis results: Console output above")

def main():
    """Main analysis function"""
    print("="*80)
    print("COMPREHENSIVE PM CONCENTRATION ANALYSIS")
    print("="*80)
    print("A. Indoor vs Outdoor PM2.5 comparison (Two-way ANOVA)")
    print("B. Indoor-only analysis for multiple pollutants (One-way & Two-way ANOVA)")
    print("Environmental variables analyzed as context")
    
    try:
        # PART A: Load and analyze Indoor vs Outdoor data
        print("\n" + "="*60)
        print("PART A: INDOOR VS OUTDOOR ANALYSIS")
        print("="*60)
        
        data = load_and_prepare_data()
        
        # Check if we have data for both indoor and outdoor
        space_types = data['Space_Type'].unique()
        schools = data['School'].unique()
        
        print(f"\nData verification:")
        print(f"Schools found: {schools}")
        print(f"Space types found: {space_types}")
        
        if len(space_types) < 2:
            print("⚠️ Warning: Need both Indoor and Outdoor data for comparison")
            indoor_outdoor_results = None
        else:
            # Perform Part A analyses
            desc_stats = descriptive_statistics(data)
            model, anova_table, anova_data = two_way_anova(data)
            post_hoc_analysis(data)
            environmental_variables_analysis(data)
            indoor_outdoor_results = (data, anova_table, anova_data)
        
        # PART B: Indoor-only analysis
        print("\n" + "="*60)
        print("PART B: INDOOR-ONLY ANALYSIS") 
        print("="*60)
        
        indoor_data = load_indoor_data()
        indoor_results = indoor_only_anova(indoor_data)
        
        # Create comprehensive visualizations
        if indoor_outdoor_results:
            data, anova_table, anova_data = indoor_outdoor_results
            create_visualizations(data, anova_data, indoor_results)
            generate_summary_report(data, anova_table, indoor_results)
        else:
            # Create visualizations for indoor-only analysis
            create_visualizations(pd.DataFrame(), pd.DataFrame(), indoor_results)
            print("\n⚠️ Indoor vs Outdoor comparison not possible - generating indoor-only summary")
        
        print(f"\n✅ Comprehensive analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()