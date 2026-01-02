#!/usr/bin/env python3
"""
JSON to ODS Converter for PROYECTO GUACHETÁ
Converts JSON files from exterior directory to ODS format with separate sheets
"""

import json
import os
import pandas as pd
from pathlib import Path
import glob

def read_json_file(file_path):
    """Read and parse a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Handle the malformed JSON (starts with "1.")
            if content.startswith('1.'):
                content = content[2:]  # Remove "1."
            
            data = json.loads(content)
            return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def process_json_data(data, file_name):
    """Process JSON data and convert to DataFrame"""
    if not data:
        return None
    
    try:
        # Extract sensor data from the JSON
        sensor_data = data.get('data', [])
        metadata = {
            'file_name': file_name,
            'date': data.get('date', ''),
            'device_id': data.get('deviceId', ''),
            'total_hours': data.get('hours', 0),
            'total_minutes': data.get('mins', 0),
            'total_seconds': data.get('secs', 0),
            'total_kms': data.get('kms', 0),
            'data_points': len(sensor_data)
        }
        
        # Convert sensor data to DataFrame
        if sensor_data:
            df = pd.DataFrame(sensor_data)
            
            # Add metadata columns
            for key, value in metadata.items():
                df[key] = value
                
            return df
        else:
            # If no sensor data, create a DataFrame with just metadata
            return pd.DataFrame([metadata])
            
    except Exception as e:
        print(f"Error processing data from {file_name}: {e}")
        return None

def convert_json_to_ods():
    """Main function to convert JSON files to ODS"""
    # Define paths
    exterior_path = Path("/home/chofojeda/todo/Documents/PROYECTO GUACHETÁ/Python/exterior")
    output_path = Path("/home/chofojeda/todo/Documents/PROYECTO GUACHETÁ/Python/exterior_data.ods")
    
    # Dictionary to hold DataFrames for each directory
    sheets_data = {}
    
    # Process each directory
    for directory in ['IED', 'STA Juanita']:
        dir_path = exterior_path / directory
        
        if not dir_path.exists():
            print(f"Directory {directory} not found")
            continue
            
        print(f"Processing directory: {directory}")
        
        # Find all JSON files in the directory
        json_files = list(dir_path.glob("*.json"))
        
        all_data = []
        
        for json_file in json_files:
            print(f"  Processing file: {json_file.name}")
            
            # Read and process JSON file
            data = read_json_file(json_file)
            df = process_json_data(data, json_file.name)
            
            if df is not None:
                all_data.append(df)
            else:
                print(f"    Warning: Could not process {json_file.name}")
        
        # Combine all data from this directory
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            sheets_data[directory] = combined_df
            print(f"  Combined {len(all_data)} files into {len(combined_df)} rows")
        else:
            print(f"  No data found for directory {directory}")
    
    # Write to ODS file
    if sheets_data:
        try:
            # Create Excel writer object (pandas doesn't support ODS directly, so we'll use Excel format)
            output_excel = output_path.with_suffix('.xlsx')
            
            with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                for sheet_name, df in sheets_data.items():
                    # Clean sheet name (remove invalid characters)
                    clean_sheet_name = sheet_name.replace(' ', '_').replace('/', '_')
                    df.to_excel(writer, sheet_name=clean_sheet_name, index=False)
                    print(f"Created sheet: {clean_sheet_name} with {len(df)} rows")
            
            print(f"\nExcel file created successfully: {output_excel}")
            
            # Try to convert to ODS using libreoffice if available
            try:
                import subprocess
                result = subprocess.run([
                    'libreoffice', '--headless', '--convert-to', 'ods', 
                    str(output_excel), '--outdir', str(output_path.parent)
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    print(f"ODS file created successfully: {output_path}")
                    # Remove the Excel file
                    output_excel.unlink()
                else:
                    print(f"Could not convert to ODS. Excel file available at: {output_excel}")
                    
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
                print(f"LibreOffice not available or conversion failed. Excel file available at: {output_excel}")
                
        except Exception as e:
            print(f"Error writing file: {e}")
    else:
        print("No data to write!")

def print_summary():
    """Print a summary of the data structure"""
    exterior_path = Path("/home/chofojeda/todo/Documents/PROYECTO GUACHETÁ/Python/exterior")
    
    print("="*50)
    print("SUMMARY OF EXTERIOR DATA STRUCTURE")
    print("="*50)
    
    for directory in ['IED', 'STA Juanita']:
        dir_path = exterior_path / directory
        
        if dir_path.exists():
            json_files = list(dir_path.glob("*.json"))
            print(f"\n{directory}:")
            print(f"  Found {len(json_files)} JSON files:")
            
            for json_file in json_files:
                data = read_json_file(json_file)
                if data:
                    sensor_count = len(data.get('data', []))
                    device_id = data.get('deviceId', 'Unknown')
                    date = data.get('date', 'Unknown')
                    print(f"    - {json_file.name}: {sensor_count} data points, Device: {device_id}, Date: {date}")
                else:
                    print(f"    - {json_file.name}: Could not read file")

if __name__ == "__main__":
    print("JSON to ODS Converter for PROYECTO GUACHETÁ")
    print("="*50)
    
    # Print summary first
    print_summary()
    
    print("\n" + "="*50)
    print("STARTING CONVERSION PROCESS")
    print("="*50)
    
    # Convert files
    convert_json_to_ods()
    
    print("\nConversion complete!")