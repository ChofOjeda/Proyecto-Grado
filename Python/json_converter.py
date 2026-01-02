import json
import os
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows

def process_json_files():
    """
    Process JSON files from exterior directory and create an ODS file with separate sheets
    for IED and STA Juanita data
    """
    
    # Base directory
    base_dir = "/home/chofojeda/todo/Documents/PROYECTO GUACHETÁ/Python/exterior"
    
    # Initialize data containers
    ied_data = []
    sta_juanita_data = []
    
    # Process IED directory
    ied_dir = os.path.join(base_dir, "IED")
    if os.path.exists(ied_dir):
        print("Processing IED directory...")
        for filename in os.listdir(ied_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(ied_dir, filename)
                print(f"Processing: {filepath}")
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Handle malformed JSON that starts with numbers
                        if content.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '0.')):
                            content = content[2:]  # Remove the number prefix
                        
                        data = json.loads(content)
                        
                        # Extract sensor data
                        if 'data' in data:
                            for record in data['data']:
                                record['source_file'] = filename
                                record['location'] = 'IED'
                                ied_data.append(record)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
    
    # Process STA Juanita directory
    sta_dir = os.path.join(base_dir, "STA Juanita")
    if os.path.exists(sta_dir):
        print("Processing STA Juanita directory...")
        for filename in os.listdir(sta_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(sta_dir, filename)
                print(f"Processing: {filepath}")
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Handle malformed JSON that starts with numbers
                        if content.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '0.')):
                            content = content[2:]  # Remove the number prefix
                        
                        data = json.loads(content)
                        
                        # Extract sensor data
                        if 'data' in data:
                            for record in data['data']:
                                record['source_file'] = filename
                                record['location'] = 'STA Juanita'
                                sta_juanita_data.append(record)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
    
    # Create DataFrames
    print(f"IED records found: {len(ied_data)}")
    print(f"STA Juanita records found: {len(sta_juanita_data)}")
    
    # Create Excel workbook
    wb = Workbook()
    
    # Remove default sheet
    wb.remove(wb.active)
    
    # Create IED sheet
    if ied_data:
        ied_df = pd.DataFrame(ied_data)
        ied_sheet = wb.create_sheet("IED")
        
        # Add headers with styling
        headers = list(ied_df.columns)
        for col_idx, header in enumerate(headers, 1):
            cell = ied_sheet.cell(row=1, column=col_idx, value=header)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        
        # Add data
        for row_idx, row_data in enumerate(dataframe_to_rows(ied_df, index=False, header=False), 2):
            for col_idx, value in enumerate(row_data, 1):
                ied_sheet.cell(row=row_idx, column=col_idx, value=value)
        
        # Auto-adjust column widths
        for column in ied_sheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ied_sheet.column_dimensions[column_letter].width = adjusted_width
    
    # Create STA Juanita sheet
    if sta_juanita_data:
        sta_df = pd.DataFrame(sta_juanita_data)
        sta_sheet = wb.create_sheet("STA Juanita")
        
        # Add headers with styling
        headers = list(sta_df.columns)
        for col_idx, header in enumerate(headers, 1):
            cell = sta_sheet.cell(row=1, column=col_idx, value=header)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="70AD47", end_color="70AD47", fill_type="solid")
        
        # Add data
        for row_idx, row_data in enumerate(dataframe_to_rows(sta_df, index=False, header=False), 2):
            for col_idx, value in enumerate(row_data, 1):
                sta_sheet.cell(row=row_idx, column=col_idx, value=value)
        
        # Auto-adjust column widths
        for column in sta_sheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            sta_sheet.column_dimensions[column_letter].width = adjusted_width
    
    # Save as Excel file (we'll convert to ODS afterwards)
    output_excel = "/home/chofojeda/todo/Documents/PROYECTO GUACHETÁ/Python/sensor_data_consolidated.xlsx"
    wb.save(output_excel)
    print(f"Excel file saved: {output_excel}")
    
    # Convert to ODS using pandas
    output_ods = "/home/chofojeda/todo/Documents/PROYECTO GUACHETÁ/Python/sensor_data_consolidated.ods"
    
    with pd.ExcelWriter(output_ods, engine='odf') as writer:
        if ied_data:
            ied_df = pd.DataFrame(ied_data)
            ied_df.to_excel(writer, sheet_name='IED', index=False)
        
        if sta_juanita_data:
            sta_df = pd.DataFrame(sta_juanita_data)
            sta_df.to_excel(writer, sheet_name='STA Juanita', index=False)
    
    print(f"ODS file saved: {output_ods}")
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"Total IED records: {len(ied_data)}")
    print(f"Total STA Juanita records: {len(sta_juanita_data)}")
    print(f"Total records: {len(ied_data) + len(sta_juanita_data)}")
    print(f"Files created:")
    print(f"  - {output_excel}")
    print(f"  - {output_ods}")

if __name__ == "__main__":
    process_json_files()