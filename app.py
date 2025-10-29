import os
import io
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# --- Constants & Configuration ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes, essential for frontend communication

NO_COLS = [f"No{i}" for i in range(1, 28)]

# --- Helper Functions (adapted from original script) ---

def parse_date_dayfirst(s: pd.Series) -> pd.Series:
    """
    Parses a Pandas Series containing date values, handling multiple formats
    safely and prioritizing day-first formats.
    """
    s_clean = s.astype(str).str.strip() # Ensure all are strings and strip whitespace
    parsed_dates = pd.Series(pd.NaT, index=s.index)

    # 1. Try converting Excel serial numbers (values 1 to 80000)
    numeric_s = pd.to_numeric(s_clean, errors='coerce')
    is_excel_serial = (numeric_s >= 1) & (numeric_s <= 80000)
    
    if is_excel_serial.any():
        parsed_dates.loc[is_excel_serial] = pd.to_datetime(numeric_s[is_excel_serial], unit='d', origin='1899-12-30', errors='coerce')
    
    # 2. Try common day-first string formats for remaining NaT values
    remaining_indices = parsed_dates.isna()
    remaining_s_values = s_clean[remaining_indices]

    date_formats = ['%d-%m-%Y', '%d/%m/%Y', '%d.%m.%Y', '%d-%m-%y', '%d/%m/%y', '%Y-%m-%d']

    for fmt in date_formats:
        if remaining_s_values.empty:
            break
        temp_parsed = pd.to_datetime(remaining_s_values, format=fmt, errors='coerce')
        # Fill only NaT values in parsed_dates from temp_parsed
        parsed_dates.loc[temp_parsed.index] = parsed_dates.loc[temp_parsed.index].fillna(temp_parsed)
        remaining_indices = parsed_dates.isna()
        remaining_s_values = s_clean[remaining_indices] # Update remaining values

    # 3. Fallback for any remaining unparsed dates
    if not remaining_s_values.empty:
        fallback_parsed = pd.to_datetime(remaining_s_values, errors='coerce', dayfirst=True)
        # Fill only NaT values in parsed_dates from fallback_parsed
        parsed_dates.loc[fallback_parsed.index] = parsed_dates.loc[fallback_parsed.index].fillna(fallback_parsed)

    # Normalize to date objects
    return parsed_dates.dt.normalize()

def load_csv_data(csv_file_stream: io.StringIO) -> pd.DataFrame:
    """
    Loads CSV from a stream, cleans, normalizes data, and prepares DataFrame.
    Raises ValueError on critical errors.
    """
    try:
        df = pd.read_csv(csv_file_stream, dtype=str, keep_default_na=False)
    except Exception as e:
        raise ValueError(f"Error reading CSV: {e}")

    # Clean column names
    df.columns = df.columns.str.replace('\ufeff', '').str.strip()

    # Handle auxiliary header rows (e.g., "STT", "No.", "No")
    if 'STT' in df.columns:
        df = df[~df['STT'].isin(['STT', 'No.', 'No'])].reset_index(drop=True)
    
    initial_cols = df.columns.tolist()

    # --- Process 'Date' column ---
    if 'Date' in df.columns:
        df['Date'] = parse_date_dayfirst(df['Date'])
    else:
        # Autodetect 'Date' column
        potential_date_cols = {}
        for col in initial_cols:
            if col in NO_COLS: continue # Skip NoX columns, as they contain numbers not dates
            temp_parsed_dates = parse_date_dayfirst(df[col])
            valid_dates_ratio = temp_parsed_dates.count() / len(df)
            if valid_dates_ratio > 0.5: # Must have more than 50% successfully parsed dates
                potential_date_cols[col] = valid_dates_ratio

        if not potential_date_cols:
            raise ValueError("Could not auto-detect a 'Date' column with more than 50% valid dates.")
        
        # Select the column with the highest valid date ratio
        best_date_col = max(potential_date_cols, key=potential_date_cols.get)
        app.logger.info(f"Auto-detected '{best_date_col}' as the 'Date' column.")
        df['Date'] = parse_date_dayfirst(df[best_date_col])

    # --- Ensure NO_COLS exist ---
    for col in NO_COLS:
        if col not in df.columns:
            df[col] = '' # Add missing columns as empty strings

    # --- Process 'Right' column ---
    if 'Right' in df.columns:
        df['Right'] = df['Right'].astype(str).str.replace(' ', '')
    else:
        app.logger.info("Column 'Right' not found. Building it from 'NoX' columns.")
        df['Right'] = df.apply(build_right_from_nos, axis=1)

    # Remove rows where 'Date' could not be parsed
    df = df.dropna(subset=['Date'])
    
    # If after cleaning, no valid rows remain, raise an error
    if df.empty:
        raise ValueError("No valid data rows remaining after date parsing and cleaning.")

    # Sort by 'Date'
    df = df.sort_values(by='Date').reset_index(drop=True)

    # Select and return only relevant columns (Date, Right, and NoX columns that were present or created)
    # Ensure NoX columns order as per NO_COLS
    present_no_cols = [col for col in NO_COLS if col in df.columns]
    final_cols = ['Date', 'Right'] + present_no_cols
    
    return df[final_cols]

def build_right_from_nos(row: pd.Series) -> str:
    """
    Builds the 'Right' string by extracting the units digit from 'NoX' columns.
    """
    right_digits = []
    for col in NO_COLS:
        val = str(row.get(col, '')).strip()
        if not val:
            continue

        try:
            # Try converting to integer directly (handle "1.0" by converting to float first)
            iv = int(float(val)) 
            right_digits.append(str(iv % 10))
        except ValueError:
            # If not a clean number, extract all digits and take the last one
            digits_in_str = [char for char in val if char.isdigit()]
            if digits_in_str:
                right_digits.append(digits_in_str[-1])
    return "".join(right_digits)

def count_digits_from_right(right_str: str) -> dict:
    """
    Counts the frequency of each digit (0-9) in a given 'Right' string.
    """
    counts = {str(i): 0 for i in range(10)}
    for char in right_str:
        if char.isdigit():
            counts[char] += 1
    return counts

def make_rows_for_date(date_val: pd.Timestamp, counts: dict) -> list:
    """
    Generates 10 output rows (Min1-Min5, Max1-Max5) for a given date
    and digit frequencies, including Freq=0 for Min rows.
    """
    counts_series = pd.Series(counts, dtype=int)
    
    all_output_rows = []

    def make_row(date, cb_label, freq, digits_at_freq):
        row_data = {
            'Date': date,
            'CB': cb_label,
            'Freq': freq,
            'Count': len(digits_at_freq) if digits_at_freq else ''
        }
        for d in range(10):
            row_data[str(d)] = str(d) if str(d) in digits_at_freq else ''
        return row_data

    # --- Generate Min rows (including Freq=0) ---
    # Group digits by their frequencies, considering all 0-9 digits
    freq_to_digits = {}
    for digit in range(10):
        freq = counts_series.get(str(digit), 0) # Ensure all digits 0-9 are covered
        if freq not in freq_to_digits:
            freq_to_digits[freq] = []
        freq_to_digits[freq].append(str(digit))
    
    sorted_min_freqs = sorted(freq_to_digits.keys()) # This will include 0 if present

    min_counter = 1
    for freq in sorted_min_freqs:
        if min_counter > 5: break
        digits_at_freq = sorted(freq_to_digits[freq]) # Sort digits for consistent output
        all_output_rows.append(make_row(date_val, f'Min{min_counter}', freq, digits_at_freq))
        min_counter += 1
    
    # Fill remaining Min rows if less than 5 unique frequencies
    while min_counter <= 5:
        all_output_rows.append(make_row(date_val, f'Min{min_counter}', '', []))
        min_counter += 1

    # --- Generate Max rows (only considering frequencies > 0) ---
    actual_counts = counts_series[counts_series > 0] # Max rows still only care about present digits
    
    if actual_counts.empty:
        max_freqs_values = []
    else:
        # Group digits with actual_counts > 0
        max_freq_to_digits = {}
        for digit, freq in actual_counts.items():
            if freq not in max_freq_to_digits:
                max_freq_to_digits[freq] = []
            max_freq_to_digits[freq].append(digit)
        max_freqs_values = sorted(max_freq_to_digits.keys(), reverse=True) # Sort frequencies descending

    max_counter = 1
    for freq in max_freqs_values:
        if max_counter > 5: break
        digits_at_freq = sorted(max_freq_to_digits[freq])
        all_output_rows.append(make_row(date_val, f'Max{max_counter}', freq, digits_at_freq))
        max_counter += 1
    
    # Fill remaining Max rows if less than 5 unique frequencies
    while max_counter <= 5:
        all_output_rows.append(make_row(date_val, f'Max{max_counter}', '', []))
        max_counter += 1

    return all_output_rows

# --- Main Processing Logic ---
def process_uploaded_csv(csv_file_stream: io.StringIO) -> io.BytesIO:
    """
    Main function logic adapted for web. Takes a CSV stream, processes it,
    and returns a BytesIO stream of the output CSV.
    """
    df_processed = load_csv_data(csv_file_stream)

    out_rows = []
    for index, row in df_processed.iterrows():
        date_val = row['Date']
        right_str = row['Right']
        
        counts = count_digits_from_right(right_str)
        generated_rows = make_rows_for_date(date_val, counts)
        out_rows.extend(generated_rows)

    if not out_rows:
        raise ValueError("No output data could be generated. Please check your input CSV and ensure it contains valid data for processing.")

    out_df = pd.DataFrame(out_rows)

    # Ensure all digit columns are present, even if some dates had no digits.
    # This prevents errors if a digit (e.g., '0') never appeared across all 'Right' strings.
    for d in range(10):
        if str(d) not in out_df.columns:
            out_df[str(d)] = ''

    # Reorder columns explicitly
    output_cols_order = ['Date', 'CB', 'Freq', 'Count'] + [str(d) for d in range(10)]
    out_df = out_df[output_cols_order]

    # Format Date column as string "%d-%m-%Y"
    out_df['Date'] = out_df['Date'].dt.strftime("%d-%m-%Y")

    # Write to an in-memory BytesIO object
    output_stream = io.BytesIO()
    out_df.to_csv(output_stream, index=False, encoding='utf-8')
    output_stream.seek(0) # Rewind to the beginning for reading
    
    return output_stream

# --- Flask API Endpoint ---

@app.route('/process_csv', methods=['POST'])
def handle_csv_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.csv'):
        try:
            # Read the uploaded file into a StringIO object
            # Decode using utf-8-sig to handle BOM (Byte Order Mark) which Excel often adds
            csv_content = file.read().decode('utf-8-sig') 
            csv_stream = io.StringIO(csv_content)

            output_csv_stream = process_uploaded_csv(csv_stream)

            return send_file(
                output_csv_stream,
                mimetype='text/csv',
                as_attachment=True,
                download_name='CB_output.csv'
            )
        except ValueError as ve:
            app.logger.error(f"Processing error (ValueError): {ve}")
            return jsonify({'error': str(ve)}), 400
        except Exception as e:
            app.logger.exception("An unexpected error occurred during CSV processing.")
            return jsonify({'error': 'An unexpected error occurred: ' + str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400

# Basic root endpoint for health check or info
@app.route('/')
def index():
    return "CSV Processor API is running! Upload a CSV to /process_csv to use the service."

if __name__ == '__main__':
    # For local development, use a placeholder port
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)