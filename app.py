import io
import re
import base64
from datetime import date
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# --- Constants & Global Setup ---
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

NO_COLS = [f"No{i}" for i in range(1, 28)]

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- Helper Functions (adapted from original script) ---

def parse_date_dayfirst(s: pd.Series) -> pd.Series:
    """
    Parses a Pandas Series containing date values, handling multiple formats safely,
    prioritizing day-first formats.
    """
    # 1. Try to convert to numeric (for Excel serial dates)
    s_numeric = pd.to_numeric(s, errors='coerce')
    is_excel_serial = (s_numeric >= 1) & (s_numeric <= 80000)
    parsed_dates = pd.to_datetime(s_numeric[is_excel_serial], unit='d', origin='1899-12-30', errors='coerce')

    # Initialize results with NaT
    result = pd.Series(pd.NaT, index=s.index)
    result.loc[is_excel_serial] = parsed_dates

    # 2. Try specific day-first string formats for remaining NaT values
    remaining_idx = result.isna()
    if remaining_idx.any():
        s_remaining = s[remaining_idx].astype(str).str.strip()
        date_formats = ['%d-%m-%Y', '%d/%m/%Y', '%d.%m.%Y', '%d-%m-%y', '%d/%m/%y', '%Y-%m-%d']
        for fmt in date_formats:
            if remaining_idx.any():
                temp_parsed = pd.to_datetime(s_remaining[remaining_idx], format=fmt, errors='coerce')
                result.loc[temp_parsed.notna()] = temp_parsed[temp_parsed.notna()]
                remaining_idx = result.isna()

    # 3. Fallback for any still remaining NaT values with dayfirst=True
    if remaining_idx.any():
        s_remaining = s[remaining_idx].astype(str).str.strip()
        temp_parsed = pd.to_datetime(s_remaining, errors='coerce', dayfirst=True)
        result.loc[temp_parsed.notna()] = temp_parsed[temp_parsed.notna()]

    # Normalize to datetime.date
    return result.dt.normalize()

def build_right_from_nos(row: pd.Series) -> str:
    """
    Constructs the 'Right' string by extracting the units digit from 'NoX' columns.
    """
    right_digits = []
    for col in NO_COLS:
        val = str(row.get(col, '')).strip()
        if not val:
            continue

        try:
            # Try converting to integer
            # Use float first to handle string numbers like "10.0" before converting to int
            iv = int(float(val))
            right_digits.append(str(iv % 10))
        except ValueError:
            # If not a simple number, try to extract all digits and take the last one
            digits = re.findall(r'\d', val)
            if digits:
                right_digits.append(digits[-1])
    return "".join(right_digits)

def count_digits_from_right(right_str: str) -> dict:
    """
    Counts the frequency of each digit (0-9) in a given 'Right' string.
    """
    counts = {str(i): 0 for i in range(10)}
    for char in right_str:
        if '0' <= char <= '9':
            counts[char] += 1
    return counts

def make_rows_for_date(date_val: pd.Timestamp, counts: dict) -> list:
    """
    Generates 10 output rows (Min1-Min5, Max1-Max5) for a given date and digit counts.
    """
    date_str = date_val.strftime("%d-%m-%Y")
    
    # Convert counts to Series for easier frequency analysis
    counts_series = pd.Series(counts, dtype=int)
    
    # Filter out digits that don't appear
    present_digits = counts_series[counts_series > 0]

    if present_digits.empty:
        # If no digits are present, return 10 empty rows for this date
        empty_rows = []
        for label_prefix in ['Min', 'Max']:
            for i in range(1, 6):
                empty_rows.append({'Date': date_str, 'CB': f'{label_prefix}{i}', 'Freq': '', 'Count': '', **{str(d): '' for d in range(10)}})
        return empty_rows

    unique_freqs_asc = present_digits.value_counts().sort_index().index.tolist()
    unique_freqs_desc = unique_freqs_asc[::-1]

    def make_row(label_prefix: str, freq_idx: int, unique_freqs: list) -> dict:
        row_data = {str(d): '' for d in range(10)}
        row_data['Date'] = date_str
        row_data['CB'] = f'{label_prefix}{freq_idx + 1}'
        
        if freq_idx < len(unique_freqs):
            freq_val = unique_freqs[freq_idx]
            digits_at_freq = present_digits[present_digits == freq_val].index.tolist()
            
            row_data['Freq'] = freq_val
            row_data['Count'] = len(digits_at_freq)
            for d in digits_at_freq:
                row_data[d] = d
        else:
            # Fill with empty if no more unique frequencies
            row_data['Freq'] = ''
            row_data['Count'] = ''
            
        return row_data

    out_rows = []
    # Min rows
    for i in range(5):
        out_rows.append(make_row('Min', i, unique_freqs_asc))

    # Max rows
    for i in range(5):
        out_rows.append(make_row('Max', i, unique_freqs_desc))
        
    return out_rows


def load_csv(file_stream) -> pd.DataFrame:
    """
    Loads CSV from a file-like object, cleans, normalizes data, and prepares DataFrame.
    Modified to accept a file stream instead of a Path object.
    """
    try:
        # Read the file stream as a string, assuming UTF-8 encoding
        df = pd.read_csv(io.StringIO(file_stream.read().decode('utf-8')), dtype=str, keep_default_na=False)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    # Clean column names
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')

    # Handle secondary header rows (e.g., "STT", "No.", "No")
    # If a 'STT' column exists and contains known header-like values, filter them out.
    if 'STT' in df.columns:
        df = df[~df['STT'].astype(str).str.lower().isin(['stt', 'no.', 'no', ''])].copy()
        
    # Date column processing
    if 'Date' in df.columns and not df['Date'].empty:
        df['Date'] = parse_date_dayfirst(df['Date'])
    else:
        # Auto-detect date column
        date_candidates = {}
        for col in df.columns:
            temp_dates = parse_date_dayfirst(df[col])
            valid_date_ratio = temp_dates.notna().sum() / len(df)
            if valid_date_ratio > 0.5:  # Must have more than 50% valid dates
                date_candidates[col] = valid_date_ratio
        
        if not date_candidates:
            raise ValueError("No suitable 'Date' column found or could be automatically detected (ratio > 50%).")

        best_date_col = max(date_candidates, key=date_candidates.get)
        df['Date'] = parse_date_dayfirst(df[best_date_col])
        app.logger.info(f"Auto-detected '{best_date_col}' as the Date column.")


    # Ensure all NO_COLS exist, creating empty ones if missing
    for col in NO_COLS:
        if col not in df.columns:
            df[col] = ''

    # Handle 'Right' column
    if 'Right' in df.columns and not df['Right'].empty:
        df['Right'] = df['Right'].astype(str).str.replace(' ', '').str.strip()
    else:
        # Build 'Right' column from NO_COLS
        df['Right'] = df.apply(build_right_from_nos, axis=1)

    # Remove rows where 'Date' could not be parsed
    df.dropna(subset=['Date'], inplace=True)
    if df.empty:
        raise ValueError("No valid date rows remaining after parsing. Please check your date column.")

    # Sort by Date
    df.sort_values(by='Date', inplace=True)

    # Select relevant columns for output
    return df[['Date', 'Right'] + NO_COLS]

# --- Flask Routes ---

@app.route('/')
def welcome():
    return jsonify({"message": "Welcome to the CSV Digit Frequency Analyzer API. Use /process-csv to upload and process a file."})

@app.route('/process-csv', methods=['POST'])
def process_csv_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not file.filename.lower().endswith('.csv'):
        return jsonify({"error": "Invalid file type. Please upload a CSV file."}), 400

    try:
        # Pass the file stream directly to load_csv
        input_df = load_csv(file.stream)
        
        out_rows = []
        for _, row in input_df.iterrows():
            date_val = row['Date']
            right_str = row['Right']
            
            counts = count_digits_from_right(right_str)
            out_rows.extend(make_rows_for_date(date_val, counts))
        
        # Define output columns explicitly for consistent order
        output_cols = ['Date', 'CB', 'Freq', 'Count'] + [str(d) for d in range(10)]
        out_df = pd.DataFrame(out_rows, columns=output_cols)
        
        # Format Date column for final output display
        out_df['Date'] = out_df['Date'].dt.strftime("%d-%m-%Y")
        
        # Prepare JSON response
        results_json = out_df.to_dict(orient='records')

        # Prepare CSV content for download (base64 encoded)
        csv_buffer = io.StringIO()
        out_df.to_csv(csv_buffer, index=False)
        csv_content_base64 = base64.b64encode(csv_buffer.getvalue().encode('utf-8')).decode('utf-8')

        return jsonify({
            "status": "success",
            "message": "CSV processed successfully!",
            "data": results_json,
            "csv_content": csv_content_base64
        })

    except ValueError as e:
        app.logger.warning(f"Client error during CSV processing: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred during processing."}), 500

if __name__ == '__main__':
    # For local development:
    # app.run(debug=True, port=5000)
    # For production (e.g., Render), gunicorn will handle running the app.
    app.run(debug=True, port=5000)