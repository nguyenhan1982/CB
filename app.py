import pandas as pd
from pathlib import Path
import datetime
import io
import re
import tempfile
import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# --- Original Script Constants & Helper Functions (adapted for web context) ---

# NO_COLS: A list of string representing numerical column names, from "No1" to "No27".
NO_COLS = [f"No{i}" for i in range(1, 28)]

def parse_date_dayfirst(s: pd.Series) -> pd.Series:
    """
    Parses a Pandas Series containing date values, handling multiple formats safely
    and prioritizing day-first formats.
    """
    original_series = s.astype(str) # Ensure all elements are strings for parsing
    parsed_dates = pd.Series(pd.NaT, index=s.index)

    # 1. Try to convert to numeric (for Excel serial dates)
    numeric_s = pd.to_numeric(original_series, errors='coerce')
    # Reasonable range for Excel dates (e.g., 1 to ~80000 covers 1900 to 2100)
    is_numeric_date = (numeric_s.notna()) & (numeric_s > 1) & (numeric_s < 80000) 
    parsed_dates[is_numeric_date] = pd.to_datetime(numeric_s[is_numeric_date], unit='d', origin='1899-12-30', errors='coerce')

    # 2. Try common day-first string formats on remaining NaT values
    remaining_indices = parsed_dates.isna()
    if remaining_indices.any():
        for fmt in ['%d-%m-%Y', '%d/%m/%Y', '%d.%m.%Y', '%d-%m-%y', '%d/%m/%y', '%Y-%m-%d']:
            if not remaining_indices.any():
                break # All dates parsed
            temp_parsed = pd.to_datetime(original_series[remaining_indices], format=fmt, errors='coerce')
            parsed_dates.loc[temp_parsed.notna()] = temp_parsed[temp_parsed.notna()]
            remaining_indices = parsed_dates.isna()

    # 3. Fallback: general pd.to_datetime with dayfirst=True
    if remaining_indices.any():
        fallback_parsed = pd.to_datetime(original_series[remaining_indices], errors='coerce', dayfirst=True)
        parsed_dates.loc[fallback_parsed.notna()] = fallback_parsed[fallback_parsed.notna()]

    # Normalize to datetime.date
    return parsed_dates.dt.normalize()

def build_right_from_nos(row: pd.Series) -> str:
    """
    Builds the 'Right' string by extracting the units digit from 'No1' to 'No27' columns of a row.
    """
    right_digits = []
    for col_name in NO_COLS:
        value = str(row.get(col_name, '')).strip()
        if not value:
            continue

        try:
            # Try converting to integer directly, handling potential floats like '1.0'
            iv = int(float(value)) 
            right_digits.append(str(iv % 10))
        except ValueError:
            # If not a pure number, try to extract digits from string
            digits_in_str = re.findall(r'\d', value)
            if digits_in_str:
                right_digits.append(digits_in_str[-1]) # Take the last digit found

    return "".join(right_digits)

def load_csv(file_stream) -> pd.DataFrame:
    """
    Loads a CSV file from a file stream, cleans, normalizes data, and prepares DataFrame.
    Returns the processed DataFrame or raises an exception on error.
    """
    try:
        df = pd.read_csv(file_stream, dtype=str, keep_default_na=False)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    # Clean column names
    df.columns = df.columns.str.replace('\ufeff', '', regex=False).str.strip()

    # Handle secondary header row (if any)
    # Checks if 'STT' column exists and contains values like "STT", "No.", "No"
    if 'STT' in df.columns and df['STT'].astype(str).str.contains(r'^(STT|No\.?)$', regex=True, na=False).any():
        df = df[~df['STT'].astype(str).str.contains(r'^(STT|No\.?)$', regex=True, na=False)]
        df = df.reset_index(drop=True) # Reset index after dropping rows

    # Process 'Date' column
    if 'Date' in df.columns:
        df['Date'] = parse_date_dayfirst(df['Date'])
    else:
        # Autodetect date column
        date_candidates = {}
        for col in df.columns:
            # Avoid re-parsing already known numeric columns or 'Right'
            if col not in NO_COLS and col != 'Right': 
                parsed_col = parse_date_dayfirst(df[col])
                valid_date_ratio = parsed_col.notna().sum() / len(df)
                if valid_date_ratio > 0.5: # Must have more than 50% valid dates
                    date_candidates[col] = valid_date_ratio
                    df[col] = parsed_col # Temporarily assign parsed dates to the column

        if not date_candidates:
            raise ValueError("Could not detect a suitable 'Date' column (> 50% valid dates).")

        # Select the column with the highest ratio of valid dates
        best_date_col = max(date_candidates, key=date_candidates.get)
        df['Date'] = df[best_date_col]
        # Drop the original column if it's not 'Date' and we renamed it
        if best_date_col != 'Date':
            df = df.drop(columns=[best_date_col], errors='ignore')

    # Ensure 'NoX' columns exist
    for col in NO_COLS:
        if col not in df.columns:
            df[col] = "" # Create empty column

    # Process 'Right' column
    if 'Right' in df.columns:
        df['Right'] = df['Right'].astype(str).str.strip()
    else:
        df['Right'] = df.apply(build_right_from_nos, axis=1)

    df = df.dropna(subset=['Date']) # Drop rows where Date could not be parsed
    df = df.sort_values(by='Date').reset_index(drop=True)

    # Return only relevant columns
    return df[['Date', 'Right'] + NO_COLS]

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
    Generates 10 output rows (Min1-Min5, Max1-Max5) for a given date and digit counts.
    """
    out_rows = []
    freq_series = pd.Series(counts, name='Freq').astype(int)
    
    # Get unique frequencies and their counts
    unique_freq_counts = freq_series.value_counts().sort_index()
    unique_freqs = unique_freq_counts.index.tolist()

    # Function to create a single row
    def make_row(label: str, freq_val, digits_at_freq: list):
        row_data = {
            'Date': date_val,
            'CB': label,
            'Freq': freq_val,
            'Count': len(digits_at_freq)
        }
        for i in range(10):
            row_data[str(i)] = str(i) if str(i) in digits_at_freq else ''
        return row_data

    # Min rows
    min_freqs_sorted = sorted(unique_freqs)[:5]
    for i in range(5):
        if i < len(min_freqs_sorted):
            freq = min_freqs_sorted[i]
            digits = [digit for digit, count in counts.items() if count == freq]
            digits.sort(key=int) # Ensure consistent order for digits
            out_rows.append(make_row(f'Min{i+1}', freq, digits))
        else:
            out_rows.append(make_row(f'Min{i+1}', '', [])) # Empty row if not enough unique freqs

    # Max rows
    max_freqs_sorted = sorted(unique_freqs, reverse=True)[:5]
    for i in range(5):
        if i < len(max_freqs_sorted):
            freq = max_freqs_sorted[i]
            digits = [digit for digit, count in counts.items() if count == freq]
            digits.sort(key=int) # Ensure consistent order for digits
            out_rows.append(make_row(f'Max{i+1}', freq, digits))
        else:
            out_rows.append(make_row(f'Max{i+1}', '', [])) # Empty row if not enough unique freqs

    return out_rows

# --- Flask Application Setup ---

app = Flask(__name__)
# Enable CORS for all origins. For production, consider restricting to specific frontend origins.
CORS(app) 

@app.route('/')
def index():
    return "<h1>CSV Processor API</h1><p>Upload a CSV to the /process-csv endpoint.</p>"

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
        # Read the uploaded file into an in-memory stream for pandas
        # Decode using utf-8-sig to handle BOM which might be present in CSVs from Excel
        file_content = file.read().decode('utf-8-sig') 
        file_stream = io.StringIO(file_content)
        
        # Load and preprocess the data
        processed_df = load_csv(file_stream)
        
        out_rows = []
        for _, row in processed_df.iterrows():
            date_val = row['Date']
            right_str = row['Right']
            counts = count_digits_from_right(right_str)
            out_rows.extend(make_rows_for_date(date_val, counts))
        
        # Define output columns explicitly to maintain order
        out_df_columns = ['Date', 'CB', 'Freq', 'Count'] + [str(i) for i in range(10)]
        out_df = pd.DataFrame(out_rows, columns=out_df_columns)
        
        # Format Date column
        out_df['Date'] = out_df['Date'].dt.strftime("%d-%m-%Y")
        
        # Save the resulting DataFrame to an in-memory CSV file and send it back
        output_buffer = io.StringIO()
        out_df.to_csv(output_buffer, index=False)
        output_buffer.seek(0) # Rewind to the beginning of the buffer
        
        return send_file(
            io.BytesIO(output_buffer.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='CB.csv'
        )

    except ValueError as e:
        # Catch specific data processing errors
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Catch any other unexpected errors
        app.logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred during processing."}), 500

if __name__ == '__main__':
    # For local development
    # In a production environment (like Render), gunicorn will manage the app
    app.run(debug=True, port=5000)