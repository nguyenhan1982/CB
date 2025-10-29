import pandas as pd
from pathlib import Path
import datetime
import numpy as np
import re
from collections import defaultdict
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
import shutil
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
app_logger = logging.getLogger(__name__)

# --- Constants & Utility Functions from the script ---
NO_COLS = [f"No{i}" for i in range(1, 28)]

def parse_date_dayfirst(s: pd.Series) -> pd.Series:
    """
    Parses a Pandas Series containing date values, handling various formats safely
    and prioritizing day-first formats.
    """
    parsed_dates = pd.Series(pd.NaT, index=s.index)

    # 1. Try to convert numeric values (Excel serial dates)
    numeric_s = pd.to_numeric(s, errors='coerce')
    # Filter for values that could realistically be Excel serial dates (1 to ~80000 for dates up to ~2200)
    excel_serial_mask = (numeric_s >= 1) & (numeric_s <= 80000)
    if excel_serial_mask.any():
        parsed_dates.loc[excel_serial_mask] = pd.to_datetime(numeric_s.loc[excel_serial_mask], unit='d', origin='1899-12-30')

    # 2. Try specific day-first string formats for remaining NaT values
    remaining_mask = parsed_dates.isna()
    if remaining_mask.any():
        # Ensure we only work with string types for string parsing
        remaining_series = s.loc[remaining_mask].astype(str)
        dayfirst_formats = [
            '%d-%m-%Y', '%d/%m/%Y', '%d.%m.%Y',
            '%d-%m-%y', '%d/%m/%y',
            '%Y-%m-%d' # Also include YMD, as `dayfirst=True` can still apply for ambiguous cases
        ]
        for fmt in dayfirst_formats:
            # Only try to parse if there are still NaT values
            if remaining_mask.any():
                temp_parsed = pd.to_datetime(remaining_series.loc[remaining_mask], format=fmt, errors='coerce')
                # Fill NaT values in parsed_dates with successfully parsed dates
                parsed_dates.loc[remaining_mask] = parsed_dates.loc[remaining_mask].fillna(temp_parsed)
                # Update the mask for remaining NaT values
                remaining_mask = parsed_dates.isna()
            else:
                break # All dates parsed, no need to try more formats

    # 3. Fallback: try pd.to_datetime with dayfirst=True for any still unparsed
    if remaining_mask.any():
        parsed_dates.loc[remaining_mask] = pd.to_datetime(s.loc[remaining_mask], errors='coerce', dayfirst=True)

    # Normalize to date objects (remove time component)
    return parsed_dates.dt.normalize()

def build_right_from_nos(row: pd.Series) -> str:
    """
    Constructs the 'Right' string by extracting the last digit of numerical
    values from 'NoX' columns, or the last digit found in mixed strings.
    """
    right_digits = []
    for col in NO_COLS:
        # Use .get() with default empty string to handle potentially missing columns safely
        val = str(row.get(col, '')).strip()
        if not val:
            continue

        try:
            # Try converting to integer first (using float to handle '1.0' type strings)
            iv = int(float(val))
            right_digits.append(str(iv % 10))
        except ValueError:
            # If not a pure number, try to extract all digits from the string
            digits_in_string = re.findall(r'\d', val)
            if digits_in_string:
                right_digits.append(digits_in_string[-1]) # Take the last digit found
    return "".join(right_digits)

def count_digits_from_right(right_str: str) -> dict:
    """
    Counts the frequency of each digit (0-9) in a given 'Right' string.
    """
    counts = defaultdict(int)
    # Initialize all digit counts to 0
    for digit in range(10):
        counts[digit] = 0
    for char in right_str:
        if char.isdigit():
            counts[int(char)] += 1
    return dict(counts) # Convert back to regular dict

def make_rows_for_date(date_val: pd.Timestamp, counts: dict) -> list:
    """
    Generates 10 output rows (Min1-Min5, Max1-Max5) for a given date
    and its digit frequency counts.
    """
    out_rows = []
    
    # Convert counts to a Series to easily get unique frequencies
    freq_series = pd.Series(counts)
    
    # Get unique frequencies
    unique_freqs = freq_series.unique()
    
    # Sort for Min (ascending) and Max (descending)
    asc_freqs = np.sort(unique_freqs)
    desc_freqs = np.sort(unique_freqs)[::-1]
    
    def make_row(label_prefix: str, freq: int, digits_at_freq: list, rank: int) -> dict:
        row_data = {
            'Date': date_val,
            'CB': f"{label_prefix}{rank}",
            'Freq': freq,
            'Count': len(digits_at_freq)
        }
        # Initialize digit columns to empty string
        for d in range(10):
            row_data[str(d)] = ''
        # Fill in digits that have this frequency
        for d in digits_at_freq:
            row_data[str(d)] = str(d)
        return row_data

    # Generate Min rows
    rank = 1
    for freq in asc_freqs:
        if rank > 5: break
        digits_at_freq = [d for d, f in counts.items() if f == freq]
        out_rows.append(make_row("Min", freq, digits_at_freq, rank))
        rank += 1
    
    # Pad with empty Min rows if less than 5 unique frequencies
    while rank <= 5:
        out_rows.append({
            'Date': date_val, 'CB': f"Min{rank}", 'Freq': np.nan, 'Count': np.nan,
            **{str(d): '' for d in range(10)}
        })
        rank += 1

    # Generate Max rows
    rank = 1
    for freq in desc_freqs:
        if rank > 5: break
        digits_at_freq = [d for d, f in counts.items() if f == freq]
        out_rows.append(make_row("Max", freq, digits_at_freq, rank))
        rank += 1
        
    # Pad with empty Max rows if less than 5 unique frequencies
    while rank <= 5:
        out_rows.append({
            'Date': date_val, 'CB': f"Max{rank}", 'Freq': np.nan, 'Count': np.nan,
            **{str(d): '' for d in range(10)}
        })
        rank += 1

    return out_rows

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Global temporary directory for generated files
# Using tempfile.mkdtemp() for a session-specific temp dir.
# Files in this directory will be cleaned up on app shutdown or restart.
# For persistent storage across server restarts, consider cloud storage (e.g., S3).
TEMP_DIR = tempfile.mkdtemp()
app_logger.info(f"Temporary directory for files: {TEMP_DIR}")

# Cleanup temp dir on app context teardown (best effort for local dev and some deployments)
@app.teardown_appcontext
def cleanup_temp_dir(exception=None):
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
            app_logger.info(f"Cleaned up temporary directory: {TEMP_DIR}")
        except Exception as e:
            app_logger.error(f"Error cleaning up temporary directory {TEMP_DIR}: {e}")

# --- Core Logic adapted for Flask ---
def load_csv_for_flask(input_path: Path) -> pd.DataFrame:
    """
    Loads and preprocesses the CSV file, adapting the original script's load_csv
    for use within the Flask environment. Raises exceptions for errors.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at {input_path}")

    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)

    # Clean column names (remove BOM, strip whitespace)
    df.columns = df.columns.str.replace('\ufeff', '').str.strip()

    # Handle potential secondary header rows (e.g., if 'STT' column contains 'STT')
    if 'STT' in df.columns and df['STT'].astype(str).isin(['STT', 'No.', 'No', 'no']).any():
        df = df[~df['STT'].astype(str).isin(['STT', 'No.', 'No', 'no'])]
    
    if df.empty:
        raise ValueError("CSV is empty or contains only header rows after cleanup.")


    # --- Process 'Date' column ---
    date_col_candidates = [col for col in df.columns if 'date' in col.lower() or 'ngày' in col.lower()]
    
    if 'Date' in df.columns:
        df['Date'] = parse_date_dayfirst(df['Date'])
    else:
        best_date_col = None
        max_valid_ratio = 0.0
        
        # Prioritize columns that explicitly mention 'date' or 'ngày'
        for col in date_col_candidates:
            temp_dates = parse_date_dayfirst(df[col])
            valid_ratio = temp_dates.count() / len(temp_dates)
            if valid_ratio > max_valid_ratio and valid_ratio > 0.5: # Must be more than 50% valid dates
                max_valid_ratio = valid_ratio
                best_date_col = col

        # If not found in candidates or ratio is too low, try all other columns
        if not best_date_col or max_valid_ratio <= 0.5:
            for col in df.columns:
                # Avoid trying to parse 'NoX' columns as dates to prevent false positives
                if col not in NO_COLS:
                    temp_dates = parse_date_dayfirst(df[col])
                    valid_ratio = temp_dates.count() / len(temp_dates)
                    if valid_ratio > max_valid_ratio and valid_ratio > 0.5:
                        max_valid_ratio = valid_ratio
                        best_date_col = col
        
        if best_date_col:
            df['Date'] = parse_date_dayfirst(df[best_date_col])
        else:
            raise ValueError("Could not detect a valid 'Date' column in the input CSV. Please ensure a date column exists and is recognizable.")

    # --- Ensure NO_COLS exist ---
    for col in NO_COLS:
        if col not in df.columns:
            df[col] = '' # Add empty column if missing

    # --- Process 'Right' column ---
    if 'Right' in df.columns:
        df['Right'] = df['Right'].astype(str).str.replace(' ', '')
    else:
        # If 'Right' column is missing, build it from 'NoX' columns
        df['Right'] = df.apply(build_right_from_nos, axis=1)

    # Remove rows where 'Date' couldn't be parsed (NaT values)
    df = df.dropna(subset=['Date'])
    
    if df.empty:
        raise ValueError("No valid date rows found after processing. Check your 'Date' column data.")

    # Sort DataFrame by 'Date'
    df = df.sort_values(by='Date')

    # Return a DataFrame with necessary columns for further processing
    return df[['Date', 'Right'] + NO_COLS].copy()


def process_data_and_generate_output(input_df: pd.DataFrame, output_path: Path):
    """
    Orchestrates the main processing logic: iterates through the preprocessed
    DataFrame, counts digit frequencies, and generates output rows.
    """
    out_rows = []
    
    for _, row in input_df.iterrows():
        date_val = row['Date']
        right_str = row['Right']
        
        counts = count_digits_from_right(right_str)
        rows_for_date = make_rows_for_date(date_val, counts)
        out_rows.extend(rows_for_date)

    if not out_rows:
        raise ValueError("No output rows generated. This might indicate issues with the input data or processing logic.")

    out_df = pd.DataFrame(out_rows)

    # Ensure all digit columns ('0' through '9') exist
    for d in range(10):
        if str(d) not in out_df.columns:
            out_df[str(d)] = '' # Add as empty string if somehow missed

    # Reorder columns explicitly for final output
    final_cols = ['Date', 'CB', 'Freq', 'Count'] + [str(d) for d in range(10)]
    out_df = out_df[final_cols]

    # Format 'Date' column to the desired string format and handle NaNs for Freq/Count
    out_df['Date'] = out_df['Date'].dt.strftime("%d-%m-%Y")
    out_df['Freq'] = out_df['Freq'].replace(np.nan, '')
    out_df['Count'] = out_df['Count'].replace(np.nan, '')

    # Write the final DataFrame to the specified output CSV path
    out_df.to_csv(output_path, index=False)
    app_logger.info(f"Output CSV generated at: {output_path}")
    return output_path

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def home():
    return "The backend is running. Please use the frontend to interact with the application."

@app.route('/upload_and_process', methods=['POST'])
def upload_and_process_file():
    """
    API endpoint to receive a CSV file, process it, and return the filename
    of the generated output CSV.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not file.filename.lower().endswith('.csv'):
        return jsonify({"error": "Invalid file type. Only CSV files are allowed."}), 400

    input_filename = file.filename
    temp_input_path = None
    temp_output_path = None

    try:
        # Create a unique temporary file for the uploaded input CSV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", dir=TEMP_DIR) as tmp_input_file:
            file.save(tmp_input_file.name)
            temp_input_path = Path(tmp_input_file.name)
        app_logger.info(f"Input file saved temporarily: {temp_input_path}")
        
        # Load and preprocess the CSV using the adapted function
        df = load_csv_for_flask(temp_input_path)

        # Generate a unique output filename for the processed CSV
        # Includes timestamp to ensure uniqueness and prevents overwrites
        output_basename = f"CB_Output_{os.path.splitext(input_filename)[0]}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        temp_output_path = Path(TEMP_DIR) / output_basename

        # Process data and generate the output CSV
        process_data_and_generate_output(df, temp_output_path)

        # Return the filename of the generated output for the frontend to download
        return jsonify({
            "message": "File processed successfully!",
            "output_filename": temp_output_path.name
        }), 200

    except (FileNotFoundError, ValueError) as e:
        app_logger.warning(f"Processing error: {e}")
        return jsonify({"error": str(e)}), 400 # 400 Bad Request for client-side input issues
    except Exception as e:
        # Catch any other unexpected errors and log them
        app_logger.error(f"An unexpected error occurred during processing: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected server error occurred: {e}"}), 500
    finally:
        # Ensure the temporary input file is removed after processing
        if temp_input_path and temp_input_path.exists():
            try:
                os.remove(temp_input_path)
                app_logger.info(f"Cleaned up input temporary file: {temp_input_path}")
            except Exception as e:
                app_logger.error(f"Error removing input temporary file {temp_input_path}: {e}")

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """
    API endpoint to serve the generated output CSV file.
    """
    file_path = Path(TEMP_DIR) / filename
    if file_path.exists():
        try:
            # Serve the file as an attachment with the original filename
            return send_file(file_path, as_attachment=True, download_name=filename, mimetype='text/csv')
        except Exception as e:
            app_logger.error(f"Error serving file {filename}: {e}", exc_info=True)
            return jsonify({"error": "Could not serve file due to a server error."}), 500
    else:
        app_logger.warning(f"Download request for non-existent file: {filename}")
        return jsonify({"error": "File not found. It may have expired or been removed."}), 404

if __name__ == '__main__':
    # For local development, run with debug=True
    app.run(debug=True, host='0.0.0.0')