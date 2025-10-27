import io
import os
import re
import sys
from datetime import datetime
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
# Allow CORS for all origins, adjust in production for specific frontend domain
CORS(app)

# --- Global Constants ---
NO_COLS = [f'No{i}' for i in range(1, 28)]

# --- Helper Functions (from original request, adapted) ---

def parse_date_dayfirst(s: pd.Series) -> pd.Series:
    """
    Parses date strings from a pandas Series into datetime64[ns],
    prioritizing Excel serial numbers and day-first formats.
    """
    s = s.copy() # Operate on a copy to avoid SettingWithCopyWarning

    # Handle Excel serial numbers: integer values in [1, 80000]
    # Origin 1899-12-30 means value 1 is 1899-12-31, 2 is 1900-01-01 etc.
    excel_serial_mask = s.apply(lambda x: isinstance(x, (int, float)) and 1 <= x <= 80000)
    if excel_serial_mask.any():
        excel_dates = pd.to_datetime(s[excel_serial_mask], unit='D', origin='1899-12-30')
        s.loc[excel_serial_mask] = excel_dates

    # Convert to datetime, trying common day-first formats
    known_formats = [
        "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
        "%d-%m-%y", "%d/%m/%y", "%Y-%m-%d",
    ]
    
    # Initialize a Series for parsed dates, filling with NaT
    parsed_series = pd.Series(pd.NaT, index=s.index, dtype='datetime64[ns]')

    for fmt in known_formats:
        # Only attempt to parse values that haven't been successfully parsed yet and are not NaN
        mask_to_parse = s.index[s.notna() & parsed_series.isna()]
        if not mask_to_parse.empty:
            try:
                temp_parsed = pd.to_datetime(s[mask_to_parse], format=fmt, errors='coerce')
                # Update parsed_series only where temp_parsed is not NaT
                parsed_series.loc[mask_to_parse] = temp_parsed.loc[mask_to_parse].fillna(parsed_series.loc[mask_to_parse])
            except ValueError: # Catch if format string itself is invalid for some data (less common with errors='coerce')
                pass

    # Fallback for any remaining unparsed values
    remaining_mask = s.index[s.notna() & parsed_series.isna()]
    if not remaining_mask.empty:
        fallback_parsed = pd.to_datetime(s[remaining_mask], errors='coerce', dayfirst=True)
        parsed_series.loc[remaining_mask] = fallback_parsed.loc[remaining_mask].fillna(parsed_series.loc[remaining_mask])

    return parsed_series.dt.normalize() # Remove time component

def build_right_from_nos(row: pd.Series) -> str:
    """
    Extracts the last digit from values in 'No1' to 'No27' columns for a given row.
    """
    extracted_digits = []
    for col in NO_COLS:
        value = str(row.get(col, '')).strip()
        if not value:
            continue
        try:
            # Try converting to integer and taking modulo 10
            num = int(value)
            extracted_digits.append(str(num % 10))
        except ValueError:
            # If not a pure integer, extract all digits and take the last one
            digits = re.findall(r'\d', value)
            if digits:
                extracted_digits.append(digits[-1])
    return "".join(extracted_digits)

def count_digits_from_right(right_str: str) -> dict:
    """
    Counts the frequency of each digit (0-9) in the input string.
    """
    counts = {str(i): 0 for i in range(10)}
    for char in right_str:
        if '0' <= char <= '9':
            counts[char] += 1
    return counts

def make_rows_for_date(date_val: pd.Timestamp, counts: dict) -> list:
    """
    Generates 10 output rows (5 MinX, 5 MaxX) for a given date and digit counts.
    """
    rows = []
    
    # Map frequency to a list of digits that have that frequency
    freq_to_digits = {}
    for digit, freq in counts.items():
        if freq not in freq_to_digits:
            freq_to_digits[freq] = []
        freq_to_digits[freq].append(digit)
    
    # Get unique frequencies sorted
    unique_freqs = sorted(freq_to_digits.keys())

    # Helper function to create a single output row dictionary
    def create_output_row(date, cb_type, freq, count, digits_in_group):
        row_dict = {
            "Date": date,
            "CB": cb_type,
            "Freq": freq,
            "Count": count,
        }
        # Mark digits belonging to this frequency group
        for d_int in range(10):
            d_str = str(d_int)
            row_dict[d_str] = d_str if d_str in digits_in_group else ''
        return row_dict

    # Create 5 MinX rows (lowest frequencies, ascending)
    for i in range(5):
        if i < len(unique_freqs):
            freq = unique_freqs[i]
            digits_in_group = sorted(freq_to_digits[freq]) # Sort digits for consistent output
            rows.append(create_output_row(date_val, f"Min{i+1}", freq, len(digits_in_group), digits_in_group))
        else:
            # If fewer than 5 unique frequencies, fill remaining MinX rows as empty
            rows.append(create_output_row(date_val, f"Min{i+1}", '', '', []))

    # Create 5 MaxX rows (highest frequencies, descending)
    unique_freqs_desc = sorted(freq_to_digits.keys(), reverse=True)
    for i in range(5):
        if i < len(unique_freqs_desc):
            freq = unique_freqs_desc[i]
            digits_in_group = sorted(freq_to_digits[freq]) # Sort digits for consistent output
            rows.append(create_output_row(date_val, f"Max{i+1}", freq, len(digits_in_group), digits_in_group))
        else:
            # If fewer than 5 unique frequencies, fill remaining MaxX rows as empty
            rows.append(create_output_row(date_val, f"Max{i+1}", '', '', []))

    return rows

# --- Main Processing Logic for Web App ---

def process_lucky_csv(input_stream: io.BytesIO) -> pd.DataFrame:
    """
    Loads raw CSV data from a stream, performs standardization and analysis,
    and returns a DataFrame with the 'CB' summary table.
    Raises ValueError if critical data (like Date column) cannot be found/parsed
    or if 'Right' column ends up empty.
    """
    
    # 1. Read CSV from stream
    try:
        # Use StringIO to read text-based CSV from BytesIO
        df = pd.read_csv(io.StringIO(input_stream.read().decode('utf-8')), dtype=str, keep_default_na=False)
    except Exception as e:
        raise ValueError(f"Error reading CSV file. Please ensure it's a valid CSV. Details: {e}")

    # 2. Normalize column names (remove BOM, strip whitespace)
    df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]

    # 3. Remove supplementary header rows (if 'STT' column exists and has header-like values)
    if 'STT' in df.columns:
        df = df[~df['STT'].isin(['STT', 'No.', 'No'])]

    # 4. Handle "Date" column: auto-detect if not present, then parse
    date_col_name = None
    if 'Date' in df.columns:
        date_col_name = 'Date'
    else:
        # Auto-detect date column by successful parsing ratio
        best_date_candidate = None
        max_parsed_ratio = -1.0
        for col in df.columns:
            # Only consider non-empty string values for parsing
            temp_series = df[col].replace('', np.nan).dropna()
            if temp_series.empty:
                continue
            
            parsed_dates = parse_date_dayfirst(temp_series)
            successfully_parsed_count = parsed_dates.count() # Number of non-NaT values
            
            if successfully_parsed_count > 0:
                parse_ratio = successfully_parsed_count / len(temp_series)
                if parse_ratio > 0.5 and parse_ratio > max_parsed_ratio: # Must parse >50% successfully
                    max_parsed_ratio = parse_ratio
                    best_date_candidate = col
        
        if best_date_candidate:
            date_col_name = best_date_candidate
            df.rename(columns={best_date_candidate: 'Date'}, inplace=True)
        else:
            raise ValueError("Could not find a suitable 'Date' column in the input CSV. No column named 'Date' found, and no other column had >50% successful date parsing.")

    df['Date'] = parse_date_dayfirst(df[date_col_name])

    # 5. Ensure `No1` to `No27` columns exist, adding empty ones if missing
    for col in NO_COLS:
        if col not in df.columns:
            df[col] = ''
    
    # 6. Handle "Right" column: standardize or build from `NoX` columns
    if 'Right' in df.columns:
        df['Right'] = df['Right'].astype(str).str.strip()
    else:
        df['Right'] = df.apply(build_right_from_nos, axis=1)

    # 7. Filter out rows with unparsed dates and sort by date
    df.dropna(subset=['Date'], inplace=True) # Remove rows where Date couldn't be parsed
    df.sort_values(by='Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Validate 'Right' column after processing
    if df['Right'].eq('').all():
        raise ValueError("The 'Right' column is empty for all rows after processing. Please check the input data or 'NoX' columns' values.")

    # 8. Generate all individual output rows for the 'CB' table
    out_rows = []
    for _, row in df.iterrows():
        counts = count_digits_from_right(row['Right'])
        out_rows.extend(make_rows_for_date(row['Date'], counts))

    # 9. Create final DataFrame for the output
    output_columns = ["Date", "CB", "Freq", "Count"] + [str(i) for i in range(10)]
    output_df = pd.DataFrame(out_rows, columns=output_columns)

    # 10. Format Date column in the output DataFrame to 'dd-mm-yyyy' string
    output_df['Date'] = output_df['Date'].dt.strftime('%d-%m-%Y')

    return output_df

# --- Flask API Endpoints ---

@app.route('/')
def home():
    """Simple health check / welcome endpoint."""
    return "Lucky CSV Processor API is running. Upload a CSV to /process_csv."

@app.route('/process_csv', methods=['POST'])
def process_csv_endpoint():
    """
    API endpoint to receive a CSV file, process it, and return the result as a CSV file.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request. Please upload a CSV file."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file. Please upload a CSV file."}), 400
    
    if file and file.filename.endswith('.csv'):
        try:
            # Read the uploaded file content into a BytesIO object for pandas
            file_stream = io.BytesIO(file.read())
            
            # Process the CSV data
            output_df = process_lucky_csv(file_stream)
            
            # Prepare the processed DataFrame as a CSV string in a StringIO object
            output_buffer = io.StringIO()
            output_df.to_csv(output_buffer, index=False, encoding='utf-8')
            output_buffer.seek(0) # Rewind to the beginning for reading
            
            # Send the generated CSV file as a download
            return send_file(
                io.BytesIO(output_buffer.getvalue().encode('utf-8')), # Encode to bytes for send_file
                mimetype='text/csv',
                as_attachment=True,
                download_name='CB.csv'
            )
        except ValueError as e:
            # Catch specific data processing errors and return a bad request status
            app.logger.error(f"Data processing error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            # Catch any other unexpected errors
            app.logger.error(f"An unexpected server error occurred during CSV processing: {e}", exc_info=True)
            return jsonify({"error": f"An internal server error occurred: {e}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload a CSV file with a '.csv' extension."}), 400

if __name__ == '__main__':
    # When running locally, Flask uses its built-in server.