import os
import io
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Constants ---
CSV_DIR = Path("CSV")
# Ensure the CSV directory exists
CSV_DIR.mkdir(exist_ok=True)

NO_COLS = [f"No{i}" for i in range(1, 28)]
ALLOWED_EXTENSIONS = {'csv'}

# --- Helper Functions (as per algorithm description) ---

def parse_date_dayfirst(s: pd.Series) -> pd.Series:
    """
    Phân tích cú pháp ngày tháng một cách an toàn và linh hoạt từ một Series của Pandas.
    Hỗ trợ định dạng số serial Excel, các định dạng ngày-tháng-năm phổ biến và fallback.
    """
    dt = pd.Series(pd.NaT, index=s.index, dtype='datetime64[ns]')
    
    # 1. Chuyển đổi Series thành số (có thể có NaNs)
    numeric_vals = pd.to_numeric(s, errors='coerce')
    
    # 2. Xử lý định dạng số serial của Excel (giữa 1 và 80000)
    # Excel's 1900-based date system (Windows) starts at 1899-12-30.
    # So, 1 maps to 1899-12-31, 2 maps to 1900-01-01, etc.
    # A value of 2 represents Jan 1, 1900.
    # Adjust origin to '1899-12-30' for compatibility with Excel's serial dates.
    mask_excel = (numeric_vals >= 1) & (numeric_vals <= 80000)
    if mask_excel.any():
        dt.loc[mask_excel] = pd.to_datetime(numeric_vals[mask_excel], unit='d', origin='1899-12-30')
    
    # 3. Xử lý các định dạng ngày-tháng-năm phổ biến
    mask_rest = dt.isna() # Remaining NaT values
    
    # Convert remaining non-NaN values to string for pattern matching
    str_vals_rest = s[mask_rest].astype(str).str.strip().replace('', pd.NA)
    
    date_formats = [
        "%d-%m-%Y", "%d/%m/%Y", "%d %m %Y", # Day-Month-Year variations
        "%Y-%m-%d", "%Y/%m/%d", "%Y %m %d", # Year-Month-Day variations
        "%m-%d-%Y", "%m/%d-%Y", "%m %d %Y"  # Month-Day-Year variations
    ]
    
    for fmt in date_formats:
        if str_vals_rest[mask_rest].empty:
            break
        try:
            parsed_chunk = pd.to_datetime(str_vals_rest[mask_rest], format=fmt, errors='coerce')
            dt.loc[mask_rest] = dt.loc[mask_rest].fillna(parsed_chunk)
            mask_rest = dt.isna() # Update mask for remaining NaT values
        except ValueError:
            # Format might not apply, continue to next
            pass
            
    # 4. Xử lý fallback: Nếu vẫn còn giá trị NaT, cố gắng phân tích cú pháp bằng pd.to_datetime với dayfirst=True
    if mask_rest.any():
        fallback_parsed = pd.to_datetime(s[mask_rest], dayfirst=True, errors='coerce')
        dt.loc[mask_rest] = dt.loc[mask_rest].fillna(fallback_parsed)

    return dt.dt.normalize() # Chuẩn hóa tất cả các ngày về đầu ngày

def build_right_from_nos(row: pd.Series) -> str:
    """
    Xây dựng chuỗi "Right" bằng cách trích xuất tất cả các chữ số từ các cột NO_COLS.
    Mỗi giá trị số được xử lý như một số gồm 2 chữ số (thêm 0 nếu là 1 chữ số).
    Ví dụ: '15' -> '1', '5'; '7' -> '0', '7'.
    Nếu giá trị có nhiều hơn 2 chữ số (ví dụ '123'), chỉ lấy 2 chữ số cuối (ví dụ '2', '3').
    """
    all_extracted_digits = []
    for col in NO_COLS:
        val_str = str(row.get(col, '')).strip()
        if val_str:
            # Try to find a numeric sequence in the string
            match = re.search(r'\d+', val_str)
            if match:
                num_str_found = match.group(0)
                
                # If the number found is longer than 2 digits, take the last two.
                # Otherwise, use the number as is.
                if len(num_str_found) > 2:
                    num_to_process = num_str_found[-2:]
                else:
                    num_to_process = num_str_found
                
                try:
                    # Convert to int, then format as a two-digit string (e.g., 7 -> "07", 15 -> "15")
                    formatted_num_str = f"{int(num_to_process):02d}"
                    # Append individual digits
                    all_extracted_digits.extend(list(formatted_num_str))
                except ValueError:
                    # If int conversion fails for num_to_process (shouldn't happen if regex is fine), skip
                    pass
    return "".join(all_extracted_digits)


def count_digits_from_right(right_str: str) -> dict:
    """
    Đếm tần suất xuất hiện của từng chữ số (0-9) trong một chuỗi.
    """
    counts = Counter(int(digit) for digit in right_str if digit.isdigit())
    # Ensure all digits 0-9 are present, even if their count is 0
    return {i: counts.get(i, 0) for i in range(10)}

def make_rows_for_date(date_val: pd.Timestamp, counts: dict) -> list:
    """
    Sinh 10 hàng dữ liệu báo cáo cho một ngày cụ thể dựa trên tần suất chữ số.
    Luôn tạo ra 10 dòng dữ liệu mới cho mỗi ngày (5 Min, 5 Max), 
    với các dòng trống được điền Freq=0, Count=0 và các chữ số=0.
    """
    output_rows = []
    
    # Convert counts to a Series for easier manipulation of unique frequencies
    freq_series = pd.Series(counts)
    
    # Get unique frequencies and sort them
    unique_freqs = sorted(freq_series.unique()) # Sorted unique frequencies found in 'counts'
    
    # Prepare labels for Min and Max rows
    min_labels = [f"Min{i}" for i in range(1, 6)]
    max_labels = [f"Max{i}" for i in range(1, 6)]
    
    # Helper to create a regular row based on a specific frequency
    def create_actual_row(label: str, freq: int) -> dict:
        row_data = {
            "Date": date_val,
            "CB": label,
            "Freq": freq,
            "Count": 0
        }
        for d in range(10):
            row_data[str(d)] = 0
            
        digits_at_freq = [d for d, f in counts.items() if f == freq]
        row_data["Count"] = len(digits_at_freq)
        
        for d in digits_at_freq:
            row_data[str(d)] = 1
        return row_data

    # Helper to create a placeholder row when fewer than 5 unique frequencies exist
    def create_placeholder_row(label: str) -> dict:
        row_data = {
            "Date": date_val,
            "CB": label,
            "Freq": 0, # Placeholder frequency
            "Count": 0  # Placeholder count
        }
        for d in range(10):
            row_data[str(d)] = 0 # All digits are 0
        return row_data

    # Create 5 MinX rows (lowest frequencies)
    for i in range(5):
        if i < len(unique_freqs):
            output_rows.append(create_actual_row(min_labels[i], unique_freqs[i]))
        else:
            output_rows.append(create_placeholder_row(min_labels[i]))
    
    # Create 5 MaxX rows (highest frequencies)
    # Iterate in reverse for highest frequencies
    for i in range(5):
        if i < len(unique_freqs):
            # Calculate the index for the 'i'-th highest frequency
            # For i=0, it's the highest: unique_freqs[len(unique_freqs)-1]
            # For i=1, it's the second highest: unique_freqs[len(unique_freqs)-2]
            freq_idx = len(unique_freqs) - 1 - i
            output_rows.append(create_actual_row(max_labels[i], unique_freqs[freq_idx]))
        else:
            output_rows.append(create_placeholder_row(max_labels[i]))
        
    return output_rows

def load_csv(path: Path) -> pd.DataFrame:
    """
    Tải, làm sạch và chuẩn bị DataFrame từ file CSV đầu vào.
    """
    if not path.exists():
        raise SystemExit(f"Error: Input file not found at {path}")

    # Read CSV, keeping all columns as strings initially to prevent type issues
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    # Clean column names: remove BOM character and strip whitespace
    df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]

    # --- Pre-processing for "STT" header row ---
    # If a column named "STT" exists and it contains its own header name
    if 'STT' in df.columns:
        # Filter out rows where 'STT' column contains "STT" or "No." (case-insensitive)
        df = df[~df['STT'].astype(str).str.contains(r'STT|No\.', case=False, na=False)]
        # Also remove rows where 'STT' is empty or only whitespace after filtering
        df = df[df['STT'].astype(str).str.strip() != '']

    # --- Date Column Processing ---
    if "Date" not in df.columns:
        # Attempt to find the best date column
        best_date_col = None
        highest_parse_rate = -1
        
        # Candidate columns are those not in NO_COLS and not 'STT' or 'Right'
        candidate_cols = [
            c for c in df.columns 
            if c not in NO_COLS and c != 'STT' and c != 'Right'
        ]
        
        for col in candidate_cols:
            temp_dates = parse_date_dayfirst(df[col])
            parsed_count = temp_dates.count() # Count non-NaT values
            
            if len(temp_dates) > 0:
                parse_rate = parsed_count / len(temp_dates)
            else:
                parse_rate = 0 # Empty series, 0% parse rate
                
            if parse_rate > highest_parse_rate:
                highest_parse_rate = parse_rate
                best_date_col = col
        
        if best_date_col and highest_parse_rate >= 0.5: # Require at least 50% success
            df["Date"] = parse_date_dayfirst(df[best_date_col])
        else:
            raise SystemExit("Error: Could not find a suitable 'Date' column in the CSV (no existing 'Date' column and no other column with >=50% date parse success).")
    else:
        df["Date"] = parse_date_dayfirst(df["Date"])

    # Ensure all NO_COLS exist, add if missing with empty string values
    for col in NO_COLS:
        if col not in df.columns:
            df[col] = ''

    # --- "Right" Column Processing ---
    # Always rebuild "Right" column based on the new logic
    df["Right"] = df.apply(build_right_from_nos, axis=1)

    # Final cleanup: Remove rows where "Date" could not be parsed
    df.dropna(subset=["Date"], inplace=True)
    if df.empty:
        raise SystemExit("Error: No valid date entries found after parsing. Check 'Date' column format.")

    # Sort by Date and reset index
    df.sort_values(by="Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Return only the relevant columns
    return df[["Date", "Right"] + NO_COLS]

# --- Flask Routes ---

@app.route('/process-csv', methods=['POST'])
def process_csv_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save the uploaded file to a temporary location
            input_csv_path = CSV_DIR / "lucky_temp.csv"
            file.save(input_csv_path)

            df = load_csv(input_csv_path)

            out_rows = []
            for _, row in df.iterrows():
                date_val = row["Date"]
                right_str = row["Right"]
                counts = count_digits_from_right(right_str)
                out_rows.extend(make_rows_for_date(date_val, counts))

            # Define output columns
            output_cols = ["Date", "CB", "Freq", "Count"] + [str(d) for d in range(10)]
            
            # Create DataFrame from output rows
            output_df = pd.DataFrame(out_rows, columns=output_cols)
            
            # Format Date column for output CSV
            output_df["Date"] = output_df["Date"].dt.strftime("%d-%m-%Y")

            # Use an in-memory buffer to send the file
            output_buffer = io.StringIO()
            output_df.to_csv(output_buffer, index=False, encoding='utf-8')
            output_buffer.seek(0) # Rewind to the beginning of the buffer

            # Clean up temporary input file
            os.remove(input_csv_path)
            
            return send_file(
                io.BytesIO(output_buffer.getvalue().encode('utf-8')), # Send as bytes
                mimetype='text/csv',
                as_attachment=True,
                download_name='CB.csv'
            )

        except SystemExit as e:
            # Catch controlled exits from load_csv or other functions
            if 'input_csv_path' in locals() and os.path.exists(input_csv_path):
                os.remove(input_csv_path) # Clean up temp file even on error
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            # Catch any other unexpected errors
            if 'input_csv_path' in locals() and os.path.exists(input_csv_path):
                os.remove(input_csv_path) # Clean up temp file even on error
            app.logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            return jsonify({'error': f"An unexpected error occurred: {e}"}), 500
    else:
        return jsonify({'error': 'Invalid file type. Only CSV files are allowed.'}), 400

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return jsonify({"message": "Flask backend is running. Use /process-csv endpoint for file upload."})

if __name__ == '__main__':
    # For local development, remove debug=True for production
    app.run(debug=True, port=5000)