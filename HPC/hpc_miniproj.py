import gradio as gr
import sqlite3
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from io import BytesIO
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import random
import string
import os
from concurrent.futures import ThreadPoolExecutor
import base64

# Database path
DB_PATH = "./banking_analytics.db"

def get_connection():
    """
    Returns a new SQLite connection with WAL mode enabled and 
    check_same_thread set to False (to allow sharing across threads if needed).
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

# Function to generate a banking dataset and create a SQLite database
def generate_dataset(num_rows=1000000, table_name="customer_transactions"):
    """Generate a banking dataset and store it in SQLite."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Drop table if it exists
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    
    # Create a new table for banking transactions
    cursor.execute(f"""
    CREATE TABLE {table_name} (
        transaction_id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        account_number TEXT,
        transaction_date TEXT,
        transaction_type TEXT,
        amount REAL,
        balance_after REAL,
        merchant_category TEXT,
        location TEXT,
        is_flagged INTEGER
    )
    """)
    
    # Generate random banking data
    batch_size = 10000
    num_batches = num_rows // batch_size
    transaction_types = ['DEPOSIT', 'WITHDRAWAL', 'TRANSFER', 'PAYMENT', 'REFUND']
    merchant_categories = ['RETAIL', 'GROCERY', 'DINING', 'TRAVEL', 'UTILITIES', 'ENTERTAINMENT', 'HEALTHCARE']
    locations = ['NEW YORK', 'CHICAGO', 'LOS ANGELES', 'MIAMI', 'DALLAS', 'SEATTLE', 'BOSTON', 'ONLINE']
    
    # Generate customer IDs (assuming 1-5% of total transactions)
    num_customers = max(100, int(num_rows * 0.03))
    customer_ids = list(range(10000, 10000 + num_customers))
    
    # Generate account numbers
    account_formats = [
        lambda: f"ACCT-{''.join(random.choice(string.digits) for _ in range(8))}",
        lambda: f"SA-{''.join(random.choice(string.digits) for _ in range(10))}",
        lambda: f"CA-{''.join(random.choice(string.digits) for _ in range(10))}"
    ]
    
    start_time = time.time()
    
    for batch in range(num_batches):
        data = []
        for i in range(batch_size):
            transaction_id = batch * batch_size + i
            customer_id = random.choice(customer_ids)
            account_number = random.choice(account_formats)()
            
            # Make transaction dates somewhat realistic (past 2 years)
            year = random.choice([2023, 2024])
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            transaction_date = f"{year}-{month:02d}-{day:02d}"
            
            transaction_type = random.choice(transaction_types)
            
            # Make amounts distribution more realistic
            if transaction_type == 'DEPOSIT':
                amount = random.uniform(100, 5000)
            elif transaction_type == 'WITHDRAWAL':
                amount = random.uniform(20, 1000)
            elif transaction_type == 'TRANSFER':
                amount = random.uniform(50, 3000)
            elif transaction_type == 'PAYMENT':
                amount = random.uniform(10, 2000)
            else:  # REFUND
                amount = random.uniform(5, 500)
                
            balance_after = random.uniform(500, 50000)
            merchant_category = random.choice(merchant_categories)
            location = random.choice(locations)
            is_flagged = 1 if random.random() < 0.02 else 0  # 2% of transactions are flagged
            
            data.append((transaction_id, customer_id, account_number, transaction_date, 
                        transaction_type, amount, balance_after, merchant_category, location, is_flagged))
            
        cursor.executemany(
            f"INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            data
        )
        conn.commit()
        
        if (batch + 1) % 10 == 0 or batch == num_batches - 1:
            progress = (batch + 1) / num_batches * 100
            elapsed = time.time() - start_time
            print(f"Progress: {progress:.2f}% - Elapsed time: {elapsed:.2f}s")
    
    # Create customer table as well
    cursor.execute("DROP TABLE IF EXISTS customers")
    cursor.execute("""
    CREATE TABLE customers (
        customer_id INTEGER PRIMARY KEY,
        customer_name TEXT,
        email TEXT,
        phone TEXT,
        address TEXT,
        credit_score INTEGER,
        join_date TEXT,
        customer_segment TEXT
    )
    """)
    
    # Add customer data
    segments = ['STANDARD', 'PREMIUM', 'BUSINESS', 'STUDENT', 'SENIOR']
    names_first = ["John", "Maria", "David", "Sarah", "Michael", "Jennifer", "Robert", "Linda", "William", "Elizabeth"]
    names_last = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor"]
    
    customer_data = []
    for cid in customer_ids:
        name = f"{random.choice(names_first)} {random.choice(names_last)}"
        email = f"{name.lower().replace(' ', '.')}@example.com"
        phone = f"({random.randint(100, 999)})-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
        address = f"{random.randint(1, 9999)} {random.choice(['Main', 'Oak', 'Pine', 'Maple', 'Cedar'])} St"
        credit_score = random.randint(300, 850)
        join_year = random.randint(2010, 2024)
        join_month = random.randint(1, 12)
        join_day = random.randint(1, 28)
        join_date = f"{join_year}-{join_month:02d}-{join_day:02d}"
        segment = random.choice(segments)
        
        customer_data.append((cid, name, email, phone, address, credit_score, join_date, segment))
    
    cursor.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        customer_data
    )
    
    # Create indexes for better performance
    cursor.execute(f"CREATE INDEX idx_{table_name}_customer_id ON {table_name}(customer_id)")
    cursor.execute(f"CREATE INDEX idx_{table_name}_transaction_date ON {table_name}(transaction_date)")
    cursor.execute(f"CREATE INDEX idx_{table_name}_transaction_type ON {table_name}(transaction_type)")
    cursor.execute(f"CREATE INDEX idx_{table_name}_amount ON {table_name}(amount)")
    cursor.execute(f"CREATE INDEX idx_{table_name}_merchant_category ON {table_name}(merchant_category)")
    conn.commit()
    
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    actual_rows = cursor.fetchone()[0]
    conn.close()
    
    return f"Generated {actual_rows:,} banking transactions and {len(customer_ids):,} customer records for analysis"

# Function to execute a SQL query serially
def execute_serial_query(query, params=None):
    """Execute a SQL query in serial mode and measure performance."""
    conn = get_connection()
    start_time = time.time()
    if params:
        result = pd.read_sql_query(query, conn, params=params)
    else:
        result = pd.read_sql_query(query, conn)
    
    execution_time = time.time() - start_time
    conn.close()
    return result, execution_time

# Extract table name from a SQL query (simplified)
def extract_table_name(query):
    """Extract table name from a SQL query (simplified)."""
    query_lower = query.lower()
    if 'from' in query_lower:
        parts = query_lower.split('from')[1].strip().split()
        return parts[0].strip().rstrip(';')
    return "customer_transactions"  # Default table name

# CUDA kernel for numeric aggregation operations
def get_cuda_aggregation_kernel():
    """
    Return a CUDA kernel for performing common aggregation operations used in banking analytics.
    This includes sum, average, min, max calculations on transaction amounts or other numeric fields.
    """
    return SourceModule("""
    __global__ void aggregate_values(float *input, float *output, int length) {
        // Define shared memory for block reduction
        __shared__ float shared_sum[256];
        __shared__ float shared_min[256];
        __shared__ float shared_max[256];
        
        int tid = threadIdx.x;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        // Initialize values
        shared_sum[tid] = 0.0f;
        shared_min[tid] = 1e10f;  // A large value for finding minimum
        shared_max[tid] = -1e10f; // A small value for finding maximum
        
        // Load data into shared memory
        if (i < length) {
            shared_sum[tid] = input[i];
            shared_min[tid] = input[i];
            shared_max[tid] = input[i];
        }
        __syncthreads();
        
        // Parallel reduction for sum, min, max
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sum[tid] += shared_sum[tid + s];
                shared_min[tid] = fminf(shared_min[tid], shared_min[tid + s]);
                shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
            }
            __syncthreads();
        }
        
        // Write results to output array (for each block)
        if (tid == 0) {
            output[blockIdx.x * 3] = shared_sum[0];           // Sum
            output[blockIdx.x * 3 + 1] = shared_min[0];       // Min
            output[blockIdx.x * 3 + 2] = shared_max[0];       // Max
        }
    }
    
    __global__ void filter_transactions(float *amounts, int *flags, int *results, float threshold, int length) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (i < length) {
            // Filter high-value transactions (1) or suspicious transactions (flags[i] == 1)
            results[i] = (amounts[i] > threshold || flags[i] == 1) ? 1 : 0;
        }
    }
    """)

# Function to process transaction data with CUDA
def process_transactions_with_cuda(transactions_df, query_type):
    """
    Use CUDA to accelerate processing of transaction data based on query type.
    """
    try:
        # Initialize CUDA kernel
        cuda_module = get_cuda_aggregation_kernel()
        
        # Extract numeric data from dataframe if available
        if 'amount' in transactions_df.columns:
            # Convert amount column to float32 array
            amounts = transactions_df['amount'].astype(np.float32).values
            
            # Prepare CUDA arrays
            input_gpu = cuda.mem_alloc(amounts.nbytes)
            cuda.memcpy_htod(input_gpu, amounts)
            
            # Determine number of threads and blocks
            block_size = 256
            grid_size = (len(amounts) + block_size - 1) // block_size
            
            # Allocate memory for output (3 values per block: sum, min, max)
            output = np.zeros(grid_size * 3, dtype=np.float32)
            output_gpu = cuda.mem_alloc(output.nbytes)
            
            # Run kernel
            aggregate_func = cuda_module.get_function("aggregate_values")
            aggregate_func(input_gpu, output_gpu, np.int32(len(amounts)), 
                          block=(block_size, 1, 1), grid=(grid_size, 1))
            
            # Get results back
            cuda.memcpy_dtoh(output, output_gpu)
            
            # Process final results
            block_sums = output[0::3]
            total_sum = np.sum(block_sums)
            
            block_mins = output[1::3]
            global_min = np.min(block_mins)
            
            block_maxs = output[2::3]
            global_max = np.max(block_maxs)
            
            # Free GPU memory
            input_gpu.free()
            output_gpu.free()
            
            # Return processed results
            return {
                'sum': total_sum,
                'avg': total_sum / len(amounts) if len(amounts) > 0 else 0,
                'min': global_min,
                'max': global_max
            }
            
        # Handle case where amount is not available
        return None
    
    except Exception as e:
        print(f"CUDA processing error: {str(e)}")
        return None

# Function to analyze query and determine if it can be accelerated with CUDA
def can_use_cuda_acceleration(query):
    """
    Analyze the SQL query to determine if it can be effectively accelerated with CUDA.
    Returns a tuple of (can_accelerate, acceleration_type)
    """
    query_lower = query.lower()
    
    # Check for aggregate functions that can be accelerated
    has_aggregates = any(agg in query_lower for agg in ['sum(', 'avg(', 'min(', 'max(', 'count('])
    
    # Check for operations that benefit from parallel processing
    has_groupby = 'group by' in query_lower
    has_orderby = 'order by' in query_lower
    has_where = 'where' in query_lower
    has_numeric_filter = any(term in query_lower for term in ['amount', 'balance_', 'credit_score'])
    
    # Determine acceleration type
    if has_aggregates and has_numeric_filter:
        return True, "aggregation_with_filter"
    elif has_aggregates:
        return True, "aggregation"
    elif has_groupby and has_numeric_filter:
        return True, "group_filter"
    elif has_where and has_numeric_filter:
        return True, "filtering"
    elif 'join' in query_lower and (has_where or has_orderby):
        return True, "join_optimization"
    else:
        return False, "none"

# Function to execute a SQL query with CUDA optimization
def execute_cuda_query(query, params=None):
    """Execute a SQL query with CUDA optimization where applicable."""
    conn = get_connection()
    start_time = time.time()
    
    # First determine if we can accelerate this query with CUDA
    can_accelerate, accel_type = can_use_cuda_acceleration(query)
    
    if not can_accelerate:
        # If we can't accelerate, just run it normally
        if params:
            result = pd.read_sql_query(query, conn, params=params)
        else:
            result = pd.read_sql_query(query, conn)
        parallel_time = time.time() - start_time
        conn.close()
        return result, parallel_time
    
    # Extract necessary data from the database
    table_name = extract_table_name(query)
    
    # For aggregation queries, we'll optimize by pushing down aggregations to GPU
    if accel_type == "aggregation" or accel_type == "aggregation_with_filter":
        # Extract the base data first
        if 'where' in query.lower():
            # Extract the WHERE clause to apply it before GPU processing
            where_clause = query.lower().split('where')[1].split('group by')[0].split('order by')[0]
            base_query = f"SELECT * FROM {table_name} WHERE {where_clause}"
        else:
            base_query = f"SELECT * FROM {table_name}"
        
        # Get raw data for GPU processing
        base_data = pd.read_sql_query(base_query, conn)
        
        # Process with CUDA
        gpu_results = process_transactions_with_cuda(base_data, accel_type)
        
        # If we have GPU results and we're doing a simple aggregation query
        if gpu_results is not None and accel_type == "aggregation":
            # Check if we're computing sum, avg, min, or max on amounts
            if 'sum(' in query.lower() and 'amount' in query.lower():
                # Create simplified result dataframe
                result = pd.DataFrame({'SUM(amount)': [gpu_results['sum']]})
            elif 'avg(' in query.lower() and 'amount' in query.lower():
                result = pd.DataFrame({'AVG(amount)': [gpu_results['avg']]})
            elif 'min(' in query.lower() and 'amount' in query.lower():
                result = pd.DataFrame({'MIN(amount)': [gpu_results['min']]})
            elif 'max(' in query.lower() and 'amount' in query.lower():
                result = pd.DataFrame({'MAX(amount)': [gpu_results['max']]})
            else:
                # Fallback to CPU for other aggregate queries
                if params:
                    result = pd.read_sql_query(query, conn, params=params)
                else:
                    result = pd.read_sql_query(query, conn)
        else:
            # For other queries or when GPU processing fails, fall back to CPU
            if params:
                result = pd.read_sql_query(query, conn, params=params)
            else:
                result = pd.read_sql_query(query, conn)
    else:
        # For non-aggregation queries or unrecognized acceleration types
        if params:
            result = pd.read_sql_query(query, conn, params=params)
        else:
            result = pd.read_sql_query(query, conn)
    
    parallel_time = time.time() - start_time
    conn.close()
    return result, parallel_time

# Function to suggest SQL queries based on the banking dataset
def suggest_queries(table_name="customer_transactions"):
    """Generate banking-specific SQL query suggestions."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        if not columns:
            conn.close()
            return ["-- No banking data found. Generate a dataset first."]
        
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
        sample = cursor.fetchone()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        
        # Banking-specific query suggestions
        suggestions = []
        suggestions.append(f"-- Basic transaction view\nSELECT * FROM {table_name} LIMIT 10")
        suggestions.append(f"-- Transaction count by type\nSELECT transaction_type, COUNT(*) as count FROM {table_name} GROUP BY transaction_type")
        suggestions.append(f"-- Top 10 largest transactions\nSELECT * FROM {table_name} ORDER BY amount DESC LIMIT 10")
        suggestions.append(f"-- Recent flagged transactions\nSELECT * FROM {table_name} WHERE is_flagged = 1 ORDER BY transaction_date DESC LIMIT 20")
        suggestions.append(f"-- Average transaction amount by merchant category\nSELECT merchant_category, AVG(amount) as avg_amount FROM {table_name} GROUP BY merchant_category ORDER BY avg_amount DESC")
        
        # Add GPU-optimized query suggestions (with * to indicate faster with GPU)
        suggestions.append(f"-- *GPU Optimized* Calculate total transaction volume\nSELECT SUM(amount) as total_volume FROM {table_name}")
        suggestions.append(f"-- *GPU Optimized* Find high-value transactions\nSELECT * FROM {table_name} WHERE amount > 1000 ORDER BY amount DESC LIMIT 50")
        suggestions.append(f"-- *GPU Optimized* Calculate statistics across all transactions\nSELECT COUNT(*) as count, AVG(amount) as avg_amount, MIN(amount) as min_amount, MAX(amount) as max_amount FROM {table_name}")
        
        # Check if customers table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='customers'")
        if cursor.fetchone():
            suggestions.append(f"-- Customer transactions with customer details\n" +
                             f"SELECT c.customer_name, c.customer_segment, t.transaction_date, t.transaction_type, t.amount\n" +
                             f"FROM {table_name} t\n" +
                             f"JOIN customers c ON t.customer_id = c.customer_id\n" +
                             f"ORDER BY t.transaction_date DESC\n" +
                             f"LIMIT 20")
            
            suggestions.append(f"-- Transaction summary by customer segment\n" +
                            f"SELECT c.customer_segment, COUNT(*) as num_transactions, SUM(t.amount) as total_volume, AVG(t.amount) as avg_transaction\n" +
                            f"FROM {table_name} t\n" +
                            f"JOIN customers c ON t.customer_id = c.customer_id\n" +
                            f"GROUP BY c.customer_segment\n" +
                            f"ORDER BY total_volume DESC")
                             
            suggestions.append(f"-- Premium customers with flagged transactions\n" +
                            f"SELECT c.customer_id, c.customer_name, COUNT(*) as flagged_count\n" +
                            f"FROM customers c\n" +
                            f"JOIN {table_name} t ON c.customer_id = t.customer_id\n" +
                            f"WHERE c.customer_segment = 'PREMIUM' AND t.is_flagged = 1\n" +
                            f"GROUP BY c.customer_id, c.customer_name\n" +
                            f"HAVING flagged_count > 0\n" +
                            f"ORDER BY flagged_count DESC")
                            
        suggestions.append(f"-- Monthly transaction volume\n" +
                         f"SELECT strftime('%Y-%m', transaction_date) as month, \n" +
                         f"       SUM(amount) as total_volume,\n" +
                         f"       COUNT(*) as transaction_count\n" +
                         f"FROM {table_name}\n" +
                         f"GROUP BY month\n" +
                         f"ORDER BY month DESC")
                         
        suggestions.append(f"-- Transaction pattern by day of week\n" +
                         f"SELECT CASE cast(strftime('%w', transaction_date) AS INTEGER)\n" +
                         f"    WHEN 0 THEN 'Sunday'\n" +
                         f"    WHEN 1 THEN 'Monday'\n" +
                         f"    WHEN 2 THEN 'Tuesday'\n" +
                         f"    WHEN 3 THEN 'Wednesday'\n" +
                         f"    WHEN 4 THEN 'Thursday'\n" +
                         f"    WHEN 5 THEN 'Friday'\n" +
                         f"    WHEN 6 THEN 'Saturday'\n" +
                         f"END as day_of_week,\n" +
                         f"COUNT(*) as transaction_count,\n" +
                         f"AVG(amount) as avg_amount\n" +
                         f"FROM {table_name}\n" +
                         f"GROUP BY day_of_week\n" +
                         f"ORDER BY transaction_count DESC")
                         
        suggestions.append(f"-- *GPU Optimized* Distribution of transaction amounts\n" +
                         f"SELECT\n" +
                         f"    CASE\n" +
                         f"        WHEN amount < 100 THEN 'Under $100'\n" +
                         f"        WHEN amount BETWEEN 100 AND 500 THEN '$100-$500'\n" +
                         f"        WHEN amount BETWEEN 500 AND 1000 THEN '$500-$1000'\n" +
                         f"        WHEN amount BETWEEN 1000 AND 5000 THEN '$1000-$5000'\n" +
                         f"        ELSE 'Over $5000'\n" +
                         f"    END as amount_range,\n" +
                         f"    COUNT(*) as transaction_count\n" +
                         f"FROM {table_name}\n" +
                         f"GROUP BY amount_range\n" +
                         f"ORDER BY transaction_count DESC")
                            
        conn.close()
        return suggestions
    except sqlite3.Error:
        conn.close()
        return ["-- No banking data found. Generate a dataset first."]

# Create visualization comparing performance
def create_performance_comparison(serial_time, parallel_time):
    """Create a visualization comparing serial vs parallel execution times."""
    labels = ['Standard Processing', 'GPU-Accelerated']
    times = [serial_time, parallel_time]
    if parallel_time > 0 and serial_time > 0:
        speedup = serial_time / parallel_time
    else:
        speedup = 1.0  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Use bank-themed colors
    bars = ax1.bar(labels, times, color=['#1F4E79', '#2D7D3A'])
    ax1.set_ylabel('Processing Time (seconds)')
    ax1.set_title('Query Processing Performance')
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.4f}s', ha='center', va='bottom')
    
    speedup_for_pie = max(1.0, speedup)
    if abs(speedup_for_pie - 1.0) < 0.01:
        ax2.pie([1], labels=['Baseline'], colors=['#1F4E79'], autopct='%1.1f%%',
                startangle=90)
    else:
        ax2.pie([1, speedup_for_pie-1], labels=['Baseline', f'GPU Boost'], 
                colors=['#1F4E79', '#2D7D3A'], autopct='%1.1f%%', startangle=90, explode=(0, 0.1))
    
    ax2.axis('equal')
    ax2.set_title(f'CUDA Optimization: {speedup:.2f}x Faster')
    plt.suptitle('Banking Analytics GPU Acceleration', fontsize=16)
    plt.tight_layout()
    temp_img_path = "banking_performance_comparison.png"
    plt.savefig(temp_img_path, format='png')
    plt.close()
    return temp_img_path

# Main function to run the SQL query and compare serial vs CUDA execution
def run_sql_query(query, run_parallel=True):
    """Run an SQL query and return results with performance metrics."""
    try:
        serial_result, serial_time = execute_serial_query(query)
        
        if run_parallel:
            parallel_result, parallel_time = execute_cuda_query(query)
            result_df = parallel_result
        else:
            parallel_time = 0
            result_df = serial_result
        
        if run_parallel and parallel_time > 0:
            viz_img_path = create_performance_comparison(serial_time, parallel_time)
            speedup = serial_time / parallel_time
            return (
                result_df.head(50).to_html(),
                f"Results: {result_df.shape[0]} transactions × {result_df.shape[1]} data points",
                f"Standard CPU processing time: {serial_time:.4f} seconds",
                f"GPU-accelerated processing time: {parallel_time:.4f} seconds",
                f"CUDA Optimization Speedup: {speedup:.2f}x",
                viz_img_path
            )
        else:
            return (
                result_df.head(50).to_html(),
                f"Results: {result_df.shape[0]} transactions × {result_df.shape[1]} data points",
                f"Processing time: {serial_time:.4f} seconds",
                "",
                "",
                None
            )
    except Exception as e:
        return f"Error: {str(e)}", "", "", "", "", None

def generate_and_update_suggestions(num_rows, table_name):
    """Generate a banking dataset and update the query suggestions."""
    result_msg = generate_dataset(int(num_rows), table_name)
    suggestions = suggest_queries(table_name)
    suggested_query = suggestions[0] if suggestions else ""
    return result_msg, suggested_query, gr.Dropdown(choices=suggestions)

def create_interface():
    with gr.Blocks(title="Banking Analytics Platform with CUDA Acceleration") as app:
        gr.Markdown("# Banking Analytics Platform")
        gr.Markdown("Analyze transaction data with GPU-accelerated processing, powered by CUDA optimization")
        
        with gr.Tab("Data Management"):
            with gr.Row():
                with gr.Column():
                    num_rows = gr.Slider(minimum=10000, maximum=10000000, value=1000000, step=10000, 
                                       label="Number of Transactions to Generate")
                    table_name = gr.Textbox(value="customer_transactions", label="Transaction Table Name")
                    generate_btn = gr.Button("Generate Banking Dataset", variant="primary")
                    generation_result = gr.Textbox(label="Generation Status")
        
        with gr.Tab("Banking Analytics"):
            with gr.Row():
                with gr.Column(scale=2):
                    query_suggestions = gr.Dropdown(choices=["-- Generate a banking dataset first to see analysis options"], 
                                                  label="Available Analysis Queries")
                    query_input = gr.Textbox(lines=5, label="SQL Query", 
                                          placeholder="SELECT * FROM customer_transactions LIMIT 10")
                    run_parallel_checkbox = gr.Checkbox(value=True, label="Use CUDA GPU Acceleration")
                    run_btn = gr.Button("Run Analysis", variant="primary")
                
            with gr.Row():
                results_html = gr.HTML(label="Analysis Results")
            
            with gr.Row():
                result_shape = gr.Textbox(label="Result Summary")
            
            with gr.Row():
                with gr.Column():
                    serial_time = gr.Textbox(label="CPU Processing Time")
                    parallel_time = gr.Textbox(label="GPU Processing Time")
                    speedup_text = gr.Textbox(label="Performance Speedup")
                
                with gr.Column():
                    performance_viz = gr.Image(label="Performance Comparison")
        
        # Connect the components
        generate_btn.click(
            generate_and_update_suggestions, 
            inputs=[num_rows, table_name], 
            outputs=[generation_result, query_input, query_suggestions]
        )
        
        query_suggestions.change(
            lambda x: x,
            inputs=query_suggestions,
            outputs=query_input
        )
        
        run_btn.click(
            run_sql_query,
            inputs=[query_input, run_parallel_checkbox],
            outputs=[results_html, result_shape, serial_time, parallel_time, speedup_text, performance_viz]
        )
        
        # Initialize suggestions when app loads
        app.load(
            lambda: (suggest_queries(), ""),
            outputs=[query_suggestions, query_input]
        )
    
    return app

# Start the application
if __name__ == "__main__":
    # Create database directory if it doesn't exist
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    # Check if the database already exists
    db_exists = os.path.exists(DB_PATH)
    
    # Create a small dataset if the database doesn't exist
    if not db_exists:
        print("Initializing database with sample data...")
        generate_dataset(100000)  # Generate a smaller initial dataset
    
    # Launch the Gradio app
    app = create_interface()
    app.launch(share=False)