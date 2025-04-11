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
from concurrent.futures import ThreadPoolExecutor  # Could switch to ProcessPoolExecutor if needed
import base64

# Database path
DB_PATH = "performance_test.db"

def get_connection():
    """
    Returns a new SQLite connection with WAL mode enabled and 
    check_same_thread set to False (to allow sharing across threads if needed).
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

# Function to generate a random dataset and create a SQLite database
def generate_dataset(num_rows=1000000, table_name="test_data"):
    """Generate a random dataset and store it in SQLite."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Drop table if it exists
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    
    # Create a new table
    cursor.execute(f"""
    CREATE TABLE {table_name} (
        id INTEGER PRIMARY KEY,
        name TEXT,
        value REAL,
        category TEXT,
        date TEXT,
        is_active INTEGER
    )
    """)
    
    # Generate random data
    batch_size = 10000
    num_batches = num_rows // batch_size
    categories = ['A', 'B', 'C', 'D', 'E']
    start_time = time.time()
    
    for batch in range(num_batches):
        data = []
        for i in range(batch_size):
            row_id = batch * batch_size + i
            name = ''.join(random.choice(string.ascii_letters) for _ in range(10))
            value = random.uniform(0, 1000)
            category = random.choice(categories)
            date = f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
            is_active = random.randint(0, 1)
            data.append((row_id, name, value, category, date, is_active))
            
        cursor.executemany(
            f"INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?)",
            data
        )
        conn.commit()
        
        if (batch + 1) % 10 == 0 or batch == num_batches - 1:
            progress = (batch + 1) / num_batches * 100
            elapsed = time.time() - start_time
            print(f"Progress: {progress:.2f}% - Elapsed time: {elapsed:.2f}s")
    
    # Create indexes for better performance
    cursor.execute(f"CREATE INDEX idx_{table_name}_category ON {table_name}(category)")
    cursor.execute(f"CREATE INDEX idx_{table_name}_value ON {table_name}(value)")
    cursor.execute(f"CREATE INDEX idx_{table_name}_date ON {table_name}(date)")
    conn.commit()
    
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    actual_rows = cursor.fetchone()[0]
    conn.close()
    
    return f"Generated {actual_rows:,} rows in table '{table_name}'"

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

# Function to execute a SQL query with parallel processing using GPU or CPU where applicable
def execute_parallel_query(query, params=None):
    """Execute a SQL query with parallel processing and measure performance."""
    query_lower = query.lower()
    
    # Check for aggregation queries that we try to accelerate on the GPU
    if ('sum(' in query_lower or 'avg(' in query_lower or 'count(' in query_lower or 
        'min(' in query_lower or 'max(' in query_lower):
        return execute_gpu_accelerated_query(query, params)
    else:
        return execute_parallel_cpu_query(query, params)

def execute_gpu_accelerated_query(query, params=None):
    """Execute a query using GPU acceleration for specific operations."""
    conn = get_connection()
    start_time = time.time()
    
    # Simplified parsing: extract table name from query
    query_parts = query.lower().split('from')
    if len(query_parts) > 1:
        table_name = query_parts[1].strip().split()[0]
    else:
        table_name = "test_data"
    
    # For demonstration, we retrieve the entire table.
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    
    agg_functions = ['sum', 'avg', 'count', 'min', 'max']
    used_agg = None
    agg_column = None
    
    for agg in agg_functions:
        if f"{agg}(" in query.lower():
            used_agg = agg
            start_idx = query.lower().find(f"{agg}(") + len(agg) + 1
            end_idx = query.find(")", start_idx)
            agg_column = query[start_idx:end_idx].strip()
            break
    
    if used_agg and agg_column:
        if agg_column == '*' and used_agg.lower() == 'count':
            result_value = len(df)
        else:
            if agg_column in df.columns:
                column_data = df[agg_column].astype(np.float32).values
                
                # Allocate GPU memory and transfer data
                column_gpu = cuda.mem_alloc(column_data.nbytes)
                cuda.memcpy_htod(column_gpu, column_data)
                
                # CUDA kernel for parallel reduction (here using shared memory)
                mod = SourceModule("""
                __global__ void sum_reduce(float *input, float *output, int n) {
                    extern __shared__ float sdata[];
                    unsigned int tid = threadIdx.x;
                    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                    sdata[tid] = (i < n) ? input[i] : 0;
                    __syncthreads();
                    
                    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                        if (tid < s) {
                            sdata[tid] += sdata[tid + s];
                        }
                        __syncthreads();
                    }
                    
                    if (tid == 0) output[blockIdx.x] = sdata[0];
                }
                """)
                sum_reduce = mod.get_function("sum_reduce")
                
                block_size = 256
                grid_size = (column_data.size + block_size - 1) // block_size
                partial_sums = np.zeros(grid_size, dtype=np.float32)
                partial_sums_gpu = cuda.mem_alloc(partial_sums.nbytes)
                
                sum_reduce(
                    column_gpu, partial_sums_gpu, np.int32(column_data.size),
                    block=(block_size, 1, 1), grid=(grid_size, 1),
                    shared=block_size * 4
                )
                cuda.memcpy_dtoh(partial_sums, partial_sums_gpu)
                
                if used_agg.lower() == 'sum':
                    result_value = np.sum(partial_sums)
                elif used_agg.lower() == 'avg':
                    result_value = np.sum(partial_sums) / column_data.size
                else:
                    # Fallback for other aggregations
                    if used_agg.lower() == 'count':
                        result_value = len(column_data)
                    elif used_agg.lower() == 'min':
                        result_value = column_data.min()
                    elif used_agg.lower() == 'max':
                        result_value = column_data.max()
                    else:
                        result_value = None
            else:
                result_value = None
        result = pd.DataFrame({f"{used_agg}({agg_column})": [result_value]})
    else:
        if params:
            result = pd.read_sql_query(query, conn, params=params)
        else:
            result = pd.read_sql_query(query, conn)
    
    execution_time = time.time() - start_time
    conn.close()
    return result, execution_time

# Helper function for executing a chunked query. Made top-level to support process/thread pools.
def execute_chunk(query, params=None):
    # Each call creates its own connection with WAL enabled.
    conn = get_connection()
    if params:
        df = pd.read_sql_query(query, conn, params=params)
    else:
        df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def execute_parallel_cpu_query(query, params=None):
    """Execute a query using CPU parallelism with chunking strategy."""
    conn = get_connection()
    start_time = time.time()
    query_lower = query.lower()
    
    if query_lower.startswith('select'):
        # Parse table name (simplified)
        query_parts = query_lower.split('from')
        if len(query_parts) > 1:
            table_name = query_parts[1].strip().split()[0]
        else:
            table_name = "test_data"
        
        # Get the total row count
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_rows = cursor.fetchone()[0]
        
        # Only chunk if there is sufficient data
        if total_rows > 10000:
            num_chunks = min(8, max(2, os.cpu_count() or 2))
            chunk_size = total_rows // num_chunks
            chunk_queries = []
            for i in range(num_chunks):
                offset = i * chunk_size
                # For queries with ORDER BY or GROUP BY, chunking may break ordering;
                # here we assume simple queries that support LIMIT/OFFSET.
                if ('order by' in query_lower or 'group by' in query_lower or 'limit' in query_lower):
                    chunk_queries = [query]
                    break
                else:
                    chunk_query = f"{query} LIMIT {chunk_size} OFFSET {offset}"
                    chunk_queries.append(chunk_query)
            
            # Execute the chunk queries concurrently.
            dfs = []
            with ThreadPoolExecutor(max_workers=num_chunks) as executor:
                dfs = list(executor.map(lambda q: execute_chunk(q, params), chunk_queries))
            # Combine chunks
            if len(dfs) > 1:
                result = pd.concat(dfs, ignore_index=True)
            else:
                result = dfs[0]
        else:
            if params:
                result = pd.read_sql_query(query, conn, params=params)
            else:
                result = pd.read_sql_query(query, conn)
    else:
        if params:
            result = pd.read_sql_query(query, conn, params=params)
        else:
            result = pd.read_sql_query(query, conn)
    
    execution_time = time.time() - start_time
    conn.close()
    return result, execution_time

# Function to suggest SQL queries based on the dataset
def suggest_queries(table_name="test_data"):
    """Generate SQL query suggestions based on the table structure."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        if not columns:
            conn.close()
            return ["-- No table found. Generate a dataset first."]
        
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
        sample = cursor.fetchone()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        column_names = [col[1] for col in columns]
        suggestions = []
        suggestions.append(f"-- Basic select\nSELECT * FROM {table_name} LIMIT 10")
        suggestions.append(f"-- Count all rows\nSELECT COUNT(*) FROM {table_name}")
        if 'category' in column_names:
            suggestions.append(f"-- Filter by category\nSELECT * FROM {table_name} WHERE category = 'A' LIMIT 10")
        if 'value' in column_names:
            suggestions.append(f"-- Filter by value range\nSELECT * FROM {table_name} WHERE value > 500 LIMIT 10")
        if 'value' in column_names:
            suggestions.append(f"-- Calculate average value\nSELECT AVG(value) FROM {table_name}")
            suggestions.append(f"-- Calculate sum of values\nSELECT SUM(value) FROM {table_name}")
        if 'category' in column_names and 'value' in column_names:
            suggestions.append(f"-- Group by category\nSELECT category, COUNT(*), AVG(value) FROM {table_name} GROUP BY category")
        if count > 0:
            suggestions.append(f"-- Create a second table and join\n" +
                             f"CREATE TABLE IF NOT EXISTS category_info AS SELECT DISTINCT category, AVG(value) as avg_value FROM {table_name} GROUP BY category;\n" +
                             f"SELECT t.*, c.avg_value FROM {table_name} t JOIN category_info c ON t.category = c.category LIMIT 10")
        if 'date' in column_names and 'value' in column_names and 'category' in column_names:
            suggestions.append(f"-- Complex analysis\n" +
                             f"SELECT strftime('%Y-%m', date) as month, category, COUNT(*) as count, AVG(value) as avg_value\n" +
                             f"FROM {table_name}\n" +
                             f"GROUP BY month, category\n" +
                             f"ORDER BY month, avg_value DESC")
        conn.close()
        return suggestions
    except sqlite3.Error:
        conn.close()
        return ["-- No table found. Generate a dataset first."]

# Create visualization comparing performance
def create_performance_comparison(serial_time, parallel_time):
    """Create a visualization comparing serial vs parallel execution times."""
    labels = ['Serial', 'Parallel']
    times = [serial_time, parallel_time]
    if parallel_time > 0 and serial_time > 0:
        speedup = serial_time / parallel_time
    else:
        speedup = 1.0  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    bars = ax1.bar(labels, times, color=['#3498db', '#2ecc71'])
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Query Execution Time Comparison')
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.4f}s', ha='center', va='bottom')
    
    speedup_for_pie = max(1.0, speedup)
    if abs(speedup_for_pie - 1.0) < 0.01:
        ax2.pie([1], labels=['Baseline (No Speedup)'], colors=['#3498db'], autopct='%1.1f%%',
                startangle=90)
    else:
        ax2.pie([1, speedup_for_pie-1], labels=['Baseline', f'Speedup'], 
                colors=['#3498db', '#2ecc71'], autopct='%1.1f%%', startangle=90, explode=(0, 0.1))
    
    ax2.axis('equal')
    ax2.set_title(f'Performance Improvement: {speedup:.2f}x Faster')
    plt.tight_layout()
    temp_img_path = "performance_comparison.png"
    plt.savefig(temp_img_path, format='png')
    plt.close()
    return temp_img_path

# Main function to run the SQL query and compare serial vs parallel execution
def run_sql_query(query, run_parallel=True):
    """Run an SQL query and return results with performance metrics."""
    try:
        serial_result, serial_time = execute_serial_query(query)
        if run_parallel:
            parallel_result, parallel_time = execute_parallel_query(query)
            result_df = parallel_result
        else:
            parallel_time = 0
            result_df = serial_result
        
        if run_parallel and parallel_time > 0:
            viz_img_path = create_performance_comparison(serial_time, parallel_time)
            return (
                result_df.head(50).to_html(),
                f"Results shape: {result_df.shape[0]} rows × {result_df.shape[1]} columns",
                f"Serial execution time: {serial_time:.4f} seconds",
                f"Parallel execution time: {parallel_time:.4f} seconds",
                f"Speedup: {serial_time/parallel_time:.2f}x",
                viz_img_path
            )
        else:
            return (
                result_df.head(50).to_html(),
                f"Results shape: {result_df.shape[0]} rows × {result_df.shape[1]} columns",
                f"Execution time: {serial_time:.4f} seconds",
                "",
                "",
                None
            )
    except Exception as e:
        return f"Error: {str(e)}", "", "", "", "", None

def generate_and_update_suggestions(num_rows, table_name):
    """Generate a dataset and update the query suggestions."""
    result_msg = generate_dataset(int(num_rows), table_name)
    suggestions = suggest_queries(table_name)
    suggested_query = suggestions[0] if suggestions else ""
    return result_msg, suggested_query, gr.Dropdown(choices=suggestions)

def create_interface():
    with gr.Blocks(title="SQL Query Performance Analyzer") as app:
        gr.Markdown("# SQL Query Performance Analyzer")
        gr.Markdown("Test SQL queries on large datasets and visualize performance improvements with parallel computing")
        
        with gr.Tab("Dataset Generation"):
            with gr.Row():
                with gr.Column():
                    num_rows = gr.Slider(minimum=10000, maximum=10000000, value=1000000, step=10000, label="Number of Rows")
                    table_name = gr.Textbox(value="test_data", label="Table Name")
                    generate_btn = gr.Button("Generate Dataset")
                    generation_result = gr.Textbox(label="Generation Result")
        
        with gr.Tab("Query Execution"):
            with gr.Row():
                with gr.Column(scale=2):
                    query_suggestions = gr.Dropdown(choices=["-- Generate a dataset first to see suggestions"], label="Query Suggestions")
                    query_input = gr.Textbox(lines=5, label="SQL Query", placeholder="SELECT * FROM test_data LIMIT 10")
                    run_parallel_checkbox = gr.Checkbox(value=True, label="Run with Parallel Execution")
                    run_btn = gr.Button("Run Query")
                
            with gr.Row():
                results_html = gr.HTML(label="Results")
            
            with gr.Row():
                result_shape = gr.Textbox(label="Result Information")
            
            with gr.Row():
                with gr.Column():
                    serial_time = gr.Textbox(label="Serial Execution Time")
                    parallel_time = gr.Textbox(label="Parallel Execution Time")
                    speedup = gr.Textbox(label="Performance Improvement")
                with gr.Column():
                    viz_output = gr.Image(type="filepath", label="Performance Comparison")
        
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
            outputs=[results_html, result_shape, serial_time, parallel_time, speedup, viz_output]
        )
        
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch()
