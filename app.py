"""
Flask Web Application for AI-Powered Performance Optimizer
Provides a web interface to interact with the optimization engine
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import json
import time
import psutil
import os
from threading import Lock

# Import our optimization modules
from optimization_engine import NumbaOptimizationEngine, ParallelComputingKernels, ThreadSynchronizationManager
from opencl_gpu_accelerator import OpenCLAccelerator, MemoryTransferOptimizer
from benchmark_analyzer import HardwareBottleneckAnalyzer

app = Flask(__name__)

# Global instances of our engines
numba_engine = NumbaOptimizationEngine()
parallel_kernels = ParallelComputingKernels()
sync_manager = ThreadSynchronizationManager()
opencl_accelerator = OpenCLAccelerator()
memory_optimizer = MemoryTransferOptimizer(opencl_accelerator) if opencl_accelerator.ctx else None
analyzer = HardwareBottleneckAnalyzer()

# Thread lock for safe concurrent operations
app_lock = Lock()

@app.route('/')
def index():
    """Serve the main UI page"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'cpu_count': os.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available
    })

@app.route('/api/optimize', methods=['POST'])
def optimize():
    """Handle optimization requests"""
    try:
        data = request.json
        operation = data.get('operation', 'matrix_multiply')
        matrix_size = data.get('matrix_size', 512)
        optimization_type = data.get('optimization_type', 'numba-jit')
        
        # Validate inputs
        if matrix_size < 1 or matrix_size > 5000:  # Reasonable upper limit
            return jsonify({'error': 'Matrix size must be between 1 and 5000'}), 400
        
        # Create test matrices
        A = np.random.random((matrix_size, matrix_size)).astype(np.float64)
        B = np.random.random((matrix_size, matrix_size)).astype(np.float64)
        
        # Record initial system stats
        initial_memory = psutil.Process().memory_info().rss
        initial_cpu_percent = psutil.cpu_percent(interval=None)
        start_time = time.perf_counter()
        
        # Perform the requested operation
        result = None
        execution_time = 0
        
        with app_lock:  # Ensure thread safety
            if operation == 'matrix_multiply':
                if optimization_type == 'numba-jit':
                    result = numba_engine.optimized_matrix_multiply(A, B)
                elif optimization_type == 'numpy':
                    result = np.dot(A, B)  # Baseline
                elif optimization_type == 'opencl-gpu' and opencl_accelerator.ctx:
                    result = opencl_accelerator.gpu_matrix_multiply(
                        A.astype(np.float32), B.astype(np.float32)
                    )
                else:
                    # Fallback to numba
                    result = numba_engine.optimized_matrix_multiply(A, B)
            
            elif operation == 'element_wise':
                multiplier = data.get('multiplier', 2.0)
                addend = data.get('addend', 1.0)
                if optimization_type == 'numba-jit':
                    result = numba_engine.optimized_elementwise_operation(A, multiplier, addend)
                elif optimization_type == 'numpy':
                    result = A * multiplier + addend  # Baseline
                else:
                    result = numba_engine.optimized_elementwise_operation(A, multiplier, addend)
            
            elif operation == 'convolution':
                kernel_size = data.get('kernel_size', 5)
                kernel = np.random.random((kernel_size, kernel_size)).astype(np.float64)
                if optimization_type == 'numba-jit':
                    # Use a subset of the matrix for convolution to avoid size mismatch
                    subset_size = min(matrix_size, 500)  # Limit for reasonable processing time
                    subset_A = A[:subset_size, :subset_size]
                    result = numba_engine.optimized_convolution_2d(subset_A, kernel)
                else:
                    # Fallback implementation
                    subset_size = min(matrix_size, 500)
                    subset_A = A[:subset_size, :subset_size]
                    result = numba_engine.optimized_convolution_2d(subset_A, kernel)
            
            else:
                return jsonify({'error': 'Invalid operation type'}), 400
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Calculate final system stats
        final_memory = psutil.Process().memory_info().rss
        final_cpu_percent = psutil.cpu_percent(interval=None)
        
        memory_used_mb = (final_memory - initial_memory) / (1024 * 1024)
        avg_cpu_util = (initial_cpu_percent + final_cpu_percent) / 2
        
        # Prepare response
        response = {
            'operation': operation,
            'matrix_size': matrix_size,
            'optimization_type': optimization_type,
            'execution_time': execution_time,
            'memory_used_mb': abs(memory_used_mb),  # Take absolute value
            'cpu_utilization': avg_cpu_util,
            'result_shape': result.shape if result is not None else None,
            'success': True
        }
        
        # If we have OpenCL available, also calculate speedup vs baseline
        if opencl_accelerator.ctx and optimization_type != 'numpy':
            # Calculate baseline time with NumPy for comparison
            baseline_start = time.perf_counter()
            if operation == 'matrix_multiply':
                baseline_result = np.dot(A, B)
            elif operation == 'element_wise':
                baseline_result = A * data.get('multiplier', 2.0) + data.get('addend', 1.0)
            else:
                # For convolution, use a smaller subset for baseline
                subset_size = min(matrix_size, 500)
                subset_A = A[:subset_size, :subset_size]
                kernel_size = data.get('kernel_size', 5)
                kernel = np.random.random((kernel_size, kernel_size)).astype(np.float64)
                baseline_result = np.zeros((subset_size-kernel_size+1, subset_size-kernel_size+1))
                # Manual convolution for baseline (slow but accurate)
                for i in range(baseline_result.shape[0]):
                    for j in range(baseline_result.shape[1]):
                        baseline_result[i, j] = np.sum(subset_A[i:i+kernel_size, j:j+kernel_size] * kernel)
            
            baseline_end_time = time.perf_counter()
            baseline_time = baseline_end_time - baseline_start
            speedup = baseline_time / execution_time if execution_time > 0 else float('inf')
            response['baseline_time'] = baseline_time
            response['speedup'] = speedup
            response['performance_gain_percentage'] = (speedup - 1) * 100 if speedup != float('inf') else float('inf')
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/benchmark', methods=['POST'])
def benchmark():
    """Run comprehensive benchmark across different architectures"""
    try:
        data = request.json
        matrix_size = data.get('matrix_size', 256)
        
        # Create test data
        A = np.random.random((matrix_size, matrix_size)).astype(np.float32)
        B = np.random.random((matrix_size, matrix_size)).astype(np.float32)
        
        results = {}
        
        # Benchmark NumPy (baseline)
        start_time = time.perf_counter()
        numpy_result = np.dot(A, B)
        numpy_time = time.perf_counter() - start_time
        results['numpy'] = {
            'time': numpy_time,
            'operation': 'matrix_multiply',
            'size': matrix_size
        }
        
        # Benchmark Numba JIT
        start_time = time.perf_counter()
        numba_result = numba_engine.optimized_matrix_multiply(A.astype(np.float64), B.astype(np.float64))
        numba_time = time.perf_counter() - start_time
        results['numba_jit'] = {
            'time': numba_time,
            'operation': 'matrix_multiply',
            'size': matrix_size
        }
        
        # Benchmark OpenCL GPU if available
        if opencl_accelerator.ctx:
            start_time = time.perf_counter()
            gpu_result = opencl_accelerator.gpu_matrix_multiply(A, B)
            gpu_time = time.perf_counter() - start_time
            results['opencl_gpu'] = {
                'time': gpu_time,
                'operation': 'matrix_multiply',
                'size': matrix_size
            }
        else:
            results['opencl_gpu'] = {
                'time': float('inf'),  # Indicate not available
                'operation': 'matrix_multiply',
                'size': matrix_size,
                'unavailable': True
            }
        
        # Calculate speedups
        results['speedups'] = {}
        if numba_time > 0:
            results['speedups']['numba_vs_numpy'] = numpy_time / numba_time
        if opencl_accelerator.ctx and results['opencl_gpu']['time'] > 0:
            results['speedups']['gpu_vs_numpy'] = numpy_time / results['opencl_gpu']['time']
            results['speedups']['gpu_vs_numba'] = numba_time / results['opencl_gpu']['time']
        
        return jsonify({
            'benchmark_results': results,
            'timestamp': time.time(),
            'matrix_size': matrix_size
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system-info')
def system_info():
    """Get system information"""
    try:
        info = {
            'cpu_count': os.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/')._asdict() if os.name != 'nt' else psutil.disk_usage('.')._asdict(),
            'platform': os.name,
            'opencl_available': opencl_accelerator.ctx is not None,
            'numba_available': True  # Assuming numba is available since we imported it
        }
        
        return jsonify(info)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    import os
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Copy the HTML file to templates directory with the correct name
    import shutil
    try:
        # Read the content from ui_index.html and save as index.html in templates
        with open('ui_index.html', 'r', encoding='utf-8') as src:
            content = src.read()
        
        # Modify the content to work with Flask template structure if needed
        # For now, we'll just copy it as is since it doesn't use Flask templating
        with open('templates/index.html', 'w', encoding='utf-8') as dst:
            dst.write(content)
    except FileNotFoundError:
        # If ui_index.html doesn't exist, create a simple placeholder
        with open('templates/index.html', 'w', encoding='utf-8') as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>AI-Powered Performance Optimizer</title>
</head>
<body>
    <h1>AI-Powered Performance Optimizer Backend</h1>
    <p>The UI has been set up. Access the full interface at the root URL.</p>
</body>
</html>
""")
    except UnicodeDecodeError:
        # Handle encoding issues
        with open('ui_index.html', 'r', encoding='latin-1') as src:
            content = src.read()
        
        with open('templates/index.html', 'w', encoding='utf-8') as dst:
            dst.write(content)
    
    print("Starting AI-Powered Performance Optimizer Web Interface...")
    print("Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)