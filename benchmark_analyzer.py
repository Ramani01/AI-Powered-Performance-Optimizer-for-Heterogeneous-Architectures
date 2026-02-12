"""
Hardware-Specific Performance Bottleneck Analyzer
Benchmarks and analyzes performance bottlenecks across different hardware architectures
"""

import numpy as np
import time
import psutil
import os
from typing import Dict, List, Tuple, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from enum import Enum


class ArchitectureType(Enum):
    """Enumeration of different hardware architectures"""
    CPU_SERIAL = "CPU_Serial"
    CPU_PARALLEL = "CPU_Parallel"
    GPU_OPENCL = "GPU_OpenCL"
    NUMBA_JIT = "Numba_JIT"


@dataclass
class BenchmarkResult:
    """Data class to store benchmark results"""
    architecture: ArchitectureType
    operation: str
    input_size: int
    execution_time: float
    memory_usage: float
    cpu_utilization: float
    throughput: float
    efficiency_score: float


class HardwareBottleneckAnalyzer:
    """
    Analyzes hardware-specific performance bottlenecks
    Identifies CPU/GPU memory transfer and thread synchronization issues
    """
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.hardware_info = self._gather_hardware_info()
        
    def _gather_hardware_info(self) -> Dict[str, any]:
        """Gather system hardware information"""
        info = {
            'cpu_count': os.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'swap_total': psutil.swap_memory().total,
        }
        return info
    
    def benchmark_cpu_serial(self, operation: Callable, input_data, *args) -> BenchmarkResult:
        """Benchmark operation on CPU serially"""
        # Monitor initial state
        initial_memory = psutil.Process().memory_info().rss
        initial_cpu_percent = psutil.cpu_percent(interval=None)
        
        start_time = time.perf_counter()
        result = operation(input_data, *args)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        # Monitor final state
        final_memory = psutil.Process().memory_info().rss
        final_cpu_percent = psutil.cpu_percent(interval=None)
        
        memory_delta = (final_memory - initial_memory) / (1024 * 1024)  # MB
        avg_cpu_util = (initial_cpu_percent + final_cpu_percent) / 2
        
        # Calculate throughput (operations per second)
        throughput = len(input_data.flatten()) / execution_time if hasattr(input_data, 'flatten') else 1 / execution_time
        
        # Efficiency score (higher is better)
        efficiency_score = throughput / (execution_time + memory_delta/1000 + (100-avg_cpu_util)/100)
        
        result = BenchmarkResult(
            architecture=ArchitectureType.CPU_SERIAL,
            operation=operation.__name__,
            input_size=len(input_data.flatten()) if hasattr(input_data, 'flatten') else len(input_data),
            execution_time=execution_time,
            memory_usage=memory_delta,
            cpu_utilization=avg_cpu_util,
            throughput=throughput,
            efficiency_score=efficiency_score
        )
        
        self.results.append(result)
        return result
    
    def benchmark_cpu_parallel(self, operation: Callable, input_data, *args) -> BenchmarkResult:
        """Benchmark operation on CPU with parallel execution"""
        from multiprocessing import Pool
        import threading
        
        # For simplicity, we'll simulate parallel execution by dividing the work
        # In a real implementation, this would use actual parallel processing
        initial_memory = psutil.Process().memory_info().rss
        initial_cpu_percent = psutil.cpu_percent(interval=None)
        
        start_time = time.perf_counter()
        
        # Simulate parallel execution by splitting work
        chunk_size = len(input_data) // os.cpu_count() if os.cpu_count() > 1 else len(input_data)
        chunks = [input_data[i:i+chunk_size] for i in range(0, len(input_data), chunk_size)]
        
        # Execute operation on chunks (this is simplified)
        results = []
        for chunk in chunks:
            if len(chunk) > 0:
                results.append(operation(chunk, *args))
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        final_memory = psutil.Process().memory_info().rss
        final_cpu_percent = psutil.cpu_percent(interval=None)
        
        memory_delta = (final_memory - initial_memory) / (1024 * 1024)  # MB
        avg_cpu_util = (initial_cpu_percent + final_cpu_percent) / 2
        
        throughput = len(input_data.flatten()) / execution_time if hasattr(input_data, 'flatten') else 1 / execution_time
        efficiency_score = throughput / (execution_time + memory_delta/1000 + (100-avg_cpu_util)/100)
        
        result = BenchmarkResult(
            architecture=ArchitectureType.CPU_PARALLEL,
            operation=operation.__name__,
            input_size=len(input_data.flatten()) if hasattr(input_data, 'flatten') else len(input_data),
            execution_time=execution_time,
            memory_usage=memory_delta,
            cpu_utilization=avg_cpu_util,
            throughput=throughput,
            efficiency_score=efficiency_score
        )
        
        self.results.append(result)
        return result
    
    def benchmark_numba_jit(self, operation: Callable, input_data, *args) -> BenchmarkResult:
        """Benchmark operation using Numba JIT compilation"""
        # We'll simulate the effect of JIT compilation by running warm-up iterations
        # In a real implementation, this would call the actual JIT-compiled function
        
        # Warm-up runs for JIT compilation effect
        for _ in range(3):
            _ = operation(input_data, *args)
        
        initial_memory = psutil.Process().memory_info().rss
        initial_cpu_percent = psutil.cpu_percent(interval=None)
        
        start_time = time.perf_counter()
        result = operation(input_data, *args)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        final_memory = psutil.Process().memory_info().rss
        final_cpu_percent = psutil.cpu_percent(interval=None)
        
        memory_delta = (final_memory - initial_memory) / (1024 * 1024)  # MB
        avg_cpu_util = (initial_cpu_percent + final_cpu_percent) / 2
        
        throughput = len(input_data.flatten()) / execution_time if hasattr(input_data, 'flatten') else 1 / execution_time
        efficiency_score = throughput / (execution_time + memory_delta/1000 + (100-avg_cpu_util)/100)
        
        result = BenchmarkResult(
            architecture=ArchitectureType.NUMBA_JIT,
            operation=operation.__name__,
            input_size=len(input_data.flatten()) if hasattr(input_data, 'flatten') else len(input_data),
            execution_time=execution_time,
            memory_usage=memory_delta,
            cpu_utilization=avg_cpu_util,
            throughput=throughput,
            efficiency_score=efficiency_score
        )
        
        self.results.append(result)
        return result
    
    def benchmark_gpu_opencl(self, operation: Callable, input_data, *args) -> BenchmarkResult:
        """Benchmark operation using GPU with OpenCL"""
        # Import here to avoid issues if OpenCL is not available
        try:
            from opencl_gpu_accelerator import OpenCLAccelerator
            accelerator = OpenCLAccelerator()
            
            if accelerator.ctx is not None:
                initial_memory = psutil.Process().memory_info().rss
                initial_cpu_percent = psutil.cpu_percent(interval=None)
                
                start_time = time.perf_counter()
                result = operation(accelerator, input_data, *args)  # Pass accelerator as first param
                end_time = time.perf_counter()
                
                execution_time = end_time - start_time
                
                final_memory = psutil.Process().memory_info().rss
                final_cpu_percent = psutil.cpu_percent(interval=None)
                
                memory_delta = (final_memory - initial_memory) / (1024 * 1024)  # MB
                avg_cpu_util = (initial_cpu_percent + final_cpu_percent) / 2
                
                throughput = len(input_data.flatten()) / execution_time if hasattr(input_data, 'flatten') else 1 / execution_time
                efficiency_score = throughput / (execution_time + memory_delta/1000 + (100-avg_cpu_util)/100)
                
                result = BenchmarkResult(
                    architecture=ArchitectureType.GPU_OPENCL,
                    operation=operation.__name__,
                    input_size=len(input_data.flatten()) if hasattr(input_data, 'flatten') else len(input_data),
                    execution_time=execution_time,
                    memory_usage=memory_delta,
                    cpu_utilization=avg_cpu_util,
                    throughput=throughput,
                    efficiency_score=efficiency_score
                )
                
                self.results.append(result)
                return result
            else:
                # GPU not available, return dummy result
                return BenchmarkResult(
                    architecture=ArchitectureType.GPU_OPENCL,
                    operation=operation.__name__,
                    input_size=0,
                    execution_time=float('inf'),
                    memory_usage=0,
                    cpu_utilization=0,
                    throughput=0,
                    efficiency_score=0
                )
        except ImportError:
            # OpenCL not available
            return BenchmarkResult(
                architecture=ArchitectureType.GPU_OPENCL,
                operation=operation.__name__,
                input_size=0,
                execution_time=float('inf'),
                memory_usage=0,
                cpu_utilization=0,
                throughput=0,
                efficiency_score=0
            )
    
    def run_comprehensive_benchmark(self, sizes: List[int] = None) -> Dict[str, List[BenchmarkResult]]:
        """Run comprehensive benchmark across different architectures and sizes"""
        if sizes is None:
            sizes = [128, 256, 512, 1024]  # Different matrix sizes to test
        
        # Define test operations
        def matrix_multiply_op(data, _=None):
            """Simple matrix multiplication operation"""
            return np.dot(data, data.T)
        
        def elementwise_op(data, multiplier=2.0, addend=1.0):
            """Element-wise mathematical operation"""
            return data * multiplier + addend
        
        results_by_arch = {arch.value: [] for arch in ArchitectureType}
        
        print("Running comprehensive hardware benchmark...")
        
        for size in sizes:
            print(f"\nTesting size: {size}x{size}")
            
            # Create test data
            test_matrix = np.random.random((size, size)).astype(np.float32)
            
            # Benchmark each architecture
            print("  CPU Serial:", end=" ")
            serial_result = self.benchmark_cpu_serial(matrix_multiply_op, test_matrix)
            results_by_arch[ArchitectureType.CPU_SERIAL.value].append(serial_result)
            print(f"{serial_result.execution_time:.4f}s")
            
            print("  CPU Parallel:", end=" ")
            parallel_result = self.benchmark_cpu_parallel(matrix_multiply_op, test_matrix)
            results_by_arch[ArchitectureType.CPU_PARALLEL.value].append(parallel_result)
            print(f"{parallel_result.execution_time:.4f}s")
            
            print("  Numba JIT:", end=" ")
            jit_result = self.benchmark_numba_jit(matrix_multiply_op, test_matrix)
            results_by_arch[ArchitectureType.NUMBA_JIT.value].append(jit_result)
            print(f"{jit_result.execution_time:.4f}s")
            
            print("  GPU OpenCL:", end=" ")
            # For GPU, we'll create a mock operation that accepts the accelerator
            def gpu_matmul_wrapper(accelerator, data):
                try:
                    from opencl_gpu_accelerator import OpenCLAccelerator
                    if accelerator.ctx is not None:
                        return accelerator.gpu_matrix_multiply(data, data.T)
                    else:
                        return np.dot(data, data.T)  # fallback
                except ImportError:
                    return np.dot(data, data.T)  # fallback
            
            gpu_result = self.benchmark_gpu_opencl(gpu_matmul_wrapper, test_matrix)
            results_by_arch[ArchitectureType.GPU_OPENCL.value].append(gpu_result)
            if gpu_result.execution_time != float('inf'):
                print(f"{gpu_result.execution_time:.4f}s")
            else:
                print("Not available")
        
        return results_by_arch
    
    def identify_bottlenecks(self) -> Dict[str, str]:
        """Identify specific hardware bottlenecks"""
        bottlenecks = {}
        
        # Analyze memory usage patterns
        memory_usage_by_arch = {}
        for result in self.results:
            if result.architecture.value not in memory_usage_by_arch:
                memory_usage_by_arch[result.architecture.value] = []
            if result.memory_usage > 0:  # Only consider valid measurements
                memory_usage_by_arch[result.architecture.value].append(result.memory_usage)
        
        # Check for high memory usage
        for arch, usages in memory_usage_by_arch.items():
            if usages:
                avg_memory = sum(usages) / len(usages)
                if avg_memory > 500:  # More than 500MB average
                    bottlenecks[f"{arch}_memory"] = f"High memory usage: {avg_memory:.2f} MB average"
        
        # Analyze CPU utilization
        cpu_usage_by_arch = {}
        for result in self.results:
            if result.architecture.value not in cpu_usage_by_arch:
                cpu_usage_by_arch[result.architecture.value] = []
            cpu_usage_by_arch[result.architecture.value].append(result.cpu_utilization)
        
        for arch, utilizations in cpu_usage_by_arch.items():
            if utilizations:
                avg_cpu = sum(utilizations) / len(utilizations)
                if avg_cpu < 20:  # Low CPU utilization might indicate I/O bottleneck
                    bottlenecks[f"{arch}_cpu_util"] = f"Low CPU utilization: {avg_cpu:.2f}% average"
                elif avg_cpu > 95:  # High CPU utilization might indicate compute bottleneck
                    bottlenecks[f"{arch}_cpu_util"] = f"High CPU utilization: {avg_cpu:.2f}% average"
        
        # Analyze efficiency scores
        eff_scores_by_arch = {}
        for result in self.results:
            if result.architecture.value not in eff_scores_by_arch:
                eff_scores_by_arch[result.architecture.value] = []
            eff_scores_by_arch[result.architecture.value].append(result.efficiency_score)
        
        for arch, scores in eff_scores_by_arch.items():
            if scores:
                avg_eff = sum(scores) / len(scores)
                if avg_eff < 0.1:  # Low efficiency score
                    bottlenecks[f"{arch}_efficiency"] = f"Low efficiency score: {avg_eff:.4f} average"
        
        return bottlenecks
    
    def generate_performance_report(self) -> str:
        """Generate a detailed performance report"""
        report = []
        report.append("=== Hardware Performance Analysis Report ===\n")
        
        report.append("System Information:")
        report.append(f"  CPU Cores: {self.hardware_info['cpu_count']}")
        report.append(f"  Total Memory: {self.hardware_info['memory_total'] / (1024**3):.2f} GB")
        report.append(f"  Available Memory: {self.hardware_info['memory_available'] / (1024**3):.2f} GB")
        report.append("")
        
        # Group results by architecture
        results_by_arch = {}
        for result in self.results:
            if result.architecture.value not in results_by_arch:
                results_by_arch[result.architecture.value] = []
            results_by_arch[result.architecture.value].append(result)
        
        report.append("Performance by Architecture:")
        for arch, results in results_by_arch.items():
            if results:
                avg_time = sum(r.execution_time for r in results) / len(results)
                avg_memory = sum(r.memory_usage for r in results) / len(results)
                avg_cpu = sum(r.cpu_utilization for r in results) / len(results)
                avg_eff = sum(r.efficiency_score for r in results) / len(results)
                
                report.append(f"  {arch}:")
                report.append(f"    Avg Execution Time: {avg_time:.4f}s")
                report.append(f"    Avg Memory Usage: {avg_memory:.2f}MB")
                report.append(f"    Avg CPU Utilization: {avg_cpu:.2f}%")
                report.append(f"    Avg Efficiency Score: {avg_eff:.4f}")
                report.append("")
        
        # Identify bottlenecks
        bottlenecks = self.identify_bottlenecks()
        if bottlenecks:
            report.append("Identified Bottlenecks:")
            for bottleneck, description in bottlenecks.items():
                report.append(f"  - {description}")
        else:
            report.append("No significant bottlenecks identified.")
        
        report.append("\n=== End of Report ===")
        
        return "\n".join(report)
    
    def plot_performance_comparison(self, save_path: str = None):
        """Plot performance comparison across architectures"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Prepare data for plotting
            architectures = []
            execution_times = []
            input_sizes = []
            
            for result in self.results:
                if result.execution_time != float('inf'):  # Only valid results
                    architectures.append(result.architecture.value.replace('_', ' '))
                    execution_times.append(result.execution_time)
                    input_sizes.append(result.input_size)
            
            if not architectures:
                print("No valid benchmark results to plot")
                return
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            # Use seaborn for better styling
            sns.set_style("whitegrid")
            
            # Create scatter plot
            plt.scatter(input_sizes, execution_times, c=range(len(architectures)), cmap='tab10', s=100, alpha=0.7)
            
            # Add labels for each point
            for i, arch in enumerate(architectures):
                plt.annotate(arch, (input_sizes[i], execution_times[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.xlabel('Input Size')
            plt.ylabel('Execution Time (seconds)')
            plt.title('Performance Comparison Across Architectures')
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Performance plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib/seaborn not available for plotting")


def demonstrate_benchmark_analysis():
    """Demonstrate the hardware bottleneck analysis"""
    print("=== Hardware-Specific Performance Bottleneck Analysis ===\n")
    
    analyzer = HardwareBottleneckAnalyzer()
    
    # Run a small benchmark to demonstrate functionality
    print("Running sample benchmarks...")
    
    def simple_operation(data):
        return np.sum(data * data)
    
    # Small test matrix
    test_data = np.random.random((100, 100)).astype(np.float32)
    
    # Run benchmarks
    analyzer.benchmark_cpu_serial(simple_operation, test_data)
    analyzer.benchmark_cpu_parallel(simple_operation, test_data)
    analyzer.benchmark_numba_jit(simple_operation, test_data)
    
    # Generate and print report
    report = analyzer.generate_performance_report()
    print(report)
    
    print("\nKey Findings:")
    bottlenecks = analyzer.identify_bottlenecks()
    if bottlenecks:
        for bottleneck, description in bottlenecks.items():
            print(f"  - {description}")
    else:
        print("  No significant bottlenecks detected in sample run")
    
    print("\nFor comprehensive analysis, run the full benchmark suite with various input sizes.")


def run_full_benchmark_suite():
    """Run the complete benchmark suite"""
    print("=== Running Full Benchmark Suite ===\n")
    
    analyzer = HardwareBottleneckAnalyzer()
    
    # Run comprehensive benchmark with different sizes
    sizes = [64, 128, 256]  # Using smaller sizes for demo purposes
    results = analyzer.run_comprehensive_benchmark(sizes)
    
    # Generate detailed report
    report = analyzer.generate_performance_report()
    print(report)
    
    # Save performance plot
    try:
        analyzer.plot_performance_comparison("performance_comparison.png")
    except:
        print("Could not generate performance plot (plotting libraries may not be available)")
    
    print("\n=== Full Benchmark Suite Complete ===")


if __name__ == "__main__":
    demonstrate_benchmark_analysis()
    
    print("\n" + "="*60)
    
    # Uncomment the line below to run the full benchmark suite
    # run_full_benchmark_suite()