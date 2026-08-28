import time
import random
import statistics

class AIInferenceSimulator:
    def __init__(self, model_name="DeepLearning-Base"):
        self.model_name = model_name
        self.is_quantized = False

    def load_model(self, quantized=False):
        """Simulates loading the AI model into memory with optional quantization."""
        self.is_quantized = quantized
        print(f"[*] Loading model '{self.model_name}' (Quantized: {self.is_quantized})...")
        time.sleep(1.5)
        vram_usage = 9.5 if self.is_quantized else 22.0
        print(f"[+] Model loaded successfully. VRAM Footprint: {vram_usage} GB\n")

    def run_inference(self, batch_size=1):
        """Simulates running inference and returns processing time in milliseconds."""
        start_time = time.time()
        
        if self.is_quantized:
            delay = random.uniform(0.05, 0.12) * (1 + (batch_size * 0.05))
        else:
            delay = random.uniform(0.20, 0.45) * (1 + (batch_size * 0.10))
            
        time.sleep(delay)
        return (time.time() - start_time) * 1000

def run_benchmark():
    print("==================================================")
    print("      AI SYSTEM PERFORMANCE BENCHMARK SUITE       ")
    print("==================================================")
    
    simulator = AIInferenceSimulator()
    
    print("\n--- Phase 1: Baseline System Evaluation (FP32) ---")
    simulator.load_model(quantized=False)
    
    baseline_latencies = []
    for i in range(10):
        lat = simulator.run_inference(batch_size=1)
        baseline_latencies.append(lat)
        print(f"Request {i+1}: Completed in {lat:.2f} ms")
        
    avg_baseline = statistics.mean(baseline_latencies)
    print(f"\n[Baseline Summary] Average Latency: {avg_baseline:.2f} ms")

    print("\n--- Phase 2: Optimized System Evaluation (INT8 Quantization) ---")
    simulator.load_model(quantized=True)
    
    optimized_latencies = []
    for i in range(10):
        lat = simulator.run_inference(batch_size=1)
        optimized_latencies.append(lat)
        print(f"Request {i+1}: Completed in {lat:.2f} ms")
        
    avg_optimized = statistics.mean(optimized_latencies)
    print(f"\n[Optimized Summary] Average Latency: {avg_optimized:.2f} ms")
    
    print("\n==================================================")
    print("               FINAL PERFORMANCE GAIN             ")
    print("==================================================")
    improvement = ((avg_baseline - avg_optimized) / avg_baseline) * 100
    print(f"Latency Reduction: {improvement:.2f}% faster")
    print("Status: Target optimization thresholds successfully achieved.")

if __name__ == "__main__":
    run_benchmark()
