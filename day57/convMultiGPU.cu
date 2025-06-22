#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <atomic>
#include <algorithm>

#define RADIUS 1
#define FILTER_SIZE (2 * RADIUS + 1)
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * RADIUS)
#define NUM_STREAMS 2

struct ImageTask 
{
    std::string filename;
    cv::Mat image;
    int priority;
};

template<typename T>
class MemoryQueue 
{
private:
    std::queue<T> q;
    std::mutex m;
    std::condition_variable cv_push, cv_pop;
    const size_t max_size;
    std::atomic<bool> finished{false};

public:
    MemoryQueue(size_t max_size) : max_size(max_size) {}

    void push(const T& val) 
    {
        std::unique_lock<std::mutex> lock(m);
        cv_push.wait(lock, [&]{ return q.size() < max_size; });
        q.push(val);
        cv_pop.notify_one();
    }

    bool isFinished(T &val) 
    {
        std::unique_lock<std::mutex> lock(m);
        cv_pop.wait(lock, [&]{ return !q.empty() || finished.load(); });
        if (q.empty()) return false;
        val = q.front();
        q.pop();
        cv_push.notify_one();
        return true;
    }

    void setFinished() 
    {
        finished.store(true);
        cv_pop.notify_all();
    }

    size_t size()
    {
        std::lock_guard<std::mutex> lock(m);
        return q.size();
    }
};

class GPUContext 
{
public:
    int device_id;
    cudaStream_t streams[NUM_STREAMS];
    float* d_filter;
    float* d_in[3];
    float* d_out[3];
    float* h_pinned_in[3];
    float* h_pinned_out[3];
    size_t max_image_size;
    GPUContext(int id, size_t max_size) : device_id(id), max_image_size(max_size) 
    {
        cudaSetDevice(device_id);
        
        for (int i = 0; i < NUM_STREAMS; ++i) 
        {
            cudaStreamCreate(&streams[i]);
        }
        
        cudaMalloc(&d_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));
        
        for (int c = 0; c < 3; ++c) 
        {
            cudaMalloc(&d_in[c], max_image_size);
            cudaMalloc(&d_out[c], max_image_size);
            cudaMallocHost(&h_pinned_in[c], max_image_size);
            cudaMallocHost(&h_pinned_out[c], max_image_size);
        }
    }

    ~GPUContext() 
    {
        cudaSetDevice(device_id);
        for (int i = 0; i < NUM_STREAMS; ++i) 
        {
            cudaStreamDestroy(streams[i]);
        }
        cudaFree(d_filter);
        for (int c = 0; c < 3; ++c) 
        {
            cudaFree(d_in[c]);
            cudaFree(d_out[c]);
            cudaFreeHost(h_pinned_in[c]);
            cudaFreeHost(h_pinned_out[c]);
        }
    }

    cudaStream_t getStream(int idx) const { return streams[idx]; }
    float* getInputDevicePtr(int c) const { return d_in[c]; }
    float* getOutputDevicePtr(int c) const { return d_out[c]; }
    float* getPinnedInputPtr(int c) const { return h_pinned_in[c]; }
    float* getPinnedOutputPtr(int c) const { return h_pinned_out[c]; }
    float* getFilterDevicePtr() const { return d_filter; }
    int getDeviceId() const { return device_id; }
};

__global__ void conv2D(float *input, float *output, float *filter_dev, int H, int W) 
{
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - RADIUS;
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - RADIUS;

    __shared__ float tile[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float filter[FILTER_SIZE][FILTER_SIZE];

    if (threadIdx.y < FILTER_SIZE && threadIdx.x < FILTER_SIZE) 
        filter[threadIdx.y][threadIdx.x] = filter_dev[threadIdx.y * FILTER_SIZE + threadIdx.x];

    if (row >= 0 && row < H && col >= 0 && col < W)
        tile[threadIdx.y][threadIdx.x] = input[row * W + col];
    else
        tile[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    int tile_row = threadIdx.y - RADIUS;
    int tile_col = threadIdx.x - RADIUS;

    if (tile_row >= 0 && tile_row < OUT_TILE_DIM && tile_col >= 0 && tile_col < OUT_TILE_DIM) 
    {
        int out_row = blockIdx.y * OUT_TILE_DIM + tile_row;
        int out_col = blockIdx.x * OUT_TILE_DIM + tile_col;
        if (out_row < H && out_col < W) 
        {
            float val = 0.0f;
            for (int i = 0; i < FILTER_SIZE; ++i)
                for (int j = 0; j < FILTER_SIZE; ++j)
                    val += filter[i][j] * tile[threadIdx.y + i - RADIUS][threadIdx.x + j - RADIUS];
            output[out_row * W + out_col] = val;
        }
    }
}

__global__ void he_normal(float *filter_flat, unsigned int seed) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = FILTER_SIZE * FILTER_SIZE;
    if (idx < total) 
    {
        float fan_in = total;
        float std_dev = sqrtf(2.0f / fan_in);
        curandState state;
        curand_init(seed + idx, idx, 0, &state);
        filter_flat[idx] = curand_normal(&state) * std_dev;
    }
}

std::mutex print_mutex;

void prefetch_thread(std::vector<std::string> paths, MemoryQueue<ImageTask>& queue, size_t max_image_pixels) 
{
    int skipped = 0;
    for (const auto& path : paths) 
    {
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (!img.empty()) 
        {
            if (img.rows * img.cols <= max_image_pixels) 
            {
                queue.push({path, img, 0});
            } 
            else 
            {
                skipped++;
                std::lock_guard<std::mutex> lock(print_mutex);
                std::cerr << "Skipping oversized image: " << path << " (" << img.rows << "x" << img.cols << ")\n";
            }
        } 
        else 
        {
            std::lock_guard<std::mutex> lock(print_mutex);
            std::cerr << "Failed to read " << path << "\n";
        }
    }
    
    if (skipped > 0) 
    {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cout << "Skipped " << skipped << " oversized images\n";
    }
    
    queue.setFinished();
}

void gpu_worker(int gpu_id, MemoryQueue<ImageTask>& queue, const std::string& out_dir, size_t max_image_size) 
{
    GPUContext ctx(gpu_id, max_image_size);
    auto t_start = std::chrono::high_resolution_clock::now();

    unsigned int seed = time(NULL) + gpu_id * 1000;
    he_normal<<<1, FILTER_SIZE * FILTER_SIZE, 0, ctx.streams[0]>>>(ctx.d_filter, seed);
    cudaStreamSynchronize(ctx.streams[0]);

    int processed_count = 0;
    int stream_idx = 0;
    ImageTask task;

    while (queue.isFinished(task)) 
    {
        cudaStream_t stream = ctx.streams[stream_idx];
        stream_idx = (stream_idx + 1) % NUM_STREAMS;
        
        task.image.convertTo(task.image, CV_32F, 1.0 / 255.0);
        std::vector<cv::Mat> channels(3);
        cv::split(task.image, channels);

        int H = task.image.rows, W = task.image.cols;
        size_t size = H * W * sizeof(float);

        for (int c = 0; c < 3; ++c) 
        {
            memcpy(ctx.h_pinned_in[c], channels[c].ptr<float>(), size);
            cudaMemcpyAsync(ctx.d_in[c], ctx.h_pinned_in[c], size, cudaMemcpyHostToDevice, stream);
        }

        dim3 block(IN_TILE_DIM, IN_TILE_DIM);
        dim3 grid((W + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (H + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

        for (int c = 0; c < 3; ++c) 
        {
            conv2D<<<grid, block, 0, stream>>>(ctx.d_in[c], ctx.d_out[c], ctx.d_filter, H, W);
        }

        for (int c = 0; c < 3; ++c) 
        {
            cudaMemcpyAsync(ctx.h_pinned_out[c], ctx.d_out[c], size, cudaMemcpyDeviceToHost, stream);
        }

        cudaStreamSynchronize(stream);

        std::vector<cv::Mat> out_channels(3);
        for (int c = 0; c < 3; ++c) 
        {
            out_channels[c] = cv::Mat(H, W, CV_32F, ctx.h_pinned_out[c]).clone();
        }

        cv::Mat result;
        cv::merge(out_channels, result);
        result.convertTo(result, CV_8U, 255.0);

        std::string fname = std::filesystem::path(task.filename).filename().string();
        std::string out_path = out_dir + "/conv" + std::to_string(gpu_id) + "_" + fname;
        if (!cv::imwrite(out_path, result)) 
        {
            std::lock_guard<std::mutex> lock(print_mutex);
            std::cerr << "GPU " << gpu_id << ": Failed to write " << out_path << "\n";
        }

        processed_count++;
        if (processed_count % 10 == 0) 
        {
            std::lock_guard<std::mutex> lock(print_mutex);
            std::cout << "GPU " << gpu_id << ": Processed " << processed_count << " images (Queue size: " << queue.size() << ")\n";
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t_end - t_start;
    double avg = processed_count > 0 ? elapsed.count() / processed_count : 0.0;

    std::lock_guard<std::mutex> lock(print_mutex);
    std::cout << "\nGPU " << gpu_id << " processed " << processed_count << " images\n";
    std::cout << "GPU " << gpu_id << " total time: " << elapsed.count() << " seconds\n";
    std::cout << "GPU " << gpu_id << " average/image: " << avg << " seconds\n";
    std::cout << "GPU " << gpu_id << " throughput: " << (processed_count / elapsed.count()) << " images/sec\n";
}

int main(int argc, char** argv) 
{
    if (argc < 3) 
    {
        std::cerr << "Usage: ./conMultiGPU <input_dir> <output_dir> [max_image_megapixels]\n";
        return -1;
    }

    std::string input_dir = argv[1];
    std::string output_dir = argv[2];

    size_t max_megapixels = (argc > 3) ? std::stoull(argv[3]) : 20;
    size_t max_image_pixels = max_megapixels * 1000000;
    size_t max_image_size = max_image_pixels * sizeof(float);
    
    std::filesystem::create_directories(output_dir);

    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    std::cout << "Found " << num_gpus << " GPU(s)\n";
    std::cout << "Max image size: " << max_megapixels << " megapixels\n";

    if (num_gpus == 0) 
    {
        std::cerr << "No GPUs found\n";
        return -1;
    }

    std::vector<std::string> image_paths;
    for (const auto& entry : std::filesystem::directory_iterator(input_dir)) 
    {
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg")image_paths.push_back(entry.path().string());
    }

    std::cout << "Found " << image_paths.size() << " images to process\n";
    if (image_paths.empty()) return 0;

    MemoryQueue<ImageTask> prefetch_queue(4 * num_gpus);

    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::thread prefetcher(prefetch_thread, image_paths, std::ref(prefetch_queue), max_image_pixels);

    std::vector<std::thread> workers;
    for (int i = 0; i < num_gpus; ++i)
        workers.emplace_back(gpu_worker, i, std::ref(prefetch_queue), output_dir, max_image_size);

    prefetcher.join();
    for (auto& t : workers) t.join();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = end_time - start_time;
    
    std::cout << "\n=== OVERALL SUMMARY ===\n";
    std::cout << "Total processing time: " << total_elapsed.count() << " seconds\n";
    std::cout << "Overall throughput: " << (image_paths.size() / total_elapsed.count()) << " images/sec\n";
    
    return 0;
}