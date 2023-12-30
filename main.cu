#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "glad/glad.h"
#include <cuda/std/complex>
#include <array>
#include <iostream>

using namespace std;

struct Pixel
{
    unsigned char r, g, b, a{255};

    Pixel() : r{0}, g{0}, b{0} {}
    Pixel(unsigned char r, unsigned char g, unsigned char b) : r{r}, g{g}, b{b} {}

    Pixel operator*(double val)
    {
        return Pixel(r * val, g * val, b * val);
    }
};

template <typename T>
struct MandelDim
{
    T minReal, maxReal;
    T minImag, maxImag;
};

const unsigned iterMax = 2000;

array<Pixel, 16> colors = {
    Pixel{66, 30, 15},
    Pixel{25, 7, 26},
    Pixel{9, 1, 47},
    Pixel{4, 4, 73},
    Pixel{0, 7, 100},
    Pixel{12, 44, 138},
    Pixel{24, 82, 177},
    Pixel{57, 125, 209},
    Pixel{134, 181, 229},
    Pixel{211, 236, 248},
    Pixel{241, 233, 191},
    Pixel{248, 201, 95},
    Pixel{255, 170, 0},
    Pixel{204, 128, 0},
    Pixel{153, 87, 0},
    Pixel{106, 52, 3}};

template <typename T>
__global__ void computeMandelbrot(unsigned *iterations, unsigned width, unsigned height, MandelDim<T> mDim)
{
    using cuda::std::complex;

    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    T relY = y / static_cast<T>(height);
    T relX = x / static_cast<T>(width);

    T cr = mDim.minReal + relX * (mDim.maxReal - mDim.minReal);
    T ci = mDim.minImag + relY * (mDim.maxImag - mDim.minImag);

    complex<T> c = {cr, ci};
    complex<T> z = {0, 0};

    unsigned i{0};
    for (; i < iterMax; ++i)
    {
        z = z * z + c;
        if (z.real() * z.real() + z.imag() * z.imag() > 1000)
            break;
    }

    iterations[y * width + x] = i;
}

int main(int argc, char *argv[])
{
    if (!glfwInit())
        return 1;

    GLFWwindow *window = glfwCreateWindow(1024, 1024, "Mandelbrot", NULL, NULL);
    if (!window)
        return 1;

    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
        return 1;

    glfwSetWindowSizeCallback(window, [](GLFWwindow *win, int width, int height)
                              { glViewport(0, 0, width, height); });
    glViewport(0, 0, 1024, 1024);

    unsigned frames = 0;
    double prevTime = glfwGetTime();
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        unsigned *itDev;
        cudaMalloc(&itDev, width * height * sizeof(unsigned));

        dim3 blockDim = {32, 32};
        // Do more computations if height or width don't divide evenly, kernel checks if current thread is within image
        dim3 gridDim = {ceil(static_cast<float>(width) / blockDim.x), ceil(static_cast<float>(height) / blockDim.y)};
        MandelDim<double> mDim = {-2, 1, -1, 1};
        computeMandelbrot<<<gridDim, blockDim>>>(itDev, width, height, mDim);
        cudaDeviceSynchronize();

        unsigned *itHost = new unsigned[width * height];
        cudaMemcpy(itHost, itDev, width * height * sizeof(unsigned), cudaMemcpyDeviceToHost);

        double time = glfwGetTime();
        double sinTime = sin(time);

        Pixel *image = new Pixel[width * height];
        for (size_t y{}; y < height; ++y)
        {
            for (size_t x{}; x < width; ++x)
            {
                unsigned it = itHost[y * width + x];
                image[y * width + x] = colors[it % 16] * sinTime;
            }
        }

        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, image);
        glfwSwapBuffers(window);

        frames++;
        double delta = time - prevTime;
        if (delta >= 1.)
        {
            double fps = frames / delta;
            glfwSetWindowTitle(window, ("Mandelbrot FPS: " + to_string(fps)).c_str());
            prevTime = time;
            frames = 0;
        }
    }

    // TODO: Use RAII
    glfwDestroyWindow(window);
    glfwTerminate();
}