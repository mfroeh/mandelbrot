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
    Pixel(unsigned char r, unsigned char g, unsigned char b, unsigned char a) : r{r}, g{g}, b{b}, a{a} {}

    Pixel operator*(double val)
    {
        return Pixel(r * val, g * val, b * val, a * val);
    }
};

template <typename T>
struct MandelDim
{
    T minReal, maxReal;
    T minImag, maxImag;

    T width() { return maxReal - minReal; }
    T height() { return maxImag - minImag; }
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

bool dragStart{false};
bool dragStop{false};
int x{}, y{};
int xstart{}, ystart{};
int xstop{}, ystop{};

void posHandler(GLFWwindow *win, double xpos, double ypos)
{
    int width, height;
    glfwGetFramebufferSize(win, &width, &height);
    x = max(0, min(static_cast<int>(xpos), width));
    y = max(0, min(static_cast<int>(height - ypos), height));
}

void buttonHandler(GLFWwindow *window, int button, int action, int mods)
{
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE)
    {
        xstop = x;
        ystop = y;
        dragStop = true;
    }
    else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        dragStart = true;
        dragStop = false;
        xstart = x;
        ystart = y;
    }
}

void draw(unsigned *itHost, Pixel *image, int width, int height, MandelDim<double> mDim)
{
    unsigned *itDev;
    cudaMalloc(&itDev, width * height * sizeof(unsigned));

    dim3 blockDim = {32, 32};
    // Do more computations if height or width don't divide evenly, kernel checks if current thread is within image
    dim3 gridDim = {ceil(static_cast<float>(width) / blockDim.x), ceil(static_cast<float>(height) / blockDim.y)};
    computeMandelbrot<<<gridDim, blockDim>>>(itDev, width, height, mDim);
    cudaDeviceSynchronize();
    cudaMemcpy(itHost, itDev, width * height * sizeof(unsigned), cudaMemcpyDeviceToHost);

    for (size_t y{}; y < height; ++y)
    {
        for (size_t x{}; x < width; ++x)
        {
            unsigned it = itHost[y * width + x];
            image[y * width + x] = colors[it % 16];
        }
    }
}

unsigned *itHost;
Pixel *image;
MandelDim<double> mDim{-2, 1, -1, 1};

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
                              { 
                                glViewport(0, 0, width, height); 
                              delete itHost;
                              delete image;
                              itHost = new unsigned[width * height];
                              image = new Pixel[width * height]; 
                              draw(itHost, image, width, height, mDim); });
    glfwSetCursorPosCallback(window, posHandler);
    glfwSetMouseButtonCallback(window, buttonHandler);

    glViewport(0, 0, 1024, 1024);

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    itHost = new unsigned[width * height];
    image = new Pixel[width * height];
    draw(itHost, image, width, height, mDim);

    // xmin, xmax, ymin, ymax

    unsigned frames = 0;
    double prevTime = glfwGetTime();
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        glfwGetFramebufferSize(window, &width, &height);

        for (size_t y{}; y < height; ++y)
        {
            for (size_t x{}; x < width; ++x)
            {
                unsigned it = itHost[y * width + x];
                image[y * width + x] = colors[it % 16];
            }
        }

        double time = glfwGetTime();
        double sinTime = sin(time);

        // zoom
        if (dragStart && !dragStop)
        {
            for (size_t y{}; y < height; ++y)
            {
                for (size_t x{}; x < width; ++x)
                {
                    unsigned it = itHost[y * width + x];
                    image[y * width + x] = colors[it % 16];
                }
            }

            int xmin{min(xstart, x)};
            int ymin{min(ystart, y)};
            int xmax{max(xstart, x)};
            int ymax{max(ystart, y)};

            // top + bottom
            for (int x{xmin}; x < xmax; ++x)
            {
                image[ymin * width + x] = Pixel{0, 0, 255, 1};
                image[ymax * width + x] = Pixel{0, 0, 255, 1};
            }

            // left + right
            for (int y{ymin}; y < ymax; ++y)
            {
                image[y * width + xmin] = Pixel{0, 0, 255, 1};
                image[y * width + xmax] = Pixel{0, 0, 255, 1};
            }
        }
        else if (dragStart && dragStop)
        {
            dragStop = false;
            dragStart = false;

            int xmin{min(xstart, xstop)};
            int ymin{min(ystart, ystop)};
            int xmax{max(xstart, xstop)};
            int ymax{max(ystart, ystop)};

            double xfac{mDim.width() / width};
            double yfac{mDim.height() / height};

            mDim.minReal += xfac * xmin;
            mDim.maxReal = mDim.minReal + xfac * (xmax - xmin);
            mDim.minImag += yfac * ymin;
            mDim.maxImag = mDim.minImag + yfac * (ymax - ymin);
            draw(itHost, image, width, height, mDim);
        }

        // actually draw
        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, image);
        glfwSwapBuffers(window);

        // fps
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
