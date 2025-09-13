#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>
#include <complex>
#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <omp.h>


const double PI = 3.141592653589793238460;
typedef std::complex<double> Complex;
typedef std::vector<Complex> CArray;

// ---------------- FFT ----------------
void bitReverse(CArray &a) {
    size_t N = a.size();
    size_t j = 0;
    for (size_t i = 0; i < N; ++i) {
        if (i < j) std::swap(a[i], a[j]);
        size_t m = N >> 1;
        while (m >= 1 && j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
}

void fft(CArray &a) {
    size_t N = a.size();
    bitReverse(a);

    for (size_t len = 2; len <= N; len <<= 1) {
        double angle = -2 * PI / len;
        Complex wlen(cos(angle), sin(angle));

        // Paralelizacija po blokovima
        #pragma omp parallel for
        for (size_t i = 0; i < N; i += len) {
            Complex w(1);
            for (size_t j = 0; j < len/2; ++j) {
                Complex u = a[i + j];
                Complex v = a[i + j + len/2] * w;
                a[i + j] = u + v;
                a[i + j + len/2] = u - v;
                w *= wlen;
            }
        }
    }
}


// ---------------- Glavni program ----------------
int main() {
    // Učitavanje audio fajla
    sf::SoundBuffer buffer;
    if (!buffer.loadFromFile("input2.wav")) {
        std::cerr << "Greska: ne mogu ucitati audio.wav!" << std::endl;
        return -1;
    }

    typedef short SampleType;
const SampleType* samples = buffer.getSamples();
    std::size_t sampleCount = buffer.getSampleCount();

    // FFT zahteva da broj uzoraka bude potencija od 2
    std::size_t N = 1;
    while (N < sampleCount) N <<= 1;

    CArray data(N);
    for (std::size_t i = 0; i < sampleCount; i++) {
        data[i] = static_cast<double>(samples[i]) / 32768.0; // normalizacija
    }
    for (std::size_t i = sampleCount; i < N; i++) {
        data[i] = 0.0; // zero padding
    }

    // Izmeri vreme FFT-a
    auto start = std::chrono::high_resolution_clock::now();
    fft(data);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "FFT trajala: " << elapsed.count() << " sekundi" << std::endl;

    // ---------------- SFML prozor ----------------
    sf::RenderWindow window(sf::VideoMode(sf::Vector2u{1600, 400}), "FFT Magnitude");

    while (window.isOpen()) {

        while (auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                window.close();
        }

        window.clear(sf::Color::Black);

        // Prikaz prvih 256 frekvencija
        size_t displayN = 256;
        size_t step = N / displayN;
        float yScale = 50.0f;

        for (size_t i = 1; i < displayN; ++i) {
            size_t idx1 = (i - 1) * step;
            size_t idx2 = i * step;

            float x1 = static_cast<float>(i - 1) * (1600.0f / displayN);
            float x2 = static_cast<float>(i) * (1600.0f / displayN);

            float y1 = 400.0f - std::log10(1 + std::abs(data[idx1])) * yScale;
            float y2 = 400.0f - std::log10(1 + std::abs(data[idx2])) * yScale;

            sf::Vertex line[2];
            line[0].position = sf::Vector2f(x1, y1);
            line[0].color = sf::Color::Green;
            line[1].position = sf::Vector2f(x2, y2);
            line[1].color = sf::Color::Green;

            window.draw(line, 2, sf::PrimitiveType::Lines);
        }

        window.display();
    }

    return 0;
}
