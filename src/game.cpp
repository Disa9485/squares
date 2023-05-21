#include <SFML/Graphics.hpp>
#include <SFML/System/Clock.hpp>
#include <SFML/Window/Event.hpp>
#include <TGUI/TGUI.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
//
//using json = nlohmann::json;
//
//enum class GameState {
//    MENU,
//    PLAY,
//    LOAD,
//    SETTINGS,
//    ERROR
//};
//
//// Default config values
//json default_config = {
//        {"resolution_option", 0},
//        {"fullscreen", false},
//        {"fps", 60},
//};
//
//// Resolution options
//std::vector<sf::Vector2u> RESOLUTION_OPTIONS = {
//        sf::Vector2u(1280, 720),
//        sf::Vector2u(1600, 900),
//        sf::Vector2u(1920, 1080),
//        sf::Vector2u(2560, 1440),
//        sf::Vector2u(3840, 2160)
//};
//
//// The current game state
//GameState current_state = GameState::MENU;
//std::string message;
//
//// Font
//sf::Font PIXEL_FONT;
//
//// Colors
//sf::Color BACKGROUND_COLOR(87, 135, 207);
//sf::Color BUTTON_COLOR(60, 129, 207);
//sf::Color TEXT_COLOR(197, 209, 237);
//
//// Load the configuration from the config.json file
//json load_config() {
//    std::ifstream configFile("config.json");
//    if (!configFile) {
//        std::cout << "ERROR: Config does not exist, using defaults." << std::endl;
//        return nullptr;
//    }
//
//    json new_config;
//    configFile >> new_config;
//    return new_config;
//}
//
//// Save the configuration to the config.json file
//void save_config(const json& new_config) {
//    std::ofstream configFile("config.json");
//    configFile << new_config.dump(4);
//}
//
//void menu(sf::RenderWindow& window, tgui::GuiSFML& gui) {
//    message = "Created by Collin Miller";
//    //auto message_time = clock.getElapsedTime().asMilliseconds() + 5000;
//
//    // Instead of creating Rectangles, we create buttons directly
//    auto new_game_button = tgui::Button::create();
//    new_game_button->setSize({"50%", "12.5%"});
//    new_game_button->setPosition({"25%", "25%"});
//    new_game_button->setText("New Game"); // Set the text of the button
//    new_game_button->onPress([](){ current_state = GameState::PLAY; message = ""; });
//    gui.add(new_game_button);
//
//    auto load_game_button = tgui::Button::create();
//    load_game_button->setSize({"50%", "12.5%"});
//    load_game_button->setPosition({"25%", "40%"});
//    load_game_button->setText("Load Game");
//    load_game_button->onPress([](){ current_state = GameState::LOAD; message = ""; });
//    gui.add(load_game_button);
//
//    auto settings_button = tgui::Button::create();
//    settings_button->setSize({"50%", "12.5%"});
//    settings_button->setPosition({"25%", "55%"});
//    settings_button->setText("Settings");
//    settings_button->onPress([](){ current_state = GameState::SETTINGS; message = ""; });
//    gui.add(settings_button);
//
//    auto exit_button = tgui::Button::create();
//    exit_button->setSize({"50%", "12.5%"});
//    exit_button->setPosition({"25%", "70%"});
//    exit_button->setText("Exit");
//    exit_button->setTextSize(48);
//    exit_button->onPress([&](){ window.close(); });
//    gui.add(exit_button);
//
//    // Handle events
//    sf::Event event;
//    while (window.pollEvent(event)) {
//        if (event.type == sf::Event::Closed)
//            window.close();
//
//        gui.handleEvent(event);
//    }
//
//    window.clear();
//    gui.draw();
//    window.display();
//}
//
//int main() {
//
//    // Load the configuration from the config.json file
//    std::ifstream configFile("config.json");
//    json config = load_config();
//    if (config.is_null()) { // Use default values
//        config = default_config;
//        save_config(default_config);
//    }
//
//    sf::RenderWindow window;
//    if (config["fullscreen"].get<bool>()) {
//        window.create(sf::VideoMode::getDesktopMode(), "Squares", sf::Style::Fullscreen);
//    } else {
//        window.create(sf::VideoMode::getDesktopMode(), "Squares", sf::Style::Default);
//    }
//
//    tgui::GuiSFML gui(window);
//    sf::Clock clock;
//
//    while (window.isOpen()) {
//        sf::Event event;
//        while (window.pollEvent(event)) {
//            if (event.type == sf::Event::Closed) {
//                window.close();
//            }
//        }
//
//        switch (current_state) {
//            case GameState::MENU:
//                menu(window, gui);
//                break;
//            case GameState::PLAY:
//                //play();
//                break;
//            case GameState::LOAD:
//                //load();
//                break;
//            case GameState::SETTINGS:
//                //settings();
//                break;
//            case GameState::ERROR:
//                //error();
//                break;
//        }
//
//        // Cap the framerate
//        sf::sleep(sf::seconds(1.f / config["fps"].get<int8_t>()));
//    }
//
//    return 0;
//}

// Imports
#include <iostream>
#include <vector>
#include <atomic>
#include "FastNoiseLite.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <thread>
#include <random>
#include <cmath>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <future>
#include <set>
#include <unordered_map>

// Map dimensions
const int mapWidth = 1000;
const int mapHeight = 1000;
const int channels = 3;  // RGB channels

// Generate noise heightmap parameters
const float frequency = 0.25; // Smaller = larger lands, bigger = smaller lands
const int octaves = 12;
const float lacunarity = 2.0f;
const float persistence = 0.5f;
const float distanceFromCenter = 0.8f;

// Generate noise heightmap function
void generateNoiseHeightMap(std::vector<float>& data) {
    // Indicate start
    std::cout << "Generating Simplex Noise Height Map..." << std::endl;
    std::atomic<int> pixelsProcessed(0);
    auto start = std::chrono::high_resolution_clock::now();

    // Generate a random seed
    int seed = rand();

    // Get threads
    unsigned int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(numThreads);

    // Calculate center of the image
    float centerX = mapWidth / 2.0f;
    float centerY = mapHeight / 2.0f;

    // Initialize Simplex noise generator
    FastNoiseLite noiseGenerator;
    noiseGenerator.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    noiseGenerator.SetSeed(seed);

    // Generate noise height map thread
    auto generateNoiseHeightMapPixels = [&](int startY, int endY) {
        for (int y = startY; y < endY; ++y) {
            for (int x = 0; x < mapWidth; ++x) {
                float noise = 0.0f;
                float amplitude = 1.0f;
                float maxAmplitude = 0.0f;
                float freq = frequency;

                for (int i = 0; i < octaves; ++i) {
                    noise += amplitude * noiseGenerator.GetNoise((float)x * freq, (float)y * freq);
                    maxAmplitude += amplitude;
                    amplitude *= persistence;
                    freq *= lacunarity;
                }

                noise /= maxAmplitude;
                noise = (noise + 1) / 2.0f;  // map noise from 0.0 - 1.0

                float dx = centerX - x;
                float dy = centerY - y;
                float distance = std::sqrt(dx * dx + dy * dy) / (((mapWidth + mapHeight) / 2) / 2.0f);
                distance = 1 / (1 + std::exp(-10 * (distance - distanceFromCenter)));
                noise = (1 - distance) * noise;
                data[y * mapWidth + x] = noise;

                // Print progress
                pixelsProcessed++;
                if ((pixelsProcessed % ((mapWidth * mapHeight) / 10)) == 0) {  // Update every 1000 pixels to avoid slowing down the computation
                    float progress = (float)pixelsProcessed / (mapWidth * mapHeight);
                    std::cout << "\rMap Generation Progress: " << progress * 100 << "%" << std::endl;
                }
            }
        }
    };

    // Start threads
    int sliceHeight = mapHeight / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int startY = i * sliceHeight;
        int endY = (i == numThreads - 1) ? mapHeight : (i + 1) * sliceHeight;  // ensure last slice goes to end
        threads[i] = std::thread(generateNoiseHeightMapPixels, startY, endY);
    }

    // Wait for threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Indicate stop
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Map Generation Complete: " << elapsed.count() << "s Elapsed, " << pixelsProcessed << " Pixels Processed" << std::endl;
}

// Global variables
int riverCount = (mapWidth * mapHeight) / 5000;
float maxRiverSpawnHeight = 0.75f; // Default: 0.85
float minRiverSpawnHeight = 0.6f; // Default: 0.7
float minRiverDespawnHeight = 0.30f; // Default: 0.30
int randomPointSpacing = 10; // Default: 10
float maxDeviation = 15.0; // Default: 15.0
float minDeviation = 3.0; // Default: 3.0
int minSearchRiverPointDistance = 10; // Default: 10
int riverStraightnessThreshold = 15; // Default: 6
const int minRiverLength = 50; // Default: 10
const int intersectionRange = 10; // Default: 10
float startRadius = 1.0f; // Default: 1.0
float maxRadius = 5.0f; // Default: 5.0
float sizeScaleRate = 1.5f; // Default: 1.5
const int maxRiverRadiusLength = 300; // Default: 300
float riverMinDepth = 0.39f; // Default: 0.39
float riverMaxDepth = 0.37f; // Default: 0.37
float terrainMinDepth = 0.40f; // Default: 0.40
float terrainDistortion = 0.025f; // Default: 0.025
float startTerrainCarveRadius = 1.5f; // Default: 1.5
float maxTerrainCarveRadius = 8.0f; // Default: 8.0
float sizeTerrainCarveScaleRate = 1.5f; // Default: 1.25

// River path struct
struct RiverPath {
    std::vector<std::pair<int, int>> path;  // Store the path of the river as (x,y) pairs
};

// Initialize rivers
void initializeRivers(std::vector<RiverPath>& rivers, std::vector<float>& heightMap, std::mt19937& gen, std::uniform_int_distribution<>& distr) {
    for (auto& river : rivers) {
        int index;
        do {
            index = distr(gen);
        } while (heightMap[index] < minRiverSpawnHeight || heightMap[index] > maxRiverSpawnHeight);

        river.path.push_back({index / mapHeight, index % mapWidth});
    }
}

// Perform circular search to find nearest lower point from given point
std::pair<int, int> circularSearch(int x, int y, const std::vector<float>& heightMap) {
    std::pair<int, int> lowestPoint = {x, y};
    float lowestHeight = heightMap[x * mapHeight + y];
    bool found = false;
    int radius = minSearchRiverPointDistance;
    while (!found) {
        for (int dx = -radius; dx <= radius; dx++) {
            for (int dy = -radius; dy <= radius; dy++) {
                if (dx * dx + dy * dy >= radius * radius) { // Check if the point is within the circular distance
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx >= 0 && nx < mapWidth && ny >= 0 && ny < mapHeight) {
                        int index = nx * mapHeight + ny;
                        if (heightMap[index] < lowestHeight) {
                            lowestPoint = {nx, ny};
                            lowestHeight = heightMap[index];
                            found = true;
                        }
                    }
                }
            }
        }
        radius++;
    }
    return lowestPoint;
}

// Create river paths
void createRiverPaths(std::vector<RiverPath>& rivers, std::vector<float>& heightMap) {
    for (auto& river : rivers) {
        while (true) {
            // Get the last point in the path
            auto [x, y] = river.path.back();
            int index = x * mapHeight + y;

            // If the current point is below minRiverDespawnHeight, break
            if (heightMap[index] < minRiverDespawnHeight)
                break;

            // Search for the lower nearest point using the circular search
            auto lowestPoint = circularSearch(x, y, heightMap);

            // If no point is found (should not happen as the search expands indefinitely), stop carving the path
            if (lowestPoint.first == x && lowestPoint.second == y)
                break;

            // Add the new point to the path
            river.path.push_back(lowestPoint);
        }
    }
}

// Get Euclidean distance between two points
float getDistance(const std::pair<int, int>& p1, const std::pair<int, int>& p2) {
    return std::sqrt(std::pow(p2.first - p1.first, 2) + std::pow(p2.second - p1.second, 2));
}

// Generate a random point between two points
std::pair<int, int> generateNewPoint(std::pair<int, int> point1, std::pair<int, int> point2, float t, std::mt19937& gen) {
    std::uniform_real_distribution<> distr(minDeviation, maxDeviation);

    // Calculate the midpoint between point1 and point2
    float xMid = (1 - t) * point1.first + t * point2.first;
    float yMid = (1 - t) * point1.second + t * point2.second;

    // Generate a random orthogonal displacement
    float theta = atan2(point2.second - point1.second, point2.first - point1.first) + M_PI_2;
    float randFactor = distr(gen);
    float dx = randFactor * cos(theta);
    float dy = randFactor * sin(theta);

    return {static_cast<int>(xMid + dx), static_cast<int>(yMid + dy)};
}

// Generate random points along paths
void generateRandomPoints(std::vector<RiverPath>& rivers, std::mt19937& gen) {
    for (auto& river : rivers) {
        // New path with added points
        std::vector<std::pair<int, int>> newPath;
        newPath.push_back(river.path[0]);

        for (size_t i = 0; i < river.path.size() - 1; i++) {
            // Calculate distance between two points
            float distance = getDistance(river.path[i], river.path[i+1]);

            // Adjusting the calculation of numNewPoints
            int numNewPoints = std::max(2, static_cast<int>(std::ceil(distance / randomPointSpacing)));

            // Generate and add new points
            for (int j = 0; j < numNewPoints; j++) {
                float t = static_cast<float>(j) / numNewPoints;
                auto newPoint = generateNewPoint(river.path[i], river.path[i+1], t, gen);
                newPath.push_back(newPoint);
            }

            newPath.push_back(river.path[i+1]);
        }

        // Replace the old path with the new path
        river.path = newPath;
    }
}

// Recursively find b-spline curve points
float BSplineBasis(int i, int degree, float t, const std::vector<float>& knots) {
    if (degree == 0) {
        return (knots[i] <= t && t < knots[i+1]) ? 1.0f : 0.0f;
    } else {
        float coeff1 = (t - knots[i]) / (knots[i+degree] - knots[i]);
        float coeff2 = (knots[i+degree+1] - t) / (knots[i+degree+1] - knots[i+1]);

        return coeff1 * BSplineBasis(i, degree-1, t, knots) + coeff2 * BSplineBasis(i+1, degree-1, t, knots);
    }
}

// Find b-spline curve points
std::pair<int, int> BSpline(float t, const std::vector<std::pair<int, int>>& points, const std::vector<float>& knots) {
    std::pair<float, float> point(0.0f, 0.0f);
    int degree = 3; // Degree of the B-spline

    for (size_t i = 0; i < points.size(); i++) {
        float basis = BSplineBasis(i, degree, t, knots);
        point.first += basis * points[i].first;
        point.second += basis * points[i].second;
    }

    return {static_cast<int>(point.first), static_cast<int>(point.second)};
}

// Get the points on a line between two points
void BresenhamLine(int x1, int y1, int x2, int y2, std::vector<std::pair<int, int>>& line) {
    int dx = abs(x2 - x1), sx = x1 < x2 ? 1 : -1;
    int dy = -abs(y2 - y1), sy = y1 < y2 ? 1 : -1;
    int err = dx + dy, e2;

    while (true) {
        line.push_back({x1, y1});
        if (x1 == x2 && y1 == y2) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x1 += sx; }
        if (e2 <= dx) { err += dx; y1 += sy; }
    }
}

// Redraw paths along B-spline curves
void redrawPaths(std::vector<RiverPath>& rivers) {
    for (auto& river : rivers) {
        // Generate uniform knot vector
        std::vector<float> knots(river.path.size() + 4);
        for (size_t i = 0; i < knots.size(); i++) {
            knots[i] = static_cast<float>(i) / (knots.size() - 1);
        }

        // New path for B-spline
        std::vector<std::pair<int, int>> splinePath;
        std::set<std::pair<int, int>> addedPoints; // used to check for duplicates

        // Generate the B-spline path
        for (float t = knots[3]; t <= knots[knots.size()-4]; t += 0.01) {
            std::pair<int, int> B = BSpline(t, river.path, knots);

            // Ensure that the point hasn't already been added
            if (addedPoints.count(B) == 0) {
                splinePath.push_back(B);
                addedPoints.insert(B);
            }
        }

        // Interpolate points along each segment of the B-spline path
        std::vector<std::pair<int, int>> newPath;
        for (size_t i = 1; i < splinePath.size(); i++) {
            BresenhamLine(splinePath[i-1].first, splinePath[i-1].second, splinePath[i].first, splinePath[i].second, newPath);
        }

        // Replace the old path with the new B-spline path
        river.path = newPath;
    }
}

// Pair hash struct
struct pair_hash {
    inline std::size_t operator()(const std::pair<int, int> & v) const {
        return v.first * 31 + v.second;
    }
};

// Remove loops and duplicate points along river path
void removeLoopsAndDuplicates(std::vector<RiverPath>& rivers) {
    // Check for self intersections and duplicate points
    for (auto& river : rivers) {
        std::unordered_map<std::pair<int, int>, int, pair_hash> pointIndices;

        for (int i = 0; i < river.path.size(); ++i) {
            if (pointIndices.count(river.path[i]) > 0) {
                // Intersection found: keep only the intersection point
                if (i - pointIndices[river.path[i]] > 1) {
                    river.path.erase(river.path.begin() + pointIndices[river.path[i]] + 1, river.path.begin() + i);
                }
            }
            pointIndices[river.path[i]] = i;
        }
    }
}

// Remove rivers that are considered too straight
void removeStraightRivers(std::vector<RiverPath>& rivers) {
    std::vector<RiverPath> filteredRivers;
    for (const auto& river : rivers) {
        // Initialize direction with a large number for the first comparison
        std::pair<int, int> lastDirection{INT_MAX, INT_MAX};

        int straightCount = 0;

        for (size_t i = 1; i < river.path.size(); ++i) {
            // Calculate direction from current point to previous point
            std::pair<int, int> currentDirection{
                    river.path[i].first - river.path[i - 1].first,
                    river.path[i].second - river.path[i - 1].second};

            // If the current direction is the same as the last one, increment the counter
            if (currentDirection == lastDirection) {
                ++straightCount;
                // If the counter exceeds the threshold, this river is too straight
                if (straightCount >= riverStraightnessThreshold) {
                    break;
                }
            } else {
                // If the current direction is different, reset the counter and update lastDirection
                straightCount = 1;
                lastDirection = currentDirection;
            }
        }

        // If straightCount is below the threshold, the river is not too straight
        if (straightCount < riverStraightnessThreshold) {
            filteredRivers.push_back(river);
        }
    }
    rivers = std::move(filteredRivers);
}

// Combine rivers which intersect
void combineIntersectingRivers(std::vector<RiverPath>& rivers) {
    for (size_t i = 0; i < rivers.size(); ++i) {
        for (size_t j = i + 1; j < rivers.size(); ++j) {
            auto& river1 = rivers[i];
            auto& river2 = rivers[j];

            bool intersectionFound = false;
            int intersectionIndexRiver1 = 0;
            int intersectionIndexRiver2 = 0;

            for (int k = 0; k < river1.path.size(); ++k) {
                for (int l = 0; l < river2.path.size(); ++l) {
                    if (getDistance(river1.path[k], river2.path[l]) <= intersectionRange) {
                        // Intersection found
                        intersectionFound = true;
                        intersectionIndexRiver1 = k;
                        intersectionIndexRiver2 = l;
                        break;
                    }
                }
                if (intersectionFound) {
                    // Create a new vector for the intersecting river
                    std::vector<std::pair<int, int>> newPath;

                    // Add points before the intersection
                    newPath.insert(newPath.end(), river2.path.begin(), river2.path.begin() + intersectionIndexRiver2);

                    // Draw line from river2's intersection point to 10 points further down river1
                    int nextPointIndex = std::min(intersectionIndexRiver1 + 10, (int) river1.path.size() - 1);
                    std::vector<std::pair<int, int>> line;
                    BresenhamLine(river2.path[intersectionIndexRiver2].first, river2.path[intersectionIndexRiver2].second,
                                  river1.path[nextPointIndex].first, river1.path[nextPointIndex].second, line);

                    // Add points from the line to the intersecting river
                    newPath.insert(newPath.end(), line.begin(), line.end());

                    // Replace the path of river2 with the new path
                    river2.path = newPath;

                    break;
                }
            }
        }
    }
}

// Remove rivers which are considered too short
void checkRiverLength(std::vector<RiverPath>& rivers) {
    rivers.erase(std::remove_if(rivers.begin(), rivers.end(), [&](const RiverPath& river){
        return river.path.size() < minRiverLength;
    }), rivers.end());
}

// Carve rivers into the heightmap
void carveRivers(std::vector<RiverPath>& rivers, std::vector<float>& heightMap) {
    for (auto& river : rivers) {
        int riverPointCount = river.path.size();
        for (int i = 0; i < riverPointCount; i++) {
            auto& point = river.path[i];

            float t = std::min(1.0f, static_cast<float>(i) / maxRiverRadiusLength);  // normalize the current river point index
            float currentRadius = startRadius + (maxRadius - startRadius) * std::pow(t, sizeScaleRate);  // interpolate the radius
            int roundedRadius = static_cast<int>(std::round(currentRadius));
            float currentTerrainCarveRadius = startTerrainCarveRadius + (maxTerrainCarveRadius - startTerrainCarveRadius) * std::pow(t, sizeTerrainCarveScaleRate);  // interpolate the terrain carving radius
            int roundedTerrainCarveRadius = static_cast<int>(std::round(currentTerrainCarveRadius));

            // Terrain carving
            for (int y = std::max(0, point.first - roundedTerrainCarveRadius); y < std::min(mapHeight, point.first + roundedTerrainCarveRadius); y++) {
                for (int x = std::max(0, point.second - roundedTerrainCarveRadius); x < std::min(mapWidth, point.second + roundedTerrainCarveRadius); x++) {
                    float dx = point.first - y;
                    float dy = point.second - x;
                    float distance = std::sqrt(dx * dx + dy * dy);

                    if (distance > currentRadius && distance <= currentTerrainCarveRadius) {
                        bool insideRiver = false;
                        for (auto& riverPoint : river.path) {
                            float dxRiver = riverPoint.first - y;
                            float dyRiver = riverPoint.second - x;
                            float distanceRiver = std::sqrt(dxRiver * dxRiver + dyRiver * dyRiver);
                            if (distanceRiver <= currentRadius) {
                                insideRiver = true;
                                break;
                            }
                        }

                        if (!insideRiver) {
                            float distortion = (currentTerrainCarveRadius - distance) / (currentTerrainCarveRadius - currentRadius) * terrainDistortion;  // interpolate the terrain distortion
                            if (heightMap[y * mapWidth + x] > terrainMinDepth) {
                                heightMap[y * mapWidth + x] = std::max(heightMap[y * mapWidth + x] - distortion, terrainMinDepth);  // subtract distortion and prevent it from going below riverMinDepth
                            }
                        }
                    }
                }
            }

            // River carving
            for (int y = std::max(0, point.first - roundedRadius); y < std::min(mapHeight, point.first + roundedRadius); y++) {
                for (int x = std::max(0, point.second - roundedRadius); x < std::min(mapWidth, point.second + roundedRadius); x++) {
                    float dx = point.first - y;
                    float dy = point.second - x;
                    float distance = std::sqrt(dx * dx + dy * dy);

                    if (distance <= currentRadius) {
                        float normalizedDistance = distance / currentRadius;  // distance normalized to [0, 1], 0 being the center, 1 being the edge
                        float newHeight = riverMaxDepth * (1 - normalizedDistance) + riverMinDepth * normalizedDistance;  // interpolate between riverMaxDepth and riverMinDepth
                        if (heightMap[y * mapWidth + x] > newHeight) { // This is to prevent heights which are lower than the riverMinDepth from being effected
                            heightMap[y * mapWidth + x] = newHeight;
                        }
                    }
                }
            }
        }
    }
}

// Generate rivers in heightmap function
void generateRiversInHeightMap(std::vector<float>& heightMap) {
    std::cout << "Initializing Rivers..." << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, mapWidth * mapHeight - 1);
    std::vector<RiverPath> rivers(riverCount);
    initializeRivers(rivers, heightMap, gen, distr);
    std::cout << rivers.size() << std::endl;

    std::cout << "Creating river paths..." << std::endl;
    createRiverPaths(rivers, heightMap);
    std::cout << rivers.size() << std::endl;

    std::cout << "Generating random points along paths..." << std::endl;
    generateRandomPoints(rivers, gen);
    std::cout << rivers.size() << std::endl;

    std::cout << "Redrawing paths along B-spline curves..." << std::endl;
    redrawPaths(rivers);
    std::cout << rivers.size() << std::endl;

    std::cout << "Removing loops and duplicate points in rivers..." << std::endl;
    removeLoopsAndDuplicates(rivers);
    std::cout << rivers.size() << std::endl;

    std::cout << "Removing straight rivers..." << std::endl;
    removeStraightRivers(rivers);
    std::cout << rivers.size() << std::endl;

    std::cout << "Combining rivers that intersect..." << std::endl;
    combineIntersectingRivers(rivers);
    std::cout << rivers.size() << std::endl;

    std::cout << "Removing short rivers..." << std::endl;
    checkRiverLength(rivers);
    std::cout << rivers.size() << std::endl;

    std::cout << "Carving heightmap..." << std::endl;
    carveRivers(rivers, heightMap);
    std::cout << rivers.size() << std::endl;
}

// Erode heightmap parameters
const int numDrops = mapWidth * mapHeight / 3;
const float minDropSpawnHeight = 0.5; // Default: 0.5
const int erosionRadius = 3; // Default: 3
const float inertia = 0.05f; // Default: 0.05
const float sedimentCapacityFactor = 4.0f; // Default: 4.0
const float minSedimentCapacity = 0.01f; // Default: 0.01
const float erodeSpeed = 0.3f; // Default: 0.3
const float depositSpeed = 0.3f; // Default: 0.3
const float evaporateSpeed = 0.01f; // Default: 0.01
const float gravity = 4.0f; // Default: 4.0
const int maxDropLifeTime = 100; // Default: 30
const float minDropDespawnHeight = 0.4f; // Default: 0.35;
const float initialWaterVolume = 1.0f; // Default: 1.0
const float minWaterVolume = 0.01f; // Default 0.01
const float initialSpeed = 1.0f; // Default: 1.0

// Initialize erosion brush function
void initializeErosionBrush(std::vector<std::vector<int>>& erosionBrushIndices, std::vector<std::vector<float>>& erosionBrushWeights) {

    // Indicate start
    std::cout << "Initializing Erosion Brush..." << std::endl;
    std::atomic<int> pixelsProcessed(0);
    auto start = std::chrono::high_resolution_clock::now();

    // Resize the erosion brush vectors according to map size
    erosionBrushIndices.resize(mapWidth * mapHeight);
    erosionBrushWeights.resize(mapWidth * mapHeight);

    // Initialize offset and weight vectors
    std::vector<int> xOffsets(erosionRadius * erosionRadius * 4);
    std::vector<int> yOffsets(erosionRadius * erosionRadius * 4);
    std::vector<float> weights(erosionRadius * erosionRadius * 4);

    // Get threads
    unsigned int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(numThreads);
    int tasksPerThread = erosionBrushIndices.size() / numThreads;

    // Initialize erosion brush thread
    auto initializeErosionBrushPixels = [&erosionBrushIndices, &erosionBrushWeights, &weights, &xOffsets, &yOffsets, &pixelsProcessed](int startIndex, int endIndex) {
        // Variables to hold the sum of weights and index position
        float weightSum = 0;
        int addIndex = 0;

        // Capture `i` by value to avoid issues with its changing value in the loop
        for (int i = startIndex; i < endIndex; i++) {

            // Calculate the centre position of the point
            int centreX = i % mapWidth;
            int centreY = i / mapHeight;

            // Check if point lies within the brush's reach
            if (centreY <= erosionRadius
                || centreY >= mapHeight - erosionRadius
                || centreX <= erosionRadius + 1
                || centreX >= mapWidth - erosionRadius) {
                // Reset weight sum and index position
                weightSum = 0;
                addIndex = 0;

                // Loop through each point in the radius of the brush
                for (int y = -erosionRadius; y <= erosionRadius; y++) {
                    for (int x = -erosionRadius; x <= erosionRadius; x++) {
                        // Calculate squared distance from the centre point
                        float sqrDst = x * x + y * y;

                        // Check if point is within the brush's radius
                        if (sqrDst < erosionRadius * erosionRadius) {
                            // Calculate the new coordinates
                            int coordX = centreX + x;
                            int coordY = centreY + y;

                            // Ensure coordinates lie within the map
                            if (coordX >= 0 && coordX < mapWidth && coordY >= 0 && coordY < mapHeight) {
                                // Calculate the weight based on the distance from the centre
                                float weight = 1 - std::sqrt(sqrDst) / erosionRadius;

                                // Update the sum of weights
                                weightSum += weight;

                                // Store the weight and offset
                                weights[addIndex] = weight;
                                xOffsets[addIndex] = x;
                                yOffsets[addIndex] = y;

                                // Increment the index position
                                addIndex++;
                            }
                        }
                    }
                }
            }

            // Update the number of entries
            int numEntries = addIndex;

            // Resize the vectors for the current point
            erosionBrushIndices[i].resize(numEntries);
            erosionBrushWeights[i].resize(numEntries);

            // Calculate and store the final indices and weights
            for (int j = 0; j < numEntries; j++) {
                erosionBrushIndices[i][j] = (yOffsets[j] + centreY) * mapWidth + xOffsets[j] + centreX;
                erosionBrushWeights[i][j] = weights[j] / weightSum;
            }

            // Print progress
            pixelsProcessed++;
            if ((pixelsProcessed % ((mapWidth * mapHeight) / 10)) == 0) {  // Update every 1000 pixels to avoid slowing down the computation
                float progress = (float)pixelsProcessed / (mapWidth * mapHeight);
                std::cout << "\rErosion Brush Initialization Progress: " << progress * 100 << "%" << std::endl;
            }
        }
    };

    // Start threads
    for (int t = 0; t < numThreads; t++) {
        // Calculate the start and end indices for this thread
        int startIndex = t * tasksPerThread;
        int endIndex = (t == numThreads - 1) ? erosionBrushIndices.size() : (t + 1) * tasksPerThread;

        // Start a new thread to process the tasks
        threads[t] = std::thread([=, &initializeErosionBrushPixels] {
            initializeErosionBrushPixels(startIndex, endIndex);
        });
    }

    // Wait for threads
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    // Indicate stop
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Initialize Erosion Brush Complete: " << elapsed.count() << "s Elapsed, " << pixelsProcessed << " Pixels Processed"  << std::endl;
}

// Height and gradient structure
struct HeightAndGradient {
    float height;
    float gradientX;
    float gradientY;

    HeightAndGradient(float _height, float _gradientX, float _gradientY) : height(_height), gradientX(_gradientX), gradientY(_gradientY) {}
};

// Calculate height and gradient function
HeightAndGradient calculateHeightAndGradient(std::vector<float>& heightMap, float posX, float posY) {
    // Get current position index
    int coordX = (int) posX;
    int coordY = (int) posY;

    // Get drop's offset inside cell
    float x = posX - coordX;
    float y = posY - coordY;

    // Calculate heights of the four nodes of the drop's cell
    int nwIndex = coordY * mapWidth + coordX;
    float nwHeight = heightMap[nwIndex];
    float neHeight = (coordX + 1 < mapWidth) ? heightMap[nwIndex + 1] : nwHeight;
    float swHeight = (coordY + 1 < mapWidth) ? heightMap[nwIndex + mapWidth] : nwHeight;
    float seHeight = ((coordX + 1 < mapWidth) && (coordY + 1 < mapWidth)) ? heightMap[nwIndex + mapWidth + 1] : nwHeight;

    //Calculate the drop's direction of flow with bilinear interpolation of height difference along the edges
    float gradientX = (neHeight - nwHeight) * (1 - y) + (seHeight - swHeight) * y;
    float gradientY = (swHeight - nwHeight) * (1 - y) + (seHeight - neHeight) * y;

    // Calculate the height with bilinear interpolation of the heights of the nodes of the cell
    float height = nwHeight * (1 - x) * (1 - y) + neHeight * x * (1 - y) + swHeight * (1 - x) * y + seHeight * x * y;

    // Create new height and gradient struct
    return HeightAndGradient(height, gradientX, gradientY);
}

// Erode height map function
// TODO: This breaks if the mapWidth != mapHeight
void erodeHeightMap (std::vector<float>& heightMap) {
    // Initialize erosion brush indices and weights
    std::vector<std::vector<int>> erosionBrushIndices;
    std::vector<std::vector<float>> erosionBrushWeights;
    initializeErosionBrush(erosionBrushIndices, erosionBrushWeights);

    // Indicate start
    std::cout << "Starting Erosion Simulation..." << std::endl;
    std::atomic<int> dropsProcessed(0);
    auto start = std::chrono::high_resolution_clock::now();

    // Get threads
    unsigned int numThreads = std::thread::hardware_concurrency();
    std::vector<std::future<void>> futures(numThreads);

    // Drop simulation thread
    auto simulateDrop = [&](int startIter, int endIter) {
        for (int iteration = startIter; iteration < endIter; ++iteration) {
            // Create water drop at random map point
            float posX;
            float posY;
            while(true) {
                posX = rand() % mapWidth;
                posY = rand() % mapHeight;

                // Check that random position is at least the min drop spawn height
                if (heightMap[posY * mapWidth + posX] >= minDropSpawnHeight) {
                    break;
                }
            }

            float dirX = 0;
            float dirY = 0;
            float speed = initialSpeed;
            float water = initialWaterVolume;
            float sediment = 0;

            // Go through life of water drop
            for (int lifeTime = 0; lifeTime < maxDropLifeTime; lifeTime++) {
                // Get current position index
                int nodeX = (int) posX;
                int nodeY = (int) posY;
                int dropIndex = nodeY * mapWidth + nodeX;

                // Calculate drop's offset inside the cell
                float cellOffsetX = posX - nodeX;
                float cellOffsetY = posY - nodeY;

                // Calculate the drop's height and direction of flow with bilinear interpolation of surround heights
                HeightAndGradient heightAndGradient = calculateHeightAndGradient(heightMap, posX, posY);

                // Update the drop's direction and position (move 1 position unit regardless of speed)
                dirX = (dirX * inertia - heightAndGradient.gradientX * (1 - inertia));
                dirY = (dirY * inertia - heightAndGradient.gradientY * (1 - inertia));

                // Normalize direction
                float len = std::sqrt(dirX * dirX + dirY * dirY);
                if (len != 0) {
                    dirX /= len;
                    dirY /= len;
                }
                posX += dirX;
                posY += dirY;

                // Stop simulating drop if it's not moving or has flowed over edge of map
                if ((dirX == 0 && dirY == 0) || posX < 0 || posX >= mapWidth - 1 || posY < 0 || posY >= mapHeight - 1) {
                    break;
                }

                // Find the drop's new height and calculate the deltaHeight
                float newHeight = calculateHeightAndGradient(heightMap, posX, posY).height;
                float deltaHeight = newHeight - heightAndGradient.height;

                // Calculate the drop's sediment capacity (higher when moving fast down a slope and contains lots of water)
                float sedimentCapacity = std::max(-deltaHeight * speed * water * sedimentCapacityFactor, minSedimentCapacity);

                // If carrying more sediment than capacity, or if flowing uphill:
                if (sediment > sedimentCapacity || deltaHeight > 0) {
                    // If moving uphill (deltaHeight > 0) try to fill up to current height, otherwise deposit a fraction of the excess sediment
                    float amountToDeposit = (deltaHeight > 0) ? std::min(deltaHeight, sediment) : (sediment - sedimentCapacity) * depositSpeed;
                    sediment -= amountToDeposit;

                    // Add the sediment to the four nodes of the current cell using bilinear interpolation
                    // Deposition is not distributed over a radius (like erosion) so that it can fill small pits
                    heightMap[dropIndex] += amountToDeposit * (1 - cellOffsetX) * (1 - cellOffsetY);
                    heightMap[dropIndex + 1] += amountToDeposit * cellOffsetX * (1 - cellOffsetY);
                    heightMap[dropIndex + mapWidth] += amountToDeposit * (1 - cellOffsetX) * cellOffsetY;
                    heightMap[dropIndex + mapWidth + 1] += amountToDeposit * cellOffsetX * cellOffsetY;

                } else {
                    // Erode a fraction of the drop's current carry capacity
                    // Clamp the erosion to the change in height so that it doesn't dig a hole in the terrain behind the droplet
                    float amountToErode = std::min((sedimentCapacity - sediment) * erodeSpeed, -deltaHeight);

                    // Use erosion brush to erode from all nodes inside the droplet's erosion radius
                    for (int brushPointIndex = 0; brushPointIndex < erosionBrushIndices[dropIndex].size(); brushPointIndex++) {
                        int nodeIndex = erosionBrushIndices[dropIndex][brushPointIndex];
                        float weighedErodeAmount = amountToErode * erosionBrushWeights[dropIndex][brushPointIndex];
                        float deltaSediment = (heightMap[nodeIndex] < weighedErodeAmount) ? heightMap[nodeIndex] : weighedErodeAmount;
                        heightMap[nodeIndex] -= deltaSediment;
                        sediment += deltaSediment;
                    }
                }

                // Update drop's speed and water content
                speed = std::sqrt(speed * speed + deltaHeight * gravity);
                water *= (1 - evaporateSpeed);

                // Prevent glitched values
                if (std::isnan(speed) || std::isnan(sediment)) {
                    break;
                }

                if (water < minWaterVolume || heightMap[posY * mapWidth + posX] < minDropDespawnHeight) {
                    break;
                }
            }

            // Print progress
            dropsProcessed++;
            if ((dropsProcessed % (numDrops / 20)) == 0) {  // Update every 1000 pixels to avoid slowing down the computation
                float progress = (float)dropsProcessed / numDrops;
                std::cout << "\rErosion Simulation Progress: " << progress * 100 << "%" << std::endl;
            }
        }
    };

    // Start threads
    int sliceSize = numDrops / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int startIter = i * sliceSize;
        int endIter = (i == numThreads - 1) ? numDrops : (i + 1) * sliceSize;  // ensure last slice goes to end
        futures[i] = std::async(std::launch::async, simulateDrop, startIter, endIter);
    }

    // Wait for threads
    for (auto& future : futures) {
        future.get();
    }

    // Indicate stop
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Erosion Simulation Complete: " << elapsed.count() << "s Elapsed, " << dropsProcessed << " Drops Simulated" << std::endl;
}

// Convert heightmap to colormap function
std::vector<unsigned char> convertHeightMapToColorMap(std::vector<float>& heightMap) {
    // Indicate start
    std::cout << "Converting HeightMap to RGB Map..." << std::endl;
    std::atomic<int> pixelsProcessed(0);
    auto start = std::chrono::high_resolution_clock::now();

    // Output vector
    std::vector<unsigned char> output(mapWidth * mapHeight * channels);

    // Convert heightmap to colormap
    for (int y = 0; y < mapHeight; ++y) {
        for (int x = 0; x < mapWidth; ++x) {
            unsigned char color = static_cast<unsigned char>(heightMap[y * mapWidth + x] * 255);
            output[(y * mapWidth + x) * channels + 0] = color;
            output[(y * mapWidth + x) * channels + 1] = color;
            output[(y * mapWidth + x) * channels + 2] = color;
        }
    }

    // Indicate stop
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Convert HeightMap to ColorMap Complete: " << elapsed.count() << "s Elapsed" << std::endl;

    // Return the new color map
    return output;
}

// Global level variables (0.0 - 1.0) Note: Should be at least 0.05 thick
float waterLevel = 0.40f;
float beachLevel = 0.45f;
float grassLevel = 0.70f;
float mountainLevel = 0.80f;

// Number of layers for each level
float totalLayers = 40;
int waterLayers = round(waterLevel * totalLayers);
int beachLayers = round((beachLevel - waterLevel) * totalLayers);
int grassLayers = round((grassLevel - beachLevel) * totalLayers);
int mountainLayers = round((mountainLevel - grassLevel) * totalLayers);
int snowLayers = round((1.0 - mountainLevel) * totalLayers);

// Color struct
struct Color {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

// Lerp between two colors
Color lerpColor(const Color& start, const Color& end, float t) {
    Color result;
    result.r = static_cast<unsigned char>(start.r * (1 - t) + end.r * t);
    result.g = static_cast<unsigned char>(start.g * (1 - t) + end.g * t);
    result.b = static_cast<unsigned char>(start.b * (1 - t) + end.b * t);
    return result;
}

// Create color interpolation
std::vector<Color> initializeColorInterpolation(const Color& minColor, const Color& maxColor, int layers) {
    std::vector<Color> colors(layers);

    for (int i = 0; i < layers; ++i) {
        float t = static_cast<float>(i) / (layers - 1);  // Fraction along the interpolation
        colors[i] = lerpColor(minColor, maxColor, t);
    }

    return colors;
}

// Apply color to colormap function
void applyColorToColorMap(std::vector<unsigned char>& colorMap) {
    // Indicate start
    std::cout << "Applying Colors to RGB Map..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // Maximum and minimum RGB colors for each level
    Color minWaterColor = {37, 89, 134};
    Color maxWaterColor = {140, 183, 220};

    Color minBeachColor = {230, 218, 166};
    Color maxBeachColor = {206, 190, 124};

    Color minGrassColor = {180, 218, 91};
    Color maxGrassColor = {93, 121, 30};

    Color minMountainColor = {92, 88, 70};
    Color maxMountainColor = {144, 142, 136};

    Color minSnowColor = {223, 222, 216};
    Color maxSnowColor = {245, 244, 239};

    // Initialize the color interpolations for each level
    std::vector<Color> waterColors = initializeColorInterpolation(minWaterColor, maxWaterColor, waterLayers);
    std::vector<Color> beachColors = initializeColorInterpolation(minBeachColor, maxBeachColor, beachLayers);
    std::vector<Color> grassColors = initializeColorInterpolation(minGrassColor, maxGrassColor, grassLayers);
    std::vector<Color> mountainColors = initializeColorInterpolation(minMountainColor, maxMountainColor, mountainLayers);
    std::vector<Color> snowColors = initializeColorInterpolation(minSnowColor, maxSnowColor, snowLayers);

    // Apply colors to colormap based on grayscale values and different levels
    for (size_t i = 0; i < colorMap.size(); i += 3) {
        float grayscale = static_cast<float>(colorMap[i]) / 255.0;
        Color color;

        // Determine color based on grayscale value
        if (grayscale < waterLevel) { // Apply water color
            int index = std::min(static_cast<int>((grayscale / waterLevel) * waterLayers), waterLayers - 1);
            color = waterColors[index];
        } else if (grayscale < beachLevel) { // Apply beach color
            int index = std::min(static_cast<int>(((grayscale - waterLevel) / (beachLevel - waterLevel)) * beachLayers), beachLayers - 1);
            color = beachColors[index];
        } else if (grayscale < grassLevel) { // Apply grass color
            int index = std::min(static_cast<int>(((grayscale - beachLevel) / (grassLevel - beachLevel)) * grassLayers), grassLayers - 1);
            color = grassColors[index];
        } else if (grayscale < mountainLevel) { // Apply mountain color
            int index = std::min(static_cast<int>(((grayscale - grassLevel) / (mountainLevel - grassLevel)) * mountainLayers), mountainLayers - 1);
            color = mountainColors[index];
        } else { // Apply snow color
            int index = std::min(static_cast<int>(((grayscale - mountainLevel) / (1.0 - mountainLevel)) * snowLayers), snowLayers - 1);
            color = snowColors[index];
        }

        // Set new color
        colorMap[i] = color.r;
        colorMap[i+1] = color.g;
        colorMap[i+2] = color.b;
    }

    // Indicate stop
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Apply Colors to ColorMap Complete: " << elapsed.count() << "s Elapsed" << std::endl;
}


// Main
int main() {
    // Set random number seed
    srand(static_cast<unsigned int>(std::time(0)));

    // Prepare a vector to hold pixel data
    std::vector<float> heightMap(mapWidth * mapHeight);

    // Perform map generation
    generateNoiseHeightMap(heightMap);

    // Perform erosion
    erodeHeightMap(heightMap);

    // Perform river cutting
    generateRiversInHeightMap(heightMap);

    // Convert heightmap to color map
    std::vector<unsigned char> colorMap = convertHeightMapToColorMap(heightMap);

    // Write the image to file
    if (!stbi_write_png("nocolor_map.png", mapWidth, mapHeight, channels, colorMap.data(), mapWidth * channels)) {
        std::cerr << "Error writing image to file" << std::endl;
    }

    // Apply colors
    applyColorToColorMap(colorMap);

    // Write the image to file
    if (!stbi_write_png("color_map.png", mapWidth, mapHeight, channels, colorMap.data(), mapWidth * channels)) {
        std::cerr << "Error writing image to file" << std::endl;
        return 1;
    }

    std::cout << "Image successfully written to file" << std::endl;
    return 0;
}
