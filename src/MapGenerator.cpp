// Imports
#include <iostream>
#include <vector>
#include <atomic>
#include "FastNoiseLite.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <fstream>
#include "nlohmann/json.hpp"
#include <thread>
#include <random>
#include <cmath>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <future>
#include <set>
#include <unordered_map>
#include "MapGenerator.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////  GENERATE PERLIN NOISE HEIGHTMAP
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Generate noise heightmap function
std::vector<float> generateNoiseHeightMap(MapConfig mapConfig) {
    // Create height map
    std::vector<float> heightMap(mapConfig.mapWidth * mapConfig.mapHeight);

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
    float centerX = mapConfig.mapWidth / 2.0f;
    float centerY = mapConfig.mapHeight / 2.0f;

    // Initialize Simplex noise generator
    FastNoiseLite noiseGenerator;
    noiseGenerator.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    noiseGenerator.SetSeed(seed);

    // Generate noise height map thread
    auto generateNoiseHeightMapPixels = [&](int startY, int endY) {
        for (int y = startY; y < endY; ++y) {
            for (int x = 0; x < mapConfig.mapWidth; ++x) {
                float noise = 0.0f;
                float amplitude = 1.0f;
                float maxAmplitude = 0.0f;
                float freq = mapConfig.frequency;

                for (int i = 0; i < mapConfig.octaves; ++i) {
                    noise += amplitude * noiseGenerator.GetNoise((float)x * freq, (float)y * freq);
                    maxAmplitude += amplitude;
                    amplitude *= mapConfig.persistence;
                    freq *= mapConfig.lacunarity;
                }

                noise /= maxAmplitude;
                noise = (noise + 1) / 2.0f;  // map noise from 0.0 - 1.0

                float dx = 2.0f * (centerX - x) / mapConfig.mapWidth;  // Scale the x-distance by map width
                float dy = 2.0f * (centerY - y) / mapConfig.mapHeight;  // Scale the y-distance by map height
                float distance = std::sqrt(dx * dx + dy * dy);
                distance = 1 / (1 + std::exp(-10 * (distance - mapConfig.distanceFromCenter)));
                noise = (1 - distance) * noise;
                heightMap[y * mapConfig.mapWidth + x] = noise;

                // Print progress
                pixelsProcessed++;
                if ((pixelsProcessed % ((mapConfig.mapWidth * mapConfig.mapHeight) / 10)) == 0) {  // Update every 1000 pixels to avoid slowing down the computation
                    float progress = (float)pixelsProcessed / (mapConfig.mapWidth * mapConfig.mapHeight);
                    std::cout << "\rMap Generation Progress: " << progress * 100 << "%" << std::endl;
                }
            }
        }
    };

    // Start threads
    int sliceHeight = mapConfig.mapHeight / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int startY = i * sliceHeight;
        int endY = (i == numThreads - 1) ? mapConfig.mapHeight : (i + 1) * sliceHeight;  // ensure last slice goes to end
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

    // Return created map
    return heightMap;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////  GENERATE RIVERS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// River path struct
struct RiverPath {
    std::vector<std::pair<int, int>> path;  // Store the path of the river as (x,y) pairs
};

// Initialize rivers
void initializeRivers(std::vector<RiverPath>& rivers,
                      std::vector<float>& heightMap,
                      std::mt19937& gen,
                      std::uniform_int_distribution<>& distr,
                      int mapWidth,
                      int mapHeight,
                      float minRiverSpawnHeight,
                      float maxRiverSpawnHeight) {

    for (auto& river : rivers) {
        int index;
        do {
            index = distr(gen);
        } while (heightMap[index] < minRiverSpawnHeight || heightMap[index] > maxRiverSpawnHeight);

        river.path.push_back({index % mapWidth, index / mapWidth});
    }
}

// Perform circular search to find nearest lower point from given point
std::pair<int, int> circularSearch(int x, int y,
                                   const std::vector<float>& heightMap,
                                   int mapWidth,
                                   int mapHeight,
                                   int minSearchRiverPointDistance) {

    std::pair<int, int> lowestPoint = {x, y};
    float lowestHeight = heightMap[y * mapWidth + x];  // Adjusted indexing
    bool found = false;
    int radius = minSearchRiverPointDistance;
    while (!found) {
        for (int dx = -radius; dx <= radius; dx++) {
            for (int dy = -radius; dy <= radius; dy++) {
                if (dx * dx + dy * dy >= radius * radius) { // Check if the point is within the circular distance
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx >= 0 && nx < mapWidth && ny >= 0 && ny < mapHeight) {
                        int index = ny * mapWidth + nx;  // Adjusted indexing
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
void createRiverPaths(std::vector<RiverPath>& rivers,
                      std::vector<float>& heightMap,
                      int mapWidth,
                      int mapHeight,
                      float minRiverDespawnHeight,
                      int minSearchRiverPointDistance) {

    for (auto& river : rivers) {
        while (true) {
            // Get the last point in the path
            auto [x, y] = river.path.back();
            int index = y * mapWidth + x;  // Adjusted indexing

            // If the current point is below minRiverDespawnHeight, break
            if (heightMap[index] < minRiverDespawnHeight)
                break;

            // Search for the lower nearest point using the circular search
            auto lowestPoint = circularSearch(x, y, heightMap, mapWidth, mapHeight, minSearchRiverPointDistance);

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
std::pair<int, int> generateNewPoint(std::pair<int, int> point1,
                                     std::pair<int, int> point2,
                                     float t,
                                     std::mt19937& gen,
                                     float minDeviation,
                                     float maxDeviation) {

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
void generateRandomPoints(std::vector<RiverPath>& rivers,
                          std::mt19937& gen,
                          int randomPointSpacing,
                          float minDeviation,
                          float maxDeviation) {

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
                auto newPoint = generateNewPoint(river.path[i], river.path[i+1], t, gen, minDeviation, maxDeviation);
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
        // Generate uniform-knot vector
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
void removeStraightRivers(std::vector<RiverPath>& rivers, int riverStraightnessThreshold) {
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
void combineIntersectingRivers(std::vector<RiverPath>& rivers, int intersectionRange) {
    for (size_t i = 0; i < rivers.size(); ++i) {
        for (size_t j = i + 1; j < rivers.size(); ++j) {
            auto& river1 = rivers[i];
            auto& river2 = rivers[j];

            bool intersectionFound = false;
            int intersectionIndexRiver1;
            int intersectionIndexRiver2;

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
void checkRiverLength(std::vector<RiverPath>& rivers, int minRiverLength) {
    rivers.erase(std::remove_if(rivers.begin(), rivers.end(), [&](const RiverPath& river){
        return river.path.size() < minRiverLength;
    }), rivers.end());
}

// Carve rivers into the heightmap
void carveRivers(std::vector<RiverPath>& rivers, std::vector<float>& heightMap, MapConfig mapConfig) {
    for (auto& river : rivers) {
        int riverPointCount = river.path.size();
        for (int i = 0; i < riverPointCount; i++) {
            auto& point = river.path[i];

            float t = std::min(1.0f, static_cast<float>(i) / mapConfig.maxRiverRadiusLength);  // normalize the current river point index
            float currentRadius = mapConfig.startRadius + (mapConfig.maxRadius - mapConfig.startRadius) * std::pow(t, mapConfig.sizeScaleRate);  // interpolate the radius
            int roundedRadius = static_cast<int>(std::round(currentRadius));
            float currentTerrainCarveRadius = mapConfig.startTerrainCarveRadius + (mapConfig.maxTerrainCarveRadius - mapConfig.startTerrainCarveRadius) * std::pow(t, mapConfig.sizeTerrainCarveScaleRate);  // interpolate the terrain carving radius
            int roundedTerrainCarveRadius = static_cast<int>(std::round(currentTerrainCarveRadius));

            // Terrain carving
            for (int x = std::max(0, point.first - roundedTerrainCarveRadius); x < std::min(mapConfig.mapWidth, point.first + roundedTerrainCarveRadius); x++) {
                for (int y = std::max(0, point.second - roundedTerrainCarveRadius); y < std::min(mapConfig.mapHeight, point.second + roundedTerrainCarveRadius); y++) {
                    float dx = point.first - x;
                    float dy = point.second - y;
                    float distance = std::sqrt(dx * dx + dy * dy);

                    if (distance > currentRadius && distance <= currentTerrainCarveRadius) {
                        bool insideRiver = false;
                        for (auto& riverPoint : river.path) {
                            float dxRiver = riverPoint.first - x;
                            float dyRiver = riverPoint.second - y;
                            float distanceRiver = std::sqrt(dxRiver * dxRiver + dyRiver * dyRiver);
                            if (distanceRiver <= currentRadius) {
                                insideRiver = true;
                                break;
                            }
                        }

                        if (!insideRiver) {
                            float distortion = (currentTerrainCarveRadius - distance) / (currentTerrainCarveRadius - currentRadius) * mapConfig.terrainDistortion;  // interpolate the terrain distortion
                            if (heightMap[y * mapConfig.mapWidth + x] > mapConfig.terrainMinDepth) {
                                heightMap[y * mapConfig.mapWidth + x] = std::max(heightMap[y * mapConfig.mapWidth + x] - distortion, mapConfig.terrainMinDepth);  // subtract distortion and prevent it from going below riverMinDepth
                            }
                        }
                    }
                }
            }

            // River carving
            for (int x = std::max(0, point.first - roundedRadius); x < std::min(mapConfig.mapWidth, point.first + roundedRadius); x++) {
                for (int y = std::max(0, point.second - roundedRadius); y < std::min(mapConfig.mapHeight, point.second + roundedRadius); y++) {
                    float dx = point.first - x;
                    float dy = point.second - y;
                    float distance = std::sqrt(dx * dx + dy * dy);

                    if (distance <= currentRadius) {
                        float normalizedDistance = distance / currentRadius;  // distance normalized to [0, 1], 0 being the center, 1 being the edge
                        float newHeight = mapConfig.riverMaxDepth * (1 - normalizedDistance) + mapConfig.riverMinDepth * normalizedDistance;  // interpolate between riverMaxDepth and riverMinDepth
                        if (heightMap[y * mapConfig.mapWidth + x] > newHeight) { // This is to prevent heights which are lower than the riverMinDepth from being effected
                            heightMap[y * mapConfig.mapWidth + x] = newHeight;
                        }
                    }
                }
            }
        }
    }
}

// Generate rivers in heightmap function
void generateRiversInHeightMap(MapConfig mapConfig, std::vector<float>& heightMap) {
    std::cout << "Initializing Rivers..." << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, mapConfig.mapWidth * mapConfig.mapHeight - 1);
    std::vector<RiverPath> rivers(mapConfig.riverCount);
    initializeRivers(rivers, heightMap, gen, distr, mapConfig.mapWidth, mapConfig.mapHeight, mapConfig.minRiverSpawnHeight, mapConfig.maxRiverSpawnHeight);
    std::cout << rivers.size() << std::endl;

    std::cout << "Finding river paths..." << std::endl;
    createRiverPaths(rivers, heightMap, mapConfig.mapWidth, mapConfig.mapHeight, mapConfig.minRiverDespawnHeight, mapConfig.minSearchRiverPointDistance);

    std::cout << "Adding randomness to river paths..." << std::endl;
    generateRandomPoints(rivers, gen, mapConfig.randomPointSpacing, mapConfig.minDeviation, mapConfig.maxDeviation);

    std::cout << "Redrawing river paths along B-spline curves..." << std::endl;
    redrawPaths(rivers);

    std::cout << "Removing loops and duplicate points in river paths..." << std::endl;
    removeLoopsAndDuplicates(rivers);

    std::cout << "Removing rivers that are too straight..." << std::endl;
    removeStraightRivers(rivers, mapConfig.riverStraightnessThreshold);

    std::cout << "Combining rivers that intersect..." << std::endl;
    combineIntersectingRivers(rivers, mapConfig.intersectionRange);

    std::cout << "Removing rivers that are too short..." << std::endl;
    checkRiverLength(rivers, mapConfig.minRiverLength);

    std::cout << "Placing rivers in heightmap..." << std::endl;
    carveRivers(rivers, heightMap, mapConfig);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////  SIMULATE EROSION
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// By: https://github.com/SebLague/Hydraulic-Erosion/tree/Coding-Adventure-E01

// Initialize erosion brush function
void initializeErosionBrush(std::vector<std::vector<int>>& erosionBrushIndices, std::vector<std::vector<float>>& erosionBrushWeights, int mapWidth, int mapHeight, int erosionRadius) {

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
    auto initializeErosionBrushPixels = [&erosionBrushIndices, &erosionBrushWeights, &weights, &xOffsets, &yOffsets, &pixelsProcessed, &mapWidth, &mapHeight, &erosionRadius](int startIndex, int endIndex) {
        // Variables to hold the sum of weights and index position
        float weightSum = 0;
        int addIndex = 0;

        // Capture `i` by value to avoid issues with its changing value in the loop
        for (int i = startIndex; i < endIndex; i++) {

            // Calculate the centre position of the point
            int centreX = i % mapWidth;
            int centreY = i / mapWidth;

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
HeightAndGradient calculateHeightAndGradient(std::vector<float>& heightMap, float posX, float posY, int mapWidth, int mapHeight) {
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
    float swHeight = (coordY + 1 < mapHeight) ? heightMap[nwIndex + mapWidth] : nwHeight;
    float seHeight = ((coordX + 1 < mapWidth) && (coordY + 1 < mapHeight)) ? heightMap[nwIndex + mapWidth + 1] : nwHeight;

    //Calculate the drop's direction of flow with bi-linear interpolation of height difference along the edges
    float gradientX = (neHeight - nwHeight) * (1 - y) + (seHeight - swHeight) * y;
    float gradientY = (swHeight - nwHeight) * (1 - y) + (seHeight - neHeight) * y;

    // Calculate the height with bi-linear interpolation of the heights of the nodes of the cell
    float height = nwHeight * (1 - x) * (1 - y) + neHeight * x * (1 - y) + swHeight * (1 - x) * y + seHeight * x * y;

    // Create new height and gradient struct
    return HeightAndGradient(height, gradientX, gradientY);
}

// Simulate erosion function
// TODO: This breaks if the mapWidth != mapHeight
void simulateErosion(MapConfig mapConfig, std::vector<float>& heightMap) {
    // Initialize erosion brush indices and weights
    std::vector<std::vector<int>> erosionBrushIndices;
    std::vector<std::vector<float>> erosionBrushWeights;
    initializeErosionBrush(erosionBrushIndices, erosionBrushWeights, mapConfig.mapWidth, mapConfig.mapHeight, mapConfig.erosionRadius);

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
                posX = rand() % mapConfig.mapWidth;
                posY = rand() % mapConfig.mapHeight;

                // Check that random position is at least the min drop spawn height
                if (heightMap[posY * mapConfig.mapWidth + posX] >= mapConfig.minDropSpawnHeight) {
                    break;
                }
            }

            float dirX = 0;
            float dirY = 0;
            float speed = mapConfig.initialSpeed;
            float water = mapConfig.initialWaterVolume;
            float sediment = 0;

            // Go through life of water drop
            for (int lifeTime = 0; lifeTime < mapConfig.maxDropLifeTime; lifeTime++) {
                // Get current position index
                int nodeX = (int) posX;
                int nodeY = (int) posY;
                int dropIndex = nodeY * mapConfig.mapWidth + nodeX;

                // Calculate drop's offset inside the cell
                float cellOffsetX = posX - nodeX;
                float cellOffsetY = posY - nodeY;

                // Calculate the drop's height and direction of flow with bilinear interpolation of surround heights
                HeightAndGradient heightAndGradient = calculateHeightAndGradient(heightMap, posX, posY, mapConfig.mapWidth, mapConfig.mapHeight);

                // Update the drop's direction and position (move 1 position unit regardless of speed)
                dirX = (dirX * mapConfig.inertia - heightAndGradient.gradientX * (1 - mapConfig.inertia));
                dirY = (dirY * mapConfig.inertia - heightAndGradient.gradientY * (1 - mapConfig.inertia));

                // Normalize direction
                float len = std::sqrt(dirX * dirX + dirY * dirY);
                if (len != 0) {
                    dirX /= len;
                    dirY /= len;
                }
                posX += dirX;
                posY += dirY;

                // Stop simulating drop if it's not moving or has flowed over edge of map
                if ((dirX == 0 && dirY == 0) || posX < 0 || posX >= mapConfig.mapWidth - 1 || posY < 0 || posY >= mapConfig.mapHeight - 1) {
                    break;
                }

                // Find the drop's new height and calculate the deltaHeight
                float newHeight = calculateHeightAndGradient(heightMap, posX, posY, mapConfig.mapWidth, mapConfig.mapHeight).height;
                float deltaHeight = newHeight - heightAndGradient.height;

                // Calculate the drop's sediment capacity (higher when moving fast down a slope and contains lots of water)
                float sedimentCapacity = std::max(-deltaHeight * speed * water * mapConfig.sedimentCapacityFactor, mapConfig.minSedimentCapacity);

                // If carrying more sediment than capacity, or if flowing uphill:
                if (sediment > sedimentCapacity || deltaHeight > 0) {
                    // If moving uphill (deltaHeight > 0) try to fill up to current height, otherwise deposit a fraction of the excess sediment
                    float amountToDeposit = (deltaHeight > 0) ? std::min(deltaHeight, sediment) : (sediment - sedimentCapacity) * mapConfig.depositSpeed;
                    sediment -= amountToDeposit;

                    // Add the sediment to the four nodes of the current cell using bi-linear interpolation
                    // Deposition is not distributed over a radius (like erosion) so that it can fill small pits
                    heightMap[dropIndex] += amountToDeposit * (1 - cellOffsetX) * (1 - cellOffsetY);
                    heightMap[dropIndex + 1] += amountToDeposit * cellOffsetX * (1 - cellOffsetY);
                    heightMap[dropIndex + mapConfig.mapWidth] += amountToDeposit * (1 - cellOffsetX) * cellOffsetY;
                    heightMap[dropIndex + mapConfig.mapWidth + 1] += amountToDeposit * cellOffsetX * cellOffsetY;

                } else {
                    // Erode a fraction of the drop's current carry capacity
                    // Clamp the erosion to the change in height so that it doesn't dig a hole in the terrain behind the droplet
                    float amountToErode = std::min((sedimentCapacity - sediment) * mapConfig.erodeSpeed, -deltaHeight);

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
                speed = std::sqrt(speed * speed + deltaHeight * mapConfig.gravity);
                water *= (1 - mapConfig.evaporateSpeed);

                // Prevent glitched values
                if (std::isnan(speed) || std::isnan(sediment)) {
                    break;
                }

                if (water < mapConfig.minWaterVolume || heightMap[posY * mapConfig.mapWidth + posX] < mapConfig.minDropDespawnHeight) {
                    break;
                }
            }

            // Print progress
            dropsProcessed++;
            if ((dropsProcessed % (mapConfig.numDrops / 20)) == 0) {  // Update every 1000 pixels to avoid slowing down the computation
                float progress = (float)dropsProcessed / mapConfig.numDrops;
                std::cout << "\rErosion Simulation Progress: " << progress * 100 << "%" << std::endl;
            }
        }
    };

    // Start threads
    int sliceSize = mapConfig.numDrops / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int startIter = i * sliceSize;
        int endIter = (i == numThreads - 1) ? mapConfig.numDrops : (i + 1) * sliceSize;  // ensure last slice goes to end
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////  CONVERT TO COLOR
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Create color map from height map function
std::vector<unsigned char> createColorMap(MapConfig mapConfig, std::vector<float>& heightMap) {
    // Indicate start
    std::cout << "Converting HeightMap to RGB Map..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // Output vector
    std::vector<unsigned char> output(mapConfig.mapWidth * mapConfig.mapHeight * mapConfig.channels);

    // Convert height map to colormap
    for (int x = 0; x < mapConfig.mapWidth; ++x) {
        for (int y = 0; y < mapConfig.mapHeight; ++y) {
            unsigned char color = static_cast<unsigned char>(heightMap[y * mapConfig.mapWidth + x] * 255);
            output[(y * mapConfig.mapWidth + x) * mapConfig.channels + 0] = color;
            output[(y * mapConfig.mapWidth + x) * mapConfig.channels + 1] = color;
            output[(y * mapConfig.mapWidth + x) * mapConfig.channels + 2] = color;
        }
    }

    // Indicate stop
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Convert HeightMap to ColorMap Complete: " << elapsed.count() << "s Elapsed" << std::endl;

    // Return the new color map
    return output;
}

// Lerp between two colors
MapColor lerpColor(const MapColor& start, const MapColor& end, float t) {
    MapColor result;
    result.r = static_cast<unsigned char>(start.r * (1 - t) + end.r * t);
    result.g = static_cast<unsigned char>(start.g * (1 - t) + end.g * t);
    result.b = static_cast<unsigned char>(start.b * (1 - t) + end.b * t);
    return result;
}

// Create color interpolation
std::vector<MapColor> initializeColorInterpolation(const MapColor& minColor, const MapColor& maxColor, int layers) {
    std::vector<MapColor> colors(layers);

    if (layers <= 1) {
        colors[0] = minColor;
        return colors;
    }

    for (int i = 0; i < layers; ++i) {
        float t = static_cast<float>(i) / (layers - 1);  // Fraction along the interpolation
        colors[i] = lerpColor(minColor, maxColor, t);
    }

    return colors;
}

// Paint color map
void paintColorMap(MapConfig mapConfig, std::vector<unsigned char>& colorMap) {
    // Indicate start
    std::cout << "Applying Colors to RGB Map..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // Initialize the color interpolations for each level
    std::vector<MapColor> waterColors = initializeColorInterpolation(mapConfig.minWaterColor, mapConfig.maxWaterColor, mapConfig.waterLayers);
    std::vector<MapColor> beachColors = initializeColorInterpolation(mapConfig.minBeachColor, mapConfig.maxBeachColor, mapConfig.beachLayers);
    std::vector<MapColor> grassColors = initializeColorInterpolation(mapConfig.minGrassColor, mapConfig.maxGrassColor, mapConfig.grassLayers);
    std::vector<MapColor> mountainColors = initializeColorInterpolation(mapConfig.minMountainColor, mapConfig.maxMountainColor, mapConfig.mountainLayers);
    std::vector<MapColor> snowColors = initializeColorInterpolation(mapConfig.minSnowColor, mapConfig.maxSnowColor, mapConfig.snowLayers);

    // Apply colors to color map based on grayscale values and different levels
    for (size_t i = 0; i < colorMap.size(); i += 3) {
        float grayscale = static_cast<float>(colorMap[i]) / 255.0;
        MapColor color;

        // Determine color based on grayscale value
        if (grayscale < mapConfig.waterLevel) { // Apply water color
            int index = std::min(static_cast<int>((grayscale / mapConfig.waterLevel) * mapConfig.waterLayers), mapConfig.waterLayers - 1);
            color = waterColors[index];
        } else if (grayscale < mapConfig.beachLevel) { // Apply beach color
            int index = std::min(static_cast<int>(((grayscale - mapConfig.waterLevel) / (mapConfig.beachLevel - mapConfig.waterLevel)) * mapConfig.beachLayers), mapConfig.beachLayers - 1);
            color = beachColors[index];
        } else if (grayscale < mapConfig.grassLevel) { // Apply grass color
            int index = std::min(static_cast<int>(((grayscale - mapConfig.beachLevel) / (mapConfig.grassLevel - mapConfig.beachLevel)) * mapConfig.grassLayers), mapConfig.grassLayers - 1);
            color = grassColors[index];
        } else if (grayscale < mapConfig.mountainLevel) { // Apply mountain color
            int index = std::min(static_cast<int>(((grayscale - mapConfig.grassLevel) / (mapConfig.mountainLevel - mapConfig.grassLevel)) * mapConfig.mountainLayers), mapConfig.mountainLayers - 1);
            color = mountainColors[index];
        } else { // Apply snow color
            int index = std::min(static_cast<int>(((grayscale - mapConfig.mountainLevel) / (1.0 - mapConfig.mountainLevel)) * mapConfig.snowLayers), mapConfig.snowLayers - 1);
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////  GENERATE
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Generate map function
MapData generate(MapConfig mapConfig) {
    // Initialize map
    MapData mapData;

    // Set random number seed
    srand(static_cast<unsigned int>(std::time(0)));

    // Create height map
    std::vector<float> heightMap = generateNoiseHeightMap(mapConfig);

    // Perform erosion
    simulateErosion(mapConfig, heightMap);

    // Perform river generation
    generateRiversInHeightMap(mapConfig, heightMap);

    // Save height map
    mapData.heightMap = heightMap;

    // Create color map
    std::vector<unsigned char> colorMap = createColorMap(mapConfig, heightMap);

    // Save grayscale map
    mapData.grayscaleMap = colorMap;

    // Set colors in color map
    paintColorMap(mapConfig, colorMap);

    // Save color map
    mapData.colorMap = colorMap;

    // Save config
    mapData.mapConfig = mapConfig;

    // Return map data
    return mapData;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////  MISCELLANEOUS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Save maps function
bool saveColorAndHeightMaps(MapData mapData) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Write the map data images to file
    if (!stbi_write_png("map_gen_output\\color_map.png", mapData.mapConfig.mapWidth, mapData.mapConfig.mapHeight, mapData.mapConfig.channels, mapData.colorMap.data(), mapData.mapConfig.mapWidth * mapData.mapConfig.channels)
    || !stbi_write_png("map_gen_output\\grayscale_map.png", mapData.mapConfig.mapWidth, mapData.mapConfig.mapHeight, mapData.mapConfig.channels, mapData.grayscaleMap.data(), mapData.mapConfig.mapWidth * mapData.mapConfig.channels)) {
        std::cerr << "Error writing maps to file!" << std::endl;
        return false;
    }

    std::cout << "Maps saved successfully!" << std::endl;
    return true;
}

// Write map color to json
void to_json(nlohmann::json& j, const MapColor& p) {
    j = nlohmann::json{{"r", p.r}, {"g", p.g}, {"b", p.b}};
}

// Read map color from json
void from_json(const nlohmann::json& j, MapColor& p) {
    j.at("r").get_to(p.r);
    j.at("g").get_to(p.g);
    j.at("b").get_to(p.b);
}

// Write map config to json
void to_json(nlohmann::json& j, const MapConfig& p) {
    j = nlohmann::json{
            {"mapWidth", p.mapWidth},
            {"mapHeight", p.mapHeight},
            {"channels", p.channels},
            {"seaLevel", p.seaLevel},
            {"frequency", p.frequency},
            {"octaves", p.octaves},
            {"lacunarity", p.lacunarity},
            {"persistence", p.persistence},
            {"distanceFromCenter", p.distanceFromCenter},
            {"riverCount", p.riverCount},
            {"maxRiverSpawnHeight", p.maxRiverSpawnHeight},
            {"minRiverSpawnHeight", p.minRiverSpawnHeight},
            {"minSearchRiverPointDistance", p.minSearchRiverPointDistance},
            {"minRiverDespawnHeight", p.minRiverDespawnHeight},
            {"randomPointSpacing", p.randomPointSpacing},
            {"maxDeviation", p.maxDeviation},
            {"minDeviation", p.minDeviation},
            {"riverStraightnessThreshold", p.riverStraightnessThreshold},
            {"minRiverLength", p.minRiverLength},
            {"intersectionRange", p.intersectionRange},
            {"startRadius", p.startRadius},
            {"maxRadius", p.maxRadius},
            {"sizeScaleRate", p.sizeScaleRate},
            {"maxRiverRadiusLength", p.maxRiverRadiusLength},
            {"riverMinDepth", p.riverMinDepth},
            {"riverMaxDepth", p.riverMaxDepth},
            {"terrainMinDepth", p.terrainMinDepth},
            {"terrainDistortion", p.terrainDistortion},
            {"startTerrainCarveRadius", p.startTerrainCarveRadius},
            {"maxTerrainCarveRadius", p.maxTerrainCarveRadius},
            {"sizeTerrainCarveScaleRate", p.sizeTerrainCarveScaleRate},
            {"numDrops", p.numDrops},
            {"minDropSpawnHeight", p.minDropSpawnHeight},
            {"erosionRadius", p.erosionRadius},
            {"inertia", p.inertia},
            {"sedimentCapacityFactor", p.sedimentCapacityFactor},
            {"minSedimentCapacity", p.minSedimentCapacity},
            {"erodeSpeed", p.erodeSpeed},
            {"depositSpeed", p.depositSpeed},
            {"evaporateSpeed", p.evaporateSpeed},
            {"gravity", p.gravity},
            {"maxDropLifeTime", p.maxDropLifeTime},
            {"minDropDespawnHeight", p.minDropDespawnHeight},
            {"initialWaterVolume", p.initialWaterVolume},
            {"minWaterVolume", p.minWaterVolume},
            {"initialSpeed", p.initialSpeed},
            {"waterLevel", p.waterLevel},
            {"beachLevel", p.beachLevel},
            {"grassLevel", p.grassLevel},
            {"mountainLevel", p.mountainLevel},
            {"totalLayers", p.totalLayers},
            {"waterLayers", p.waterLayers},
            {"beachLayers", p.beachLayers},
            {"grassLayers", p.grassLayers},
            {"mountainLayers", p.mountainLayers},
            {"snowLayers", p.snowLayers},
            {"minWaterColor", p.minWaterColor},
            {"maxWaterColor", p.maxWaterColor},
            {"minBeachColor", p.minBeachColor},
            {"maxBeachColor", p.maxBeachColor},
            {"minGrassColor", p.minGrassColor},
            {"maxGrassColor", p.maxGrassColor},
            {"minMountainColor", p.minMountainColor},
            {"maxMountainColor", p.maxMountainColor},
            {"minSnowColor", p.minSnowColor},
            {"maxSnowColor", p.maxSnowColor}
    };
}

// Read map config from json
void from_json(const nlohmann::json& j, MapConfig& p) {
    j.at("mapWidth").get_to(p.mapWidth);
    j.at("mapHeight").get_to(p.mapHeight);
    j.at("channels").get_to(p.channels);
    j.at("seaLevel").get_to(p.seaLevel);

    j.at("frequency").get_to(p.frequency);
    j.at("octaves").get_to(p.octaves);
    j.at("lacunarity").get_to(p.lacunarity);
    j.at("persistence").get_to(p.persistence);
    j.at("distanceFromCenter").get_to(p.distanceFromCenter);

    j.at("riverCount").get_to(p.riverCount);
    j.at("maxRiverSpawnHeight").get_to(p.maxRiverSpawnHeight);
    j.at("minRiverSpawnHeight").get_to(p.minRiverSpawnHeight);
    j.at("minSearchRiverPointDistance").get_to(p.minSearchRiverPointDistance);
    j.at("minRiverDespawnHeight").get_to(p.minRiverDespawnHeight);
    j.at("randomPointSpacing").get_to(p.randomPointSpacing);
    j.at("maxDeviation").get_to(p.maxDeviation);
    j.at("minDeviation").get_to(p.minDeviation);
    j.at("riverStraightnessThreshold").get_to(p.riverStraightnessThreshold);
    j.at("minRiverLength").get_to(p.minRiverLength);
    j.at("intersectionRange").get_to(p.intersectionRange);
    j.at("startRadius").get_to(p.startRadius);
    j.at("maxRadius").get_to(p.maxRadius);
    j.at("sizeScaleRate").get_to(p.sizeScaleRate);
    j.at("maxRiverRadiusLength").get_to(p.maxRiverRadiusLength);
    j.at("riverMinDepth").get_to(p.riverMinDepth);
    j.at("riverMaxDepth").get_to(p.riverMaxDepth);
    j.at("terrainMinDepth").get_to(p.terrainMinDepth);
    j.at("terrainDistortion").get_to(p.terrainDistortion);
    j.at("startTerrainCarveRadius").get_to(p.startTerrainCarveRadius);
    j.at("maxTerrainCarveRadius").get_to(p.maxTerrainCarveRadius);
    j.at("sizeTerrainCarveScaleRate").get_to(p.sizeTerrainCarveScaleRate);

    j.at("numDrops").get_to(p.numDrops);
    j.at("minDropSpawnHeight").get_to(p.minDropSpawnHeight);
    j.at("erosionRadius").get_to(p.erosionRadius);
    j.at("inertia").get_to(p.inertia);
    j.at("sedimentCapacityFactor").get_to(p.sedimentCapacityFactor);
    j.at("minSedimentCapacity").get_to(p.minSedimentCapacity);
    j.at("erodeSpeed").get_to(p.erodeSpeed);
    j.at("depositSpeed").get_to(p.depositSpeed);
    j.at("evaporateSpeed").get_to(p.evaporateSpeed);
    j.at("gravity").get_to(p.gravity);
    j.at("maxDropLifeTime").get_to(p.maxDropLifeTime);
    j.at("minDropDespawnHeight").get_to(p.minDropDespawnHeight);
    j.at("initialWaterVolume").get_to(p.initialWaterVolume);
    j.at("minWaterVolume").get_to(p.minWaterVolume);
    j.at("initialSpeed").get_to(p.initialSpeed);

    j.at("waterLevel").get_to(p.waterLevel);
    j.at("beachLevel").get_to(p.beachLevel);
    j.at("grassLevel").get_to(p.grassLevel);
    j.at("mountainLevel").get_to(p.mountainLevel);
    j.at("totalLayers").get_to(p.totalLayers);
    j.at("waterLayers").get_to(p.waterLayers);
    j.at("beachLayers").get_to(p.beachLayers);
    j.at("grassLayers").get_to(p.grassLayers);
    j.at("mountainLayers").get_to(p.mountainLayers);
    j.at("snowLayers").get_to(p.snowLayers);

    j.at("minWaterColor").get_to(p.minWaterColor);
    j.at("maxWaterColor").get_to(p.maxWaterColor);
    j.at("minBeachColor").get_to(p.minBeachColor);
    j.at("maxBeachColor").get_to(p.maxBeachColor);
    j.at("minGrassColor").get_to(p.minGrassColor);
    j.at("maxGrassColor").get_to(p.maxGrassColor);
    j.at("minMountainColor").get_to(p.minMountainColor);
    j.at("maxMountainColor").get_to(p.maxMountainColor);
    j.at("minSnowColor").get_to(p.minSnowColor);
    j.at("maxSnowColor").get_to(p.maxSnowColor);
}

// Load map config
MapConfig loadMapConfig() {
    std::ifstream file("resources\\MapConfig.json");
    if(!file) {
        std::cerr << "Failed to open the file. Using default config." << std::endl;
        return MapConfig();
    } else {
        std::cout << "MapConfig.json read successfully." << std::endl;
        nlohmann::json j;
        file >> j;
        return j.get<MapConfig>();
    }
}