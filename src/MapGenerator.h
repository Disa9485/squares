#include <vector>
#include <cmath>

#ifndef SQUARES_MAP_GENERATOR_H
#define SQUARES_MAP_GENERATOR_H

// A struct to store color
struct MapColor {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

// Struct that stores map configuration
struct MapConfig {
    //// Map general config ////
    int mapWidth = 1500;
    int mapHeight = 1500;
    int channels = 3; // RGB channels
    float seaLevel = 0.4f; // Dictates where water ends and land starts

    //// Noise generation config ////
    float frequency = 0.25; // 0.25 frequency at 1000*1000, 0.15 at 2000*2000, 0.10 at 3000*3000, 0.08 at 4000*4000, 0.07 at 5000*5000
    int octaves = 12; // Default 12
    float lacunarity = 2.0f; // Default 2.0
    float persistence = 0.5f; // Default 0.5
    float distanceFromCenter = 0.8f; // Default 0.8

    //// River generation config ////
    // River spawning
    int riverCount = (mapWidth * mapHeight) / 10000; // Should be proportional to map size
    float maxRiverSpawnHeight = 0.75f; // Default: 0.85
    float minRiverSpawnHeight = 0.50f; // Default: 0.7
    // River path finding
    int minSearchRiverPointDistance = 20; // Default: 10
    float minRiverDespawnHeight = 0.30f; // Default: 0.30
    // River random point placement
    int randomPointSpacing = 10; // Default: 10
    float maxDeviation = 15.0; // Default: 15.0
    float minDeviation = 3.0; // Default: 3.0
    // River filtering
    int riverStraightnessThreshold = 15; // Default: 10
    int minRiverLength = 0; // Default: 10 (This can lead to rivers that go nowhere)
    // River intersection stitching
    int intersectionRange = 10; // Default: 10
    // River carving
    float startRadius = 1.0f; // Default: 1.0
    float maxRadius = 5.0f; // Default: 5.0
    float sizeScaleRate = 1.5f; // Default: 1.5
    int maxRiverRadiusLength = 300; // Default: 300
    float riverMinDepth = 0.39f; // Default: 0.39
    float riverMaxDepth = 0.37f; // Default: 0.37
    float terrainMinDepth = 0.40f; // Default: 0.40
    float terrainDistortion = 0.025f; // Default: 0.025
    float startTerrainCarveRadius = 1.5f; // Default: 1.5
    float maxTerrainCarveRadius = 8.0f; // Default: 8.0
    float sizeTerrainCarveScaleRate = 1.5f; // Default: 1.25

    // Erode heightmap parameters
    int numDrops = mapWidth * mapHeight / 3; // Should be proportional to map size
    float minDropSpawnHeight = 0.5; // Default: 0.5
    int erosionRadius = 3; // Default: 3
    float inertia = 0.05f; // Default: 0.05
    float sedimentCapacityFactor = 4.0f; // Default: 4.0
    float minSedimentCapacity = 0.01f; // Default: 0.01
    float erodeSpeed = 0.3f; // Default: 0.3
    float depositSpeed = 0.3f; // Default: 0.3
    float evaporateSpeed = 0.01f; // Default: 0.01
    float gravity = 4.0f; // Default: 4.0
    int maxDropLifeTime = 100; // Default: 30
    float minDropDespawnHeight = 0.4f; // Default: 0.35;
    float initialWaterVolume = 1.0f; // Default: 1.0
    float minWaterVolume = 0.01f; // Default 0.01
    float initialSpeed = 1.0f; // Default: 1.0

    // Global level variables (0.0 - 1.0) Note: Should be at least 0.05 thick
    float waterLevel = seaLevel;
    float beachLevel = seaLevel + 0.02;
    float grassLevel = 0.70f;
    float mountainLevel = 0.80f;

    // Number of layers for each level
    float totalLayers = 40; // Combined level layers
    int waterLayers = round(waterLevel * totalLayers);
    int beachLayers = round((beachLevel - waterLevel) * totalLayers);
    int grassLayers = round((grassLevel - beachLevel) * totalLayers);
    int mountainLayers = round((mountainLevel - grassLevel) * totalLayers);
    int snowLayers = round((1.0 - mountainLevel) * totalLayers);

    // Maximum and minimum RGB colors for each level
    MapColor minWaterColor = {37, 89, 134};
    MapColor maxWaterColor = {140, 183, 220};

    MapColor minBeachColor = {230, 218, 166};
    MapColor maxBeachColor = {206, 190, 124};

    MapColor minGrassColor = {180, 218, 91};
    MapColor maxGrassColor = {93, 121, 30};

    MapColor minMountainColor = {92, 88, 70};
    MapColor maxMountainColor = {144, 142, 136};

    MapColor minSnowColor = {223, 222, 216};
    MapColor maxSnowColor = {245, 244, 239};
};

// Struct that stores map data
struct MapData {
    MapConfig mapConfig;
    std::vector<float> heightMap;  // Store the world map height map
    std::vector<unsigned char> grayscaleMap; // Store the world map with color
    std::vector<unsigned char> colorMap; // Store the world map with color
};

MapData generate(MapConfig mapConfig);

bool saveColorAndHeightMaps(MapData mapData);

#endif
